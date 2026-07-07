from __future__ import annotations

import logging
import gc
import os
import resource
import subprocess
import sys

from kb.embeddings.base import DenseEmbeddingProvider, SparseEmbeddingProvider


logger = logging.getLogger(__name__)


class SentenceTransformerDenseProvider(DenseEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        backend: str = "torch",
        torch_dtype: str | None = None,
        torch_compile: bool = False,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Avoid a non-daemon Transformers safetensors auto-conversion thread that can block CLI shutdown.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the KB extra with "
                "`uv sync --extra kb` or run with `uv run --extra kb ...`."
            ) from exc
        model_kwargs = _model_kwargs(backend=backend, torch_dtype=torch_dtype)
        self.model_name = model_name
        self.model_version = _model_version(
            "sentence-transformers",
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
            torch_compile=torch_compile,
        )
        logger.info(
            "loading dense sentence-transformers model name=%s device=%s backend=%s dtype=%s compile=%s",
            model_name,
            device,
            backend,
            torch_dtype or "auto",
            torch_compile,
        )
        self._model = SentenceTransformer(model_name, device=device, backend=backend, model_kwargs=model_kwargs)
        _compile_model(self._model, backend=backend, enabled=torch_compile)
        logger.info("loaded dense sentence-transformers model name=%s", model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.debug("dense encode start batch_size=%d", len(texts))
        with _inference_mode():
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        logger.debug("dense encode done batch_size=%d", len(texts))
        return [vector.astype(float).tolist() for vector in vectors]


class SentenceTransformerSparseProvider(SparseEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        *,
        top_k: int = 128,
        device: str | None = None,
        backend: str = "torch",
        torch_dtype: str | None = None,
        torch_compile: bool = False,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Avoid a non-daemon Transformers safetensors auto-conversion thread that can block CLI shutdown.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers with SparseEncoder is not installed. Install the KB extra with "
                "`uv sync --extra kb` or run with `uv run --extra kb ...`."
            ) from exc
        model_kwargs = _model_kwargs(backend=backend, torch_dtype=torch_dtype)
        self.model_name = model_name
        self.model_version = _model_version(
            "sentence-transformers-sparse",
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
            torch_compile=torch_compile,
        )
        self.top_k = top_k
        logger.info(
            "loading sparse sentence-transformers model name=%s device=%s backend=%s dtype=%s compile=%s top_k=%d",
            model_name,
            device,
            backend,
            torch_dtype or "auto",
            torch_compile,
            top_k,
        )
        self._model = SparseEncoder(model_name, device=device, backend=backend, model_kwargs=model_kwargs)
        _compile_model(self._model, backend=backend, enabled=torch_compile)
        logger.info("loaded sparse sentence-transformers model name=%s", model_name)

    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        _log_memory("sparse encode_document start", batch_size=len(texts))
        with _inference_mode():
            embeddings = self._model.encode_document(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_tensor=True,
                convert_to_sparse_tensor=False,
            )
            _log_memory(
                "sparse encode_document done",
                batch_size=len(texts),
                embedding_type=type(embeddings).__name__,
            )
            decoded = self._model.decode(embeddings, top_k=self.top_k)
            _log_memory("sparse decode done", batch_size=len(texts), top_k=self.top_k)
        terms_by_text = [
            {token: float(weight) for token, weight in terms}
            for terms in decoded
        ]
        _log_memory("sparse terms materialized", batch_size=len(texts))
        del embeddings, decoded
        _release_torch_memory()
        _log_memory("sparse tensors deleted", batch_size=len(texts))
        return terms_by_text


def _inference_mode():
    try:
        import torch

        return torch.inference_mode()
    except Exception:  # noqa: BLE001
        from contextlib import nullcontext

        return nullcontext()


def _model_kwargs(*, backend: str, torch_dtype: str | None) -> dict | None:
    if not torch_dtype or torch_dtype == "auto":
        return None
    if backend != "torch":
        raise ValueError("--*-torch-dtype is only supported with the torch backend.")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required when --*-torch-dtype is set.") from exc
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return {"torch_dtype": dtype_map[torch_dtype]}


def _compile_model(model, *, backend: str, enabled: bool) -> None:
    if not enabled:
        return
    if backend != "torch":
        raise ValueError("--*-torch-compile is only supported with the torch backend.")
    try:
        model.compile()
    except AttributeError as exc:
        raise RuntimeError("This sentence-transformers model does not expose torch compile support.") from exc


def _model_version(
    prefix: str,
    *,
    backend: str,
    device: str | None,
    torch_dtype: str | None,
    torch_compile: bool = False,
) -> str:
    parts = [prefix, f"backend={backend}"]
    if device:
        parts.append(f"device={device}")
    if torch_dtype and torch_dtype != "auto":
        parts.append(f"dtype={torch_dtype}")
    if torch_compile:
        parts.append("compile=true")
    return ";".join(parts)


def _log_memory(event: str, **fields: object) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    stats = _memory_stats()
    payload = " ".join(
        f"{key}={value}" for key, value in {**fields, **stats}.items() if value is not None
    )
    logger.debug("%s %s", event, payload)


def _memory_stats() -> dict[str, float | None]:
    rss_mb = None
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        rss_mb = _rss_mb_from_ps()
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    max_rss_mb = max_rss / (1024 * 1024) if sys.platform == "darwin" else max_rss / 1024
    stats: dict[str, float | None] = {
        "rss_mb": round(rss_mb, 1) if rss_mb is not None else None,
        "max_rss_mb": round(max_rss_mb, 1),
        "mps_current_mb": None,
        "mps_driver_mb": None,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
    }
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            stats["mps_current_mb"] = round(torch.mps.current_allocated_memory() / (1024 * 1024), 1)
            if hasattr(torch.mps, "driver_allocated_memory"):
                stats["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 1)
        if torch.cuda.is_available():
            stats["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            stats["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        logger.debug("torch memory stats unavailable", exc_info=True)
    return stats


def _rss_mb_from_ps() -> float | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        return float(output) / 1024
    except Exception:  # noqa: BLE001
        return None


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        return
