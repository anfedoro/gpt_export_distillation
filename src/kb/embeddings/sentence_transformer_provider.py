from __future__ import annotations

import logging
import os

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
        logger.debug("sparse encode_document start batch_size=%d", len(texts))
        with _inference_mode():
            embeddings = self._model.encode_document(texts)
            logger.debug("sparse encode_document done batch_size=%d embedding_type=%s", len(texts), type(embeddings).__name__)
            decoded = self._model.decode(embeddings, top_k=self.top_k)
            logger.debug("sparse decode done batch_size=%d top_k=%d", len(texts), self.top_k)
        terms_by_text = [
            {token: float(weight) for token, weight in terms}
            for terms in decoded
        ]
        del embeddings, decoded
        logger.debug("sparse terms materialized batch_size=%d", len(texts))
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
