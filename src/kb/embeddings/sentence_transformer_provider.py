from __future__ import annotations

import logging
import gc
import os
import resource
import subprocess
import sys

from kb.embeddings.base import DenseEmbeddingProvider, EmbeddingProviderContract, SparseEmbeddingProvider


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
        effective_max_seq_length: int | None = None,
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
        self.embedding_space_id = _dense_embedding_space_id(
            model_name,
            normalize_embeddings=True,
            output_dim=None,
        )
        self.runtime_metadata = _runtime_metadata(
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
        self._tokenizer = getattr(self._model, "tokenizer", None)
        st_max_seq_length = int(getattr(self._model, "max_seq_length", 256) or 256)
        self.effective_max_sequence_length = int(effective_max_seq_length or st_max_seq_length)
        if effective_max_seq_length is not None:
            self._model.max_seq_length = self.effective_max_sequence_length
        if hasattr(self._model, "get_embedding_dimension"):
            output_dim = self._model.get_embedding_dimension()
        else:
            output_dim = self._model.get_sentence_embedding_dimension()
        if output_dim is not None:
            self.embedding_space_id = _dense_embedding_space_id(
                model_name,
                normalize_embeddings=True,
                output_dim=int(output_dim),
                max_seq_length=self.effective_max_sequence_length,
                max_seq_override=effective_max_seq_length,
            )
        self.provider_contract = _provider_contract(
            model_name=model_name,
            model=self._model,
            tokenizer=self._tokenizer,
            embedding_dimension=int(output_dim) if output_dim is not None else None,
            effective_max_seq_length=self.effective_max_sequence_length,
            max_seq_length_override=effective_max_seq_length,
            document_prefix=self.document_prefix,
            query_prefix=self.query_prefix,
            safety_reserve=4,
            token_counter=self.token_count,
        )
        _compile_model(self._model, backend=backend, enabled=torch_compile)
        logger.info("loaded dense sentence-transformers model name=%s", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.debug("dense encode start batch_size=%d", len(texts))
        with _inference_mode():
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        logger.debug("dense encode done batch_size=%d", len(texts))
        result = [vector.astype(float).tolist() for vector in vectors]
        return result

    def token_count(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError(f"Dense provider {self.model_name} does not expose a tokenizer.")
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def fits_token_budget(self, text: str, budget: int) -> bool:
        if self._tokenizer is None:
            raise RuntimeError(f"Dense provider {self.model_name} does not expose a tokenizer.")
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=True, max_length=budget + 1)
        return len(encoded["input_ids"]) <= budget

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        if self._tokenizer is None or not getattr(self._tokenizer, "is_fast", False):
            raise RuntimeError(f"Dense provider {self.model_name} does not expose a fast tokenizer with offsets.")
        encoded = self._tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        return [(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]


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
        effective_max_seq_length: int | None = None,
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
        self.embedding_space_id = _sparse_embedding_space_id(
            model_name,
            top_k=top_k,
            document_encoder="encode_document",
            query_encoder="encode_query",
        )
        self.runtime_metadata = _runtime_metadata(
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
        self._tokenizer = getattr(self._model, "tokenizer", None)
        st_max_seq_length = int(getattr(self._model, "max_seq_length", 256) or 256)
        self.effective_max_sequence_length = int(effective_max_seq_length or st_max_seq_length)
        self.embedding_space_id = _sparse_embedding_space_id(
            model_name,
            top_k=top_k,
            document_encoder="encode_document",
            query_encoder="encode_query",
            max_seq_length=self.effective_max_sequence_length,
            max_seq_override=effective_max_seq_length,
        )
        self.provider_contract = _provider_contract(
            model_name=model_name,
            model=self._model,
            tokenizer=self._tokenizer,
            embedding_dimension=None,
            effective_max_seq_length=self.effective_max_sequence_length,
            max_seq_length_override=effective_max_seq_length,
            document_prefix=self.document_prefix,
            query_prefix=self.query_prefix,
            safety_reserve=4,
            token_counter=self.token_count,
        )
        _compile_model(self._model, backend=backend, enabled=torch_compile)
        logger.info("loaded sparse sentence-transformers model name=%s", model_name)

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
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

    def embed_query(self, query: str) -> dict[str, float]:
        logger.debug("sparse encode_query start")
        with _inference_mode():
            embeddings = self._model.encode_query(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_tensor=True,
                convert_to_sparse_tensor=False,
            )
            decoded = self._model.decode(embeddings, top_k=self.top_k)
        terms = {token: float(weight) for token, weight in decoded[0]}
        del embeddings, decoded
        _release_torch_memory()
        logger.debug("sparse encode_query done terms=%d", len(terms))
        return terms

    def token_count(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError(f"Sparse provider {self.model_name} does not expose a tokenizer.")
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def fits_token_budget(self, text: str, budget: int) -> bool:
        if self._tokenizer is None:
            raise RuntimeError(f"Sparse provider {self.model_name} does not expose a tokenizer.")
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=True, max_length=budget + 1)
        return len(encoded["input_ids"]) <= budget

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        if self._tokenizer is None or not getattr(self._tokenizer, "is_fast", False):
            raise RuntimeError(f"Sparse provider {self.model_name} does not expose a fast tokenizer with offsets.")
        encoded = self._tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        return [(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]


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


def _dense_embedding_space_id(
    model_name: str,
    *,
    normalize_embeddings: bool,
    output_dim: int | None,
    max_seq_length: int | None = None,
    max_seq_override: int | None = None,
) -> str:
    parts = [
        "sentence-transformers-dense",
        f"model={model_name}",
        "pooling=model-default",
        f"normalize={str(normalize_embeddings).lower()}",
        "query_document=symmetric",
    ]
    if output_dim is not None:
        parts.append(f"dim={output_dim}")
    if max_seq_length is not None:
        parts.append(f"max_seq={max_seq_length}")
    if max_seq_override is not None:
        parts.append(f"max_seq_override={max_seq_override}")
    return ";".join(parts)


def _sparse_embedding_space_id(
    model_name: str,
    *,
    top_k: int,
    document_encoder: str,
    query_encoder: str,
    max_seq_length: int | None = None,
    max_seq_override: int | None = None,
) -> str:
    parts = [
            "sentence-transformers-sparse",
            f"model={model_name}",
            f"document_encoder={document_encoder}",
            f"query_encoder={query_encoder}",
            f"top_k={top_k}",
    ]
    if max_seq_length is not None:
        parts.append(f"max_seq={max_seq_length}")
    if max_seq_override is not None:
        parts.append(f"max_seq_override={max_seq_override}")
    return ";".join(parts)


def _provider_contract(
    *,
    model_name: str,
    model,
    tokenizer,
    embedding_dimension: int | None,
    effective_max_seq_length: int,
    max_seq_length_override: int | None,
    document_prefix: str,
    query_prefix: str,
    safety_reserve: int,
    token_counter,
) -> EmbeddingProviderContract:
    tokenizer_name = getattr(tokenizer, "name_or_path", None) if tokenizer is not None else None
    tokenizer_limit = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 1_000_000_000:
        tokenizer_limit = None
    st_limit = int(getattr(model, "max_seq_length", effective_max_seq_length) or effective_max_seq_length)
    backbone_limit = _backbone_max_position_embeddings(model)
    special_token_overhead = int(token_counter(""))
    return EmbeddingProviderContract(
        model_name=model_name,
        model_revision=None,
        embedding_dimension=embedding_dimension,
        tokenizer_name=str(tokenizer_name) if tokenizer_name else None,
        tokenizer_model_max_length=int(tokenizer_limit) if isinstance(tokenizer_limit, int) else None,
        backbone_max_position_embeddings=backbone_limit,
        sentence_transformer_max_seq_length=st_limit,
        configured_effective_max_seq_length=effective_max_seq_length,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
        special_token_overhead=special_token_overhead,
        configured_safety_reserve=safety_reserve,
        computed_content_budget=max(0, effective_max_seq_length - special_token_overhead - safety_reserve),
        max_seq_length_override=max_seq_length_override,
    )


def _backbone_max_position_embeddings(model) -> int | None:
    candidates = []
    for attr in ("_first_module",):
        try:
            module = getattr(model, attr)()
            candidates.append(module)
        except Exception:  # noqa: BLE001
            pass
    candidates.extend(getattr(model, "_modules", {}).values() if hasattr(model, "_modules") else [])
    for candidate in candidates:
        auto_model = getattr(candidate, "auto_model", None)
        config = getattr(auto_model, "config", None)
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int):
            return value
    return None


def _runtime_metadata(
    *,
    backend: str,
    device: str | None,
    torch_dtype: str | None,
    torch_compile: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {"backend": backend, "torch_compile": torch_compile}
    if device:
        metadata["device"] = device
    if torch_dtype and torch_dtype != "auto":
        metadata["torch_dtype"] = torch_dtype
    return metadata


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
