"""MLX BGE-M3 dense and lexical embedding providers."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from kb.embeddings.base import (
    DenseEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingProviderContract,
    EmbeddingResult,
    SparseEmbeddingProvider,
)

MLX_EMBEDDINGS_REVISION = "4a8277aa523eb34ff29a5a832fa3f3f654336b54"


def _cached_snapshot(
    repo_id: str,
    revision: str,
    required: set[str],
    *,
    cache_dir: Path | None,
    try_to_load_from_cache: Any,
) -> Path | None:
    """Return a complete cached snapshot without invoking Hub download logic."""
    cached_paths: list[Path] = []
    for filename in sorted(required):
        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        if not isinstance(cached, (str, Path)):
            return None
        path = Path(cached)
        if not path.is_file():
            return None
        cached_paths.append(path)
    if not cached_paths or any(path.parent != cached_paths[0].parent for path in cached_paths):
        return None
    return cached_paths[0].parent


class MlxBgeM3Provider(EmbeddingProvider):
    """Produce BGE-M3 dense and lexical vectors from one MLX forward."""

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str,
        dtype: str = "float16",
        device: str = "gpu",
        max_seq_length: int = 512,
        sparse_top_k: int = 128,
        batch_size: int = 4,
        max_padded_tokens: int | None = None,
        sparse_head: str = "sparse_linear.safetensors",
        colbert_head: str = "colbert_linear.safetensors",
        model_cache: Path | None = None,
    ) -> None:
        if not model_revision:
            raise ValueError("MLX model revision must be pinned to a Hugging Face commit SHA.")
        if dtype != "float16":
            raise ValueError("Production MLX BGE-M3 supports embedding_dtype=float16 only.")
        if device != "gpu":
            raise ValueError("Production MLX BGE-M3 supports embedding_device=gpu only.")
        if batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive.")
        if max_padded_tokens is not None and max_padded_tokens <= 0:
            raise ValueError("embedding_max_padded_tokens must be positive when set.")

        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download, try_to_load_from_cache
            from huggingface_hub.utils import disable_progress_bars
            from mlx_embeddings.sparse import load_sparse_linear
            from mlx_embeddings.utils import load_model, load_tokenizer
        except ImportError as exc:
            raise RuntimeError("MLX embedding runtime is unavailable; install PTHA on Apple Silicon macOS.") from exc

        if not hasattr(mx.fast, "scaled_dot_product_attention"):
            raise RuntimeError("MLX fused scaled-dot-product attention is unavailable.")
        mx.set_default_device(mx.gpu)
        required = {
            "model.safetensors",
            sparse_head,
            colbert_head,
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        local_model = Path(model_name).expanduser()
        if local_model.is_dir():
            snapshot = local_model.resolve()
            source = "local model directory"
            self._announce_model(model_name, model_revision, source, snapshot)
        else:
            snapshot = _cached_snapshot(
                model_name,
                model_revision,
                required,
                cache_dir=model_cache,
                try_to_load_from_cache=try_to_load_from_cache,
            )
            if snapshot is not None:
                source = "local Hugging Face cache"
                self._announce_model(model_name, model_revision, source, snapshot)
            else:
                self._announce_model(model_name, model_revision, "Hugging Face Hub", None, downloading=True)
                # The application owns the single indexing progress bar. Hugging Face's
                # per-file bars otherwise interleave with it and create broken terminal output.
                disable_progress_bars()
                snapshot = Path(snapshot_download(
                    repo_id=model_name,
                    revision=model_revision,
                    cache_dir=str(model_cache) if model_cache else None,
                ))
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if missing:
            raise RuntimeError(f"MLX BGE-M3 artifact is incomplete; missing: {', '.join(missing)}")

        self.model_name = model_name
        self.model_revision = model_revision
        self.model_path = snapshot
        self.max_seq_length = max_seq_length
        self.sparse_top_k = sparse_top_k
        self.batch_size = batch_size
        self.max_padded_tokens = max_padded_tokens
        self.device = device
        self.model = load_model(snapshot, path_to_repo=model_name)
        self.tokenizer = load_tokenizer(snapshot)
        self.sparse_weight, self.sparse_bias = load_sparse_linear(snapshot / sparse_head)
        self.dimension = int(self.model.config.hidden_size)
        self.forward_calls = 0
        self.last_batch_metrics: dict[str, float | int] = {}
        self.special_ids = {
            value for value in (
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ) if value is not None
        }
        self._validate_runtime(mx)
        print(
            "  status: ready\n"
            f"  dimension: {self.dimension}\n"
            "  precision: float16\n"
            "  quantization: none\n"
            "  attention: fused scaled-dot-product\n"
            "  representations: dense + sparse from one backbone forward",
            file=sys.stderr,
            flush=True,
        )
        self.runtime_metadata = {
            "backend": "mlx-bge-m3",
            "framework": "mlx",
            "device": "gpu",
            "dtype": "float16",
            "attention": "fused-scaled-dot-product-attention",
            "model_revision": model_revision,
            "mlx_embeddings_revision": MLX_EMBEDDINGS_REVISION,
            "shared_backbone": True,
            "batch_size": batch_size,
            "max_padded_tokens": max_padded_tokens,
        }

    @staticmethod
    def _announce_model(
        model_name: str,
        model_revision: str,
        source: str,
        snapshot: Path | None,
        *,
        downloading: bool = False,
    ) -> None:
        print(
            "PTHA embedding model\n"
            f"  repository: {model_name}\n"
            f"  revision: {model_revision}\n"
            f"  source: {source}\n"
            f"  cache: {'not found' if downloading else 'found'}\n"
            f"  action: {'downloading model artifacts' if downloading else 'using installed model'}"
            + (f"\n  path: {snapshot}" if snapshot is not None else ""),
            file=sys.stderr,
            flush=True,
        )

    def _validate_runtime(self, mx: Any) -> None:
        from mlx.utils import tree_flatten

        if self.dimension != 1024:
            raise RuntimeError(f"Expected BGE-M3 hidden size 1024, received {self.dimension}.")
        if tuple(self.sparse_weight.shape) != (self.dimension, 1) or self.sparse_bias.size != 1:
            raise RuntimeError("Sparse head is incompatible with the BGE-M3 hidden size.")
        parameters = [value for _, value in tree_flatten(self.model.parameters())]
        if not parameters or any(value.dtype != mx.float16 for value in parameters):
            raise RuntimeError("Configured FP16 MLX checkpoint contains non-FP16 or quantized backbone weights.")
        if self.sparse_weight.dtype != mx.float16 or self.sparse_bias.dtype != mx.float16:
            raise RuntimeError("Configured FP16 MLX checkpoint contains a non-FP16 sparse head.")
        vocab_size = int(getattr(self.model.config, "vocab_size", 0))
        tokenizer_size = len(getattr(self.tokenizer, "_tokenizer", self.tokenizer))
        if vocab_size != tokenizer_size:
            raise RuntimeError(f"Tokenizer vocabulary ({tokenizer_size}) does not match model ({vocab_size}).")
        positions = int(getattr(self.model.config, "max_position_embeddings", 0))
        if positions < self.max_seq_length:
            raise RuntimeError(f"Model supports {positions} positions, below configured {self.max_seq_length}.")

    def embed_batch(self, texts: Sequence[str]) -> Sequence[EmbeddingResult]:
        import mlx.core as mx
        from mlx_embeddings.sparse import sparse_token_weights

        if not texts:
            return []
        started = time.perf_counter()
        encoded = [
            self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_length)
            for text in texts
        ]
        order = sorted(range(len(encoded)), key=lambda index: (len(encoded[index]), index))
        restored: list[EmbeddingResult | None] = [None] * len(encoded)
        real_tokens = padded_tokens = 0
        cursor = 0
        while cursor < len(order):
            end = min(cursor + self.batch_size, len(order))
            if self.max_padded_tokens is not None:
                while end > cursor + 1:
                    longest = len(encoded[order[end - 1]])
                    if longest * (end - cursor) <= self.max_padded_tokens:
                        break
                    end -= 1
            indices = order[cursor:end]
            max_length = max(len(encoded[index]) for index in indices)
            pad_id = int(self.tokenizer.pad_token_id)
            ids = [encoded[index] + [pad_id] * (max_length - len(encoded[index])) for index in indices]
            mask = [[1] * len(encoded[index]) + [0] * (max_length - len(encoded[index])) for index in indices]
            input_ids = mx.array(ids, dtype=mx.int32)
            attention_mask = mx.array(mask, dtype=mx.int32)
            output = self.model(input_ids, attention_mask=attention_mask)
            self.forward_calls += 1
            hidden = output.last_hidden_state
            dense = hidden[:, 0, :]
            dense = dense / mx.maximum(mx.sqrt(mx.sum(dense * dense, axis=-1, keepdims=True)), mx.array(1e-12))
            sparse = sparse_token_weights(hidden, self.sparse_weight, self.sparse_bias, attention_mask)
            mx.eval(dense, sparse)
            dense_values = dense.astype(mx.float32).tolist()
            sparse_values = sparse.astype(mx.float32).tolist()
            for local, original in enumerate(indices):
                restored[original] = EmbeddingResult(
                    dense=[float(value) for value in dense_values[local]],
                    sparse=self._aggregate_sparse(
                        encoded[original], sparse_values[local][: len(encoded[original])]
                    ),
                )
            real_tokens += sum(len(encoded[index]) for index in indices)
            padded_tokens += max_length * len(indices)
            cursor = end
        elapsed = time.perf_counter() - started
        self.last_batch_metrics = {
            "chunks": len(texts),
            "real_tokens": real_tokens,
            "padded_tokens": padded_tokens,
            "padding_efficiency": real_tokens / padded_tokens if padded_tokens else 1.0,
            "tokens_per_second": real_tokens / elapsed if elapsed else 0.0,
            "chunks_per_second": len(texts) / elapsed if elapsed else 0.0,
            "seconds": elapsed,
        }
        return [result for result in restored if result is not None]

    def _aggregate_sparse(self, token_ids: Sequence[int], weights: Sequence[float]) -> dict[str, float]:
        by_id: dict[int, float] = {}
        for token_id, weight in zip(token_ids, weights, strict=True):
            if token_id in self.special_ids or weight <= 0 or not math.isfinite(weight):
                continue
            by_id[token_id] = max(by_id.get(token_id, 0.0), float(weight))
        strongest = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))[: self.sparse_top_k]
        return {self.tokenizer.convert_ids_to_tokens(token_id): weight for token_id, weight in strongest}

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def fits_token_budget(self, text: str, budget: int) -> bool:
        return self.token_count(text) <= budget

    def diagnostic_snapshot(self) -> dict[str, Any]:
        return {**self.runtime_metadata, "model_dtype": "float16", "sparse_head_dtype": "float16"}


class BgeM3DenseProvider(DenseEmbeddingProvider):
    def __init__(self, backend: MlxBgeM3Provider) -> None:
        self.backend = backend
        self.model_name = backend.model_name
        self.effective_max_sequence_length = backend.max_seq_length
        self.embedding_space_id = (
            f"bge-m3-dense;model={self.model_name};revision={backend.model_revision};dim={backend.dimension};"
            f"normalize=true;max_seq={backend.max_seq_length}"
        )
        self.runtime_metadata = backend.runtime_metadata
        self.provider_contract = _contract(backend, embedding_dimension=backend.dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [result.dense for result in self.backend.embed_batch(texts)]

    def token_count(self, text: str) -> int:
        return self.backend.token_count(text)

    def fits_token_budget(self, text: str, budget: int) -> bool:
        return self.backend.fits_token_budget(text, budget)


class BgeM3SparseProvider(SparseEmbeddingProvider):
    def __init__(self, backend: MlxBgeM3Provider) -> None:
        self.backend = backend
        self.model_name = backend.model_name
        self.effective_max_sequence_length = backend.max_seq_length
        self.embedding_space_id = (
            f"bge-m3-lexical;model={self.model_name};revision={backend.model_revision};"
            f"top_k={backend.sparse_top_k};max_seq={backend.max_seq_length}"
        )
        self.runtime_metadata = backend.runtime_metadata
        self.provider_contract = _contract(backend, embedding_dimension=None)

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        return [result.sparse for result in self.backend.embed_batch(texts)]

    def embed_query(self, query: str) -> dict[str, float]:
        return self.embed_documents([query])[0]

    def token_count(self, text: str) -> int:
        return self.backend.token_count(text)

    def fits_token_budget(self, text: str, budget: int) -> bool:
        return self.backend.fits_token_budget(text, budget)


def build_bge_m3_providers(
    model_name: str,
    *,
    model_revision: str,
    device: str = "gpu",
    dtype: str = "float16",
    max_seq_length: int = 512,
    sparse_top_k: int = 128,
    batch_size: int = 4,
    max_padded_tokens: int | None = None,
    sparse_head: str = "sparse_linear.safetensors",
    colbert_head: str = "colbert_linear.safetensors",
    model_cache: Path | None = None,
) -> tuple[BgeM3DenseProvider, BgeM3SparseProvider]:
    backend = MlxBgeM3Provider(
        model_name,
        model_revision=model_revision,
        device=device,
        dtype=dtype,
        max_seq_length=max_seq_length,
        sparse_top_k=sparse_top_k,
        batch_size=batch_size,
        max_padded_tokens=max_padded_tokens,
        sparse_head=sparse_head,
        colbert_head=colbert_head,
        model_cache=model_cache,
    )
    return BgeM3DenseProvider(backend), BgeM3SparseProvider(backend)


def embed_joint_documents(
    dense_provider: Any,
    sparse_provider: Any,
    texts: list[str],
) -> tuple[list[list[float]], list[dict[str, float]]]:
    """Use one shared backend forward while retaining separate storage contracts."""
    dense_backend = getattr(dense_provider, "backend", None)
    sparse_backend = getattr(sparse_provider, "backend", None)
    if dense_backend is not None and dense_backend is sparse_backend and hasattr(dense_backend, "embed_batch"):
        results = dense_backend.embed_batch(texts)
        return [result.dense for result in results], [result.sparse for result in results]
    return dense_provider.embed_documents(texts), sparse_provider.embed_documents(texts)


def _contract(backend: MlxBgeM3Provider, *, embedding_dimension: int | None) -> EmbeddingProviderContract:
    overhead = backend.token_count("")
    return EmbeddingProviderContract(
        model_name=backend.model_name,
        model_revision=backend.model_revision,
        embedding_dimension=embedding_dimension,
        tokenizer_name=str(getattr(backend.tokenizer, "name_or_path", backend.model_name)),
        tokenizer_model_max_length=int(getattr(backend.tokenizer, "model_max_length", backend.max_seq_length)),
        backbone_max_position_embeddings=int(backend.model.config.max_position_embeddings),
        sentence_transformer_max_seq_length=None,
        configured_effective_max_seq_length=backend.max_seq_length,
        document_prefix="",
        query_prefix="",
        special_token_overhead=overhead,
        configured_safety_reserve=4,
        computed_content_budget=backend.max_seq_length - overhead - 4,
        max_seq_length_override=backend.max_seq_length,
    )
