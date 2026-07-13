"""Shared-backbone BGE-M3 dense and lexical embedding providers."""

from __future__ import annotations

import math
from typing import Any

from kb.embeddings.base import DenseEmbeddingProvider, EmbeddingProviderContract, SparseEmbeddingProvider


def resolve_embedding_device(requested: str | None) -> str:
    """Resolve one device for every BGE-M3 representation."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BgeM3Backend:
    """Load one BGE-M3 backbone and expose dense and lexical batch passes."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        torch_dtype: str | None = None,
        max_seq_length: int = 512,
        sparse_top_k: int = 128,
        model: Any | None = None,
        tokenizer: Any | None = None,
        sparse_linear: Any | None = None,
    ) -> None:
        import torch

        self.model_name = model_name
        self.device = resolve_embedding_device(device)
        self.max_seq_length = max_seq_length
        self.sparse_top_k = sparse_top_k
        dtype = _resolve_dtype(torch_dtype, self.device)
        self.from_pretrained_dtype = dtype
        self.sparse_checkpoint_dtype = None
        if model is None or tokenizer is None or sparse_linear is None:
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, dtype=dtype)
            sparse_linear = torch.nn.Linear(model.config.hidden_size, 1, dtype=dtype)
            sparse_path = hf_hub_download(model_name, "sparse_linear.pt")
            state = torch.load(sparse_path, map_location="cpu", weights_only=True)
            self.sparse_checkpoint_dtype = str(next(iter(state.values())).dtype).removeprefix("torch.")
            sparse_linear.load_state_dict(state)
        self.tokenizer = tokenizer
        self.model = model.eval().to(self.device)
        self.sparse_linear = sparse_linear.eval().to(self.device)
        self.dimension = int(self.model.config.hidden_size)
        self.runtime_metadata = {
            "backend": "transformers-bge-m3",
            "device": self.device,
            "torch_dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "shared_backbone": True,
        }
        self.special_ids = {
            value for value in (
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ) if value is not None
        }

    def diagnostic_snapshot(self) -> dict[str, Any]:
        """Return runtime facts without exposing model weights or input content."""
        import torch

        def dtype_of(value: Any) -> str | None:
            if isinstance(value, torch.dtype):
                return str(value).removeprefix("torch.")
            dtype = getattr(value, "dtype", None)
            return str(dtype).removeprefix("torch.") if dtype is not None else None

        def module_dtype(module: Any) -> str | None:
            try:
                return dtype_of(next(module.parameters()))
            except (StopIteration, AttributeError, TypeError):
                return None

        embeddings = getattr(self.model, "embeddings", None)
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        transformer_layer = layers[0] if layers else None
        dense_projection = getattr(self.model, "pooler", None)
        autocast_enabled = bool(torch.is_autocast_enabled())
        try:
            autocast_enabled = autocast_enabled or bool(torch.is_autocast_enabled(self.device))
        except (TypeError, RuntimeError):
            pass
        autocast_dtype = None
        if autocast_enabled and hasattr(torch, "get_autocast_dtype"):
            try:
                autocast_dtype = dtype_of(torch.get_autocast_dtype(self.device))
            except (TypeError, RuntimeError):
                autocast_dtype = None
        return {
            "framework": "pytorch",
            "device": self.device,
            "model_device": str(next(self.model.parameters()).device),
            "input_device": self.device,
            "sparse_head_device": str(next(self.sparse_linear.parameters()).device),
            "model_dtype": dtype_of(next(self.model.parameters())),
            "from_pretrained_dtype": dtype_of(self.from_pretrained_dtype),
            "from_pretrained_dtype_argument": "dtype",
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": autocast_dtype,
            "layers": {
                "embeddings": module_dtype(embeddings),
                "transformer_layer_0": module_dtype(transformer_layer),
                "dense_projection": module_dtype(dense_projection),
                "dense_projection_description": "none; CLS pooling + L2 normalization"
                if dense_projection is None else type(dense_projection).__name__,
                "dense_projection_used_by_pipeline": False,
                "sparse_head": module_dtype(self.sparse_linear),
            },
            "sparse_checkpoint_dtype": self.sparse_checkpoint_dtype,
            "model_eval": not self.model.training,
            "sparse_head_eval": not self.sparse_linear.training,
            "inference_mode_in_provider": True,
        }

    def dense_documents(self, texts: list[str]) -> list[list[float]]:
        import torch

        inputs = self._inputs(texts)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            vectors = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
        return vectors.float().cpu().numpy().tolist()

    def sparse_documents(self, texts: list[str]) -> list[dict[str, float]]:
        import torch

        inputs = self._inputs(texts)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu()
            input_ids = inputs["input_ids"].cpu()
            attention = inputs["attention_mask"].cpu()
        return [self._lexical(ids, values, mask) for ids, values, mask in zip(input_ids, weights, attention, strict=True)]

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def fits_token_budget(self, text: str, budget: int) -> bool:
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=budget + 1)
        return len(encoded["input_ids"]) <= budget

    def _inputs(self, texts: list[str]) -> dict[str, Any]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

    def _lexical(self, ids: Any, weights: Any, attention: Any) -> dict[str, float]:
        by_id: dict[int, float] = {}
        for token_id, weight, active in zip(ids.tolist(), weights.tolist(), attention.tolist(), strict=True):
            if not active or token_id in self.special_ids or weight <= 0 or not math.isfinite(weight):
                continue
            by_id[token_id] = max(by_id.get(token_id, 0.0), float(weight))
        strongest = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))[: self.sparse_top_k]
        return {self.tokenizer.convert_ids_to_tokens(token_id): weight for token_id, weight in strongest}


class BgeM3DenseProvider(DenseEmbeddingProvider):
    def __init__(self, backend: BgeM3Backend) -> None:
        self.backend = backend
        self.model_name = backend.model_name
        self.effective_max_sequence_length = backend.max_seq_length
        self.embedding_space_id = (
            f"bge-m3-dense;model={self.model_name};dim={backend.dimension};normalize=true;"
            f"max_seq={backend.max_seq_length}"
        )
        self.runtime_metadata = backend.runtime_metadata
        self.provider_contract = _contract(backend, embedding_dimension=backend.dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.backend.dense_documents(texts)

    def token_count(self, text: str) -> int:
        return self.backend.token_count(text)

    def fits_token_budget(self, text: str, budget: int) -> bool:
        return self.backend.fits_token_budget(text, budget)


class BgeM3SparseProvider(SparseEmbeddingProvider):
    def __init__(self, backend: BgeM3Backend) -> None:
        self.backend = backend
        self.model_name = backend.model_name
        self.effective_max_sequence_length = backend.max_seq_length
        self.embedding_space_id = (
            f"bge-m3-lexical;model={self.model_name};top_k={backend.sparse_top_k};"
            f"max_seq={backend.max_seq_length}"
        )
        self.runtime_metadata = backend.runtime_metadata
        self.provider_contract = _contract(backend, embedding_dimension=None)

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        return self.backend.sparse_documents(texts)

    def embed_query(self, query: str) -> dict[str, float]:
        return self.embed_documents([query])[0]

    def token_count(self, text: str) -> int:
        return self.backend.token_count(text)

    def fits_token_budget(self, text: str, budget: int) -> bool:
        return self.backend.fits_token_budget(text, budget)


def build_bge_m3_providers(
    model_name: str,
    *,
    device: str | None,
    torch_dtype: str | None,
    max_seq_length: int = 512,
    sparse_top_k: int = 128,
) -> tuple[BgeM3DenseProvider, BgeM3SparseProvider]:
    backend = BgeM3Backend(
        model_name,
        device=device,
        torch_dtype=torch_dtype,
        max_seq_length=max_seq_length,
        sparse_top_k=sparse_top_k,
    )
    return BgeM3DenseProvider(backend), BgeM3SparseProvider(backend)


def _resolve_dtype(value: str | None, device: str) -> Any:
    import torch

    if value and value != "auto":
        return getattr(torch, value)
    return torch.float16 if device in {"mps", "cuda"} else torch.float32


def _contract(backend: BgeM3Backend, *, embedding_dimension: int | None) -> EmbeddingProviderContract:
    overhead = backend.token_count("")
    return EmbeddingProviderContract(
        model_name=backend.model_name,
        model_revision=None,
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
