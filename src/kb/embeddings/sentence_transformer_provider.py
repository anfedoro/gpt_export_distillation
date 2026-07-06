from __future__ import annotations

import os

from kb.embeddings.base import DenseEmbeddingProvider, SparseEmbeddingProvider


class SentenceTransformerDenseProvider(DenseEmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the KB extra with "
                "`uv sync --extra kb` or run with `uv run --extra kb ...`."
            ) from exc
        self.model_name = model_name
        self.model_version = "sentence-transformers"
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in vectors]


class SentenceTransformerSparseProvider(SparseEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        *,
        top_k: int = 128,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers with SparseEncoder is not installed. Install the KB extra with "
                "`uv sync --extra kb` or run with `uv run --extra kb ...`."
            ) from exc
        self.model_name = model_name
        self.model_version = "sentence-transformers-sparse"
        self.top_k = top_k
        self._model = SparseEncoder(model_name)

    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        embeddings = self._model.encode_document(texts)
        decoded = self._model.decode(embeddings, top_k=self.top_k)
        return [
            {token: float(weight) for token, weight in terms}
            for terms in decoded
        ]
