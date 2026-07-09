from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from kb.embeddings.base import DenseEmbeddingProvider, SparseEmbeddingProvider


TOKEN_RE = re.compile(r"[A-Za-z0-9А-Яа-яЁё_]{2,}")


class MockDenseProvider(DenseEmbeddingProvider):
    model_name = "mock-dense"
    embedding_space_id = "mock-dense;dim=16;normalize=true;symmetric=true"
    runtime_metadata: dict[str, object] = {"backend": "mock"}
    effective_max_sequence_length = 64

    def __init__(self, dim: int = 16, max_sequence_length: int = 64) -> None:
        self.dim = dim
        self.effective_max_sequence_length = max_sequence_length
        self.embedding_space_id = f"mock-dense;dim={dim};normalize=true;symmetric=true;max_seq={max_sequence_length}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_mock_dense_vector(text, self.dim) for text in texts]

    def token_count(self, text: str) -> int:
        return len(TOKEN_RE.findall(text)) or (1 if text else 0)


class MockSparseProvider(SparseEmbeddingProvider):
    model_name = "mock-sparse"
    embedding_space_id = "mock-sparse;document=query;top_k=all"
    runtime_metadata: dict[str, object] = {"backend": "mock"}
    effective_max_sequence_length = 64

    def __init__(self, max_sequence_length: int = 64) -> None:
        self.effective_max_sequence_length = max_sequence_length
        self.embedding_space_id = f"mock-sparse;document=query;top_k=all;max_seq={max_sequence_length}"

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        return [_mock_sparse_terms(text) for text in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        return _mock_sparse_terms(query)

    def token_count(self, text: str) -> int:
        return len(TOKEN_RE.findall(text)) or (1 if text else 0)


def _mock_dense_vector(text: str, dim: int) -> list[float]:
    values: list[float] = []
    for idx in range(dim):
        digest = hashlib.sha256(f"{idx}\x1f{text}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], "big")
        values.append((integer / ((1 << 64) - 1)) * 2.0 - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _mock_sparse_terms(text: str) -> dict[str, float]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {token: count / total for token, count in sorted(counts.items())}
