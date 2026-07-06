from __future__ import annotations

from abc import ABC, abstractmethod


class DenseEmbeddingProvider(ABC):
    model_name: str
    model_version: str

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SparseEmbeddingProvider(ABC):
    model_name: str
    model_version: str

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        raise NotImplementedError
