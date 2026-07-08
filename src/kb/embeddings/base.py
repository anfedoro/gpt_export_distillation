from __future__ import annotations

from abc import ABC, abstractmethod


class DenseEmbeddingProvider(ABC):
    model_name: str
    embedding_space_id: str
    runtime_metadata: dict[str, object]

    @property
    def model_version(self) -> str:
        return self.embedding_space_id

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class SparseEmbeddingProvider(ABC):
    model_name: str
    embedding_space_id: str
    runtime_metadata: dict[str, object]

    @property
    def model_version(self) -> str:
        return self.embedding_space_id

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> dict[str, float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        return self.embed_documents(texts)
