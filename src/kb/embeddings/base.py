from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbeddingProviderContract:
    model_name: str
    model_revision: str | None
    embedding_dimension: int | None
    tokenizer_name: str | None
    tokenizer_model_max_length: int | None
    backbone_max_position_embeddings: int | None
    sentence_transformer_max_seq_length: int | None
    configured_effective_max_seq_length: int | None
    document_prefix: str
    query_prefix: str
    special_token_overhead: int
    configured_safety_reserve: int
    computed_content_budget: int | None
    max_seq_length_override: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DenseEmbeddingProvider(ABC):
    model_name: str
    embedding_space_id: str
    runtime_metadata: dict[str, object]
    document_prefix: str = ""
    query_prefix: str = ""
    effective_max_sequence_length: int | None = None
    provider_contract: EmbeddingProviderContract | None = None

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

    def embedding_input(self, text: str) -> str:
        return f"{self.document_prefix}{text}"

    def query_input(self, text: str) -> str:
        return f"{self.query_prefix}{text}"

    def contract_dict(self) -> dict[str, object]:
        if self.provider_contract is None:
            return {}
        return self.provider_contract.as_dict()

    def token_count(self, text: str) -> int:
        raise RuntimeError(f"Provider {self.model_name} does not expose a reliable tokenizer.")

    def fits_token_budget(self, text: str, budget: int) -> bool:
        """Return whether text fits, without requiring a full unbounded tokenization."""
        return self.token_count(text) <= budget

    def assert_fits(self, text: str, *, chunk_id: str, block_id: str, source_identity: str) -> int:
        if self.effective_max_sequence_length is None:
            raise RuntimeError(f"Provider {self.model_name} does not expose an effective max sequence length.")
        count = self.token_count(self.embedding_input(text))
        if count > self.effective_max_sequence_length:
            raise ValueError(
                f"Embedding input exceeds provider limit provider={self.model_name} chunk_id={chunk_id} "
                f"block_id={block_id} source={source_identity} actual_tokens={count} "
                f"allowed_tokens={self.effective_max_sequence_length}"
            )
        return count


class SparseEmbeddingProvider(ABC):
    model_name: str
    embedding_space_id: str
    runtime_metadata: dict[str, object]
    document_prefix: str = ""
    query_prefix: str = ""
    effective_max_sequence_length: int | None = None
    provider_contract: EmbeddingProviderContract | None = None

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

    def embedding_input(self, text: str) -> str:
        return f"{self.document_prefix}{text}"

    def query_input(self, text: str) -> str:
        return f"{self.query_prefix}{text}"

    def contract_dict(self) -> dict[str, object]:
        if self.provider_contract is None:
            return {}
        return self.provider_contract.as_dict()

    def token_count(self, text: str) -> int:
        raise RuntimeError(f"Provider {self.model_name} does not expose a reliable tokenizer.")

    def fits_token_budget(self, text: str, budget: int) -> bool:
        """Return whether text fits, without requiring a full unbounded tokenization."""
        return self.token_count(text) <= budget

    def assert_fits(self, text: str, *, chunk_id: str, block_id: str, source_identity: str) -> int:
        if self.effective_max_sequence_length is None:
            raise RuntimeError(f"Provider {self.model_name} does not expose an effective max sequence length.")
        count = self.token_count(self.embedding_input(text))
        if count > self.effective_max_sequence_length:
            raise ValueError(
                f"Embedding input exceeds provider limit provider={self.model_name} chunk_id={chunk_id} "
                f"block_id={block_id} source={source_identity} actual_tokens={count} "
                f"allowed_tokens={self.effective_max_sequence_length}"
            )
        return count
