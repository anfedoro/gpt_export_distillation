"""Stable identity and manifest primitives for future incremental imports.

This module deliberately does not compare exports or mutate an active database.
It defines the durable metadata contract that a future delta importer will use.
"""

from __future__ import annotations

import json
import uuid
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from blake3 import blake3


INCREMENTAL_METADATA_SCHEMA_VERSION = 2
CANONICAL_REPRESENTATION_VERSION = 1
PARSER_CONTRACT = f"markdown-it-py:{version('markdown-it-py')}"
CANONICALIZER_VERSION = "ptha.canonical-blocks.v1"
SOURCE_TRANSFORM_VERSION = "ptha.chat-md-source-transform.v1"
BLOCK_BUILDER_VERSION = "ptha.block-builder.v2"
CHUNKER_VERSION = "ptha.chunker.v2"
IDENTITY_VERSION = "ptha.source-identity.v1"
SPARSE_REPRESENTATION_VERSION = "ptha.sparse-compact.v1"


@dataclass(frozen=True)
class SourceIdentity:
    """A stable, source-level identity independent of local paths and row IDs."""

    id: str
    entity_type: str
    source_type: str
    external_id: str
    identity_method: str
    identity_version: str = IDENTITY_VERSION


@dataclass(frozen=True)
class SourceRevision:
    """An immutable canonical content revision of one source identity."""

    id: str
    source_identity_id: str
    canonical_hash: str
    canonicalizer_version: str = CANONICALIZER_VERSION


def canonical_bytes(value: Any) -> bytes:
    """Serialize semantic data deterministically for content-addressed metadata.

    Strings use NFC, LF line endings, no trailing horizontal whitespace per line,
    and no leading/trailing blank lines. Mapping keys are serialized in lexical
    order. Lists retain their semantic order.
    """
    normalized = _canonicalize(value)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_hash(domain: str, value: Any) -> str:
    """Return the full BLAKE3-256 hex digest with explicit domain separation."""
    return blake3(domain.encode("utf-8") + b"\0" + canonical_bytes(value)).hexdigest()


def source_identity(
    *, entity_type: str, source_type: str, native_id: str | None, fallback: Mapping[str, Any],
) -> SourceIdentity:
    """Prefer a native ID; otherwise use a deterministic, labelled fallback."""
    native = _nonempty(native_id)
    if native is not None:
        external_id = native
        method = "native_id"
    else:
        external_id = content_hash(f"ptha:{entity_type}:fallback-identity:v1", dict(fallback))
        method = "deterministic_fallback"
    identifier = content_hash(
        "ptha:source-identity-record:v1",
        {
            "entity_type": entity_type,
            "source_type": source_type,
            "external_id": external_id,
            "identity_method": method,
            "identity_version": IDENTITY_VERSION,
        },
    )
    return SourceIdentity(
        id=f"sid_{identifier}",
        entity_type=entity_type,
        source_type=source_type,
        external_id=external_id,
        identity_method=method,
    )


def conversation_identity(conversation: Any) -> SourceIdentity:
    """Build a stable conversation identity without document paths or mtimes."""
    return source_identity(
        entity_type="conversation",
        source_type="chatgpt_export",
        native_id=getattr(conversation, "conversation_id", None),
        fallback={
            "conversation_template_id": getattr(conversation, "conversation_template_id", None),
            "title": getattr(conversation, "title", None),
            "create_time_utc": getattr(conversation, "create_time_utc", None),
        },
    )


def message_identity(message: Any, *, conversation_identity_id: str) -> SourceIdentity:
    """Build a message identity, using ordinal only when the native ID is absent."""
    return source_identity(
        entity_type="message",
        source_type="chatgpt_export",
        native_id=getattr(message, "message_id", None),
        fallback={
            "conversation_identity_id": conversation_identity_id,
            "role": getattr(message, "role", None),
            "time_utc": getattr(message, "time_utc", None),
            "ordinal": getattr(message, "ordinal", None),
        },
    )


def conversation_revision(conversation: Any, *, identity: SourceIdentity) -> SourceRevision:
    return _revision(identity, "ptha:canonical-conversation:v1", {
        "conversation_id": getattr(conversation, "conversation_id", None),
        "conversation_template_id": getattr(conversation, "conversation_template_id", None),
        "title": getattr(conversation, "title", None),
        "create_time_utc": getattr(conversation, "create_time_utc", None),
        "update_time_utc": getattr(conversation, "update_time_utc", None),
        "metadata": getattr(conversation, "metadata_json", {}),
    })


def message_revision(
    message: Any, *, identity: SourceIdentity, canonical_blocks: Sequence[Any] | None = None,
) -> SourceRevision:
    semantic_content: Any
    if canonical_blocks is None:
        semantic_content = {"raw_text": getattr(message, "raw_text", None)}
    else:
        semantic_content = [
            {
                "ordinal": getattr(block, "ordinal", None),
                "block_type": getattr(block, "block_type", None),
                "language": getattr(block, "language", None),
                "canonical_content_hash": getattr(block, "canonical_content_hash", None),
            }
            for block in canonical_blocks
        ]
    return _revision(identity, "ptha:canonical-message:v1", {
        "message_id": getattr(message, "message_id", None),
        "role": getattr(message, "role", None),
        "time_utc": getattr(message, "time_utc", None),
        "canonical_blocks": semantic_content,
        "metadata": getattr(message, "metadata_json", {}),
    })


def block_identity(block: Any, *, message_revision_id: str) -> str:
    return "block_" + content_hash("ptha:block-identity:v2", {
        "source_revision_id": message_revision_id,
        "block_builder_version": BLOCK_BUILDER_VERSION,
        "ordinal": getattr(block, "ordinal", None),
        "block_type": getattr(block, "block_type", None),
        "language": getattr(block, "language", None),
        "canonical_content_hash": getattr(block, "canonical_content_hash", None),
    })


def chunk_derivation_fingerprint(*, source_revision_id: str, block_identity_id: str, chunk_policy_id: str) -> str:
    return content_hash("ptha:chunk-derivation:v1", {
        "source_revision_id": source_revision_id,
        "block_identity": block_identity_id,
        "canonicalizer_version": CANONICALIZER_VERSION,
        "block_builder_version": BLOCK_BUILDER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "chunk_policy_id": chunk_policy_id,
    })


def chunk_identity(
    *, source_revision_id: str, block_identity_id: str, chunk_policy_id: str,
    ordinal: int, source_char_start: int, source_char_end: int,
) -> str:
    return "chunk_" + content_hash("ptha:chunk-identity:v1", {
        "derivation_fingerprint": chunk_derivation_fingerprint(
            source_revision_id=source_revision_id,
            block_identity_id=block_identity_id,
            chunk_policy_id=chunk_policy_id,
        ),
        "ordinal": ordinal,
        "source_char_start": source_char_start,
        "source_char_end": source_char_end,
    })


def chunk_content_hash(text: str) -> str:
    return content_hash("ptha:chunk-content:v1", {"text": text})


def embedding_contract_fingerprint(*, dense: Any, sparse: Any) -> tuple[dict[str, Any], str]:
    """Fingerprint semantic embedding behavior, excluding batch/device settings."""
    dense_contract = dict(getattr(dense, "contract_dict", lambda: {})())
    runtime = dict(getattr(dense, "runtime_metadata", {}))
    payload = {
        "provider_type": runtime.get("backend", type(getattr(dense, "backend", dense)).__name__),
        "model_repository": getattr(dense, "model_name", None),
        "model_revision": runtime.get("model_revision") or dense_contract.get("model_revision"),
        "precision": runtime.get("dtype"),
        "tokenizer": {
            "name": dense_contract.get("tokenizer_name"),
            "model_max_length": dense_contract.get("tokenizer_model_max_length"),
            "backbone_max_position_embeddings": dense_contract.get("backbone_max_position_embeddings"),
        },
        "dense": {
            "dimension": dense_contract.get("embedding_dimension"),
            "normalization": True,
            "pooling": "cls",
            "document_prefix": getattr(dense, "document_prefix", ""),
            "query_prefix": getattr(dense, "query_prefix", ""),
        },
        "sparse": {
            "representation_version": SPARSE_REPRESENTATION_VERSION,
            "space": getattr(sparse, "embedding_space_id", None),
            "top_k": _sparse_top_k(getattr(sparse, "embedding_space_id", "")),
            "document_prefix": getattr(sparse, "document_prefix", ""),
            "query_prefix": getattr(sparse, "query_prefix", ""),
        },
        "max_sequence_length": getattr(dense, "effective_max_sequence_length", None),
        "mlx_embeddings_revision": runtime.get("mlx_embeddings_revision"),
    }
    return payload, content_hash("ptha:embedding-contract:v1", payload)


def compare_source_revisions(old: Mapping[str, str], new: Mapping[str, str]) -> dict[str, list[str]]:
    """Classify inventories only; absent IDs are diagnostic and never deletions."""
    unchanged = sorted(identity for identity, digest in new.items() if old.get(identity) == digest)
    changed = sorted(identity for identity, digest in new.items() if identity in old and old[identity] != digest)
    added = sorted(identity for identity in new if identity not in old)
    absent = sorted(identity for identity in old if identity not in new)
    return {"unchanged": unchanged, "changed": changed, "new": added, "absent_in_new_export": absent}


def new_generation_id() -> str:
    return f"gen_{uuid.uuid4().hex}"


def _revision(identity: SourceIdentity, domain: str, payload: Mapping[str, Any]) -> SourceRevision:
    digest = content_hash(domain, payload)
    identifier = content_hash("ptha:source-revision-record:v1", {
        "source_identity_id": identity.id,
        "canonical_hash": digest,
        "canonicalizer_version": CANONICALIZER_VERSION,
    })
    return SourceRevision(id=f"rev_{identifier}", source_identity_id=identity.id, canonical_hash=digest)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, str):
        text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip(" \t") for line in text.split("\n")]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    return _canonicalize(str(value))


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _canonicalize(value)
    return normalized or None


def _sparse_top_k(space: str) -> int | None:
    for piece in str(space).split(";"):
        if piece.startswith("top_k="):
            try:
                return int(piece.removeprefix("top_k="))
            except ValueError:
                return None
    return None
