from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class InventoryItem:
    relative_path: str
    file_name: str
    extension: str
    size: int
    sha256: str
    detected_kind: str
    folder_kind: str | None
    project_path: str | None
    is_attachment: bool
    interest_tier: str


@dataclass(frozen=True)
class Conversation:
    id: str
    source_document_id: str
    conversation_id: str | None
    conversation_template_id: str | None
    title: str | None
    create_time_utc: str | None
    update_time_utc: str | None
    message_count: int
    assistant_messages: int
    user_messages: int
    text_chars: int
    estimated_code_blocks: int
    project_id: str | None
    folder_kind: str | None
    metadata_json: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    ordinal: int
    role: str
    message_id: str | None
    time_utc: str | None
    raw_text: str
    metadata_json: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    id: str
    message_id: str
    conversation_id: str
    ordinal: int
    block_type: str
    language: str | None
    raw_text: str
    normalized_text: str
    char_start: int
    char_end: int
    metadata_json: JsonDict = field(default_factory=dict)
    canonical_content_hash: str = ""
    parser_version: str = ""
    canonicalizer_version: str = ""
    semantic_status: str = "graph_eligible"
    dense_index_policy: str = "include"
    sparse_index_policy: str = "include"
    graph_eligibility: bool = True
    artifact_policy: str = "no"
    context_policy: str = "include"
    exclusion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockRelationship:
    source_block_id: str
    target_block_id: str
    relation_type: str
    ordinal: int
    metadata_json: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedChat:
    conversation: Conversation
    messages: list[Message]
    blocks: list[Block]
    metadata: JsonDict
    relationships: list[BlockRelationship] = field(default_factory=list)
