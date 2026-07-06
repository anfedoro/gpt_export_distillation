from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportSource:
    label: str
    root_dir: Path
    output_dir: Path
    kind: str


@dataclass(frozen=True)
class InputBundle:
    source: ExportSource
    conversations: list[dict[str, Any]]
    asset_file_names: dict[str, str] = field(default_factory=dict)
    library_files: list["LibraryFileRecord"] = field(default_factory=list)
    shared_conversations: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class LibraryFileRecord:
    raw: dict[str, Any]
    file_id: str
    file_name: str
    normalized_name: str
    mime_type: str | None
    library_file_category: str | None
    directory_id: str | None
    knowledge_store_id: str | None
    knowledge_store_kind: str | None
    origination_thread_id: str | None
    origination_message_id: str | None
    is_project: bool | None
    pinned_at: str | None
    context_scopes: tuple[str, ...] = ()
    context_scopes_v2: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class MessageRow:
    role: str
    text: str
    timestamp: float | None
    message_id: str


@dataclass(frozen=True)
class ChatMetrics:
    assistant_messages: int
    user_messages: int
    total_messages: int
    text_chars: int
    code_blocks: int
    urls: int


@dataclass(frozen=True)
class ChatDocument:
    conversation: dict[str, Any]
    rows: list[MessageRow]
    metrics: ChatMetrics
    group_name: str
    source_label: str
    attachment_ids: tuple[str, ...] = ()
