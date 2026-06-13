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
    library_files: list[dict[str, Any]] = field(default_factory=list)
    shared_conversations: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None


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
