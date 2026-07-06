from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kb.model.entities import Block, Conversation, Message, ParsedChat
from kb.model.ids import stable_id


MESSAGE_HEADING_RE = re.compile(
    r"^###\s+(\d+)\.\s+(USER|ASSISTANT|SYSTEM|TOOL|UNKNOWN)\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
KV_RE = re.compile(r"^-\s+`?([A-Za-z0-9_ -]+)`?:\s*(.*)$")
FENCE_RE = re.compile(r"^```([A-Za-z0-9_+.-]*)\s*$")
ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def parse_chat_file(path: Path, source_document_id: str, project_id: str | None = None, folder_kind: str | None = None) -> ParsedChat:
    text = path.read_text(encoding="utf-8")
    metadata_text, conversation_text = _split_sections(text)
    metadata = _parse_metadata(metadata_text)
    messages = _parse_messages(conversation_text, source_document_id)
    blocks: list[Block] = []
    for message in messages:
        blocks.extend(_parse_blocks(message))
    conversation = _build_conversation(
        source_document_id=source_document_id,
        metadata=metadata,
        messages=messages,
        blocks=blocks,
        project_id=project_id,
        folder_kind=folder_kind,
    )
    return ParsedChat(conversation=conversation, messages=messages, blocks=blocks, metadata=metadata)


def _split_sections(text: str) -> tuple[str, str]:
    metadata_match = re.search(r"^##\s+Metadata\s*$", text, flags=re.MULTILINE)
    conversation_match = re.search(r"^##\s+Conversation\s*$", text, flags=re.MULTILINE)
    if not conversation_match:
        return "", text
    metadata_text = ""
    if metadata_match and metadata_match.start() < conversation_match.start():
        metadata_text = text[metadata_match.end() : conversation_match.start()]
    return metadata_text, text[conversation_match.end() :]


def _parse_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        match = KV_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        value = match.group(2).strip()
        if value.startswith("`") and value.endswith("`"):
            value = value[1:-1]
        metadata[key] = value
    return metadata


def _parse_messages(text: str, source_document_id: str) -> list[Message]:
    matches = _message_heading_matches(text)
    messages: list[Message] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading_ordinal = int(match.group(1))
        ordinal = idx + 1
        role = ROLE_MAP.get(match.group(2).strip().lower(), "unknown")
        section = text[start:end].strip("\n")
        attrs, body = _split_message_attrs(section)
        metadata_json = {k: v for k, v in attrs.items() if k not in {"message_id", "time_utc"}}
        if heading_ordinal != ordinal:
            metadata_json["heading_ordinal"] = heading_ordinal
        message_id = attrs.get("message_id")
        message = Message(
            id=stable_id(source_document_id, ordinal, message_id, prefix="msg"),
            conversation_id=stable_id(source_document_id, "conversation", prefix="conv"),
            ordinal=ordinal,
            role=role,
            message_id=message_id,
            time_utc=attrs.get("time_utc"),
            raw_text=body,
            metadata_json=metadata_json,
        )
        messages.append(message)
    return messages


def _message_heading_matches(text: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    in_fence = False
    for line_match in re.finditer(r"^.*(?:\n|$)", text, flags=re.MULTILINE):
        line = line_match.group(0)
        if line == "":
            continue
        stripped = line.rstrip("\n")
        fence_match = FENCE_RE.match(stripped)
        if fence_match:
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading_match = MESSAGE_HEADING_RE.match(text, line_match.start())
        if heading_match:
            matches.append(heading_match)
    return matches


def _split_message_attrs(section: str) -> tuple[dict[str, str], str]:
    attrs: dict[str, str] = {}
    body_lines: list[str] = []
    in_attrs = True
    for line in section.splitlines():
        if in_attrs:
            match = KV_RE.match(line.strip())
            if match:
                key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
                value = match.group(2).strip()
                if value.startswith("`") and value.endswith("`"):
                    value = value[1:-1]
                attrs[key] = value
                continue
            if not line.strip():
                continue
            in_attrs = False
        body_lines.append(line)
    return attrs, "\n".join(body_lines).strip("\n")


def _build_conversation(
    source_document_id: str,
    metadata: dict[str, Any],
    messages: list[Message],
    blocks: list[Block],
    project_id: str | None,
    folder_kind: str | None,
) -> Conversation:
    assistant_messages = sum(1 for message in messages if message.role == "assistant")
    user_messages = sum(1 for message in messages if message.role == "user")
    return Conversation(
        id=stable_id(source_document_id, "conversation", prefix="conv"),
        source_document_id=source_document_id,
        conversation_id=_first(metadata, "id", "conversation_id"),
        conversation_template_id=metadata.get("conversation_template_id"),
        title=metadata.get("title"),
        create_time_utc=_first(metadata, "create_time_utc", "create_time"),
        update_time_utc=_first(metadata, "update_time_utc", "update_time"),
        message_count=len(messages),
        assistant_messages=assistant_messages,
        user_messages=user_messages,
        text_chars=sum(len(message.raw_text) for message in messages),
        estimated_code_blocks=sum(1 for block in blocks if block.block_type in {"code", "mermaid"}),
        project_id=project_id,
        folder_kind=folder_kind,
        metadata_json=metadata,
    )


def _first(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _parse_blocks(message: Message) -> list[Block]:
    text = message.raw_text
    if not text:
        return []
    spans: list[tuple[str, str | None, int, int]] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    prose_start: int | None = None
    prose_parts: list[str] = []
    in_fence = False
    fence_language: str | None = None
    fence_start = 0
    fence_parts: list[str] = []

    def flush_prose(end_offset: int) -> None:
        nonlocal prose_start, prose_parts
        if prose_start is None:
            return
        raw = "".join(prose_parts).strip("\n")
        if raw.strip():
            spans.append((_classify_non_code(raw), None, prose_start, end_offset))
        prose_start = None
        prose_parts = []

    for line in lines:
        match = FENCE_RE.match(line.rstrip("\n"))
        if match and not in_fence:
            flush_prose(offset)
            in_fence = True
            fence_language = match.group(1).strip().lower() or None
            fence_start = offset
            fence_parts = [line]
        elif match and in_fence:
            fence_parts.append(line)
            raw = "".join(fence_parts)
            spans.append(("mermaid" if fence_language == "mermaid" else "code", fence_language, fence_start, offset + len(line)))
            in_fence = False
            fence_language = None
            fence_parts = []
        elif in_fence:
            fence_parts.append(line)
        else:
            if prose_start is None:
                prose_start = offset
            prose_parts.append(line)
        offset += len(line)
    if in_fence:
        raw = "".join(fence_parts)
        spans.append(("mermaid" if fence_language == "mermaid" else "code", fence_language, fence_start, len(text)))
    else:
        flush_prose(len(text))

    blocks: list[Block] = []
    for ordinal, (block_type, language, start, end) in enumerate(spans, start=1):
        raw = text[start:end].strip("\n")
        blocks.append(
            Block(
                id=stable_id(message.id, ordinal, start, end, prefix="block"),
                message_id=message.id,
                conversation_id=message.conversation_id,
                ordinal=ordinal,
                block_type=block_type,
                language=language,
                raw_text=raw,
                normalized_text=_normalize_block(raw, block_type),
                char_start=start,
                char_end=end,
                metadata_json={},
            )
        )
    return blocks


def _classify_non_code(raw: str) -> str:
    stripped_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if stripped_lines and all(line.startswith("|") and line.endswith("|") for line in stripped_lines):
        return "table"
    if stripped_lines and all(line.startswith(">") for line in stripped_lines):
        return "quote"
    if stripped_lines and all(re.match(r"^([-*+]|\d+\.)\s+", line) for line in stripped_lines):
        return "list"
    if stripped_lines and all(line.startswith("#") for line in stripped_lines):
        return "heading"
    return "prose"


def _normalize_block(raw: str, block_type: str) -> str:
    if block_type in {"code", "mermaid"}:
        return raw.strip("\n")
    return re.sub(r"\s+", " ", raw).strip()


def parsed_chat_to_jsonable(parsed: ParsedChat) -> dict[str, Any]:
    return {
        "conversation": parsed.conversation.__dict__,
        "messages": [message.__dict__ for message in parsed.messages],
        "blocks": [block.__dict__ for block in parsed.blocks],
        "metadata": parsed.metadata,
    }


def write_parsed_chat_json(parsed: ParsedChat, output_path: Path) -> None:
    output_path.write_text(json.dumps(parsed_chat_to_jsonable(parsed), ensure_ascii=False, indent=2), encoding="utf-8")
