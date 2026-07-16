"""Deterministic Markdown-to-canonical-block transformation."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from kb.model.entities import Block, BlockRelationship, Message
from kb.model.ids import stable_id
from ptha.incremental import PARSER_CONTRACT, content_hash


CANONICAL_REPRESENTATION_VERSION = 1
CANONICALIZER_VERSION = "ptha.canonical-blocks.v1"
STRUCTURED_FORMATS = {"json", "jsonc", "xml", "yaml", "yml", "toml"}
PLAIN_TEXT_FORMATS = {"text", "txt", "plaintext", "plain"}
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUNCTUATION_ONLY_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)
LANGUAGE_LABELS = {
    "bash", "c", "cpp", "csharp", "css", "html", "java", "javascript", "json",
    "markdown", "mermaid", "powershell", "python", "ruby", "rust", "shell",
    "sql", "text", "toml", "typescript", "xml", "yaml",
}
GENERIC_TRANSITIONS = {
    "далее", "дальше", "например", "пример", "итак", "минимум", "лучше так",
    "вот так", "да, так лучше", "или", "или даже", "and", "next", "for example",
    "example", "minimum", "better", "better this way", "note",
}
GENERIC_INTRO_RE = re.compile(
    r"^(?:"
    r"(?:а\s+)?(?:вот|примерно|лучше|корректнее|точнее)\b.*|"
    r"(?:хороший|практический|реалистичный|доработанный)\s+(?:вариант|вывод)\b.*|"
    r"(?:сейчас|тогда|потом)\s+(?:сделай|надо|добавь|проверяй)\b.*|"
    r"(?:попробуй|сделай)\s+так\b.*|"
    r"(?:мой|наш)\s+(?:вывод|итог)\b.*|"
    r"(?:команда|команды|варианты|пример|итог|резюме)\b.*|"
    r"(?:for example|example|next|the result|my conclusion)\b.*"
    r")$",
    flags=re.IGNORECASE,
)
WRITING_WRAPPER_OPEN_RE = re.compile(r"^\s*:::writing\{[^}\n]*\}\s*", flags=re.IGNORECASE)
WRITING_WRAPPER_CLOSE_RE = re.compile(r"\s*:::\s*$")


@dataclass
class _Candidate:
    block_type: str
    content: str
    start: int
    end: int
    language: str | None = None
    metadata: dict[str, Any] | None = None


def canonicalize_message(message: Message) -> tuple[list[Block], list[BlockRelationship]]:
    """Convert one source message into ordered, policy-bearing canonical blocks."""
    source = _normalize_source(message.raw_text)
    if not source:
        return [], []
    parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    tokens = parser.parse(source)
    offsets = _line_offsets(source)
    candidates = _tokens_to_candidates(tokens, source, offsets)
    candidates = _coalesce_prose(candidates)
    section_ordinal = 0
    for candidate in candidates:
        if (candidate.metadata or {}).get("source_node") in {"heading", "heading_with_body"}:
            section_ordinal += 1
        candidate.metadata = dict(candidate.metadata or {})
        candidate.metadata["section_ordinal"] = section_ordinal
    offsets_basis = "raw_message" if source == message.raw_text else "normalized_message"
    for candidate in candidates:
        candidate.metadata = dict(candidate.metadata or {})
        candidate.metadata["source_offsets_basis"] = offsets_basis
        candidate.metadata["raw_content_reference"] = {
            "message_id": message.id,
            "char_start": candidate.start,
            "char_end": candidate.end,
            "basis": offsets_basis,
        }
    blocks = [_make_block(message, ordinal, candidate) for ordinal, candidate in enumerate(candidates, 1)]
    relationships = _relationships(blocks)
    return blocks, relationships


def _tokens_to_candidates(tokens: list[Token], source: str, offsets: list[int]) -> list[_Candidate]:
    result: list[_Candidate] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        start, end = _token_offsets(token, offsets, len(source))
        if token.type == "fence":
            info = token.info.strip().split(maxsplit=1)[0].lower() if token.info.strip() else None
            content = token.content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
            if info == "mermaid":
                result.append(_Candidate("diagram", content, start, end, info, {"format": "mermaid"}))
            elif info in STRUCTURED_FORMATS:
                canonical, parse_status = _canonicalize_structured(content, info)
                result.extend(_unwrap_transport_or_structured(canonical, content, info, parse_status, start, end))
            elif info in PLAIN_TEXT_FORMATS:
                result.append(_Candidate("prose", _normalize_prose(content), start, end, info))
            elif content.strip():
                result.append(_Candidate("code", content, start, end, info))
            index += 1
            continue
        if token.type == "table_open":
            closing = _matching_close(tokens, index, "table_close")
            result.append(_table_candidate(tokens[index : closing + 1], start, _token_offsets(tokens[closing], offsets, len(source))[1]))
            index = closing + 1
            continue
        if token.type == "blockquote_open":
            closing = _matching_close(tokens, index, "blockquote_close")
            content = _render_inline_range(tokens[index + 1 : closing])
            result.append(_Candidate("quote_or_external_content", content, start, _token_offsets(tokens[closing], offsets, len(source))[1]))
            index = closing + 1
            continue
        if token.type in {"heading_open", "paragraph_open"}:
            inline = tokens[index + 1] if index + 1 < len(tokens) and tokens[index + 1].type == "inline" else None
            if inline is not None:
                prose, media = _render_inline(inline)
                if prose:
                    source_node = token.type.removesuffix("_open")
                    if token.type == "paragraph_open" and token.level > 0:
                        source_node = "list_item"
                    result.append(_Candidate("prose", prose, start, end, metadata={"source_node": source_node}))
                result.extend(_Candidate("media_reference", "", start, end, metadata=item) for item in media)
            index += 1
            continue
        if token.type == "inline" and token.level > 0:
            # Inline content is consumed by its owning paragraph/list item.
            index += 1
            continue
        index += 1
    return [item for item in result if item.content or item.block_type in {"media_reference", "attachment_reference"}]


def _render_inline(token: Token) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    media: list[dict[str, Any]] = []
    for child in token.children or []:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        elif child.type == "image":
            attrs = dict(child.attrs or {})
            alt = child.content.strip()
            media.append({"pointer": attrs.get("src"), "alt_text": alt or None, "title": attrs.get("title")})
            if alt:
                parts.append(alt)
    return _normalize_prose("".join(parts)), media


def _render_inline_range(tokens: list[Token]) -> str:
    values: list[str] = []
    for token in tokens:
        if token.type == "inline":
            text, _media = _render_inline(token)
            if text:
                values.append(text)
    return _normalize_prose(" ".join(values))


def _table_candidate(tokens: list[Token], start: int, end: int) -> _Candidate:
    columns: list[str] = []
    rows: list[list[str]] = []
    current: list[str] | None = None
    in_head = False
    for token in tokens:
        if token.type == "thead_open":
            in_head = True
        elif token.type == "thead_close":
            in_head = False
        elif token.type == "tr_open":
            current = []
        elif token.type == "tr_close" and current is not None:
            if in_head and not columns:
                columns = current
            else:
                rows.append(current)
            current = None
        elif token.type == "inline" and current is not None:
            text, _media = _render_inline(token)
            current.append(text)
    payload = {"columns": columns, "rows": rows}
    return _Candidate("table", json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), start, end, metadata=payload)


def _unwrap_transport_or_structured(
    canonical: str, raw: str, format_name: str, parse_status: str, start: int, end: int,
) -> list[_Candidate]:
    if format_name in {"json", "jsonc"} and parse_status == "parsed":
        value = json.loads(canonical)
        if isinstance(value, dict) and value.get("content_type") == "audio_transcription" and isinstance(value.get("text"), str):
            prose = _Candidate("prose", _normalize_prose(value["text"]), start, end, metadata={"extracted_from": "audio_transcription"})
            media = _Candidate("media_reference", "", start, end, metadata={"transport": value, "pointer": value.get("audio_asset_pointer")})
            return [prose, media]
        if isinstance(value, dict) and _looks_like_asset_pointer(value):
            return [_Candidate("media_reference", "", start, end, metadata={"transport": value})]
    return [_Candidate(
        "structured_data", canonical if parse_status == "parsed" else raw.rstrip("\n"), start, end, format_name,
        {"format": format_name, "parse_status": parse_status},
    )]


def _canonicalize_structured(content: str, format_name: str) -> tuple[str, str]:
    if format_name in {"json", "jsonc"}:
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return content.rstrip("\n"), "malformed"
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), "parsed"
    # XML/YAML/TOML are preserved losslessly in v1; no extra parser dependency is required.
    return content.rstrip("\n"), "preserved_unparsed"


def _looks_like_asset_pointer(value: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    return bool(keys & {"asset_pointer", "image_asset_pointer", "audio_asset_pointer"}) or (
        "content_type" in keys and str(value.get("content_type", "")).lower() in {"image_asset_pointer", "asset_pointer"}
    )


def _coalesce_prose(candidates: list[_Candidate]) -> list[_Candidate]:
    result: list[_Candidate] = []
    for candidate in candidates:
        if candidate.block_type == "prose":
            candidate.content = _normalize_prose(candidate.content)
            if not candidate.content:
                continue
            previous_node = result[-1].metadata.get("source_node") if result and result[-1].metadata else None
            candidate_node = candidate.metadata.get("source_node") if candidate.metadata else None
            should_join = (
                previous_node in {"heading", "heading_with_body", "list_item"}
                and candidate_node == "list_item"
            )
            if result and result[-1].block_type == "prose" and should_join:
                previous = result[-1]
                if previous.metadata and previous.metadata.get("source_node") == "heading":
                    separator = ": "
                    previous.metadata["source_node"] = "heading_with_body"
                elif previous.metadata and previous.metadata.get("source_node") == "heading_with_body":
                    separator = "; "
                else:
                    separator = " "
                previous.content = _normalize_prose(previous.content.rstrip(":") + separator + candidate.content)
                previous.end = candidate.end
                continue
        result.append(candidate)
    return result


def _make_block(message: Message, ordinal: int, candidate: _Candidate) -> Block:
    status, dense, sparse, graph, artifact, context, reasons = _policy(candidate)
    digest = content_hash(
        "ptha:canonical-block-content:v1",
        {"type": candidate.block_type, "content": candidate.content, "language": candidate.language},
    )
    identifier = stable_id(message.id, CANONICAL_REPRESENTATION_VERSION, ordinal, digest, prefix="block")
    metadata = dict(candidate.metadata or {})
    metadata["canonical_representation_version"] = CANONICAL_REPRESENTATION_VERSION
    return Block(
        id=identifier,
        message_id=message.id,
        conversation_id=message.conversation_id,
        ordinal=ordinal,
        block_type=candidate.block_type,
        language=candidate.language,
        raw_text=(
            message.raw_text[candidate.start:candidate.end]
            if (candidate.metadata or {}).get("source_offsets_basis") == "raw_message"
            else _normalize_source(message.raw_text)[candidate.start:candidate.end]
        ),
        normalized_text=candidate.content,
        char_start=candidate.start,
        char_end=candidate.end,
        metadata_json=metadata,
        canonical_content_hash=digest,
        parser_version=PARSER_CONTRACT,
        canonicalizer_version=CANONICALIZER_VERSION,
        semantic_status=status,
        dense_index_policy=dense,
        sparse_index_policy=sparse,
        graph_eligibility=graph,
        artifact_policy=artifact,
        context_policy=context,
        exclusion_reasons=tuple(reasons),
    )


def _policy(candidate: _Candidate) -> tuple[str, str, str, bool, str, str, list[str]]:
    if candidate.block_type == "prose":
        reasons: list[str] = []
        text = candidate.content.strip()
        if not text:
            return "excluded", "exclude", "exclude", False, "no", "exclude", ["empty"]
        if PUNCTUATION_ONLY_RE.fullmatch(text):
            return "excluded", "exclude", "exclude", False, "no", "exclude", ["punctuation_only"]
        if text.casefold() in LANGUAGE_LABELS:
            return "context_only", "exclude", "exclude", False, "no", "include", ["standalone_language_label"]
        lexical = text.casefold().strip(" :;,.!?—-")
        source_node = (candidate.metadata or {}).get("source_node")
        if lexical in GENERIC_TRANSITIONS:
            return "context_only", "exclude", "exclude", False, "no", "include", ["generic_transition"]
        if len(re.findall(r"\w+", text, flags=re.UNICODE)) <= 12 and GENERIC_INTRO_RE.fullmatch(lexical):
            return "context_only", "exclude", "exclude", False, "no", "include", ["generic_contextual_intro"]
        if source_node == "heading" and len(re.findall(r"\w+", text, flags=re.UNICODE)) <= 4:
            return "context_only", "exclude", "exclude", False, "no", "include", ["short_heading_without_body"]
        return "graph_eligible", "include", "include", True, "no", "include", reasons
    if candidate.block_type == "quote_or_external_content":
        return "artifact", "scoped", "scoped", False, "store", "structurally_reachable", ["quoted_or_external"]
    if candidate.block_type in {"code", "structured_data", "table", "diagram", "media_reference", "attachment_reference"}:
        return "artifact", "exclude", "exclude", False, "store", "structurally_reachable", [f"artifact_type:{candidate.block_type}"]
    return "artifact", "exclude", "exclude", False, "store", "structurally_reachable", ["unknown_type"]


def _relationships(blocks: list[Block]) -> list[BlockRelationship]:
    relationships: list[BlockRelationship] = []
    ordinal = 1
    for left, right in zip(blocks, blocks[1:]):
        for source, target, kind in (
            (left, right, "next_block"), (right, left, "previous_block"),
            (left, right, "adjacent_block"), (right, left, "adjacent_block"),
            (left, right, "same_message"), (right, left, "same_message"),
            (left, right, "same_document"), (right, left, "same_document"),
        ):
            relationships.append(BlockRelationship(source.id, target.id, kind, ordinal))
            ordinal += 1
        if left.metadata_json.get("section_ordinal") == right.metadata_json.get("section_ordinal"):
            relationships.append(BlockRelationship(left.id, right.id, "same_section", ordinal))
            ordinal += 1
            relationships.append(BlockRelationship(right.id, left.id, "same_section", ordinal))
            ordinal += 1
        if left.artifact_policy != right.artifact_policy:
            prose = left if left.artifact_policy == "no" else right
            artifact = right if prose is left else left
            relationships.append(BlockRelationship(prose.id, artifact.id, "has_adjacent_artifact", ordinal))
            ordinal += 1
    return relationships


def _normalize_source(text: str) -> str:
    return CONTROL_RE.sub("", ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")))


def _normalize_prose(text: str) -> str:
    text = _normalize_source(text)
    text = WRITING_WRAPPER_OPEN_RE.sub("", text)
    text = WRITING_WRAPPER_CLOSE_RE.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not PUNCTUATION_ONLY_RE.fullmatch(line)]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    offsets.append(len(text))
    return offsets


def _token_offsets(token: Token, offsets: list[int], text_length: int) -> tuple[int, int]:
    if token.map is None:
        return 0, text_length
    start_line, end_line = token.map
    return offsets[min(start_line, len(offsets) - 1)], offsets[min(end_line, len(offsets) - 1)]


def _matching_close(tokens: list[Token], start: int, close_type: str) -> int:
    depth = 0
    for index in range(start + 1, len(tokens)):
        token = tokens[index]
        if token.type == tokens[start].type:
            depth += 1
        elif token.type == close_type:
            if depth == 0:
                return index
            depth -= 1
    return len(tokens) - 1
