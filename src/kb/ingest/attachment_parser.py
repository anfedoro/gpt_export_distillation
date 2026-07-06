from __future__ import annotations

import json
import mimetypes
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


TEXT_EXTENSIONS = {"md", "txt", "json", "jsonl", "yaml", "yml"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {"pdf", "docx", "pptx"}


@dataclass(frozen=True)
class AttachmentBlock:
    block_type: str
    text: str
    ordinal: int
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedAttachment:
    mime_type: str | None
    extraction_status: str
    blocks: list[AttachmentBlock]
    metadata_json: dict[str, Any] = field(default_factory=dict)


def parse_attachment(path: Path) -> ParsedAttachment:
    extension = path.suffix.lower().lstrip(".")
    mime_type = mimetypes.guess_type(path.name)[0]
    if extension not in SUPPORTED_EXTENSIONS:
        return ParsedAttachment(
            mime_type=mime_type,
            extraction_status="unsupported",
            blocks=[],
            metadata_json={"reason": "unsupported_extension"},
        )
    try:
        if extension in TEXT_EXTENSIONS:
            return _parse_text_attachment(path, extension, mime_type)
        if extension == "docx":
            return _parse_docx(path, mime_type)
        if extension == "pptx":
            return _parse_pptx(path, mime_type)
        if extension == "pdf":
            return _parse_pdf(path, mime_type)
    except Exception as exc:  # noqa: BLE001
        return ParsedAttachment(
            mime_type=mime_type,
            extraction_status="failed",
            blocks=[],
            metadata_json={"error": str(exc), "error_type": type(exc).__name__},
        )
    return ParsedAttachment(
        mime_type=mime_type,
        extraction_status="unsupported",
        blocks=[],
        metadata_json={"reason": "unsupported_extension"},
    )


def _parse_text_attachment(path: Path, extension: str, mime_type: str | None) -> ParsedAttachment:
    text = path.read_text(encoding="utf-8")
    block_type = "markdown_text" if extension == "md" else "text"
    if extension in {"json", "jsonl"}:
        block_type = "json_text"
    elif extension in {"yaml", "yml"}:
        block_type = "yaml_text"
    return ParsedAttachment(
        mime_type=mime_type,
        extraction_status="extracted",
        blocks=[AttachmentBlock(block_type=block_type, text=text, ordinal=1, metadata_json={})] if text.strip() else [],
        metadata_json={},
    )


def _parse_docx(path: Path, mime_type: str | None) -> ParsedAttachment:
    blocks: list[AttachmentBlock] = []
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    ordinal = 1
    for paragraph in root.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = _joined_xml_text(paragraph)
        if not text.strip():
            continue
        blocks.append(
            AttachmentBlock(
                block_type="docx_paragraph",
                text=text,
                ordinal=ordinal,
                metadata_json={"paragraph_number": ordinal},
            )
        )
        ordinal += 1
    return ParsedAttachment(mime_type=mime_type, extraction_status="extracted", blocks=blocks, metadata_json={})


def _parse_pptx(path: Path, mime_type: str | None) -> ParsedAttachment:
    blocks: list[AttachmentBlock] = []
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            name
            for name in archive.namelist()
            if re.match(r"ppt/slides/slide\d+\.xml$", name)
        )
        for ordinal, name in enumerate(slide_names, start=1):
            root = ElementTree.fromstring(archive.read(name))
            text = _joined_xml_text(root)
            if not text.strip():
                continue
            blocks.append(
                AttachmentBlock(
                    block_type="slide_text",
                    text=text,
                    ordinal=ordinal,
                    metadata_json={"slide_number": ordinal, "zip_member": name},
                )
            )
    return ParsedAttachment(mime_type=mime_type, extraction_status="extracted", blocks=blocks, metadata_json={})


def _parse_pdf(path: Path, mime_type: str | None) -> ParsedAttachment:
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return ParsedAttachment(
            mime_type=mime_type,
            extraction_status="failed",
            blocks=[],
            metadata_json={"error": f"pdfplumber unavailable: {exc}", "error_type": type(exc).__name__},
        )
    blocks: list[AttachmentBlock] = []
    with pdfplumber.open(path) as pdf:
        for ordinal, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            blocks.append(
                AttachmentBlock(
                    block_type="pdf_page_text",
                    text=text,
                    ordinal=ordinal,
                    metadata_json={"page_number": ordinal},
                )
            )
    return ParsedAttachment(mime_type=mime_type, extraction_status="extracted", blocks=blocks, metadata_json={})


def _joined_xml_text(root: ElementTree.Element) -> str:
    chunks = [text for text in root.itertext() if text and text.strip()]
    return "\n".join(chunk.strip() for chunk in chunks)
