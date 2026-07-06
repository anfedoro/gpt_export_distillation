from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from kb.model.entities import InventoryItem


CHAT_INDEX_NAMES = {"INDEX.md", "SUMMARY.md", "ATTACHMENTS.md", "FILES.md", "LIBRARY_FILES.md"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_folder_kind(relative_path: Path) -> tuple[str | None, str | None]:
    parts = relative_path.parts
    if len(parts) >= 2 and parts[0] == "Common" and parts[1] == "useful":
        return "common_useful", None
    if len(parts) >= 2 and parts[0] == "Common" and parts[1] == "potential_trash":
        return "common_trash", None
    if parts and parts[0] == "Pinned":
        return "pinned", None
    if len(parts) >= 2 and parts[0] == "Projects":
        return "project", parts[1]
    return None, None


def detect_kind(path: Path, relative_path: Path) -> tuple[str, bool]:
    if path.name in CHAT_INDEX_NAMES:
        if path.name == "INDEX.md":
            return "index_md", False
        if path.name == "SUMMARY.md":
            return "summary_md", False
        return "attachment_index_md", False
    if "attachments" in relative_path.parts:
        return "attachment", True
    if path.suffix.lower() == ".md":
        return "chat_md", False
    return "attachment", True


def interest_tier_for_folder(folder_kind: str | None) -> str:
    if folder_kind == "common_trash":
        return "low"
    return "normal"


def scan_tree(input_dir: Path) -> Iterable[InventoryItem]:
    root = input_dir.expanduser().resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        folder_kind, project_path = detect_folder_kind(relative_path)
        detected_kind, is_attachment = detect_kind(path, relative_path)
        yield InventoryItem(
            relative_path=relative_path.as_posix(),
            file_name=path.name,
            extension=path.suffix.lower().lstrip("."),
            size=path.stat().st_size,
            sha256=sha256_file(path),
            detected_kind=detected_kind,
            folder_kind=folder_kind,
            project_path=project_path,
            is_attachment=is_attachment,
            interest_tier=interest_tier_for_folder(folder_kind),
        )


def write_inventory_jsonl(items: Iterable[InventoryItem], output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
