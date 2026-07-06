from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

from .config import AppConfig
from .models import ExportSource, InputBundle, LibraryFileRecord


def discover_inputs(config: AppConfig, explicit_input: str | None) -> list[Path]:
    if explicit_input:
        return [Path(explicit_input).expanduser().resolve()]
    base = Path(config.input.search_dir).expanduser().resolve()
    paths = sorted(base.glob(config.input.conversations_glob))
    if config.input.include_zip:
        paths.extend(sorted(base.glob(config.input.zip_glob)))
    return sorted(set(paths))


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _string_tuple(value) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items = [item for item in value if isinstance(item, str) and item]
    return tuple(items)


def _string_value(value) -> str | None:
    return value if isinstance(value, str) and value else None


def _parse_library_file(item: dict) -> LibraryFileRecord:
    knowledge_store = item.get("knowledge_store_id")
    if isinstance(knowledge_store, dict):
        knowledge_store_id = knowledge_store.get("id")
        knowledge_store_kind = knowledge_store.get("kind")
    else:
        knowledge_store_id = None
        knowledge_store_kind = None
    file_id = _string_value(item.get("file_id")) or ""
    file_name = _string_value(item.get("file_name")) or _string_value(item.get("normalized_name")) or file_id or "unknown"
    normalized_name = _string_value(item.get("normalized_name")) or file_name
    is_project = item.get("is_project")
    if not isinstance(is_project, bool):
        is_project = None
    pinned_at = _string_value(item.get("pinned_at"))
    mime_type = _string_value(item.get("mime_type"))
    library_file_category = _string_value(item.get("library_file_category"))
    directory_id = _string_value(item.get("directory_id"))
    origination_thread_id = _string_value(item.get("origination_thread_id"))
    origination_message_id = _string_value(item.get("origination_message_id"))
    return LibraryFileRecord(
        raw=item,
        file_id=file_id,
        file_name=file_name,
        normalized_name=normalized_name,
        mime_type=mime_type,
        library_file_category=library_file_category,
        directory_id=directory_id,
        knowledge_store_id=knowledge_store_id if isinstance(knowledge_store_id, str) else None,
        knowledge_store_kind=knowledge_store_kind if isinstance(knowledge_store_kind, str) else None,
        origination_thread_id=origination_thread_id,
        origination_message_id=origination_message_id,
        is_project=is_project,
        pinned_at=pinned_at,
        context_scopes=_string_tuple(item.get("context_scopes")),
        context_scopes_v2=_string_tuple(item.get("context_scopes_v2")),
        created_at=item.get("created_at") if isinstance(item.get("created_at"), str) else None,
        updated_at=item.get("updated_at") if isinstance(item.get("updated_at"), str) else None,
    )


def _bundle_from_dir(root_dir: Path, output_dir: Path, label: str, kind: str) -> InputBundle:
    conversations: list[dict] = []
    for path in sorted(root_dir.glob("conversations-*.json")):
        data = _read_json(path)
        if isinstance(data, list):
            conversations.extend(item for item in data if isinstance(item, dict))

    asset_file_names = {}
    asset_path = root_dir / "conversation_asset_file_names.json"
    if asset_path.exists():
        data = _read_json(asset_path)
        if isinstance(data, dict):
            asset_file_names = {
                str(key): str(value) for key, value in data.items() if isinstance(value, str)
            }

    library_files: list[LibraryFileRecord] = []
    library_path = root_dir / "library_files.json"
    if library_path.exists():
        data = _read_json(library_path)
        if isinstance(data, list):
            library_files = [_parse_library_file(item) for item in data if isinstance(item, dict)]

    shared_conversations: list[dict] = []
    shared_path = root_dir / "shared_conversations.json"
    if shared_path.exists():
        data = _read_json(shared_path)
        if isinstance(data, list):
            shared_conversations = [item for item in data if isinstance(item, dict)]

    manifest = None
    manifest_path = root_dir / "export_manifest.json"
    if manifest_path.exists():
        data = _read_json(manifest_path)
        if isinstance(data, dict):
            manifest = data

    return InputBundle(
        source=ExportSource(label=label, root_dir=root_dir, output_dir=output_dir, kind=kind),
        conversations=conversations,
        asset_file_names=asset_file_names,
        library_files=library_files,
        shared_conversations=shared_conversations,
        manifest=manifest,
    )


def load_bundle(path: Path) -> InputBundle:
    path = path.expanduser().resolve()
    if path.is_dir():
        return _bundle_from_dir(path, output_dir=path, label=path.name, kind="directory")
    if path.suffix.lower() == ".json":
        return _bundle_from_dir(
            path.parent,
            output_dir=path.parent,
            label=path.name,
            kind="json-file",
        )
    if path.suffix.lower() == ".zip":
        tmp_dir = mkdtemp(prefix="gpt_export_distill_")
        with ZipFile(path) as archive:
            archive.extractall(tmp_dir)
        root_dir = Path(tmp_dir)
        return _bundle_from_dir(
            root_dir,
            output_dir=path.parent,
            label=path.stem,
            kind="zip",
        )
    raise ValueError(f"Unsupported input path: {path}")
