from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

from .config import AppConfig
from .models import ExportSource, InputBundle


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

    library_files: list[dict] = []
    library_path = root_dir / "library_files.json"
    if library_path.exists():
        data = _read_json(library_path)
        if isinstance(data, list):
            library_files = [item for item in data if isinstance(item, dict)]

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
