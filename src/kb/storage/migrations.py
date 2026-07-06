from __future__ import annotations

from pathlib import Path

from kb.storage.sqlite_store import init_db


def ensure_schema(db_path: Path) -> None:
    init_db(db_path)
