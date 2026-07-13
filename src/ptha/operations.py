"""Maintenance locking and recoverable operation state."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from ptha.config import PthaConfig
from ptha.errors import PthaError
from ptha.process import inspect_process


class MaintenanceError(PthaError):
    code = "maintenance_busy"
    exit_code = 8


@dataclass(frozen=True)
class MaintenanceState:
    operation: str
    phase: str
    started_at: str
    source_database: str
    temporary_database: str | None
    process_id: int
    process_start_time: float

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "operation": self.operation, "phase": self.phase,
                "started_at": self.started_at, "source_database": self.source_database,
                "temporary_database": self.temporary_database, "process_id": self.process_id,
                "process_start_time": self.process_start_time}


def maintenance_state_path(config: PthaConfig) -> Path:
    return config.paths.state_dir / "maintenance-state.json"


@contextmanager
def maintenance_lock(config: PthaConfig, *, timeout: float = 0.0) -> Iterator[None]:
    path = config.paths.state_dir / "maintenance.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MaintenanceError("Maintenance lock path must not be a symlink.")
    with path.open("a+b") as handle:
        os.chmod(path, 0o600)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MaintenanceError("Another PTHA import or reindex operation is running.")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def new_maintenance_state(config: PthaConfig, operation: str, source: Path, temporary: Path | None) -> MaintenanceState:
    identity = inspect_process(os.getpid())
    return MaintenanceState(operation, "starting", datetime.now(UTC).isoformat(), str(source),
                            str(temporary) if temporary else None, os.getpid(), identity.create_time if identity else 0.0)


def write_maintenance_state(config: PthaConfig, state: MaintenanceState, *, phase: str | None = None) -> MaintenanceState:
    if phase:
        state = MaintenanceState(state.operation, phase, state.started_at, state.source_database,
                                 state.temporary_database, state.process_id, state.process_start_time)
    path = maintenance_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MaintenanceError("Maintenance state path must not be a symlink.")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return state


def read_maintenance_state(config: PthaConfig) -> dict[str, Any] | None:
    path = maintenance_state_path(config)
    if not path.exists():
        return None
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise MaintenanceError("Maintenance state is not a safe regular file.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaintenanceError("Maintenance state is unreadable.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise MaintenanceError("Maintenance state schema is unsupported.")
    return value


def clear_maintenance_state(config: PthaConfig) -> None:
    path = maintenance_state_path(config)
    if not path.exists():
        return
    if path.is_symlink() or path.lstat().st_uid != os.geteuid():
        raise MaintenanceError("Maintenance state is not safe to remove automatically.")
    path.unlink()


def cleanup_stale_operations(config: PthaConfig, *, force_state: bool = False) -> list[str]:
    """Remove only inactive, owned operation markers and expected temporary DB files."""
    removed: list[str] = []
    with maintenance_lock(config):
        marker = read_maintenance_state(config)
        if marker:
            pid = int(marker.get("process_id", 0))
            expected_start = float(marker.get("process_start_time", 0.0))
            current = inspect_process(pid) if pid > 0 else None
            matching = bool(current and abs(current.create_time - expected_start) <= 0.01)
            if matching and not force_state:
                raise MaintenanceError("Maintenance state still belongs to a live process.")
            clear_maintenance_state(config)
            removed.append(str(maintenance_state_path(config)))
        import_marker = config.paths.state_dir / "import-state.json"
        if import_marker.exists():
            _unlink_owned_regular(import_marker)
            removed.append(str(import_marker))
        candidates = {config.database.with_name(config.database.name + ".reindexing")}
        candidates.update(config.paths.data_dir.glob("*.building"))
        candidates.update(config.paths.data_dir.glob(".*.building"))
        for candidate in candidates:
            if candidate.exists():
                _unlink_owned_regular(candidate)
                removed.append(str(candidate))
    return removed


def _unlink_owned_regular(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise MaintenanceError("Operation state path is not safe to remove automatically.")
    path.unlink()
