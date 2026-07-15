"""Authoritative raw-export to clean-native PTHA import workflow."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import closing
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from gpt_export_distillation.config import DEFAULT_CONFIG as DISTILL_CONFIG
from gpt_export_distillation.loader import load_bundle
from gpt_export_distillation.pipeline import build_documents, write_output
from kb.storage.native_pre_mvp import build_native_pre_mvp_db
from ptha import application_version
from ptha.config import PthaConfig
from ptha.database import inspect_database
from ptha.errors import DatabaseExistsError, PthaError
from ptha.operations import maintenance_lock


class ImportFailedError(PthaError):
    code = "import_failed"
    exit_code = 6


def import_archive(
    source: Path,
    config: PthaConfig,
    *,
    replace: bool = False,
    keep_distilled: bool = False,
    include_low_interest: bool = False,
    discard_failed: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    with maintenance_lock(config):
        return _import_archive(config=config, source=source, replace=replace, keep_distilled=keep_distilled,
                               include_low_interest=include_low_interest, discard_failed=discard_failed, progress=progress)


def _import_archive(
    source: Path,
    config: PthaConfig,
    *,
    replace: bool,
    keep_distilled: bool,
    include_low_interest: bool,
    discard_failed: bool,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise ImportFailedError(f"Archive source does not exist: {source}")
    active = config.database
    if replace:
        from ptha.lifecycle import service_status
        lifecycle = service_status(config)
        if lifecycle["state"] in {"ready", "starting", "degraded", "unknown-process"}:
            raise ImportFailedError(
                "PTHA service must be stopped before replacing the database.\n\nRun:\n  ptha service stop"
            )
    if active.exists() and not replace:
        raise DatabaseExistsError(f"Database already exists: {active}. Use --replace to rebuild it safely.")
    active.parent.mkdir(parents=True, exist_ok=True)
    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
    state_file = config.paths.state_dir / "import-state.json"
    previous_state = _read_state(state_file)
    source_identity = _source_identity(source)
    resume_state = None
    if previous_state and previous_state.get("status") in {"failed", "interrupted", "running"}:
        if previous_state.get("source_identity") == source_identity and previous_state.get("workspace"):
            resume_state = previous_state
        elif not discard_failed:
            raise ImportFailedError(
                "An incomplete import exists for another source. "
                "Resume that import or rerun with --discard-failed to remove its staging data."
            )
    if discard_failed and previous_state:
        _discard_state_files(previous_state)
        state_file.unlink(missing_ok=True)
        previous_state = None
    protected = []
    if resume_state and resume_state.get("staging"):
        protected.extend(_staging_family(Path(str(resume_state["staging"]))))
    removed_orphans = cleanup_orphan_import_files(active, protected=protected)
    if removed_orphans:
        _stage(progress, f"Removed {removed_orphans} stale import files.")
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    resume = resume_state is not None
    if resume:
        staging = Path(str(resume_state["staging"]))
        workspace = Path(str(resume_state["workspace"]))
        if not workspace.exists():
            raise ImportFailedError("Import checkpoint workspace is missing; rerun with --discard-failed.")
    else:
        staging = active.with_name(f".{active.name}.{os.getpid()}.staging")
        workspace_parent = config.working_dir or (config.paths.state_dir / "imports")
        workspace_parent.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="ptha-import-", dir=workspace_parent))
    distilled = workspace / "distilled"
    kept_path: Path | None = None
    _write_state(state_file, "running", "reading_export", started_at, source_identity=source_identity,
                 staging=staging, workspace=workspace, distilled=distilled)
    try:
        distilled_root = distilled
        if resume and distilled.exists():
            _stage(progress, "[resume] Reusing completed distillation workspace")
        else:
            _stage(progress, "[1/7] Reading export")
            bundle = load_bundle(source)
            _stage(progress, "[2/7] Distilling conversations")
            documents = build_documents(bundle, DISTILL_CONFIG)
            distilled_root = Path(write_output(bundle, documents, DISTILL_CONFIG, str(distilled)))
        _write_state(state_file, "running", "building_native_database", started_at,
                     source_identity=source_identity, staging=staging, workspace=workspace, distilled=distilled)
        resume_build = resume and staging.with_name(staging.name + ".building").exists()
        _stage(progress, "[resume] Reusing canonical/chunk checkpoint" if resume_build else "[3/7] Importing canonical content")
        audit = build_native_pre_mvp_db(
            export_path=distilled_root,
            output_db=staging,
            dense_model=config.dense_model,
            sparse_model=config.sparse_model,
            model_revision=config.embedding_model_revision,
            embedding_device=config.embedding_device,
            embedding_dtype=config.embedding_dtype,
            embedding_max_padded_tokens=config.embedding_max_padded_tokens,
            sparse_head=config.embedding_sparse_head,
            colbert_head=config.embedding_colbert_head,
            model_cache=config.model_cache,
            sparse_top_k=config.sparse_top_k,
            batch_size=config.batch_size,
            skip_low_interest=not include_low_interest,
            progress=progress is not None,
            resume=resume_build,
        )
        with closing(sqlite3.connect(staging)) as conn:
            conn.execute("UPDATE native_build_audit SET export_path=? WHERE id=1", (source.name,))
            conn.execute("UPDATE source_documents SET path=relative_path")
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _stage(progress, "[7/7] Validating database")
        validation = inspect_database(staging, integrity=True)
        if validation.get("state") != "ready" or validation.get("integrity_check") != "ok":
            raise ImportFailedError("The replacement database failed validation.")
        os.replace(staging, active)
        published = True
        os.chmod(active, 0o600)
        completed_at = datetime.now(UTC)
        metadata = {
            "schema_version": 1,
            "source_type": "chatgpt_export_zip" if source.is_file() else "chatgpt_export_directory",
            "source_name": source.name,
            "import_started_at": started_at.isoformat(),
            "import_completed_at": completed_at.isoformat(),
            "application_version": application_version(),
            "database_schema_version": validation.get("schema_version"),
            "dense_model": config.dense_model,
            "sparse_model": config.sparse_model,
            "chunk_policy": audit.get("contracts", {}).get("chunk_policy"),
            "conversation_count": validation["counts"]["conversations"],
            "message_count": validation["counts"]["messages"],
            "retrieval_chunk_count": validation["counts"]["retrieval_chunks"],
        }
        (config.paths.data_dir / "archive-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(config.paths.data_dir / "archive-metadata.json", 0o600)
        if keep_distilled:
            kept_path = config.paths.data_dir / "distilled-archive"
            if kept_path.exists():
                shutil.rmtree(kept_path)
            shutil.move(str(distilled_root), kept_path)
        state_file.unlink(missing_ok=True)
        return {"metadata": metadata, "database": str(active), "size_bytes": active.stat().st_size,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "distilled_archive": str(kept_path) if kept_path else None,
                "resumed": resume}
    except KeyboardInterrupt:
        _write_state(state_file, "interrupted", "building_native_database", started_at,
                     source_identity=source_identity, staging=staging, workspace=workspace, distilled=distilled)
        raise
    except PthaError:
        _write_state(state_file, "failed", "building_native_database", started_at,
                     source_identity=source_identity, staging=staging, workspace=workspace, distilled=distilled)
        raise
    except Exception as exc:  # noqa: BLE001
        _write_state(state_file, "failed", "building_native_database", started_at,
                     source_identity=source_identity, staging=staging, workspace=workspace, distilled=distilled)
        raise ImportFailedError(
            "Archive import failed; the active database was not changed. "
            "A resumable checkpoint was preserved. Re-run the same command to continue, "
            "or add --discard-failed to start over."
        ) from exc
    finally:
        if locals().get("published", False):
            for candidate in _staging_family(staging):
                candidate.unlink(missing_ok=True)
            shutil.rmtree(workspace, ignore_errors=True)


def _stage(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def cleanup_orphan_import_files(active: Path, *, protected: list[Path] | None = None) -> int:
    """Remove import staging files only when their recorded process is dead."""
    removed = 0
    prefix = f".{active.name}."
    protected_resolved = {path.expanduser().resolve() for path in (protected or [])}
    for candidate in active.parent.glob(f".{active.name}.*.staging*"):
        if candidate.expanduser().resolve() in protected_resolved:
            continue
        if candidate.is_symlink() or not candidate.is_file():
            continue
        suffix = candidate.name.removeprefix(prefix)
        pid_text = suffix.split(".", 1)[0]
        if not pid_text.isdigit() or _pid_is_alive(int(pid_text)):
            continue
        candidate.unlink()
        removed += 1
    return removed


def _staging_family(staging: Path) -> tuple[Path, ...]:
    building = staging.with_name(staging.name + ".building")
    return (
        staging, staging.with_name(staging.name + "-wal"), staging.with_name(staging.name + "-shm"),
        building, building.with_name(building.name + "-wal"), building.with_name(building.name + "-shm"),
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _source_identity(source: Path) -> dict[str, Any]:
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "is_file": source.is_file()}


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _discard_state_files(state: dict[str, Any]) -> None:
    for key in ("staging", "workspace"):
        value = state.get(key)
        if not value:
            continue
        path = Path(str(value))
        if key == "workspace":
            shutil.rmtree(path, ignore_errors=True)
        else:
            for candidate in _staging_family(path):
                candidate.unlink(missing_ok=True)


def _write_state(path: Path, status: str, stage: str, started_at: datetime, **extra: Any) -> None:
    payload = {"schema_version": 1, "status": status, "stage": stage, "started_at": started_at.isoformat()}
    for key, value in extra.items():
        payload[key] = str(value) if isinstance(value, Path) else value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
