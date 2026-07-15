"""Lightweight and full operational diagnostics for PTHA."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from ptha import application_version
from ptha.config import PthaConfig
from ptha.database import inspect_database
from ptha.ipc import IPCError, request
from ptha.lifecycle import service_status
from ptha.operations import MaintenanceError, read_maintenance_state
from kb.storage.native_pre_mvp import NATIVE_PRE_MVP_SCHEMA_VERSION


@dataclass(frozen=True)
class Check:
    id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None


def run_doctor(config: PthaConfig, *, full: bool = False, query: str | None = None) -> dict[str, Any]:
    checks: list[Check] = []
    checks.append(_check("configuration.file", "pass" if config.config_file.exists() else "warn",
                         "Configuration loaded." if config.config_file.exists() else "Built-in defaults are active.",
                         {"path": str(config.config_file)}, "Run: ptha init"))
    checks.extend(_filesystem_checks(config))
    database = inspect_database(config.database, integrity=True)
    checks.extend(_database_checks(config, database))
    checks.extend(_operation_checks(config))
    lifecycle = service_status(config)
    checks.extend(_service_checks(config, lifecycle))
    checks.extend(_dependency_checks())
    checks.append(Check("archive.attachments", "warn", "Attachment content is not indexed in PTHA v1."))
    if full:
        checks.extend(_full_checks(config, lifecycle, query=query))
    counts = {name: sum(item.status == name for item in checks) for name in ("pass", "warn", "fail")}
    return {
        "schema_version": 1,
        "result": "fail" if counts["fail"] else "pass",
        "checks": [asdict(item) for item in checks],
        "summary": {"passed": counts["pass"], "warnings": counts["warn"], "failed": counts["fail"]},
    }


def _filesystem_checks(config: PthaConfig) -> list[Check]:
    checks: list[Check] = []
    for name, path in (("data", config.paths.data_dir), ("cache", config.paths.cache_dir),
                       ("state", config.paths.state_dir), ("runtime", config.paths.runtime_dir),
                       ("logs", config.paths.log_dir)):
        exists = path.is_dir()
        writable = exists and os.access(path, os.W_OK)
        checks.append(_check(f"filesystem.{name}", "pass" if writable else "fail",
                             f"{name.capitalize()} directory is writable." if writable else f"{name.capitalize()} directory is unavailable.",
                             {"path": str(path)}, "Run: ptha init"))
    runtime = config.paths.runtime_dir
    if runtime.exists():
        mode = runtime.stat().st_mode & 0o777
        checks.append(_check("filesystem.runtime_permissions", "pass" if mode & 0o077 == 0 else "warn",
                             "Runtime directory permissions are private." if mode & 0o077 == 0 else "Runtime directory is accessible by other users.",
                             {"mode": oct(mode)}, f"Run: chmod 700 '{runtime}'"))
    usage = shutil.disk_usage(config.paths.data_dir)
    checks.append(_check("filesystem.free_space", "pass" if usage.free >= 512 * 1024 * 1024 else "warn",
                         "Data filesystem has available working space.", {"free_bytes": usage.free},
                         "Free at least 512 MiB before import or reindex."))
    return checks


def _database_checks(config: PthaConfig, database: dict[str, Any]) -> list[Check]:
    state = database.get("state")
    checks = [_check("database.exists", "pass" if database.get("exists") else "fail",
                     "PTHA database exists." if database.get("exists") else "PTHA database is missing.",
                     {"path": str(config.database)}, "Run: ptha import /path/to/chatgpt-export.zip")]
    if not database.get("exists"):
        metadata = config.paths.data_dir / "archive-metadata.json"
        if metadata.exists():
            checks.append(_check("database.archive_metadata", "warn", "Archive metadata exists without a database.",
                                 {"path": str(metadata)}, "Inspect and remove stale metadata after confirming no DB is expected."))
        return checks
    checks.append(_check("database.layout", "pass" if state == "ready" else "fail",
                         "Clean-native database layout is ready." if state == "ready" else f"Database state is {state}.",
                         {"missing_tables": database.get("missing_tables", [])}, "Re-import the archive into a replacement database."))
    checks.append(_check("database.integrity", "pass" if database.get("integrity_check") == "ok" else "fail",
                         "SQLite integrity check passed." if database.get("integrity_check") == "ok" else "SQLite integrity check failed.",
                         {}, "Restore or re-import the database."))
    checks.append(_check("database.schema_version", "pass" if database.get("schema_version") == NATIVE_PRE_MVP_SCHEMA_VERSION else "fail",
                         "Database schema version is supported." if database.get("schema_version") == NATIVE_PRE_MVP_SCHEMA_VERSION else "Database schema version is unsupported.",
                         {"schema_version": database.get("schema_version")}, "Re-import with the current PTHA version."))
    mode = config.database.stat().st_mode & 0o777
    checks.append(_check("database.permissions", "pass" if mode & 0o077 == 0 else "warn",
                         "Database permissions are private." if mode & 0o077 == 0 else "Database is readable by other local users.",
                         {"mode": oct(mode)}, f"Run: chmod 600 '{config.database}'"))
    metadata = config.paths.data_dir / "archive-metadata.json"
    checks.append(_check("database.archive_metadata", "pass" if metadata.exists() else "warn",
                         "Archive metadata is present." if metadata.exists() else "Database exists without archive metadata.",
                         {"path": str(metadata)}, "Re-import to regenerate archive metadata."))
    counts = database.get("counts", {})
    chunks = counts.get("retrieval_chunks", -1)
    dense = counts.get("dense_native_metadata", -2)
    sparse = counts.get("sparse_vector_metadata", -3)
    checks.append(_check("database.derived_counts", "pass" if chunks == dense == sparse and chunks >= 0 else "fail",
                         "Canonical chunks and derived vector counts are consistent.",
                         {"chunks": chunks, "dense": dense, "sparse": sparse}, "Run: ptha reindex"))
    models = database.get("models", {})
    compatible = models.get("dense") == config.dense_model and models.get("sparse") == config.sparse_model
    checks.append(_check("database.model_metadata", "pass" if compatible else "fail",
                         "Configured models match database metadata." if compatible else "Configured models do not match database metadata.",
                         {"dense": models.get("dense"), "sparse": models.get("sparse")},
                         "Use the models recorded by the DB or run ptha reindex."))
    checks.append(_check("database.chunk_policy", "pass" if database.get("chunk_policy") else "fail",
                         "Chunk policy metadata is present." if database.get("chunk_policy") else "Chunk policy metadata is missing.",
                         {"chunk_policy": database.get("chunk_policy")}, "Re-import or reindex the database."))
    try:
        uri = f"file:{config.database.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            legacy = [name for name in ("knowledge_blocks", "dense_vectors", "sparse_terms")
                      if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()]
    except sqlite3.Error:
        legacy = ["unknown"]
    checks.append(_check("database.no_legacy_fallback", "pass" if not legacy else "fail",
                         "No legacy retrieval fallback tables are required." if not legacy else "Legacy retrieval tables are present.",
                         {"legacy_tables": legacy}, "Build a clean-native PTHA database."))
    return checks


def _operation_checks(config: PthaConfig) -> list[Check]:
    checks: list[Check] = []
    try:
        marker = read_maintenance_state(config)
    except MaintenanceError as exc:
        marker = {"error": type(exc).__name__}
    checks.append(_check("operations.maintenance_state", "warn" if marker else "pass",
                         "Incomplete maintenance operation is recorded." if marker else "No incomplete maintenance operation is recorded.",
                         marker or {}, "Inspect doctor output, then rerun the operation or remove proven stale state."))
    import_marker = config.paths.state_dir / "import-state.json"
    checks.append(_check("operations.import_state", "warn" if import_marker.exists() else "pass",
                         "Incomplete import state is present." if import_marker.exists() else "No incomplete import state is present.",
                         {"path": str(import_marker)}, "Rerun ptha import after confirming the service is stopped."))
    temporary = ([str(path) for path in config.paths.data_dir.glob("*.building")] +
                 [str(path) for path in config.paths.data_dir.glob(".*.building")] +
                 [str(path) for path in config.paths.data_dir.glob("*.reindexing")])
    checks.append(_check("operations.temporary_databases", "warn" if temporary else "pass",
                         "Orphan temporary databases are present." if temporary else "No orphan build databases are present.",
                         {"paths": temporary}, "Use ptha service cleanup and retry the interrupted operation."))
    return checks


def _service_checks(config: PthaConfig, lifecycle: dict[str, Any]) -> list[Check]:
    state = lifecycle["state"]
    status = "pass" if state in {"ready", "stopped"} else "warn"
    checks = [_check("service.lifecycle", status, f"Service state is {state}.",
                     {"state": state, "pid": lifecycle.get("pid")}, "Run: ptha service cleanup")]
    if lifecycle.get("ipc_ready"):
        try:
            internal = request(config.paths.socket, "status", timeout_ms=500)
        except IPCError:
            internal = {}
        same_database = internal.get("database_path") == str(config.database)
        checks.append(_check("service.database", "pass" if same_database else "fail",
                             "Service uses the configured database." if same_database else "Service database differs from configuration.",
                             {}, "Stop the service and start it with the intended configuration."))
    return checks


def _dependency_checks() -> list[Check]:
    checks = [_check("runtime.python", "pass" if sys.version_info >= (3, 13) else "fail",
                     "Python version is supported.", {"version": sys.version.split()[0]}, "Install Python 3.13 or newer."),
              Check("runtime.package", "pass", "PTHA package metadata is available.", {"version": application_version()})]
    runtime_modules = ["platformdirs", "psutil", "sqlite_vec", "huggingface_hub", "transformers"]
    if sys.platform == "darwin" and platform.machine() == "arm64":
        runtime_modules.extend(("mlx", "mlx_embeddings"))
    for module in runtime_modules:
        try:
            importlib.import_module(module)
            status = "pass"
        except ImportError:
            status = "fail"
        checks.append(_check(f"dependency.{module}", status, f"Dependency {module} is importable." if status == "pass" else f"Dependency {module} is missing.",
                             {}, "Reinstall PTHA with uv tool install."))
    try:
        from kb.storage.dense_native import load_sqlite_vec
        with closing(sqlite3.connect(":memory:")) as conn:
            load_sqlite_vec(conn)
        vector_status = "pass"
    except Exception:  # noqa: BLE001
        vector_status = "fail"
    checks.append(_check("dependency.sqlite_vec_load", vector_status,
                         "sqlite-vec extension loads successfully." if vector_status == "pass" else "sqlite-vec extension cannot be loaded.",
                         {}, "Reinstall sqlite-vec for this Python runtime."))
    return checks


def _full_checks(config: PthaConfig, lifecycle: dict[str, Any], *, query: str | None) -> list[Check]:
    probe = query or "PTHA runtime compatibility smoke test"
    metrics: dict[str, Any] = {}
    try:
        if lifecycle.get("ipc_ready") and lifecycle.get("state") == "ready":
            started = time.perf_counter()
            request(config.paths.socket, "search_archive", {"query": probe, "limit": 1, "max_tokens": 100}, timeout_ms=30_000)
            metrics["focused_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            request(config.paths.socket, "construct_archive_context", {"current_context": probe, "max_tokens": 100}, timeout_ms=30_000)
            metrics["broad_seconds"] = time.perf_counter() - started
            metrics["mode"] = "live_service"
        else:
            from kb.mcp.archive import ArchiveConfig, ArchiveSession
            from ptha.service import build_providers
            started = time.perf_counter()
            dense, sparse = build_providers(config)
            metrics["model_load_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            session = ArchiveSession(ArchiveConfig(config.database, config.candidate_pool,
                                                   config.default_output_tokens, config.max_output_tokens), dense, sparse)
            metrics["session_initialization_seconds"] = time.perf_counter() - started
            try:
                started = time.perf_counter()
                session.search_archive({"query": probe, "limit": 1, "max_tokens": 100, "timeout_ms": 30_000})
                metrics["focused_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                session.construct_archive_context({"current_context": probe, "max_tokens": 100, "timeout_ms": 30_000})
                metrics["broad_seconds"] = time.perf_counter() - started
            finally:
                session.close()
            metrics["mode"] = "local_session"
        metrics["rss_bytes"] = psutil.Process().memory_info().rss
        return [Check("runtime.full_retrieval", "pass", "Models, embedding spaces, and archive operations are compatible.", metrics)]
    except Exception as exc:  # noqa: BLE001
        return [Check("runtime.full_retrieval", "fail", "Full retrieval compatibility check failed.",
                      {"error_class": type(exc).__name__}, "Verify model cache/configuration and run ptha doctor --full --debug.")]


def _check(identifier: str, status: str, message: str, details: dict[str, Any], remediation: str | None) -> Check:
    return Check(identifier, status, message, details, remediation if status != "pass" else None)
