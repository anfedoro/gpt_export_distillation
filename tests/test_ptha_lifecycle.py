from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ptha.config import load_config
from ptha.lifecycle import (
    LifecycleError,
    ServiceState,
    lifecycle_lock,
    read_state,
    service_status,
    state_path,
    write_state,
)
from ptha.paths import PthaPaths
from ptha.process import ProcessIdentity, identity_matches, inspect_process


class LifecycleUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        paths = PthaPaths(root / "config", root / "data", root / "cache", root / "state", root / "logs", root / "run")
        paths.config_dir.mkdir()
        paths.config_file.write_text("config_version = 1\n", encoding="utf-8")
        self.config = replace(load_config(paths.config_file), paths=paths, database=paths.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_state_metadata_round_trip_and_permissions(self) -> None:
        identity = inspect_process(os.getpid())
        self.assertIsNotNone(identity)
        state = ServiceState(identity, str(self.config.database), str(self.config.paths.socket),
                             "2026-07-13T00:00:00+00:00", instance_id="test-instance-identity-token")
        write_state(self.config, state)
        self.assertEqual(read_state(self.config), state)
        self.assertEqual(state_path(self.config).stat().st_mode & 0o777, 0o600)
        payload = json.loads(state_path(self.config).read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("process_start_time", payload)

    def test_identity_comparison_rejects_pid_reuse(self) -> None:
        current = inspect_process(os.getpid())
        self.assertIsNotNone(current)
        reused = ProcessIdentity(current.pid, current.create_time + 10, current.executable, current.command)
        self.assertFalse(identity_matches(reused, current))

    def test_status_classifies_unknown_process_without_signalling(self) -> None:
        current = inspect_process(os.getpid())
        self.assertIsNotNone(current)
        wrong = ProcessIdentity(current.pid, current.create_time + 10, current.executable, current.command)
        write_state(self.config, ServiceState(wrong, str(self.config.database), str(self.config.paths.socket),
                                              "2026-07-13T00:00:00+00:00", instance_id="test-instance-identity-token"))
        status = service_status(self.config)
        self.assertEqual(status["state"], "unknown-process")
        self.assertFalse(status["process_identity_valid"])

    def test_status_classifies_dead_state_as_stale(self) -> None:
        dead = ProcessIdentity(999_999_999, 1.0, "/missing/python", ("ptha", "service", "run"))
        write_state(self.config, ServiceState(dead, str(self.config.database), str(self.config.paths.socket),
                                              "2026-07-13T00:00:00+00:00", instance_id="test-instance-identity-token"))
        self.assertEqual(service_status(self.config)["state"], "stale-state")

    def test_lifecycle_lock_is_os_level_and_bounded(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def owner() -> None:
            with lifecycle_lock(self.config):
                entered.set()
                release.wait(2)

        thread = threading.Thread(target=owner)
        thread.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaises(LifecycleError) as raised:
            with lifecycle_lock(self.config, timeout=0.05):
                pass
        self.assertEqual(raised.exception.code, "service_busy")
        release.set()
        thread.join(timeout=1)

    def test_logging_rotation_config_loads(self) -> None:
        self.config.config_file.write_text(
            "config_version = 1\n[logging]\nservice_max_bytes = 1234\nservice_backup_count = 2\n", encoding="utf-8"
        )
        loaded = load_config(self.config.config_file)
        self.assertEqual(loaded.service_max_bytes, 1234)
        self.assertEqual(loaded.service_backup_count, 2)

    def test_stop_identity_mismatch_never_sends_signal(self) -> None:
        from ptha.lifecycle import stop_service
        current = inspect_process(os.getpid())
        self.assertIsNotNone(current)
        wrong = ProcessIdentity(current.pid, current.create_time + 10, current.executable, current.command)
        write_state(self.config, ServiceState(wrong, str(self.config.database), str(self.config.paths.socket),
                                              "2026-07-13T00:00:00+00:00", instance_id="test-instance-identity-token"))
        with patch("ptha.lifecycle.send_termination") as terminate, self.assertRaises(LifecycleError) as raised:
            stop_service(self.config)
        self.assertEqual(raised.exception.code, "service_identity_mismatch")
        terminate.assert_not_called()


class DetachedLifecycleIntegrationTests(unittest.TestCase):
    def test_real_detached_start_restart_stop(self) -> None:
        from ptha.lifecycle import restart_service, start_service, stop_service
        from ptha.paths import platform_paths

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            bin_dir = root_path / "bin"
            bin_dir.mkdir()
            wrapper = bin_dir / "ptha"
            repository = Path(__file__).resolve().parents[1]
            wrapper.write_text(
                f"#!{os.sys.executable}\nimport sys\nsys.path.insert(0, {str(repository)!r})\n"
                "from tests.ptha_fake_service import main\nmain()\n", encoding="utf-8"
            )
            wrapper.chmod(0o700)
            runtime = Path("/tmp") / f"ptha-test-{os.getpid()}-{time.time_ns()}"
            with patch.dict(os.environ, {"HOME": str(root_path), "PTHA_RUNTIME_DIR": str(runtime),
                                         "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}):
                paths = platform_paths()
                paths.config_dir.mkdir(parents=True)
                paths.data_dir.mkdir(parents=True, exist_ok=True)
                paths.config_file.write_text(
                    f"config_version = 1\n[paths]\ndatabase = '{paths.database}'\n"
                    "[models]\ndense_model = 'synthetic-dense'\nsparse_model = 'synthetic-sparse'\n",
                    encoding="utf-8",
                )
                _ready_database(paths.database)
                config = load_config(paths.config_file)
                starts: list[dict[str, object]] = []
                errors: list[Exception] = []

                def concurrent_start() -> None:
                    try:
                        starts.append(start_service(config, timeout=5))
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)

                starters = [threading.Thread(target=concurrent_start) for _ in range(2)]
                for starter in starters:
                    starter.start()
                for starter in starters:
                    starter.join(timeout=6)
                self.assertEqual(errors, [])
                self.assertEqual(len(starts), 2)
                self.assertEqual(sum(not bool(item.get("already_running")) for item in starts), 1)
                started = next(item for item in starts if not item.get("already_running"))
                first_pid = started["pid"]
                self.assertEqual(started["state"], "ready")
                self.assertTrue(config.paths.service_log.exists())
                duplicate = start_service(config, timeout=5)
                self.assertTrue(duplicate["already_running"])
                self.assertEqual(duplicate["pid"], first_pid)
                from ptha.mcp import MCPAdapter
                adapter_response = MCPAdapter(config).handle(
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": "search_archive", "arguments": {"query": "synthetic"}}}
                )
                self.assertFalse(adapter_response["result"]["isError"])
                from ptha.importer import ImportFailedError, import_archive
                source = root_path / "export.zip"
                source.touch()
                with self.assertRaises(ImportFailedError):
                    import_archive(source, config, replace=True)
                restarted = restart_service(config, start_timeout=5, stop_timeout=5)
                self.assertEqual(restarted["old_pid"], first_pid)
                self.assertNotEqual(restarted["new_pid"], first_pid)
                stopped = stop_service(config, timeout=5)
                self.assertEqual(stopped["state"], "stopped")
                self.assertFalse(config.paths.socket.exists())
                self.assertFalse(state_path(config).exists())


def _ready_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
            CREATE TABLE source_documents(id TEXT);
            CREATE TABLE conversations(id TEXT);
            CREATE TABLE messages(id TEXT);
            CREATE TABLE blocks(id TEXT);
            CREATE TABLE retrieval_chunks(id TEXT);
            CREATE TABLE dense_native_metadata(id TEXT);
            CREATE TABLE sparse_vector_metadata(id TEXT);
            CREATE TABLE sparse_vectors_compact(id TEXT);
            CREATE TABLE native_build_audit(id INTEGER, schema_version TEXT, status TEXT, started_at TEXT,
                                            finished_at TEXT, contracts_json TEXT);
            INSERT INTO native_build_audit VALUES(1, 'synthetic-v1', 'completed', '2026-01-01', '2026-01-01', '{}');
        """)
        conn.commit()


if __name__ == "__main__":
    unittest.main()
