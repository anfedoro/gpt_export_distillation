from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import unittest
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ptha.config import load_config
from ptha.doctor import run_doctor
from ptha.lifecycle import cleanup_service_state
from ptha.operations import maintenance_state_path
from ptha.paths import PthaPaths
from ptha.reindex import canonical_fingerprint, reindex_database
from ptha.importer import cleanup_orphan_import_files, import_archive
from ptha.mcp import MCPAdapter
from ptha.service import RetrievalService, build_providers
from ptha.ipc import request
from tests.test_ptha_runtime import (
    FakeSession,
    SyntheticDenseProvider,
    SyntheticSparseProvider,
    _create_retrievable_database,
    _wait_ready,
)


class OperationalCompletenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        paths = PthaPaths(root / "config", root / "data", root / "cache", root / "state", root / "logs", root / "run")
        for directory in (paths.config_dir, paths.data_dir, paths.cache_dir, paths.state_dir, paths.log_dir, paths.runtime_dir):
            directory.mkdir(parents=True, exist_ok=True)
        paths.config_file.write_text("config_version = 1\n", encoding="utf-8")
        _create_retrievable_database(paths.database)
        base = load_config(paths.config_file)
        self.config = replace(base, paths=paths, database=paths.database, dense_model="synthetic-dense",
                              sparse_model="synthetic-sparse", candidate_pool=10)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_ptha_dense_provider_uses_bounded_effective_sequence_length(self) -> None:
        config = replace(self.config, sparse_model=self.config.dense_model)
        with patch("ptha.service.build_bge_m3_providers", return_value=(object(), object())) as providers:
            build_providers(config)

        self.assertEqual(providers.call_args.kwargs["max_seq_length"], 512)

    def test_reindex_preserves_canonical_hash_and_rebuilds_derived(self) -> None:
        before = canonical_fingerprint(self.config.database)
        with patch("ptha.service.build_providers", return_value=(SyntheticDenseProvider(), SyntheticSparseProvider())):
            report = reindex_database(self.config)
        after = canonical_fingerprint(self.config.database)
        self.assertEqual(before, after)
        self.assertEqual(report["canonical_sha256"], before["sha256"])
        self.assertTrue(all(report["conditions"].values()))
        self.assertFalse(maintenance_state_path(self.config).exists())
        self.assertFalse(Path(str(self.config.database) + ".reindexing").exists())
        with patch("ptha.service.build_providers", return_value=(SyntheticDenseProvider(), SyntheticSparseProvider())):
            doctor = run_doctor(self.config, full=True)
        full = next(item for item in doctor["checks"] if item["id"] == "runtime.full_retrieval")
        self.assertEqual(full["status"], "pass")

    def test_reindex_failure_preserves_active_database_and_marker(self) -> None:
        before = hashlib.sha256(self.config.database.read_bytes()).hexdigest()
        dense = SyntheticDenseProvider()

        def fail(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("synthetic model failure")

        dense.embed_documents = fail  # type: ignore[method-assign]
        with patch("ptha.service.build_providers", return_value=(dense, SyntheticSparseProvider())), self.assertRaises(RuntimeError):
            reindex_database(self.config)
        self.assertEqual(hashlib.sha256(self.config.database.read_bytes()).hexdigest(), before)
        self.assertTrue(maintenance_state_path(self.config).exists())
        with patch("ptha.service.build_providers", return_value=(SyntheticDenseProvider(), SyntheticSparseProvider())):
            reindex_database(self.config, force=True)

    def test_active_service_blocks_reindex_and_full_doctor_uses_ipc(self) -> None:
        service = RetrievalService(self.config, session_factory=lambda _config: FakeSession())
        thread = threading.Thread(target=service.run)
        thread.start()
        _wait_ready(self.config.paths.socket)
        try:
            with self.assertRaises(Exception):
                reindex_database(self.config)
            report = run_doctor(self.config, full=True)
            full = next(item for item in report["checks"] if item["id"] == "runtime.full_retrieval")
            self.assertEqual(full["status"], "pass")
            serialized = json.dumps(report)
            self.assertNotIn("synthetic result", serialized)
            self.assertNotIn("synthetic context", serialized)
        finally:
            request(self.config.paths.socket, "shutdown")
            thread.join(timeout=3)

    def test_doctor_reports_missing_database_and_stale_marker(self) -> None:
        self.config.database.unlink()
        maintenance_state_path(self.config).write_text('{"schema_version":1,"operation":"reindex"}', encoding="utf-8")
        report = run_doctor(self.config)
        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(checks["database.exists"]["status"], "fail")
        self.assertEqual(checks["operations.maintenance_state"]["status"], "warn")
        self.assertEqual(report["result"], "fail")

    def test_cleanup_stale_state_signals_no_process(self) -> None:
        orphan = self.config.paths.data_dir / ".ptha.db.old.building"
        orphan.write_bytes(b"orphan")
        import_marker = self.config.paths.state_dir / "import-state.json"
        import_marker.write_text("{}", encoding="utf-8")
        result = cleanup_service_state(self.config)
        self.assertTrue(result["cleaned"])
        self.assertEqual(result["state"], "stopped")
        self.assertFalse(orphan.exists())
        self.assertFalse(import_marker.exists())

    def test_import_cleanup_removes_dead_pid_staging_family_only(self) -> None:
        dead = self.config.database.with_name(f".{self.config.database.name}.999999.staging.building-wal")
        unrelated = self.config.database.with_name("unrelated.building")
        dead.write_bytes(b"stale")
        unrelated.write_bytes(b"keep")

        removed = cleanup_orphan_import_files(self.config.database)

        self.assertEqual(removed, 1)
        self.assertFalse(dead.exists())
        self.assertTrue(unrelated.exists())

    def test_full_synthetic_operational_e2e(self) -> None:
        self.config.database.unlink()
        source = Path(self.temporary.name) / "synthetic-export.zip"
        source.touch()

        def write_distilled(_bundle: object, _documents: object, _config: object, output: str) -> Path:
            path = Path(output)
            path.mkdir(parents=True)
            return path

        def build_native(**kwargs: object) -> dict[str, object]:
            output = Path(kwargs["output_db"])
            _create_retrievable_database(output)
            with closing(sqlite3.connect(output)) as conn:
                contracts = {"dense": {"model": "synthetic-dense"}, "sparse": {"model": "synthetic-sparse"},
                             "chunk_policy": "synthetic-policy"}
                conn.execute("UPDATE native_build_audit SET contracts_json=? WHERE id=1", (json.dumps(contracts),))
                conn.commit()
            return {"contracts": {"chunk_policy": "synthetic-policy"}}

        with patch("ptha.importer.load_bundle", return_value=object()), \
             patch("ptha.importer.build_documents", return_value=[]), \
             patch("ptha.importer.write_output", side_effect=write_distilled), \
             patch("ptha.importer.build_native_pre_mvp_db", side_effect=build_native):
            import_archive(source, self.config)
        lightweight = run_doctor(self.config)
        self.assertEqual(lightweight["result"], "pass", lightweight)
        with patch("ptha.service.build_providers", return_value=(SyntheticDenseProvider(), SyntheticSparseProvider())):
            self.assertEqual(run_doctor(self.config, full=True)["result"], "pass")

        def runtime_round_trip() -> None:
            from kb.mcp.archive import ArchiveConfig, ArchiveSession
            session_factory = lambda _config: ArchiveSession(ArchiveConfig(self.config.database, candidate_pool=10),
                                                               SyntheticDenseProvider(), SyntheticSparseProvider())
            service = RetrievalService(self.config, session_factory=session_factory)
            thread = threading.Thread(target=service.run)
            thread.start()
            _wait_ready(self.config.paths.socket)
            try:
                response = MCPAdapter(self.config, stderr=io.StringIO()).handle(
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": "search_archive", "arguments": {"query": "synthetic"}}}
                )
                self.assertFalse(response["result"]["isError"])
            finally:
                request(self.config.paths.socket, "shutdown")
                thread.join(timeout=3)

        runtime_round_trip()
        with patch("ptha.service.build_providers", return_value=(SyntheticDenseProvider(), SyntheticSparseProvider())):
            reindex_database(self.config)
            self.assertEqual(run_doctor(self.config, full=True)["result"], "pass")
        runtime_round_trip()


if __name__ == "__main__":
    unittest.main()
