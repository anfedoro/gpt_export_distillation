from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

from kb.index.chunk_builder import build_chunk_policy
from kb.storage import native_pre_mvp as native
from kb.storage.native_pre_mvp import NativeBuildStore, build_native_pre_mvp_db


class BatchFault(RuntimeError):
    pass


class CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def embed_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        self.calls += 1
        return [
            SimpleNamespace(dense=[1.0] + [0.0] * 1023, sparse={f"token-{index}": 1.0})
            for index, _text in enumerate(texts)
        ]


class FakeProvider:
    effective_max_sequence_length = 64
    document_prefix = ""
    runtime_metadata = {"device": "synthetic"}

    def __init__(self, backend: CountingBackend, *, space: str) -> None:
        self.backend = backend
        self.model_name = "synthetic-bge"
        self.embedding_space_id = space

    def contract_dict(self) -> dict[str, object]:
        return {"provider": "synthetic"}

    def embedding_input(self, text: str) -> str:
        return text

    def token_count(self, text: str) -> int:
        return max(1, len(text.split()))

    def assert_fits(self, text: str, **_kwargs: object) -> int:
        return self.token_count(text)


class ProbeTqdm:
    calls: list[dict[str, object]] = []

    def __init__(self, *_args: object, **kwargs: object) -> None:
        type(self).calls.append(dict(kwargs))

    def __enter__(self) -> "ProbeTqdm":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def update(self, *_args: object) -> None:
        return None

    def set_postfix(self, **_kwargs: object) -> None:
        return None


class ResumableIndexBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ptha-resumable-index-")
        self.root = Path(self.temporary.name).resolve()
        self.export = self.root / "distilled"
        self.export.mkdir()
        self.seed_number = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _providers(self) -> tuple[CountingBackend, FakeProvider, FakeProvider]:
        backend = CountingBackend()
        return backend, FakeProvider(backend, space="synthetic-dense"), FakeProvider(backend, space="synthetic-sparse")

    def _seed_candidate(
        self, *, total: int = 4, complete: int = 0, partial_dense: bool = False,
        legacy_audit: bool = False,
    ) -> tuple[Path, CountingBackend, FakeProvider, FakeProvider]:
        self.seed_number += 1
        target = self.root / f"candidate-{self.seed_number}.db"
        self._assert_inside(target)
        building = target.with_name(target.name + ".building")
        backend, dense, sparse = self._providers()
        policy = build_chunk_policy([dense, sparse], version="v2", content_budget_override=16)
        contracts = {
            "dense": {"model": dense.model_name, "embedding_space_id": dense.embedding_space_id},
            "sparse": {"model": sparse.model_name, "embedding_space_id": sparse.embedding_space_id},
            "chunk_policy": policy.id,
        }
        chunk_audit = {
            "total_retrieval_chunks": total,
            "uncovered_characters": 0,
            "chunks_over_limit": 0,
            "truncated_chunks": 0,
            "blocks_with_coverage_gaps": 0,
        }
        audit_json: dict[str, object] = {"stage": "embedding", "chunk_audit": chunk_audit, "contracts": contracts}
        if not legacy_audit:
            audit_json.update({"processed": complete, "complete_pair_count": complete, "updated_at": "old"})
        with NativeBuildStore(building) as store:
            store.conn.execute("INSERT INTO source_documents VALUES('src','synthetic.md','synthetic.md','chat_md',NULL,'normal',NULL,NULL,'synthetic.md','md','hash',NULL,NULL,'{}')")
            store.conn.execute("INSERT INTO conversations VALUES('conv','src','conversation',NULL,'Synthetic',NULL,NULL,1,0,1,64,0,NULL,NULL,'{}')")
            store.conn.execute("INSERT INTO messages VALUES('msg','conv',1,'user','source-msg',NULL,'synthetic text','{}')")
            store.conn.execute("INSERT INTO blocks VALUES('block','msg',NULL,1,'prose',NULL,0,64,'{}')")
            store.conn.executemany(
                "INSERT INTO retrieval_chunks VALUES(?,?,?,?,?,?,?,?,?)",
                [(f"chunk-{index}", "block", index, 0, 10, 3, f"synthetic chunk {index}", policy.id, "{}") for index in range(1, total + 1)],
            )
            if complete:
                rows = store.conn.execute(
                    "SELECT id,block_id,text FROM retrieval_chunks WHERE ordinal<=? ORDER BY ordinal", (complete,)
                ).fetchall()
                store.write_embedding_batch(
                    rows=rows,
                    dense_vectors=[[1.0] + [0.0] * 1023 for _row in rows],
                    sparse_vectors=[{f"seed-{index}": 1.0} for index, _row in enumerate(rows)],
                    dense_model=dense.model_name,
                    dense_space="seed-dense",
                    sparse_model=sparse.model_name,
                    sparse_space="seed-sparse",
                )
            if partial_dense:
                row = store.conn.execute(
                    "SELECT id,block_id,text FROM retrieval_chunks WHERE ordinal=?", (complete + 1,)
                ).fetchall()
                store.write_dense_batch(
                    rows=row,
                    vectors=[[1.0] + [0.0] * 1023],
                    model=dense.model_name,
                    space="seed-dense",
                )
            store.conn.execute(
                "INSERT INTO native_build_audit VALUES(1,?,?,?,?,?,?,?)",
                ("synthetic-v1", "synthetic", "failed", datetime.now(UTC).isoformat(), None,
                 json.dumps(contracts), json.dumps(audit_json)),
            )
            store.commit()
        return target, backend, dense, sparse

    def _resume(
        self, target: Path, dense: FakeProvider, sparse: FakeProvider,
        *, fault: Callable[[str], None] | None = None, progress: bool = False,
    ) -> dict[str, object]:
        self._assert_inside(target)
        with patch.object(native, "build_bge_m3_providers", return_value=(dense, sparse)):
            return build_native_pre_mvp_db(
                export_path=self.export,
                output_db=target,
                dense_model=dense.model_name,
                sparse_model=sparse.model_name,
                model_revision="synthetic",
                chunk_content_budget=16,
                batch_size=2,
                progress=progress,
                resume=True,
                batch_fault_injector=fault,
            )

    def _counts(self, path: Path) -> dict[str, int]:
        self._assert_inside(path)
        with closing(sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)) as conn:
            cursor = conn.execute("""
                SELECT
                  (SELECT COUNT(*) FROM retrieval_chunks) AS chunks,
                  (SELECT COUNT(*) FROM dense_native_metadata) AS dense_rows,
                  (SELECT COUNT(*) FROM sparse_vector_metadata) AS sparse_rows,
                  (SELECT COUNT(*) FROM sparse_vectors_compact) AS sparse_payloads,
                  (SELECT COUNT(*) FROM sparse_vocabulary) AS vocabulary_rows,
                  (SELECT COUNT(*) FROM retrieval_chunks rc
                    WHERE EXISTS (SELECT 1 FROM dense_native_metadata d WHERE d.chunk_id=rc.id)
                      AND EXISTS (SELECT 1 FROM sparse_vector_metadata s WHERE s.chunk_id=rc.id)) AS complete_pairs,
                  (SELECT COUNT(*) FROM retrieval_chunks rc
                    LEFT JOIN dense_native_metadata d ON d.chunk_id=rc.id
                    LEFT JOIN sparse_vector_metadata s ON s.chunk_id=rc.id
                    WHERE (d.chunk_id IS NULL) <> (s.chunk_id IS NULL)) AS partial_pairs
            """)
            row = cursor.fetchone()
            names = [str(column[0]) for column in cursor.description]
        assert row is not None
        return dict(zip(names, row, strict=True))

    def _audit(self, path: Path) -> tuple[str, dict[str, object]]:
        self._assert_inside(path)
        with closing(sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT status,audit_json FROM native_build_audit WHERE id=1").fetchone()
        return str(row[0]), json.loads(str(row[1]))

    def _assert_inside(self, path: Path) -> None:
        self.assertTrue(path.resolve().is_relative_to(self.root), path)

    def test_complete_pair_is_not_reembedded_after_resume(self) -> None:
        target, backend, dense, sparse = self._seed_candidate(complete=2)

        self._resume(target, dense, sparse)

        self.assertEqual(backend.calls, 1)
        self.assertEqual(self._counts(target)["complete_pairs"], 4)

    def test_audit_cursor_is_not_resume_truth(self) -> None:
        target, backend, dense, sparse = self._seed_candidate(complete=2)
        building = target.with_name(target.name + ".building")
        with closing(sqlite3.connect(building)) as conn:
            row = conn.execute("SELECT audit_json FROM native_build_audit WHERE id=1").fetchone()
            audit = json.loads(str(row[0]))
            audit["processed"] = 0
            audit["processed_count"] = 0
            conn.execute("UPDATE native_build_audit SET audit_json=? WHERE id=1", (json.dumps(audit),))
            conn.commit()

        self._resume(target, dense, sparse)

        self.assertEqual(backend.calls, 1)
        self.assertEqual(self._counts(target)["complete_pairs"], 4)

    def test_partial_pair_is_repaired_and_retried(self) -> None:
        target, backend, dense, sparse = self._seed_candidate(complete=1, partial_dense=True)

        report = self._resume(target, dense, sparse)

        self.assertEqual(backend.calls, 2)
        self.assertEqual(self._counts(target)["partial_pairs"], 0)
        self.assertEqual(report["embedding_build"]["partial_embeddings_removed"], 1)
        _status, audit = self._audit(target)
        self.assertEqual(audit["dense_only_pairs_repaired"], 1)
        self.assertEqual(audit["sparse_only_pairs_repaired"], 0)

    def test_batch_rows_and_audit_commit_atomically(self) -> None:
        target, _backend, dense, sparse = self._seed_candidate()

        with self.assertRaises(BatchFault):
            self._resume(target, dense, sparse, fault=_raise_at("after_audit_update_before_commit"))

        building = target.with_name(target.name + ".building")
        counts = self._counts(building)
        _status, audit = self._audit(building)
        self.assertEqual(counts["complete_pairs"], 0)
        self.assertEqual(counts["dense_rows"], 0)
        self.assertEqual(counts["sparse_rows"], 0)
        self.assertEqual(counts["sparse_payloads"], 0)
        self.assertEqual(counts["vocabulary_rows"], 0)
        self.assertNotIn("last_committed_batch_id", audit)

    def test_failure_before_commit_rolls_back_entire_batch(self) -> None:
        boundaries = (
            "before_batch_begin", "after_sparse_vocabulary", "after_dense_metadata", "after_dense_vector",
            "after_sparse_metadata", "after_sparse_payload", "before_audit_update", "after_audit_update_before_commit",
        )
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                target, _backend, dense, sparse = self._seed_candidate(complete=2)
                with self.assertRaises(BatchFault):
                    self._resume(target, dense, sparse, fault=_raise_at(boundary))
                counts = self._counts(target.with_name(target.name + ".building"))
                self.assertEqual(counts["complete_pairs"], 2)
                self.assertEqual(counts["partial_pairs"], 0)

    def test_failure_after_commit_preserves_batch(self) -> None:
        target, backend, dense, sparse = self._seed_candidate(total=2)
        with self.assertRaises(BatchFault):
            self._resume(target, dense, sparse, fault=_raise_at("after_commit"))

        building = target.with_name(target.name + ".building")
        self.assertEqual(self._counts(building)["complete_pairs"], 2)
        _status, audit = self._audit(building)
        self.assertEqual(audit["processed_count"], 2)
        self.assertTrue(audit["last_committed_batch_id"])

        backend.calls = 0
        self._resume(target, dense, sparse)
        self.assertEqual(backend.calls, 0)

    def test_resume_repeats_at_most_last_uncommitted_batch(self) -> None:
        target, _backend, dense, sparse = self._seed_candidate(complete=2)
        with self.assertRaises(KeyboardInterrupt):
            self._resume(target, dense, sparse, fault=_raise_at("after_dense_vector", KeyboardInterrupt))

        backend_after_restart, dense_after_restart, sparse_after_restart = self._providers()
        self._resume(target, dense_after_restart, sparse_after_restart)
        self.assertEqual(backend_after_restart.calls, 1)
        self.assertEqual(self._counts(target)["complete_pairs"], 4)

    def test_legacy_audit_resumes_without_rebuild(self) -> None:
        target, backend, dense, sparse = self._seed_candidate(complete=2, legacy_audit=True)

        self._resume(target, dense, sparse)

        self.assertEqual(backend.calls, 1)
        _status, audit = self._audit(target)
        self.assertEqual(audit["complete_pair_count"], 4)

    def test_progress_initial_equals_complete_pair_count(self) -> None:
        target, _backend, dense, sparse = self._seed_candidate(complete=2)
        ProbeTqdm.calls.clear()
        with patch.object(native, "tqdm", ProbeTqdm):
            self._resume(target, dense, sparse, progress=True)
        progress = next(call for call in ProbeTqdm.calls if call.get("desc") == "Building dense+sparse index")
        self.assertEqual(progress["total"], 4)
        self.assertEqual(progress["initial"], 2)

    def test_active_database_unchanged_after_interrupted_staging_build(self) -> None:
        active = self.root / "active.db"
        active.write_bytes(b"active-database-sentinel")
        target, _backend, dense, sparse = self._seed_candidate()

        with self.assertRaises(BatchFault):
            self._resume(target, dense, sparse, fault=_raise_at("after_sparse_metadata"))

        self.assertEqual(active.read_bytes(), b"active-database-sentinel")

    def test_no_duplicate_dense_or_sparse_metadata_after_retry(self) -> None:
        target, _backend, dense, sparse = self._seed_candidate()
        with self.assertRaises(BatchFault):
            self._resume(target, dense, sparse, fault=_raise_at("after_audit_update_before_commit"))

        _backend_after, dense_after, sparse_after = self._providers()
        self._resume(target, dense_after, sparse_after)
        with closing(sqlite3.connect(f"file:{target.resolve().as_posix()}?mode=ro", uri=True)) as conn:
            dense_duplicates = conn.execute("SELECT COUNT(*) FROM (SELECT chunk_id FROM dense_native_metadata GROUP BY chunk_id HAVING COUNT(*)>1)").fetchone()[0]
            sparse_duplicates = conn.execute("SELECT COUNT(*) FROM (SELECT chunk_id FROM sparse_vector_metadata GROUP BY chunk_id HAVING COUNT(*)>1)").fetchone()[0]
        self.assertEqual(dense_duplicates, 0)
        self.assertEqual(sparse_duplicates, 0)

    def test_connection_close_without_commit_leaves_batch_pending(self) -> None:
        target, _backend, dense, sparse = self._seed_candidate()
        building = target.with_name(target.name + ".building")
        store = NativeBuildStore(building, create_schema=False)
        rows = store.conn.execute("SELECT id,block_id,text FROM retrieval_chunks ORDER BY ordinal LIMIT 2").fetchall()
        store.conn.execute("BEGIN")
        store.write_dense_batch(rows=rows, vectors=[[1.0] + [0.0] * 1023 for _row in rows], model=dense.model_name, space="dense")
        store.write_sparse_batch(rows=rows, vectors=[{"abrupt": 1.0} for _row in rows], model=sparse.model_name, space="sparse")
        store.close()

        self.assertEqual(self._counts(building)["complete_pairs"], 0)

    def test_staging_uses_wal_with_full_synchronous_mode(self) -> None:
        target, _backend, _dense, _sparse = self._seed_candidate()
        building = target.with_name(target.name + ".building")
        with closing(sqlite3.connect(building)) as conn:
            journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            synchronous = int(conn.execute("PRAGMA synchronous").fetchone()[0])
        self.assertEqual(journal_mode, "wal")
        self.assertEqual(synchronous, 2)


def _raise_at(boundary: str, exception_type: type[BaseException] = BatchFault):
    def inject(point: str) -> None:
        if point == boundary:
            raise exception_type(f"synthetic fault at {point}")
    return inject


if __name__ == "__main__":
    unittest.main()
