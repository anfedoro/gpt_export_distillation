from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


NATIVE_SCHEMA_VERSION = "kb.dense_native.v1"
NATIVE_DTYPE = "float32-le"


class DenseNativeError(RuntimeError):
    """Raised when the native dense-vector backend cannot be used safely."""


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec explicitly and never substitute a fallback."""
    try:
        import sqlite_vec
    except ImportError as exc:
        raise DenseNativeError("sqlite-vec is not installed.") from exc
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as exc:  # noqa: BLE001
        raise DenseNativeError(f"Unable to load sqlite-vec: {exc}") from exc
    finally:
        conn.enable_load_extension(False)
    conn.execute("SELECT vec_version()").fetchone()


def serialize_float32(vector: Iterable[float]) -> bytes:
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise DenseNativeError("Dense vector contains a non-finite value.")
    try:
        import sqlite_vec
    except ImportError as exc:
        raise DenseNativeError("sqlite-vec is required for float32 vector serialization.") from exc
    return bytes(sqlite_vec.serialize_float32(values))


def deserialize_float32(blob: bytes, *, dimension: int) -> list[float]:
    if len(blob) != dimension * 4:
        raise DenseNativeError(f"Native vector has {len(blob)} bytes, expected {dimension * 4}.")
    return list(struct.unpack(f"<{dimension}f", blob))


def native_schema_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'dense_vectors_native'"
    ).fetchone() is not None


def create_native_schema(conn: sqlite3.Connection, *, dimension: int = 1024) -> None:
    load_sqlite_vec(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dense_native_metadata (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE REFERENCES retrieval_chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            embedding_space_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            dtype TEXT NOT NULL CHECK(dtype = 'float32-le'),
            legacy_vector_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dense_native_migrations (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            schema_version TEXT NOT NULL,
            source_db_path TEXT NOT NULL,
            source_dense_count INTEGER NOT NULL,
            migrated_count INTEGER NOT NULL DEFAULT 0,
            expected_dim INTEGER NOT NULL,
            model_name TEXT,
            embedding_space_id TEXT,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            audit_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_dense_native_metadata_space
          ON dense_native_metadata(model_name, embedding_space_id, chunk_id);
        """
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS dense_vectors_native USING vec0(embedding float[{dimension}] distance_metric=cosine)"
    )


def migrate_dense_native(
    *,
    source_db: Path,
    target_db: Path,
    batch_size: int = 256,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy source DB and migrate dense JSON rows into native float32 vectors."""
    if source_db.resolve() == target_db.resolve():
        raise DenseNativeError("Source and target databases must use different paths.")
    if not source_db.exists():
        raise DenseNativeError(f"Source database does not exist: {source_db}")
    if target_db.exists():
        raise DenseNativeError(f"Target database already exists: {target_db}")
    if batch_size <= 0:
        raise DenseNativeError("batch_size must be positive.")

    target_db.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(target_db)
    target.row_factory = sqlite3.Row
    try:
        source.backup(target)
        target.execute("PRAGMA foreign_keys = ON")
        create_native_schema(target)
        source_count = int(source.execute("SELECT COUNT(*) FROM dense_vectors").fetchone()[0])
        owners = {
            str(row["owner_type"]): int(row["count"])
            for row in source.execute("SELECT owner_type, COUNT(*) AS count FROM dense_vectors GROUP BY owner_type")
        }
        duplicate_owners = int(source.execute(
            "SELECT COUNT(*) FROM (SELECT owner_id FROM dense_vectors WHERE owner_type='retrieval_chunk' GROUP BY owner_id HAVING COUNT(*) > 1)"
        ).fetchone()[0])
        if set(owners) != {"retrieval_chunk"}:
            raise DenseNativeError(f"Expected only retrieval_chunk owners, found {owners}.")
        spaces = source.execute(
            "SELECT model_name, embedding_space_id, dim, COUNT(*) AS count FROM dense_vectors "
            "WHERE owner_type='retrieval_chunk' GROUP BY model_name, embedding_space_id, dim"
        ).fetchall()
        if len(spaces) != 1:
            raise DenseNativeError(f"Expected one dense embedding space, found {len(spaces)}.")
        space = spaces[0]
        dimension = int(space["dim"])
        if dimension != 1024:
            raise DenseNativeError(f"Native migration currently requires dim=1024, source has {dimension}.")
        model_name = str(space["model_name"])
        embedding_space_id = str(space["embedding_space_id"] or "")
        target.execute(
            "INSERT INTO dense_native_migrations(id,schema_version,source_db_path,source_dense_count,migrated_count,expected_dim,model_name,embedding_space_id,status,started_at,audit_json) "
            "VALUES(1,?,?,?,?,?,?,?,?,?,?)",
            (NATIVE_SCHEMA_VERSION, str(source_db), source_count, 0, dimension, model_name, embedding_space_id, "running", datetime.now(UTC).isoformat(), "{}"),
        )
        target.commit()

        audit: dict[str, int] = {
            "source_vectors": source_count, "migrated_vectors": 0, "missing_vectors": 0,
            "duplicate_owners": duplicate_owners, "invalid_dimensions": 0,
            "non_finite_values": 0, "orphan_owners": 0,
        }
        cursor = source.execute(
            "SELECT id,owner_id,model_name,embedding_space_id,dim,vector_json,created_at FROM dense_vectors "
            "WHERE owner_type='retrieval_chunk' ORDER BY owner_id,id"
        )
        pending: list[tuple[int, str, str, str, int, bytes, str, str]] = []
        for ordinal, row in enumerate(cursor, start=1):
            vector = json.loads(str(row["vector_json"]))
            if int(row["dim"]) != dimension or len(vector) != dimension:
                audit["invalid_dimensions"] += 1
                continue
            if not all(math.isfinite(float(value)) for value in vector):
                audit["non_finite_values"] += 1
                continue
            chunk_id = str(row["owner_id"])
            if target.execute("SELECT 1 FROM retrieval_chunks WHERE id=?", (chunk_id,)).fetchone() is None:
                audit["orphan_owners"] += 1
                continue
            pending.append((
                ordinal, chunk_id, str(row["model_name"]), str(row["embedding_space_id"] or ""),
                dimension, serialize_float32(vector), str(row["id"]), str(row["created_at"]),
            ))
            if len(pending) == batch_size:
                _write_batch(target, pending)
                audit["migrated_vectors"] += len(pending)
                _save_progress(target, audit)
                pending.clear()
        if pending:
            _write_batch(target, pending)
            audit["migrated_vectors"] += len(pending)
            _save_progress(target, audit)

        metadata_count = int(target.execute("SELECT COUNT(*) FROM dense_native_metadata").fetchone()[0])
        vector_count = int(target.execute("SELECT COUNT(*) FROM dense_vectors_native").fetchone()[0])
        audit["native_metadata_count"] = metadata_count
        audit["native_vector_count"] = vector_count
        audit["missing_vectors"] = source_count - audit["migrated_vectors"]
        valid = (
            audit["migrated_vectors"] == source_count and audit["missing_vectors"] == 0
            and audit["duplicate_owners"] == 0 and audit["invalid_dimensions"] == 0
            and audit["non_finite_values"] == 0 and audit["orphan_owners"] == 0
            and metadata_count == source_count and vector_count == source_count
        )
        status = "completed" if valid else "failed"
        target.execute(
            "UPDATE dense_native_migrations SET migrated_count=?,status=?,finished_at=?,audit_json=? WHERE id=1",
            (audit["migrated_vectors"], status, datetime.now(UTC).isoformat(), json.dumps(audit, separators=(",", ":"))),
        )
        target.commit()
        if not valid:
            raise DenseNativeError(f"Native migration audit failed: {audit}")
        report = audit_dense_native(target_db, source_db=source_db)
        report["migration"] = audit
        report["timing_ms"] = {"total": (time.perf_counter() - started) * 1000}
        if report_dir:
            write_dense_native_report(report, report_dir)
        return report
    except Exception:
        target.rollback()
        raise
    finally:
        target.close()
        source.close()


def _write_batch(conn: sqlite3.Connection, rows: list[tuple[int, str, str, str, int, bytes, str, str]]) -> None:
    with conn:
        for rowid, chunk_id, model_name, space_id, dimension, blob, legacy_id, created_at in rows:
            conn.execute(
                "INSERT INTO dense_native_metadata(rowid,chunk_id,model_name,embedding_space_id,dim,dtype,legacy_vector_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (rowid, chunk_id, model_name, space_id, dimension, NATIVE_DTYPE, legacy_id, created_at),
            )
            conn.execute("INSERT INTO dense_vectors_native(rowid,embedding) VALUES(?,?)", (rowid, blob))


def _save_progress(conn: sqlite3.Connection, audit: dict[str, int]) -> None:
    conn.execute(
        "UPDATE dense_native_migrations SET migrated_count=?,audit_json=? WHERE id=1",
        (audit["migrated_vectors"], json.dumps(audit, separators=(",", ":"))),
    )


@dataclass(frozen=True)
class NativeDenseHit:
    chunk_id: str
    score: float
    distance: float


class NativeDenseSearchBackend:
    """Exact sqlite-vec cosine search over migrated retrieval chunks."""

    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        load_sqlite_vec(self.conn)
        if not native_schema_exists(self.conn):
            self.close()
            raise DenseNativeError("Database has no native dense schema. Run kb-index migrate-dense-native first.")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "NativeDenseSearchBackend":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def validate_embedding_space(self, *, model_name: str, embedding_space_id: str, dimension: int) -> int:
        rows = self.conn.execute(
            "SELECT dim,COUNT(*) AS count FROM dense_native_metadata WHERE model_name=? AND embedding_space_id=? AND dtype=? GROUP BY dim",
            (model_name, embedding_space_id, NATIVE_DTYPE),
        ).fetchall()
        if not rows:
            raise DenseNativeError(f"No native vectors for model={model_name!r}, embedding_space_id={embedding_space_id!r}.")
        if len(rows) != 1 or int(rows[0]["dim"]) != dimension:
            raise DenseNativeError(f"Native dense dimension mismatch: query={dimension}, stored={[(int(r['dim']), int(r['count'])) for r in rows]}.")
        return int(rows[0]["count"])

    def search(self, query_vector: Iterable[float], *, limit: int, model_name: str, embedding_space_id: str) -> list[NativeDenseHit]:
        values = [float(value) for value in query_vector]
        count = self.validate_embedding_space(model_name=model_name, embedding_space_id=embedding_space_id, dimension=len(values))
        if limit <= 0:
            raise DenseNativeError("Search limit must be positive.")
        rows = self.conn.execute(
            "SELECT metadata.chunk_id,native.distance FROM dense_vectors_native AS native "
            "JOIN dense_native_metadata AS metadata ON metadata.rowid=native.rowid "
            "WHERE native.embedding MATCH ? AND k=? AND metadata.model_name=? AND metadata.embedding_space_id=? AND metadata.dtype=? "
            "ORDER BY native.distance,metadata.chunk_id",
            (serialize_float32(values), min(limit, count), model_name, embedding_space_id, NATIVE_DTYPE),
        ).fetchall()
        return [NativeDenseHit(str(row["chunk_id"]), 1.0 - float(row["distance"]), float(row["distance"])) for row in rows]

    def scores_for_chunk_ids(
        self, query_vector: Iterable[float], chunk_ids: Iterable[str], *, model_name: str, embedding_space_id: str
    ) -> dict[str, float]:
        values = [float(value) for value in query_vector]
        self.validate_embedding_space(model_name=model_name, embedding_space_id=embedding_space_id, dimension=len(values))
        ids = sorted(set(chunk_ids))
        result: dict[str, float] = {}
        for offset in range(0, len(ids), 900):
            batch = ids[offset : offset + 900]
            if not batch:
                continue
            marks = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT metadata.chunk_id,vec_distance_cosine(native.embedding,?) AS distance "
                f"FROM dense_vectors_native AS native JOIN dense_native_metadata AS metadata ON metadata.rowid=native.rowid "
                f"WHERE metadata.model_name=? AND metadata.embedding_space_id=? AND metadata.dtype=? AND metadata.chunk_id IN ({marks})",
                (serialize_float32(values), model_name, embedding_space_id, NATIVE_DTYPE, *batch),
            )
            result.update({str(row["chunk_id"]): 1.0 - float(row["distance"]) for row in rows})
        return result

    def get_vector(self, chunk_id: str, *, model_name: str, embedding_space_id: str) -> list[float]:
        row = self.conn.execute(
            "SELECT native.embedding,metadata.dim FROM dense_vectors_native AS native "
            "JOIN dense_native_metadata AS metadata ON metadata.rowid=native.rowid "
            "WHERE metadata.chunk_id=? AND metadata.model_name=? AND metadata.embedding_space_id=?",
            (chunk_id, model_name, embedding_space_id),
        ).fetchone()
        if row is None:
            raise DenseNativeError(f"No native vector for chunk {chunk_id!r}.")
        return deserialize_float32(bytes(row["embedding"]), dimension=int(row["dim"]))


def audit_dense_native(db_path: Path, *, source_db: Path | None = None, sample_size: int = 1000) -> dict[str, Any]:
    with NativeDenseSearchBackend(db_path) as backend:
        conn = backend.conn
        count = int(conn.execute("SELECT COUNT(*) FROM dense_vectors_native").fetchone()[0])
        metadata_count = int(conn.execute("SELECT COUNT(*) FROM dense_native_metadata").fetchone()[0])
        group = dict(conn.execute(
            "SELECT model_name,embedding_space_id,dim,dtype,COUNT(*) AS count FROM dense_native_metadata GROUP BY model_name,embedding_space_id,dim,dtype"
        ).fetchone() or {})
        native_bytes = _dbstat_prefix_bytes(conn, "dense_vectors_native")
        metadata_bytes = _dbstat_bytes(conn, "dense_native_metadata")
        metadata_index_bytes = _dbstat_index_bytes(conn, "dense_native_metadata")
        report: dict[str, Any] = {
            "schema_version": NATIVE_SCHEMA_VERSION,
            "database": str(db_path),
            "sqlite_vec_version": str(conn.execute("SELECT vec_version()").fetchone()[0]),
            "native_vector_count": count,
            "native_metadata_count": metadata_count,
            "group": group,
            "storage": {
                "native_vector_bytes": native_bytes,
                "metadata_bytes": metadata_bytes,
                "metadata_index_bytes": metadata_index_bytes,
                "total_bytes": native_bytes + metadata_bytes + metadata_index_bytes,
                "bytes_per_vector": (native_bytes + metadata_bytes + metadata_index_bytes) / count if count else 0,
                "theoretical_float32_bytes_per_vector": int(group.get("dim", 0)) * 4,
            },
            "migration": dict(conn.execute("SELECT * FROM dense_native_migrations WHERE id=1").fetchone() or {}),
        }
    if source_db:
        report["numeric_parity"] = numeric_parity(source_db=source_db, native_db=db_path, sample_size=sample_size)
    return report


def numeric_parity(*, source_db: Path, native_db: Path, sample_size: int = 1000) -> dict[str, Any]:
    source = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        rows = source.execute(
            "SELECT owner_id,model_name,embedding_space_id,vector_json FROM dense_vectors "
            "WHERE owner_type='retrieval_chunk' ORDER BY owner_id,id LIMIT ?",
            (sample_size,),
        ).fetchall()
    finally:
        source.close()
    max_abs = total_abs = max_cosine_difference = 0.0
    element_count = 0
    with NativeDenseSearchBackend(native_db) as backend:
        for row in rows:
            original = [float(value) for value in json.loads(str(row["vector_json"]))]
            native = backend.get_vector(str(row["owner_id"]), model_name=str(row["model_name"]), embedding_space_id=str(row["embedding_space_id"] or ""))
            for before, after in zip(original, native, strict=True):
                error = abs(before - after)
                max_abs = max(max_abs, error)
                total_abs += error
                element_count += 1
            original_norm = math.sqrt(sum(value * value for value in original))
            native_norm = math.sqrt(sum(value * value for value in native))
            if original_norm and native_norm:
                cosine = sum(before * after for before, after in zip(original, native, strict=True)) / (original_norm * native_norm)
                max_cosine_difference = max(max_cosine_difference, abs(1.0 - cosine))
    return {
        "sample_size": len(rows),
        "max_absolute_error": max_abs,
        "mean_absolute_error": total_abs / element_count if element_count else 0.0,
        "max_cosine_difference": max_cosine_difference,
    }


def write_dense_native_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    storage = report["storage"]
    parity = report.get("numeric_parity", {})
    lines = [
        "# Native Dense Migration", "",
        f"Database: {report['database']}",
        f"sqlite-vec: {report['sqlite_vec_version']}",
        f"Native vectors: {report['native_vector_count']}",
        f"Native storage GiB: {storage['total_bytes'] / 1024**3:.3f}",
        f"Bytes/vector: {storage['bytes_per_vector']:.1f}",
        f"Numeric parity sample: {parity.get('sample_size', 0)}",
        f"Max absolute error: {parity.get('max_absolute_error', 0):.9g}",
        f"Max cosine difference: {parity.get('max_cosine_difference', 0):.9g}",
    ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dbstat_bytes(conn: sqlite3.Connection, name: str) -> int:
    return int(conn.execute("SELECT COALESCE(SUM(pgsize),0) FROM dbstat WHERE name=?", (name,)).fetchone()[0])


def _dbstat_prefix_bytes(conn: sqlite3.Connection, prefix: str) -> int:
    return int(conn.execute("SELECT COALESCE(SUM(pgsize),0) FROM dbstat WHERE name LIKE ?", (prefix + "%",)).fetchone()[0])


def _dbstat_index_bytes(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(
        "SELECT COALESCE(SUM(ds.pgsize),0) FROM dbstat ds JOIN sqlite_master sm ON sm.name=ds.name "
        "WHERE sm.type='index' AND sm.tbl_name=?",
        (table_name,),
    ).fetchone()[0])
