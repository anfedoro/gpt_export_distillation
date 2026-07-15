from __future__ import annotations

import json
import math
import os
import gc
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from kb.embeddings.bge_m3_provider import build_bge_m3_providers, embed_joint_documents
from kb.index.chunk_builder import ChunkPolicy, StrictestTokenizer, build_chunk_policy, build_retrieval_chunks
from kb.ingest.chat_md_parser import parse_chat_file
from kb.ingest.tree_walker import scan_tree
from kb.model.entities import Block, Conversation, Message, ParsedChat
from kb.model.ids import stable_id
from kb.storage.dense_native import NATIVE_DTYPE, NativeDenseSearchBackend, load_sqlite_vec, serialize_float32


NATIVE_PRE_MVP_SCHEMA_VERSION = "kb.native_pre_mvp.v1"
CANONICAL_TABLES = ("source_documents", "conversations", "messages", "blocks", "retrieval_chunks")
LEGACY_TABLES = (
    "dense_vectors", "sparse_terms", "knowledge_blocks", "semantic_nodes", "semantic_node_members",
    "semantic_edges", "retrieval_traces", "dense_native_migrations", "attachment_documents", "ingestion_runs",
)


class NativePreMvpError(RuntimeError):
    """Raised when the clean native build or runtime cannot meet its contract."""


@dataclass(frozen=True)
class SparseNativeHit:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class NativeRetrievalHit:
    chunk_id: str
    dense_score: float
    sparse_score: float
    final_score: float
    provenance: dict[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pack_uint32(values: Iterable[int]) -> bytes:
    items = list(values)
    return struct.pack(f"<{len(items)}I", *items)


def _pack_float32(values: Iterable[float]) -> bytes:
    items = [float(value) for value in values]
    if not all(math.isfinite(value) for value in items):
        raise NativePreMvpError("Sparse vector contains non-finite weights.")
    return struct.pack(f"<{len(items)}f", *items)


def _unpack_array(blob: bytes, dtype: np.dtype[Any]) -> np.ndarray:
    if len(blob) % dtype.itemsize:
        raise NativePreMvpError(f"Invalid {dtype} BLOB length: {len(blob)}.")
    return np.frombuffer(blob, dtype=dtype).copy()


def _covered_length(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
        return 0
    total = 0
    start, end = sorted(ranges)[0]
    for next_start, next_end in sorted(ranges)[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def create_clean_native_schema(conn: sqlite3.Connection, *, dimension: int = 1024) -> None:
    """Create the final clean schema; it deliberately has no legacy compatibility tables."""
    load_sqlite_vec(conn)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE source_documents (
            id TEXT PRIMARY KEY, path TEXT NOT NULL, relative_path TEXT NOT NULL, source_kind TEXT NOT NULL,
            folder_kind TEXT, interest_tier TEXT NOT NULL DEFAULT 'normal', project_id TEXT, project_name TEXT,
            file_name TEXT NOT NULL, extension TEXT NOT NULL, sha256 TEXT NOT NULL, created_at TEXT, updated_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}', UNIQUE(relative_path, sha256)
        );
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY, source_document_id TEXT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
            conversation_id TEXT, conversation_template_id TEXT, title TEXT, create_time_utc TEXT, update_time_utc TEXT,
            message_count INTEGER NOT NULL, assistant_messages INTEGER NOT NULL, user_messages INTEGER NOT NULL,
            text_chars INTEGER NOT NULL, estimated_code_blocks INTEGER NOT NULL, project_id TEXT, folder_kind TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, role TEXT NOT NULL, message_id TEXT, time_utc TEXT, raw_text TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}', UNIQUE(conversation_id, ordinal)
        );
        CREATE TABLE blocks (
            id TEXT PRIMARY KEY, message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            parent_block_id TEXT REFERENCES blocks(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, block_type TEXT NOT NULL,
            language TEXT, source_char_start INTEGER NOT NULL, source_char_end INTEGER NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}', UNIQUE(message_id, ordinal)
        );
        CREATE TABLE retrieval_chunks (
            id TEXT PRIMARY KEY, block_id TEXT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL,
            source_char_start INTEGER NOT NULL, source_char_end INTEGER NOT NULL, token_count INTEGER NOT NULL,
            text TEXT NOT NULL, chunk_policy_id TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(block_id, chunk_policy_id, ordinal)
        );
        CREATE TABLE dense_native_metadata (
            rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE REFERENCES retrieval_chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL, embedding_space_id TEXT NOT NULL, dim INTEGER NOT NULL CHECK(dim > 0),
            dtype TEXT NOT NULL CHECK(dtype = 'float32-le'), created_at TEXT NOT NULL
        );
        CREATE TABLE sparse_vector_metadata (
            rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE REFERENCES retrieval_chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL, embedding_space_id TEXT NOT NULL, term_count INTEGER NOT NULL CHECK(term_count >= 0),
            norm REAL NOT NULL CHECK(norm >= 0), dtype TEXT NOT NULL CHECK(dtype = 'uint32-float32-le'),
            created_at TEXT NOT NULL
        );
        CREATE TABLE sparse_vectors_compact (
            rowid INTEGER PRIMARY KEY REFERENCES sparse_vector_metadata(rowid) ON DELETE CASCADE,
            indices_blob BLOB NOT NULL, weights_blob BLOB NOT NULL,
            CHECK(length(indices_blob) = length(weights_blob))
        );
        CREATE TABLE sparse_vocabulary (
            term_id INTEGER PRIMARY KEY, token_text TEXT NOT NULL UNIQUE
        ) WITHOUT ROWID;
        CREATE TABLE native_build_audit (
            id INTEGER PRIMARY KEY CHECK(id = 1), schema_version TEXT NOT NULL, export_path TEXT NOT NULL,
            status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, contracts_json TEXT NOT NULL,
            audit_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_messages_conversation ON messages(conversation_id, ordinal);
        CREATE INDEX idx_blocks_message ON blocks(message_id, ordinal);
        CREATE INDEX idx_retrieval_chunks_block ON retrieval_chunks(block_id, chunk_policy_id, ordinal);
        CREATE INDEX idx_dense_native_metadata_space ON dense_native_metadata(model_name, embedding_space_id, chunk_id);
        CREATE INDEX idx_sparse_vector_metadata_space ON sparse_vector_metadata(model_name, embedding_space_id, chunk_id);
        """
    )
    conn.execute(f"CREATE VIRTUAL TABLE dense_vectors_native USING vec0(embedding float[{dimension}] distance_metric=cosine)")


class NativeBuildStore:
    """Write-only canonical and native-representation store for one clean build."""

    def __init__(self, db_path: Path, *, create_schema: bool = True) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 60000")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA cache_size = -32768")
        if create_schema:
            create_clean_native_schema(self.conn)
        else:
            load_sqlite_vec(self.conn)
        self._term_ids: dict[str, int] = {
            str(row["token_text"]): int(row["term_id"])
            for row in self.conn.execute("SELECT term_id,token_text FROM sparse_vocabulary")
        } if not create_schema else {}
        self._dense_rowid = int(self.conn.execute("SELECT COALESCE(MAX(rowid),0) FROM dense_native_metadata").fetchone()[0]) if not create_schema else 0
        self._sparse_rowid = int(self.conn.execute("SELECT COALESCE(MAX(rowid),0) FROM sparse_vector_metadata").fetchone()[0]) if not create_schema else 0

    def reset_derived(self) -> None:
        """Delete only reproducible retrieval structures from an existing clean DB."""
        self.conn.execute("DELETE FROM dense_vectors_native")
        self.conn.execute("DELETE FROM dense_native_metadata")
        self.conn.execute("DELETE FROM sparse_vectors_compact")
        self.conn.execute("DELETE FROM sparse_vector_metadata")
        self.conn.execute("DELETE FROM sparse_vocabulary")
        self.conn.execute("DELETE FROM retrieval_chunks")
        self.conn.commit()
        self._term_ids.clear()
        self._dense_rowid = 0
        self._sparse_rowid = 0

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "NativeBuildStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def commit(self) -> None:
        self.conn.commit()

    def insert_source_document(self, root: Path, item: Any) -> str:
        source_id = stable_id(item.relative_path, item.sha256, prefix="src")
        self.conn.execute(
            "INSERT INTO source_documents(id,path,relative_path,source_kind,folder_kind,interest_tier,project_id,project_name,file_name,extension,sha256,metadata_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (source_id, str(root / item.relative_path), item.relative_path, item.detected_kind, item.folder_kind,
             item.interest_tier, item.project_path, item.project_path, item.file_name, item.extension, item.sha256,
             _json({"size": item.size, "is_attachment": item.is_attachment})),
        )
        return source_id

    def insert_parsed_chat(self, parsed: ParsedChat) -> None:
        self._insert_conversation(parsed.conversation)
        for message in parsed.messages:
            self._insert_message(message)
        for block in parsed.blocks:
            self._insert_block(block)

    def _insert_conversation(self, item: Conversation) -> None:
        self.conn.execute(
            "INSERT INTO conversations(id,source_document_id,conversation_id,conversation_template_id,title,create_time_utc,update_time_utc,message_count,assistant_messages,user_messages,text_chars,estimated_code_blocks,project_id,folder_kind,metadata_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item.id, item.source_document_id, item.conversation_id, item.conversation_template_id, item.title,
             item.create_time_utc, item.update_time_utc, item.message_count, item.assistant_messages, item.user_messages,
             item.text_chars, item.estimated_code_blocks, item.project_id, item.folder_kind, _json(item.metadata_json)),
        )

    def _insert_message(self, item: Message) -> None:
        self.conn.execute(
            "INSERT INTO messages(id,conversation_id,ordinal,role,message_id,time_utc,raw_text,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (item.id, item.conversation_id, item.ordinal, item.role, item.message_id, item.time_utc, item.raw_text, _json(item.metadata_json)),
        )

    def _insert_block(self, item: Block) -> None:
        self.conn.execute(
            "INSERT INTO blocks(id,message_id,parent_block_id,ordinal,block_type,language,source_char_start,source_char_end,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (item.id, item.message_id, None, item.ordinal, item.block_type, item.language, item.char_start, item.char_end, _json(item.metadata_json)),
        )

    def create_chunks(self, *, policy: ChunkPolicy, tokenizer_provider: Any, skip_low_interest: bool, progress: bool = False) -> dict[str, Any]:
        where = "b.source_char_end > b.source_char_start"
        if skip_low_interest:
            where += " AND sd.interest_tier NOT IN ('low','quarantine')"
        block_rows = self.conn.execute(
            "SELECT b.id,m.raw_text,b.source_char_start,b.source_char_end FROM blocks b "
            "JOIN messages m ON m.id=b.message_id JOIN conversations c ON c.id=m.conversation_id "
            "JOIN source_documents sd ON sd.id=c.source_document_id WHERE " + where + " ORDER BY b.id"
        )
        total_blocks = int(self.conn.execute(
            "SELECT COUNT(*) FROM blocks b JOIN messages m ON m.id=b.message_id "
            "JOIN conversations c ON c.id=m.conversation_id "
            "JOIN source_documents sd ON sd.id=c.source_document_id WHERE " + where
        ).fetchone()[0])
        token_counts: list[int] = []
        total_source_characters = covered_unique_characters = 0
        blocks_with_coverage_gaps = chunks_with_overlap = overlap_token_count_total = 0
        chunks_split_on_natural_boundary = chunks_split_by_token_fallback = total_chunks = 0
        pending: list[tuple[str, str, int, int, int, int, str, str, str]] = []
        with tqdm(total=total_blocks, desc="Building retrieval chunks", unit="block", file=sys.stderr,
                  dynamic_ncols=True, disable=not progress) as bar:
            for row in block_rows:
                text = str(row["raw_text"])[int(row["source_char_start"]):int(row["source_char_end"])]
                chunks = build_retrieval_chunks(
                    block_id=str(row["id"]), block_text=text, block_char_start=int(row["source_char_start"]),
                    policy=policy, tokenizer_provider=tokenizer_provider,
                )
                block_start, block_end = int(row["source_char_start"]), int(row["source_char_end"])
                total_source_characters += block_end - block_start
                covered = _covered_length([(chunk.source_char_start, chunk.source_char_end) for chunk in chunks])
                covered_unique_characters += covered
                if covered != block_end - block_start:
                    blocks_with_coverage_gaps += 1
                for chunk in chunks:
                    pending.append((chunk.id, chunk.block_id, chunk.ordinal, chunk.source_char_start, chunk.source_char_end,
                                    chunk.token_count, chunk.text, chunk.chunk_policy_id,
                                    _json({"split_reason": chunk.split_reason, "overlap_token_count": chunk.overlap_token_count})))
                    token_counts.append(chunk.token_count)
                    total_chunks += 1
                    chunks_with_overlap += int(chunk.overlap_token_count > 0)
                    overlap_token_count_total += chunk.overlap_token_count
                    chunks_split_by_token_fallback += int(chunk.split_reason == "token_window_fallback")
                    chunks_split_on_natural_boundary += int(chunk.split_reason != "token_window_fallback")
                if len(pending) >= 1024:
                    self.conn.executemany("INSERT INTO retrieval_chunks(id,block_id,ordinal,source_char_start,source_char_end,token_count,text,chunk_policy_id,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)", pending)
                    self.conn.commit()
                    pending.clear()
                    bar.set_postfix(chunks=total_chunks)
                bar.update(1)
            bar.set_postfix(chunks=total_chunks)
        if pending:
            self.conn.executemany("INSERT INTO retrieval_chunks(id,block_id,ordinal,source_char_start,source_char_end,token_count,text,chunk_policy_id,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)", pending)
        values = np.asarray(token_counts, dtype=np.int64)
        return {
            "chunk_policy_id": policy.id, "chunk_policy_version": policy.version,
            "chunk_policy_max_input_tokens": policy.max_input_tokens, "chunk_policy_content_token_budget": policy.content_token_budget,
            "chunk_policy_overlap_tokens": policy.overlap_tokens, "chunk_policy_safety_reserve": policy.safety_reserve,
            "total_source_characters": total_source_characters, "total_indexable_characters": total_source_characters,
            "covered_unique_characters": covered_unique_characters,
            "uncovered_characters": max(0, total_source_characters - covered_unique_characters),
            "total_retrieval_chunks": total_chunks, "maximum_chunk_token_count": int(values.max()) if len(values) else 0,
            "p50_chunk_token_count": float(np.percentile(values, 50)) if len(values) else 0.0,
            "p95_chunk_token_count": float(np.percentile(values, 95)) if len(values) else 0.0,
            "p99_chunk_token_count": float(np.percentile(values, 99)) if len(values) else 0.0,
            "chunks_over_limit": int(np.sum(values > policy.max_input_tokens)) if len(values) else 0,
            "truncated_chunks": 0, "blocks_with_coverage_gaps": blocks_with_coverage_gaps,
            "chunks_with_overlap": chunks_with_overlap, "overlap_token_count_total": overlap_token_count_total,
            "chunks_split_on_natural_boundary": chunks_split_on_natural_boundary,
            "chunks_split_by_token_fallback": chunks_split_by_token_fallback,
        }

    def embedding_batches(self, *, policy_id: str, batch_size: int) -> Iterable[list[sqlite3.Row]]:
        after_id: str | None = None
        while True:
            params: list[Any] = [policy_id]
            predicate = "chunk_policy_id=?"
            if after_id is not None:
                predicate += " AND id>?"
                params.append(after_id)
            params.append(batch_size)
            rows = self.conn.execute(
                "SELECT id,block_id,text FROM retrieval_chunks WHERE " + predicate + " ORDER BY id LIMIT ?", params
            ).fetchall()
            if not rows:
                return
            yield rows
            after_id = str(rows[-1]["id"])

    def embedding_batches_by_length(self, *, policy_id: str, batch_size: int, missing_only: bool = False) -> Iterable[list[sqlite3.Row]]:
        """Yield deterministic length buckets to minimize transformer padding."""
        after_tokens = -1
        after_id = ""
        while True:
            missing_predicate = ""
            if missing_only:
                missing_predicate = (
                    " AND NOT EXISTS (SELECT 1 FROM dense_native_metadata d WHERE d.chunk_id=rc.id)"
                    " AND NOT EXISTS (SELECT 1 FROM sparse_vector_metadata s WHERE s.chunk_id=rc.id)"
                )
            rows = self.conn.execute(
                "SELECT rc.id,rc.block_id,rc.text,rc.token_count FROM retrieval_chunks rc "
                "WHERE rc.chunk_policy_id=?" + missing_predicate +
                " AND (rc.token_count>? OR (rc.token_count=? AND rc.id>?)) "
                "ORDER BY rc.token_count,rc.id LIMIT ?",
                (policy_id, after_tokens, after_tokens, after_id, batch_size),
            ).fetchall()
            if not rows:
                return
            yield rows
            after_tokens = int(rows[-1]["token_count"])
            after_id = str(rows[-1]["id"])

    def remove_partial_embeddings(self) -> int:
        """Remove chunks with only one of the dense/sparse representations."""
        rows = self.conn.execute(
            "SELECT rc.id,d.rowid AS dense_rowid,s.rowid AS sparse_rowid "
            "FROM retrieval_chunks rc "
            "LEFT JOIN dense_native_metadata d ON d.chunk_id=rc.id "
            "LEFT JOIN sparse_vector_metadata s ON s.chunk_id=rc.id "
            "WHERE (d.chunk_id IS NULL) <> (s.chunk_id IS NULL)"
        ).fetchall()
        for row in rows:
            if row["dense_rowid"] is not None:
                self.conn.execute("DELETE FROM dense_vectors_native WHERE rowid=?", (row["dense_rowid"],))
                self.conn.execute("DELETE FROM dense_native_metadata WHERE rowid=?", (row["dense_rowid"],))
            if row["sparse_rowid"] is not None:
                self.conn.execute("DELETE FROM sparse_vectors_compact WHERE rowid=?", (row["sparse_rowid"],))
                self.conn.execute("DELETE FROM sparse_vector_metadata WHERE rowid=?", (row["sparse_rowid"],))
        if rows:
            self.conn.commit()
        return len(rows)

    def write_embedding_batch(
        self, *, rows: list[sqlite3.Row], dense_vectors: list[list[float]], sparse_vectors: list[dict[str, float]],
        dense_model: str, dense_space: str, sparse_model: str, sparse_space: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        for row, dense, sparse in zip(rows, dense_vectors, sparse_vectors, strict=True):
            chunk_id = str(row["id"])
            if len(dense) != 1024 or not all(math.isfinite(float(value)) for value in dense):
                raise NativePreMvpError(f"Non-finite or invalid dense vector chunk_id={chunk_id} block_id={row['block_id']}.")
            self._dense_rowid += 1
            self.conn.execute(
                "INSERT INTO dense_native_metadata(rowid,chunk_id,model_name,embedding_space_id,dim,dtype,created_at) VALUES(?,?,?,?,?,?,?)",
                (self._dense_rowid, chunk_id, dense_model, dense_space, 1024, NATIVE_DTYPE, now),
            )
            self.conn.execute("INSERT INTO dense_vectors_native(rowid,embedding) VALUES(?,?)", (self._dense_rowid, serialize_float32(dense)))
            pairs: list[tuple[int, float]] = []
            for token, weight in sorted(sparse.items()):
                numeric = self._term_ids.get(token)
                if numeric is None:
                    numeric = len(self._term_ids) + 1
                    self._term_ids[token] = numeric
                    self.conn.execute("INSERT INTO sparse_vocabulary(term_id,token_text) VALUES(?,?)", (numeric, token))
                pairs.append((numeric, float(weight)))
            if not pairs or not all(math.isfinite(weight) for _, weight in pairs):
                raise NativePreMvpError(f"Invalid sparse vector chunk_id={chunk_id} block_id={row['block_id']}.")
            norm = math.sqrt(sum(weight * weight for _, weight in pairs))
            if not math.isfinite(norm) or norm == 0:
                raise NativePreMvpError(f"Invalid sparse norm chunk_id={chunk_id} block_id={row['block_id']}.")
            self._sparse_rowid += 1
            self.conn.execute(
                "INSERT INTO sparse_vector_metadata(rowid,chunk_id,model_name,embedding_space_id,term_count,norm,dtype,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (self._sparse_rowid, chunk_id, sparse_model, sparse_space, len(pairs), norm, "uint32-float32-le", now),
            )
            self.conn.execute(
                "INSERT INTO sparse_vectors_compact(rowid,indices_blob,weights_blob) VALUES(?,?,?)",
                (self._sparse_rowid, _pack_uint32(term for term, _ in pairs), _pack_float32(weight for _, weight in pairs)),
            )

    def write_dense_batch(
        self, *, rows: list[sqlite3.Row], vectors: list[list[float]], model: str, space: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        for row, vector in zip(rows, vectors, strict=True):
            chunk_id = str(row["id"])
            if len(vector) != 1024 or not all(math.isfinite(float(value)) for value in vector):
                raise NativePreMvpError(f"Non-finite or invalid dense vector chunk_id={chunk_id} block_id={row['block_id']}.")
            self._dense_rowid += 1
            self.conn.execute(
                "INSERT INTO dense_native_metadata(rowid,chunk_id,model_name,embedding_space_id,dim,dtype,created_at) VALUES(?,?,?,?,?,?,?)",
                (self._dense_rowid, chunk_id, model, space, 1024, NATIVE_DTYPE, now),
            )
            self.conn.execute("INSERT INTO dense_vectors_native(rowid,embedding) VALUES(?,?)", (self._dense_rowid, serialize_float32(vector)))

    def write_sparse_batch(
        self, *, rows: list[sqlite3.Row], vectors: list[dict[str, float]], model: str, space: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        for row, vector in zip(rows, vectors, strict=True):
            chunk_id = str(row["id"])
            pairs: list[tuple[int, float]] = []
            for token, weight in sorted(vector.items()):
                numeric = self._term_ids.get(token)
                if numeric is None:
                    numeric = len(self._term_ids) + 1
                    self._term_ids[token] = numeric
                    self.conn.execute("INSERT INTO sparse_vocabulary(term_id,token_text) VALUES(?,?)", (numeric, token))
                pairs.append((numeric, float(weight)))
            if not pairs or not all(math.isfinite(weight) for _, weight in pairs):
                raise NativePreMvpError(f"Invalid sparse vector chunk_id={chunk_id} block_id={row['block_id']}.")
            norm = math.sqrt(sum(weight * weight for _, weight in pairs))
            if not math.isfinite(norm) or norm == 0:
                raise NativePreMvpError(f"Invalid sparse norm chunk_id={chunk_id} block_id={row['block_id']}.")
            self._sparse_rowid += 1
            self.conn.execute(
                "INSERT INTO sparse_vector_metadata(rowid,chunk_id,model_name,embedding_space_id,term_count,norm,dtype,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (self._sparse_rowid, chunk_id, model, space, len(pairs), norm, "uint32-float32-le", now),
            )
            self.conn.execute(
                "INSERT INTO sparse_vectors_compact(rowid,indices_blob,weights_blob) VALUES(?,?,?)",
                (self._sparse_rowid, _pack_uint32(term for term, _ in pairs), _pack_float32(weight for _, weight in pairs)),
            )

    def audit(self) -> dict[str, Any]:
        counts = {table: int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in CANONICAL_TABLES}
        counts["dense_vectors_native"] = int(self.conn.execute("SELECT COUNT(*) FROM dense_vectors_native").fetchone()[0])
        counts["sparse_vectors_compact"] = int(self.conn.execute("SELECT COUNT(*) FROM sparse_vectors_compact").fetchone()[0])
        counts["sparse_vocabulary"] = int(self.conn.execute("SELECT COUNT(*) FROM sparse_vocabulary").fetchone()[0])
        legacy = [name for name in LEGACY_TABLES if self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()]
        orphans = int(self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM dense_native_metadata d LEFT JOIN retrieval_chunks c ON c.id=d.chunk_id WHERE c.id IS NULL) + "
            "(SELECT COUNT(*) FROM sparse_vector_metadata s LEFT JOIN retrieval_chunks c ON c.id=s.chunk_id WHERE c.id IS NULL)"
        ).fetchone()[0])
        duplicates = int(self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM (SELECT chunk_id FROM dense_native_metadata GROUP BY chunk_id HAVING COUNT(*)>1)) + "
            "(SELECT COUNT(*) FROM (SELECT chunk_id FROM sparse_vector_metadata GROUP BY chunk_id HAVING COUNT(*)>1))"
        ).fetchone()[0])
        non_finite = sum(
            1 for row in self.conn.execute("SELECT embedding FROM dense_vectors_native")
            if not all(math.isfinite(value) for value in struct.unpack("<1024f", bytes(row[0])))
        )
        return {
            "counts": counts, "legacy_tables_present": legacy, "orphan_mappings": orphans,
            "duplicate_chunk_mappings": duplicates, "non_finite_dense_vectors": non_finite,
            "integrity_check": str(self.conn.execute("PRAGMA integrity_check").fetchone()[0]),
            "foreign_key_errors": len(self.conn.execute("PRAGMA foreign_key_check").fetchall()),
        }


def build_native_pre_mvp_db(
    *, export_path: Path, output_db: Path, dense_model: str = "anfedoro/bge-m3-mlx-fp16",
    sparse_model: str = "anfedoro/bge-m3-mlx-fp16", model_revision: str = "58e70901dbba8de8f3df91b5a313bcefcb151bae",
    embedding_device: str = "gpu", embedding_dtype: str = "float16",
    embedding_max_padded_tokens: int | None = None,
    sparse_head: str = "sparse_linear.safetensors", colbert_head: str = "colbert_linear.safetensors",
    model_cache: Path | None = None, chunk_policy_version: str = "v2",
    chunk_content_budget: int = 506, sparse_top_k: int = 128, batch_size: int = 4,
    skip_low_interest: bool = True, progress: bool = True,
    dense_effective_max_seq_length: int = 512,
    dense_device: str | None = None, sparse_device: str | None = None,
    dense_torch_dtype: str | None = None, sparse_torch_dtype: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Parse a raw export and create a clean DB without opening any legacy DB."""
    if output_db.exists():
        raise NativePreMvpError(f"Output database already exists: {output_db}")
    building_db = output_db.with_name(output_db.name + ".building")
    if building_db.exists() and not resume:
        raise NativePreMvpError(
            f"Unfinished temporary build exists: {building_db}. Inspect or remove it before a new build."
        )
    if resume and not building_db.exists():
        raise NativePreMvpError(f"Cannot resume import; temporary build does not exist: {building_db}")
    if batch_size <= 0:
        raise NativePreMvpError("batch_size must be positive.")
    export_path = export_path.expanduser().resolve()
    if not export_path.is_dir():
        raise NativePreMvpError(f"Export directory does not exist: {export_path}")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if sparse_model != dense_model:
        raise NativePreMvpError("Native PTHA builds require one shared BGE-M3 model.")
    requested_devices = {value for value in (dense_device, sparse_device) if value not in {None, "auto", "mps"}}
    if requested_devices:
        embedding_device = next(iter(requested_devices))
    requested_dtypes = {value for value in (dense_torch_dtype, sparse_torch_dtype) if value not in {None, "auto"}}
    if requested_dtypes:
        embedding_dtype = next(iter(requested_dtypes))
    dense, sparse = build_bge_m3_providers(
        dense_model,
        model_revision=model_revision,
        device=embedding_device,
        dtype=embedding_dtype,
        max_seq_length=dense_effective_max_seq_length,
        sparse_top_k=sparse_top_k,
        batch_size=batch_size,
        max_padded_tokens=embedding_max_padded_tokens,
        sparse_head=sparse_head,
        colbert_head=colbert_head,
        model_cache=model_cache,
    )
    policy = build_chunk_policy([dense, sparse], version=chunk_policy_version, content_budget_override=chunk_content_budget)
    contracts = {
        "dense": {"model": dense.model_name, "embedding_space_id": dense.embedding_space_id, "runtime": dense.runtime_metadata, "provider": dense.contract_dict()},
        "sparse": {"model": sparse.model_name, "embedding_space_id": sparse.embedding_space_id, "runtime": sparse.runtime_metadata, "provider": sparse.contract_dict()},
        "chunk_policy": policy.id,
    }
    try:
        with NativeBuildStore(building_db, create_schema=not resume) as store:
            previous_audit: dict[str, Any] = {}
            chunk_audit: dict[str, Any] | None = None
            if resume:
                row = store.conn.execute("SELECT contracts_json,audit_json,status FROM native_build_audit WHERE id=1").fetchone()
                if row is None:
                    raise NativePreMvpError("Cannot resume import; build checkpoint is missing.")
                previous_contracts = json.loads(str(row["contracts_json"]))
                previous_audit = json.loads(str(row["audit_json"] or "{}"))
                expected = {
                    "dense_model": dense.model_name,
                    "dense_space": dense.embedding_space_id,
                    "sparse_model": sparse.model_name,
                    "sparse_space": sparse.embedding_space_id,
                    "chunk_policy": policy.id,
                }
                actual = {
                    "dense_model": previous_contracts.get("dense", {}).get("model"),
                    "dense_space": previous_contracts.get("dense", {}).get("embedding_space_id"),
                    "sparse_model": previous_contracts.get("sparse", {}).get("model"),
                    "sparse_space": previous_contracts.get("sparse", {}).get("embedding_space_id"),
                    "chunk_policy": previous_contracts.get("chunk_policy"),
                }
                if actual != expected:
                    raise NativePreMvpError(
                        "Import checkpoint is incompatible with the current embedding configuration; "
                        "discard it and start a fresh import."
                    )
                chunk_audit = previous_audit.get("chunk_audit")
                if not isinstance(chunk_audit, dict):
                    raise NativePreMvpError("Import checkpoint has no completed chunk phase; start a fresh import.")
                scanned = int(store.conn.execute("SELECT COUNT(*) FROM source_documents").fetchone()[0])
                parsed = int(store.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
                failed = 0
            else:
                store.conn.execute(
                    "INSERT INTO native_build_audit(id,schema_version,export_path,status,started_at,contracts_json) VALUES(1,?,?,?,?,?)",
                    (NATIVE_PRE_MVP_SCHEMA_VERSION, str(export_path), "running", datetime.now(UTC).isoformat(), _json(contracts)),
                )
                scanned = parsed = failed = 0
                items = list(scan_tree(export_path))
                total_chats = sum(item.detected_kind == "chat_md" for item in items)
                if progress:
                    print("[4/7] Parsing conversations", file=sys.stderr, flush=True)
                with tqdm(total=total_chats, desc="Parsing conversations", unit="chat", file=sys.stderr,
                          dynamic_ncols=True, disable=not progress) as bar:
                    for item in items:
                        scanned += 1
                        source_id = store.insert_source_document(export_path, item)
                        if item.detected_kind != "chat_md":
                            continue
                        try:
                            parsed_chat = parse_chat_file(export_path / item.relative_path, source_document_id=source_id, project_id=item.project_path, folder_kind=item.folder_kind)
                            store.insert_parsed_chat(parsed_chat)
                            parsed += 1
                        except Exception as exc:  # noqa: BLE001
                            failed += 1
                            raise NativePreMvpError(f"Raw export parse failed path={item.relative_path}: {exc}") from exc
                        if parsed % 100 == 0:
                            store.commit()
                            bar.set_postfix(scanned=scanned)
                        bar.update(1)
                store.commit()
                if progress:
                    print("[5/7] Building search chunks", file=sys.stderr, flush=True)
                chunk_audit = store.create_chunks(
                    policy=policy,
                    tokenizer_provider=StrictestTokenizer([dense, sparse]),
                    skip_low_interest=skip_low_interest,
                    progress=progress,
                )
                required_chunk_conditions = {key: chunk_audit[key] == 0 for key in ("uncovered_characters", "chunks_over_limit", "truncated_chunks", "blocks_with_coverage_gaps")}
                if not all(required_chunk_conditions.values()):
                    raise NativePreMvpError(f"Chunk audit failed: {required_chunk_conditions}")
                store.commit()
                store.conn.execute(
                    "UPDATE native_build_audit SET audit_json=? WHERE id=1",
                    (_json({"stage": "chunks_ready", "chunk_audit": chunk_audit, "contracts": contracts}),),
                )
                store.commit()
            assert chunk_audit is not None
            required_chunk_conditions = {key: chunk_audit.get(key, 0) == 0 for key in ("uncovered_characters", "chunks_over_limit", "truncated_chunks", "blocks_with_coverage_gaps")}
            if not all(required_chunk_conditions.values()):
                raise NativePreMvpError(f"Chunk audit failed: {required_chunk_conditions}")
            partial_embeddings = store.remove_partial_embeddings()
            dense_space = _chunked_space(dense.embedding_space_id, policy.id)
            sparse_space = _chunked_space(sparse.embedding_space_id, policy.id)
            joint_started = time.perf_counter()
            complete_before = int(store.conn.execute(
                "SELECT COUNT(*) FROM retrieval_chunks rc "
                "WHERE rc.chunk_policy_id=? AND EXISTS (SELECT 1 FROM dense_native_metadata d WHERE d.chunk_id=rc.id) "
                "AND EXISTS (SELECT 1 FROM sparse_vector_metadata s WHERE s.chunk_id=rc.id)", (policy.id,)
            ).fetchone()[0])
            dense_processed = complete_before
            sparse_processed = complete_before
            real_tokens = 0
            padded_tokens = 0
            total_chunks = chunk_audit["total_retrieval_chunks"]
            if progress:
                print("[6/7] Building dense+sparse search index", file=sys.stderr, flush=True)
            with tqdm(total=total_chunks, initial=complete_before, desc="Building dense+sparse index", unit="chunk", file=sys.stderr,
                      dynamic_ncols=True, disable=not progress) as bar:
                for rows in store.embedding_batches_by_length(policy_id=policy.id, batch_size=batch_size, missing_only=resume):
                    batch_len = len(rows)
                    texts = [str(row["text"]) for row in rows]
                    for row, text in zip(rows, texts, strict=True):
                        dense.assert_fits(text, chunk_id=str(row["id"]), block_id=str(row["block_id"]), source_identity=str(row["block_id"]))
                        sparse.assert_fits(text, chunk_id=str(row["id"]), block_id=str(row["block_id"]), source_identity=str(row["block_id"]))
                    dense_vectors, sparse_vectors = embed_joint_documents(dense, sparse, texts)
                    batch_metrics = getattr(getattr(dense, "backend", None), "last_batch_metrics", {})
                    real_tokens += int(batch_metrics.get("real_tokens", 0))
                    padded_tokens += int(batch_metrics.get("padded_tokens", 0))
                    store.write_dense_batch(rows=rows, vectors=dense_vectors, model=dense.model_name, space=dense_space)
                    store.write_sparse_batch(rows=rows, vectors=sparse_vectors, model=sparse.model_name, space=sparse_space)
                    store.commit()
                    store.conn.execute(
                        "UPDATE native_build_audit SET audit_json=? WHERE id=1",
                        (_json({"stage": "embedding", "processed": dense_processed + batch_len,
                                "total_chunks": total_chunks, "chunk_audit": chunk_audit, "contracts": contracts}),),
                    )
                    store.commit()
                    dense_processed += batch_len
                    sparse_processed += batch_len
                    bar.update(batch_len)
                    del dense_vectors, sparse_vectors, texts, rows
                    if dense_processed % (batch_size * 25) == 0:
                        _release_batch_memory()
            joint_seconds = time.perf_counter() - joint_started
            _release_batch_memory()
            embedding_metrics = {
                "chunks": chunk_audit["total_retrieval_chunks"],
                "reused": complete_before,
                "partial_embeddings_removed": partial_embeddings,
                "dense": {"processed": dense_processed, "seconds": joint_seconds,
                          "throughput": dense_processed / joint_seconds if joint_seconds else 0.0,
                          "device": dense.runtime_metadata.get("device"), "shared_joint_pass": True},
                "sparse": {"processed": sparse_processed, "seconds": joint_seconds,
                           "throughput": sparse_processed / joint_seconds if joint_seconds else 0.0,
                           "device": sparse.runtime_metadata.get("device"), "shared_joint_pass": True},
                "joint": {"processed": dense_processed, "seconds": joint_seconds,
                           "throughput": dense_processed / joint_seconds if joint_seconds else 0.0,
                           "device": dense.runtime_metadata.get("device"),
                           "real_tokens": real_tokens, "padded_tokens": padded_tokens,
                           "padding_efficiency": real_tokens / padded_tokens if padded_tokens else 1.0,
                           "tokens_per_second": real_tokens / joint_seconds if joint_seconds else 0.0,
                           "chunks_per_second": dense_processed / joint_seconds if joint_seconds else 0.0},
                "total_seconds": joint_seconds,
            }
            audit = store.audit()
            audit.update({"schema_version": NATIVE_PRE_MVP_SCHEMA_VERSION, "export_path": str(export_path), "output_db": str(output_db),
                          "scanned_source_documents": scanned, "parsed_chats": parsed, "failed_chats": failed,
                          "chunk_audit": chunk_audit, "embedding_build": embedding_metrics,
                          "contracts": {**contracts, "dense_embedding_space_id": dense_space, "sparse_embedding_space_id": sparse_space},
                          "timing_ms": {"total": (time.perf_counter() - started) * 1000}})
            conditions = {
                **required_chunk_conditions,
                "dense_complete": audit["counts"]["dense_vectors_native"] == audit["counts"]["retrieval_chunks"],
                "sparse_complete": audit["counts"]["sparse_vectors_compact"] == audit["counts"]["retrieval_chunks"],
                "non_finite_zero": audit["non_finite_dense_vectors"] == 0,
                "orphan_zero": audit["orphan_mappings"] == 0,
                "duplicates_zero": audit["duplicate_chunk_mappings"] == 0,
                "legacy_absent": not audit["legacy_tables_present"],
                "integrity": audit["integrity_check"] == "ok" and audit["foreign_key_errors"] == 0,
            }
            audit["conditions"] = conditions
            store.conn.execute("UPDATE native_build_audit SET status=?,finished_at=?,audit_json=? WHERE id=1",
                               ("completed" if all(conditions.values()) else "failed", datetime.now(UTC).isoformat(), _json(audit)))
            store.commit()
            if not all(conditions.values()):
                raise NativePreMvpError(f"Native build audit failed: {conditions}")
        os.replace(building_db, output_db)
        return audit
    except Exception:
        if building_db.exists():
            with sqlite3.connect(building_db) as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='native_build_audit'").fetchone():
                    conn.execute("UPDATE native_build_audit SET status='failed',finished_at=? WHERE id=1", (datetime.now(UTC).isoformat(),))
        raise


def _chunked_space(space_id: str, policy_id: str) -> str:
    return f"{space_id};chunk_policy={policy_id}" if ";chunk_policy=" not in space_id else space_id


def _release_batch_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        return


def candidate_union(dense_ids: Iterable[str], sparse_ids: Iterable[str]) -> set[str]:
    """Build the fusion candidate set by stable chunk identity."""
    return set(dense_ids) | set(sparse_ids)


def fuse_candidate_scores(
    candidates: Iterable[str], dense_scores: dict[str, float], sparse_scores: dict[str, float], *, alpha: float, beta: float,
) -> list[tuple[str, float]]:
    """Fuse only after union; missing branch scores are explicit zeroes."""
    return sorted(
        ((chunk_id, alpha * dense_scores.get(chunk_id, 0.0) + beta * sparse_scores.get(chunk_id, 0.0)) for chunk_id in candidates),
        key=lambda item: (-item[1], item[0]),
    )


class CompactSparseSearchBackend:
    """Read-only flat CSR-like scorer backed only by compact BLOB tables."""

    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        needed = {"sparse_vector_metadata", "sparse_vectors_compact", "sparse_vocabulary"}
        if not needed.issubset(tables) or "sparse_terms" in tables:
            self.close()
            raise NativePreMvpError("Compact sparse runtime requires the clean native schema; no legacy fallback exists.")
        group = self.conn.execute("SELECT model_name,embedding_space_id,COUNT(*) AS count FROM sparse_vector_metadata GROUP BY model_name,embedding_space_id").fetchall()
        if len(group) != 1:
            self.close()
            raise NativePreMvpError(f"Expected one sparse space, found {len(group)}.")
        self.model_name, self.embedding_space_id = str(group[0][0]), str(group[0][1])
        started = time.perf_counter()
        self._load()
        self.materialization_ms = (time.perf_counter() - started) * 1000

    def _load(self) -> None:
        ids: list[str] = []
        norms: list[float] = []
        offsets = [0]
        indices: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for row in self.conn.execute("SELECT m.chunk_id,m.norm,v.indices_blob,v.weights_blob FROM sparse_vector_metadata m JOIN sparse_vectors_compact v ON v.rowid=m.rowid ORDER BY m.rowid"):
            term_ids = _unpack_array(bytes(row["indices_blob"]), np.dtype("<u4"))
            term_weights = _unpack_array(bytes(row["weights_blob"]), np.dtype("<f4"))
            if len(term_ids) != len(term_weights):
                raise NativePreMvpError(f"Sparse BLOB mismatch chunk_id={row['chunk_id']}.")
            ids.append(str(row["chunk_id"])); norms.append(float(row["norm"])); indices.append(term_ids); weights.append(term_weights); offsets.append(offsets[-1] + len(term_ids))
        self.chunk_ids = np.asarray(ids, dtype=str)
        self.norms = np.asarray(norms, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.dtype("<u4"))
        self.weights = np.concatenate(weights) if weights else np.empty(0, dtype=np.dtype("<f4"))
        self.term_ids = {str(row["token_text"]): int(row["term_id"]) for row in self.conn.execute("SELECT term_id,token_text FROM sparse_vocabulary")}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CompactSparseSearchBackend":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def score(self, query_terms: dict[str, float]) -> np.ndarray:
        vector = np.zeros(len(self.term_ids) + 1, dtype=np.float32)
        # Keep legacy cosine semantics: OOV query terms contribute to the query norm
        # even though they cannot contribute to a document dot product.
        norm_sq = sum(float(weight) * float(weight) for weight in query_terms.values())
        for token, weight in query_terms.items():
            numeric = self.term_ids.get(token)
            if numeric is not None:
                vector[numeric] = np.float32(weight)
        if not norm_sq:
            return np.zeros(len(self.chunk_ids), dtype=np.float32)
        contributions = self.weights * vector[self.indices]
        dots = np.add.reduceat(contributions, self.offsets[:-1])
        denominator = self.norms * np.float32(math.sqrt(norm_sq))
        return np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator != 0)

    def search(self, query_terms: dict[str, float], *, limit: int) -> list[SparseNativeHit]:
        scores = self.score(query_terms)
        indices = np.lexsort((self.chunk_ids, -scores))[:limit]
        return [SparseNativeHit(str(self.chunk_ids[index]), float(scores[index])) for index in indices]


class NativePreMvpRetriever:
    """Native sqlite-vec dense, compact sparse, deterministic hybrid and provenance joins."""

    def __init__(self, db_path: Path) -> None:
        started = time.perf_counter()
        self.dense = NativeDenseSearchBackend(db_path)
        dense_ready = time.perf_counter()
        self.sparse = CompactSparseSearchBackend(db_path)
        sparse_ready = time.perf_counter()
        self.conn = self.sparse.conn
        self.dense_model, self.dense_space = tuple(self.dense.conn.execute("SELECT model_name,embedding_space_id FROM dense_native_metadata LIMIT 1").fetchone())
        self.load_timing_ms = {
            "dense_backend_open": (dense_ready - started) * 1000,
            "sparse_materialization": (sparse_ready - dense_ready) * 1000,
            "total": (sparse_ready - started) * 1000,
        }
        self.last_search_timing_ms: dict[str, float] = {}

    def close(self) -> None:
        self.dense.close(); self.sparse.close()

    def __enter__(self) -> "NativePreMvpRetriever":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def search(self, *, query_dense: list[float], query_sparse: dict[str, float], limit: int = 20, alpha: float = 0.65, beta: float = 0.35, dense_candidate_k: int = 500, sparse_candidate_k: int = 500) -> list[NativeRetrievalHit]:
        started = time.perf_counter()
        if alpha < 0 or beta < 0 or alpha + beta == 0:
            raise NativePreMvpError("Fusion weights must be non-negative and non-zero.")
        dense_hits = self.dense.search(query_dense, limit=max(limit, dense_candidate_k), model_name=str(self.dense_model), embedding_space_id=str(self.dense_space))
        dense_search_finished = time.perf_counter()
        sparse_scores = self.sparse.score(query_sparse)
        sparse_search_finished = time.perf_counter()
        self.last_sparse_scores = sparse_scores
        if dense_candidate_k <= 0 or sparse_candidate_k <= 0:
            raise NativePreMvpError("Candidate pool sizes must be positive.")
        sparse_order = np.lexsort((self.sparse.chunk_ids, -sparse_scores))[:sparse_candidate_k]
        sparse_candidates = [index for index in sparse_order if sparse_scores[index] > 0]
        candidates = candidate_union((hit.chunk_id for hit in dense_hits), (str(self.sparse.chunk_ids[index]) for index in sparse_candidates))
        union_finished = time.perf_counter()
        dense_by_id = self.dense.scores_for_chunk_ids(query_dense, candidates, model_name=str(self.dense_model), embedding_space_id=str(self.dense_space))
        dense_rescore_finished = time.perf_counter()
        sparse_by_id = {str(self.sparse.chunk_ids[index]): float(sparse_scores[index]) for index in sparse_candidates}
        ordered = [chunk_id for chunk_id, _ in fuse_candidate_scores(candidates, dense_by_id, sparse_by_id, alpha=alpha, beta=beta)[:limit]]
        fusion_finished = time.perf_counter()
        provenance = self._provenance(ordered)
        provenance_finished = time.perf_counter()
        self.last_search_timing_ms = {
            "dense_top_k": (dense_search_finished - started) * 1000,
            "sparse_top_k": (sparse_search_finished - dense_search_finished) * 1000,
            "candidate_union": (union_finished - sparse_search_finished) * 1000,
            "dense_rescore": (dense_rescore_finished - union_finished) * 1000,
            "fusion": (fusion_finished - dense_rescore_finished) * 1000,
            "provenance": (provenance_finished - fusion_finished) * 1000,
            "total": (provenance_finished - started) * 1000,
            "dense_candidate_count": float(len(dense_hits)),
            "sparse_candidate_count": float(len(sparse_candidates)),
            "union_candidate_count": float(len(candidates)),
        }
        return [NativeRetrievalHit(chunk_id, dense_by_id.get(chunk_id, 0.0), sparse_by_id.get(chunk_id, 0.0), alpha * dense_by_id.get(chunk_id, 0.0) + beta * sparse_by_id.get(chunk_id, 0.0), provenance[chunk_id]) for chunk_id in ordered]

    def _provenance(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            "SELECT rc.id AS chunk_id,rc.ordinal AS chunk_ordinal,rc.source_char_start,rc.source_char_end,rc.text,"
            "b.id AS block_id,b.block_type,m.id AS message_id,m.message_id AS source_message_id,m.role,m.ordinal AS message_ordinal,m.time_utc,"
            "c.id AS conversation_id,c.conversation_id AS dialogue_id,c.title AS conversation_title,c.project_id,c.create_time_utc,c.update_time_utc,sd.relative_path AS source_path "
            "FROM retrieval_chunks rc JOIN blocks b ON b.id=rc.block_id JOIN messages m ON m.id=b.message_id "
            "JOIN conversations c ON c.id=m.conversation_id JOIN source_documents sd ON sd.id=c.source_document_id "
            "WHERE rc.id IN (" + marks + ")", chunk_ids).fetchall()
        return {str(row["chunk_id"]): dict(row) for row in rows}

    def messages_for_windows(self, windows: dict[str, tuple[int, int]]) -> dict[str, list[dict[str, Any]]]:
        """Return bounded chronological message windows keyed by conversation ID."""
        result: dict[str, list[dict[str, Any]]] = {}
        for conversation_id, (start, end) in windows.items():
            rows = self.conn.execute(
                "SELECT id,message_id,ordinal,role,time_utc,raw_text FROM messages "
                "WHERE conversation_id=? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
                (conversation_id, start, end),
            ).fetchall()
            result[conversation_id] = [dict(row) for row in rows]
        return result


def native_pre_mvp_query(
    *, db_path: Path, query: str, dense: Any, sparse: Any, limit: int, alpha: float, beta: float, policy_id: str,
) -> dict[str, Any]:
    """Run the clean native retrieval path and return the stable query-result shape."""
    started = time.perf_counter()
    dense_space = _chunked_space(dense.embedding_space_id, policy_id)
    sparse_space = _chunked_space(sparse.embedding_space_id, policy_id)
    preprocessing_finished = time.perf_counter()
    dense_rows, sparse_rows = embed_joint_documents(dense, sparse, [query])
    query_dense, query_sparse = dense_rows[0], sparse_rows[0]
    dense_query_finished = time.perf_counter()
    sparse_query_finished = time.perf_counter()
    with NativePreMvpRetriever(db_path) as retriever:
        if str(retriever.dense_model) != dense.model_name or str(retriever.dense_space) != dense_space:
            raise NativePreMvpError("Dense query provider is incompatible with clean native DB embedding space.")
        if retriever.sparse.model_name != sparse.model_name or retriever.sparse.embedding_space_id != sparse_space:
            raise NativePreMvpError("Sparse query provider is incompatible with clean native DB embedding space.")
        results = retriever.search(query_dense=query_dense, query_sparse=query_sparse, limit=limit, alpha=alpha, beta=beta)
        search_timing = dict(retriever.last_search_timing_ms)
        sparse_scores = retriever.last_sparse_scores
        dense_count = int(retriever.dense.conn.execute("SELECT COUNT(*) FROM dense_native_metadata").fetchone()[0])
        sparse_count = len(retriever.sparse.chunk_ids)
    items = []
    for rank, hit in enumerate(results, start=1):
        source = hit.provenance
        items.append({
            "rank": rank, "block_id": hit.chunk_id, "chunk_id": hit.chunk_id,
            "source_path": source.get("source_path"), "project": source.get("project_id"),
            "conversation_id": source.get("conversation_id"), "message_id": source.get("message_id"),
            "conversation_title": source.get("conversation_title"), "role": source.get("role"),
            "block_type": source.get("block_type"), "chunk_ordinal": source.get("chunk_ordinal"),
            "source_char_start": source.get("source_char_start"), "source_char_end": source.get("source_char_end"),
            "dense_score": hit.dense_score, "sparse_score": hit.sparse_score, "final_score": hit.final_score,
            "overlapping_terms": [], "preview": " ".join(str(source.get("text", "")).split())[:320],
        })
    return {
        "schema_version": "kb.query.result.v1", "query": query, "candidate_blocks": sparse_count,
        "results": items,
        "diagnostics": {
            "dense": {"enabled": True, "status": "active", "query_embedding_created": True,
                      "embedding_space_id": dense_space, "query_dimension": len(query_dense),
                      "query_norm": math.sqrt(sum(value * value for value in query_dense)),
                      "candidate_blocks_with_vector": dense_count},
            "sparse": {"enabled": True, "status": "active", "query_embedding_created": True,
                       "embedding_space_id": sparse_space, "query_term_count": len(query_sparse),
                       "candidate_blocks_with_terms": sparse_count,
                       "nonzero_score_count": int(np.count_nonzero(sparse_scores)),
                       "max_score": float(sparse_scores.max()) if len(sparse_scores) else 0.0},
        },
        "latency_ms": {
            "preprocessing": (preprocessing_finished - started) * 1000,
            "dense_query_encoding": (dense_query_finished - preprocessing_finished) * 1000,
            "sparse_query_encoding": (sparse_query_finished - dense_query_finished) * 1000,
            "retriever_load": retriever.load_timing_ms["total"],
            **search_timing,
            "result_packaging": (time.perf_counter() - (sparse_query_finished + retriever.load_timing_ms["total"] / 1000)) * 1000,
            "total": (time.perf_counter() - started) * 1000,
        },
        "runtime": {
            "candidate_pool": 500,
            "dense_calls": 1,
            "sparse_calls": 1,
            "fusion_calls": 1,
            "sparse_materialized_bytes": int(retriever.sparse.indices.nbytes + retriever.sparse.weights.nbytes + retriever.sparse.offsets.nbytes + retriever.sparse.norms.nbytes + retriever.sparse.chunk_ids.nbytes),
        },
    }


def write_native_pre_mvp_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Native Pre-MVP Build", "", f"Output: `{report.get('output_db', '')}`", "", "## Counts", "", "| Object | Count |", "|---|---:|"]
    lines.extend(f"| {name} | {count} |" for name, count in report.get("counts", {}).items())
    lines += ["", "## Audit", ""]
    lines.extend(f"- `{name}`: `{value}`" for name, value in report.get("conditions", {}).items())
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
