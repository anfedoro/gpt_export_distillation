"""Atomic clone-and-rebuild maintenance for clean-native PTHA databases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from kb.index.chunk_builder import StrictestTokenizer, build_chunk_policy
from kb.embeddings.bge_m3_provider import embed_joint_documents
from kb.storage.native_pre_mvp import NativeBuildStore, NativePreMvpRetriever, _chunked_space
from ptha.config import PthaConfig
from ptha.database import inspect_database
from ptha.errors import PthaError
from ptha.incremental import embedding_contract_fingerprint
from ptha.lifecycle import service_status
from ptha.operations import (
    clear_maintenance_state,
    maintenance_lock,
    maintenance_state_path,
    new_maintenance_state,
    read_maintenance_state,
    write_maintenance_state,
)

CANONICAL_TABLES = ("source_documents", "conversations", "messages", "blocks")


class ReindexError(PthaError):
    code = "reindex_failed"
    exit_code = 9


def reindex_database(config: PthaConfig, *, force: bool = False, batch_size: int | None = None,
                     dense_device: str | None = None, sparse_device: str | None = None,
                     progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    effective = replace(config, batch_size=batch_size or config.batch_size,
                        dense_device=dense_device or config.dense_device,
                        sparse_device=sparse_device or config.sparse_device,
                        embedding_device=(dense_device or sparse_device or config.embedding_device).replace("mps", "gpu"))
    with maintenance_lock(effective):
        return _reindex(effective, force=force, progress=progress)


def _reindex(config: PthaConfig, *, force: bool, progress: Callable[[str], None] | None) -> dict[str, Any]:
    lifecycle = service_status(config)
    if lifecycle["state"] in {"starting", "ready", "degraded", "stopping", "unknown-process"}:
        raise ReindexError("PTHA service must be stopped before reindexing.\n\nRun:\n  ptha service stop")
    active = config.database
    before = inspect_database(active, integrity=True)
    if before.get("state") != "ready" or before.get("integrity_check") != "ok":
        raise ReindexError("The active PTHA database is not ready for reindexing.")
    temporary = active.with_name(active.name + ".reindexing")
    prior_marker = read_maintenance_state(config)
    if (prior_marker or temporary.exists()) and not force:
        raise ReindexError("An interrupted reindex is present. Inspect ptha doctor, then rerun with --force to clean and restart.")
    if force:
        _safe_remove_temporary(active, temporary)
        clear_maintenance_state(config)
    required = max(active.stat().st_size * 2, 64 * 1024 * 1024)
    if shutil.disk_usage(active.parent).free < required:
        raise ReindexError(f"Insufficient free space for atomic reindex; at least {required} bytes are required.")
    state = write_maintenance_state(config, new_maintenance_state(config, "reindex", active, temporary))
    started = time.perf_counter()
    canonical_before = canonical_fingerprint(active)
    old_chunks, skip_low_interest, audit_record = _existing_contract(active)
    try:
        _stage(progress, "[1/6] Cloning active database")
        _clone_database(active, temporary)
        state = write_maintenance_state(config, state, phase="reconstructing_chunks")
        from ptha.service import build_providers
        dense, sparse = build_providers(config)
        policy_details = audit_record.get("chunk_audit", {})
        budget = policy_details.get("chunk_policy_content_token_budget")
        version = str(policy_details.get("chunk_policy_version") or "v2")
        policy = build_chunk_policy([dense, sparse], version=version,
                                    content_budget_override=int(budget) if budget else None)
        _stage(progress, "[2/6] Reconstructing retrieval chunks")
        with NativeBuildStore(temporary, create_schema=False) as store:
            store.reset_derived()
            chunk_audit = store.create_chunks(
                policy=policy,
                tokenizer_provider=StrictestTokenizer([dense, sparse]),
                skip_low_interest=skip_low_interest,
                progress=False,
            )
            store.commit()
            state = write_maintenance_state(config, state, phase="building_dense_and_sparse")
            _stage(progress, "[3/6] Building dense and sparse indexes (joint forward)")
            _stage(progress, "[4/6] Publishing dense and sparse batches")
            dense_space = _chunked_space(dense.embedding_space_id, policy.id)
            sparse_space = _chunked_space(sparse.embedding_space_id, policy.id)
            joint_started = time.perf_counter()
            dense_processed = 0
            sparse_processed = 0
            real_tokens = 0
            padded_tokens = 0
            for rows in store.embedding_batches_by_length(policy_id=policy.id, batch_size=config.batch_size):
                texts = [str(row["text"]) for row in rows]
                dense_vectors, sparse_vectors = embed_joint_documents(dense, sparse, texts)
                batch_metrics = getattr(getattr(dense, "backend", None), "last_batch_metrics", {})
                real_tokens += int(batch_metrics.get("real_tokens", 0))
                padded_tokens += int(batch_metrics.get("padded_tokens", 0))
                store.write_dense_batch(rows=rows, vectors=dense_vectors, model=dense.model_name, space=dense_space)
                store.write_sparse_batch(rows=rows, vectors=sparse_vectors, model=sparse.model_name, space=sparse_space)
                store.commit()
                dense_processed += len(rows)
                sparse_processed += len(rows)
                del dense_vectors, sparse_vectors, texts, rows
                if dense_processed % (config.batch_size * 25) == 0:
                    from kb.storage.native_pre_mvp import _release_batch_memory
                    _release_batch_memory()
            joint_seconds = time.perf_counter() - joint_started
            audit = store.audit()
            audit["embedding_build"] = {
                "chunks": chunk_audit["total_retrieval_chunks"],
                "dense": {"processed": dense_processed, "seconds": joint_seconds,
                          "throughput": dense_processed / joint_seconds if joint_seconds else 0.0,
                          "device": getattr(dense, "runtime_metadata", {}).get("device"),
                          "shared_joint_pass": True},
                "sparse": {"processed": sparse_processed, "seconds": joint_seconds,
                           "throughput": sparse_processed / joint_seconds if joint_seconds else 0.0,
                           "device": getattr(sparse, "runtime_metadata", {}).get("device"),
                           "shared_joint_pass": True},
                "joint": {"processed": dense_processed, "seconds": joint_seconds,
                          "throughput": dense_processed / joint_seconds if joint_seconds else 0.0,
                          "device": getattr(dense, "runtime_metadata", {}).get("device"),
                          "real_tokens": real_tokens, "padded_tokens": padded_tokens,
                          "padding_efficiency": real_tokens / padded_tokens if padded_tokens else 1.0,
                          "tokens_per_second": real_tokens / joint_seconds if joint_seconds else 0.0,
                          "chunks_per_second": dense_processed / joint_seconds if joint_seconds else 0.0},
                "total_seconds": joint_seconds,
            }
            contracts = {"dense": {"model": dense.model_name, "embedding_space_id": dense.embedding_space_id},
                         "sparse": {"model": sparse.model_name, "embedding_space_id": sparse.embedding_space_id},
                         "chunk_policy": policy.id, "dense_embedding_space_id": dense_space,
                         "sparse_embedding_space_id": sparse_space}
            embedding_contract, embedding_contract_digest = embedding_contract_fingerprint(dense=dense, sparse=sparse)
            contracts["embedding_contract"] = embedding_contract
            contracts["embedding_contract_fingerprint"] = embedding_contract_digest
            audit.update({"chunk_audit": chunk_audit, "contracts": contracts, "reindexed": True})
            manifest = store.write_generation_manifest(embedding_contract_fingerprint=embedding_contract_digest)
            if manifest is not None:
                audit["generation_manifest"] = manifest
            store.conn.execute("UPDATE native_build_audit SET export_path=?,status='completed',finished_at=?,contracts_json=?,audit_json=? WHERE id=1",
                               ("<reindexed-from-database>", _utc_now(), _json(contracts), _json(audit)))
            store.commit()
        state = write_maintenance_state(config, state, phase="validating")
        _stage(progress, "[5/6] Validating replacement database")
        validation = inspect_database(temporary, integrity=True)
        canonical_after = canonical_fingerprint(temporary)
        counts = validation.get("counts", {})
        conditions = {
            "ready": validation.get("state") == "ready",
            "integrity": validation.get("integrity_check") == "ok" and validation.get("foreign_key_errors") == 0,
            "canonical_hash": canonical_before["sha256"] == canonical_after["sha256"],
            "canonical_counts": canonical_before["counts"] == canonical_after["counts"],
            "dense_complete": counts.get("retrieval_chunks") == counts.get("dense_native_metadata"),
            "sparse_complete": counts.get("retrieval_chunks") == counts.get("sparse_vector_metadata"),
            "chunk_reconstruction": counts.get("retrieval_chunks") == chunk_audit.get("total_retrieval_chunks"),
        }
        if not all(conditions.values()):
            raise ReindexError(f"Replacement validation failed: {conditions}")
        with NativePreMvpRetriever(temporary) as retriever:
            retriever.search(query_dense=dense.embed_query("PTHA technical reindex smoke test"),
                             query_sparse=sparse.embed_query("PTHA technical reindex smoke test"), limit=1,
                             dense_candidate_k=1, sparse_candidate_k=1)
        state = write_maintenance_state(config, state, phase="publishing")
        _stage(progress, "[6/6] Publishing replacement database")
        os.replace(temporary, active)
        reopened = inspect_database(active, integrity=True)
        if reopened.get("state") != "ready" or reopened.get("integrity_check") != "ok":
            raise ReindexError("Published database could not be reopened cleanly.")
        clear_maintenance_state(config)
        return {"schema_version": 1, "database": str(active), "duration_seconds": round(time.perf_counter() - started, 3),
                "database_size_bytes": active.stat().st_size, "canonical_sha256": canonical_after["sha256"],
                "canonical_counts": canonical_after["counts"], "previous_chunk_count": old_chunks,
                "retrieval_chunk_count": counts.get("retrieval_chunks"),
                "embedded_count": min(dense_processed, sparse_processed),
                "embedding_build": audit["embedding_build"], "conditions": conditions}
    except (KeyboardInterrupt, Exception):
        # The marker and clone intentionally remain for doctor and explicit --force recovery.
        raise


def canonical_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        for table in CANONICAL_TABLES:
            columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY id")
            count = 0
            digest.update(table.encode())
            digest.update(_json(columns).encode())
            for row in rows:
                digest.update(_json(list(row)).encode())
                digest.update(b"\n")
                count += 1
            counts[table] = count
    return {"sha256": digest.hexdigest(), "counts": counts}


def _clone_database(source: Path, target: Path) -> None:
    if target.exists():
        raise ReindexError(f"Temporary reindex database already exists: {target}")
    with closing(sqlite3.connect(source)) as source_conn, closing(sqlite3.connect(target)) as target_conn:
        source_conn.backup(target_conn)
    os.chmod(target, 0o600)


def _existing_contract(path: Path) -> tuple[int, bool, dict[str, Any]]:
    with closing(sqlite3.connect(path)) as conn:
        chunk_count = int(conn.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0])
        low_indexed = bool(conn.execute(
            "SELECT 1 FROM retrieval_chunks rc JOIN blocks b ON b.id=rc.block_id JOIN messages m ON m.id=b.message_id "
            "JOIN conversations c ON c.id=m.conversation_id JOIN source_documents sd ON sd.id=c.source_document_id "
            "WHERE sd.interest_tier IN ('low','quarantine') LIMIT 1"
        ).fetchone())
        row = conn.execute("SELECT audit_json FROM native_build_audit WHERE id=1").fetchone()
    try:
        audit = json.loads(row[0]) if row and row[0] else {}
    except json.JSONDecodeError:
        audit = {}
    return chunk_count, not low_indexed, audit


def _safe_remove_temporary(active: Path, temporary: Path) -> None:
    if temporary.parent.resolve() != active.parent.resolve() or temporary.name != active.name + ".reindexing":
        raise ReindexError("Refusing to remove an unexpected temporary database path.")
    if temporary.is_symlink():
        raise ReindexError("Refusing to follow a temporary database symlink.")
    if temporary.exists():
        if temporary.lstat().st_uid != os.geteuid() or not temporary.is_file():
            raise ReindexError("Temporary reindex database is not safe to remove.")
        temporary.unlink()
    for suffix in ("-wal", "-shm"):
        companion = Path(str(temporary) + suffix)
        if companion.exists() and not companion.is_symlink() and companion.lstat().st_uid == os.geteuid():
            companion.unlink()


def _stage(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()
