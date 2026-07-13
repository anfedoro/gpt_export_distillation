"""Diagnostic-only BGE-M3 runtime and 10k embedding breakdown.

This module deliberately does not participate in the product import or retrieval
paths. It reports runtime facts and runs a fixed, local-only benchmark without
printing chunk text or retrieval results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import psutil

from kb.embeddings.bge_m3_provider import BgeM3Backend
from kb.storage.native_pre_mvp import NativeBuildStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect BGE-M3 runtime precision and embedding timings.")
    parser.add_argument("--chunk-db", type=Path, required=True, help="Local chunk-only or native PTHA DB.")
    parser.add_argument("--output-db", type=Path, required=True, help="Temporary benchmark DB receiving vectors.")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    report = run_diagnostics(
        args.chunk_db.expanduser().resolve(), args.output_db.expanduser().resolve(),
        limit=args.limit, batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_diagnostics(
    chunk_db: Path, output_db: Path, *, limit: int = 10_000, batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Run a fixed diagnostic pass; output contains no archive text."""
    if output_db.exists():
        raise FileExistsError(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chunk_db, output_db)
    process = psutil.Process(os.getpid())
    load_started = time.perf_counter()
    backend = BgeM3Backend("BAAI/bge-m3", device=device, torch_dtype="auto", sparse_top_k=128)
    model_load_seconds = time.perf_counter() - load_started
    snapshot = backend.diagnostic_snapshot()
    policy_id = _policy_id(output_db)
    rows = _select_rows(output_db, policy_id, limit)
    if not rows:
        raise RuntimeError("Diagnostic database contains no retrieval chunks.")
    peak_rss = process.memory_info().rss
    peak_mps = _mps_driver_memory()
    timings = {
        "tokenization_seconds": 0.0,
        "dense_tokenization_seconds": 0.0,
        "sparse_tokenization_seconds": 0.0,
        "dense_forward_seconds": 0.0,
        "dense_pooling_normalization_seconds": 0.0,
        "sparse_forward_seconds": 0.0,
        "sparse_projection_seconds": 0.0,
        "sparse_head_transfer_seconds": 0.0,
        "sparse_head_seconds": 0.0,
        "sparse_decode_seconds": 0.0,
        "dense_sqlite_insert_seconds": 0.0,
        "sparse_sqlite_insert_seconds": 0.0,
        "sqlite_insert_seconds": 0.0,
        "total_seconds": 0.0,
    }
    dense_space = "diagnostic-bge-m3-dense"
    sparse_space = "diagnostic-bge-m3-sparse"
    total_started = time.perf_counter()
    with NativeBuildStore(output_db, create_schema=False) as store:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [str(row["text"]) for row in batch]
            _measure_batch(backend, texts, batch, store, dense_space, sparse_space, timings)
            store.commit()
            peak_rss = max(peak_rss, process.memory_info().rss)
            peak_mps = max(peak_mps, _mps_driver_memory())
            if (start // batch_size + 1) % 25 == 0:
                _release_accelerator_cache()
        store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    timings["total_seconds"] = time.perf_counter() - total_started
    timings["sqlite_insert_seconds"] = (
        timings["dense_sqlite_insert_seconds"] + timings["sparse_sqlite_insert_seconds"]
    )
    return {
        "schema_version": 1,
        "chunks": len(rows),
        "batch_size": batch_size,
        "model_load_seconds": model_load_seconds,
        "runtime": snapshot,
        "timings": timings,
        "throughput_chunks_per_second": len(rows) / timings["total_seconds"] if timings["total_seconds"] else 0.0,
        "peak_rss_bytes": peak_rss,
        "peak_mps_driver_bytes": peak_mps or None,
        "output_database_size_bytes": output_db.stat().st_size,
        "privacy": {"archive_text_emitted": False, "retrieval_results_emitted": False},
    }


def _measure_batch(
    backend: BgeM3Backend, texts: list[str], rows: list[Any], store: NativeBuildStore,
    dense_space: str, sparse_space: str, timings: dict[str, float],
) -> None:
    import torch

    def sync() -> None:
        try:
            if backend.device == "mps" and torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif backend.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except (AttributeError, RuntimeError):
            pass

    started = time.perf_counter()
    token_started = time.perf_counter()
    dense_inputs = backend._inputs(texts)
    dense_tokenization_seconds = time.perf_counter() - token_started
    timings["dense_tokenization_seconds"] += dense_tokenization_seconds
    sync()
    forward_started = time.perf_counter()
    with torch.inference_mode():
        dense_hidden = backend.model(**dense_inputs).last_hidden_state
    sync()
    timings["dense_forward_seconds"] += time.perf_counter() - forward_started
    pool_started = time.perf_counter()
    with torch.inference_mode():
        dense_vectors_tensor = torch.nn.functional.normalize(dense_hidden[:, 0], p=2, dim=-1)
    dense_vectors = dense_vectors_tensor.float().cpu().numpy().tolist()
    timings["dense_pooling_normalization_seconds"] += time.perf_counter() - pool_started
    dense_insert_started = time.perf_counter()
    store.write_dense_batch(rows=rows, vectors=dense_vectors, model=backend.model_name, space=dense_space)
    timings["dense_sqlite_insert_seconds"] += time.perf_counter() - dense_insert_started
    del dense_hidden, dense_vectors_tensor, dense_vectors, dense_inputs

    token_started = time.perf_counter()
    sparse_inputs = backend._inputs(texts)
    sparse_tokenization_seconds = time.perf_counter() - token_started
    timings["sparse_tokenization_seconds"] += sparse_tokenization_seconds
    sync()
    forward_started = time.perf_counter()
    with torch.inference_mode():
        sparse_hidden = backend.model(**sparse_inputs).last_hidden_state
    sync()
    timings["sparse_forward_seconds"] += time.perf_counter() - forward_started
    projection_started = time.perf_counter()
    with torch.inference_mode():
        sparse_weights = torch.relu(backend.sparse_linear(sparse_hidden)).squeeze(-1)
    sync()
    timings["sparse_projection_seconds"] += time.perf_counter() - projection_started
    head_started = time.perf_counter()
    weights_cpu = sparse_weights.float().cpu()
    ids_cpu = sparse_inputs["input_ids"].cpu()
    attention_cpu = sparse_inputs["attention_mask"].cpu()
    timings["sparse_head_seconds"] += time.perf_counter() - head_started
    timings["sparse_head_transfer_seconds"] += time.perf_counter() - head_started
    decode_started = time.perf_counter()
    sparse_vectors = [
        backend._lexical(ids, values, mask)
        for ids, values, mask in zip(ids_cpu, weights_cpu, attention_cpu, strict=True)
    ]
    timings["sparse_decode_seconds"] += time.perf_counter() - decode_started
    sparse_insert_started = time.perf_counter()
    store.write_sparse_batch(rows=rows, vectors=sparse_vectors, model=backend.model_name, space=sparse_space)
    timings["sparse_sqlite_insert_seconds"] += time.perf_counter() - sparse_insert_started
    timings["tokenization_seconds"] += dense_tokenization_seconds + sparse_tokenization_seconds
    del sparse_hidden, sparse_weights, weights_cpu, ids_cpu, attention_cpu, sparse_inputs, sparse_vectors
    timings["sparse_head_seconds"] = (
        timings["sparse_projection_seconds"] + timings["sparse_head_transfer_seconds"]
    )
    _ = time.perf_counter() - started


def _select_rows(path: Path, policy_id: str, limit: int) -> list[Any]:
    with NativeBuildStore(path, create_schema=False) as store:
        return store.conn.execute(
            "SELECT id, block_id, text, token_count FROM retrieval_chunks "
            "WHERE chunk_policy_id=? ORDER BY token_count, id LIMIT ?", (policy_id, limit)
        ).fetchall()


def _policy_id(path: Path) -> str:
    with NativeBuildStore(path, create_schema=False) as store:
        row = store.conn.execute("SELECT chunk_policy_id FROM retrieval_chunks LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("Diagnostic database has no retrieval chunks.")
    return str(row[0])


def _release_accelerator_cache() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mps_driver_memory() -> int:
    try:
        import torch
        return int(torch.mps.driver_allocated_memory()) if torch.backends.mps.is_available() else 0
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
