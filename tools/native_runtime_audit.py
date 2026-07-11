#!/usr/bin/env python3
"""Local read-only runtime audit for the clean native retrieval database."""

from __future__ import annotations

import json
import os
import resource
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path

from kb.embeddings.sentence_transformer_provider import SentenceTransformerDenseProvider, SentenceTransformerSparseProvider
from kb.storage.native_pre_mvp import NativePreMvpRetriever


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024)


def current_rss_mb() -> float:
    value = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip())
    return value / 1024


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dense = SentenceTransformerDenseProvider("BAAI/bge-m3", device="mps", torch_dtype="float16")
    sparse = SentenceTransformerSparseProvider("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1", device="mps", torch_dtype="float16", top_k=128)
    providers_ready = time.perf_counter()
    probes = json.loads(args.probes.read_text(encoding="utf-8"))["probes"]
    with NativePreMvpRetriever(args.db) as retriever:
        corpus_ready = time.perf_counter()
        steady_after_materialization = current_rss_mb()
        warm_rows = []
        for probe in probes:
            query_started = time.perf_counter()
            dense_query = dense.embed_query(probe["query"])
            dense_done = time.perf_counter()
            sparse_query = sparse.embed_query(probe["query"])
            sparse_done = time.perf_counter()
            retriever.search(query_dense=dense_query, query_sparse=sparse_query, limit=20, alpha=0.65, beta=0.35)
            search_done = time.perf_counter()
            timing = dict(retriever.last_search_timing_ms)
            warm_rows.append({
                "probe_id": probe["probe_id"],
                "query_preprocessing_ms": 0.0,
                "dense_query_encoding_ms": (dense_done - query_started) * 1000,
                "sparse_query_encoding_ms": (sparse_done - dense_done) * 1000,
                "search_total_ms": (search_done - sparse_done) * 1000,
                "end_to_end_ms": (search_done - query_started) * 1000,
                "stages": timing,
            })
        sparse_memory = int(retriever.sparse.indices.nbytes + retriever.sparse.weights.nbytes + retriever.sparse.offsets.nbytes + retriever.sparse.norms.nbytes + retriever.sparse.chunk_ids.nbytes)
        steady_after_warm = current_rss_mb()
        load_timing = dict(retriever.load_timing_ms)
    conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    objects = conn.execute("SELECT name, type, COALESCE((SELECT SUM(pgsize) FROM dbstat WHERE name=m.name),0) FROM sqlite_master m WHERE type IN ('table','index','trigger','view') ORDER BY 3 DESC LIMIT 20").fetchall()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts = {}
    for table in ("conversations", "messages", "blocks", "retrieval_chunks", "dense_native_metadata", "sparse_vector_metadata"):
        if table in tables:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    warm = {name: {"p50": statistics.median([row[name] for row in warm_rows]), "p95": sorted(row[name] for row in warm_rows)[int(len(warm_rows) * .95) - 1], "p99": sorted(row[name] for row in warm_rows)[int(len(warm_rows) * .99) - 1]} for name in ("dense_query_encoding_ms", "sparse_query_encoding_ms", "search_total_ms", "end_to_end_ms")}
    payload = {
        "schema_version": "kb.native.runtime_performance.v1", "status": "completed", "probe_count": len(probes),
        "db_path": str(args.db), "db_size_bytes": args.db.stat().st_size, "db_size_gib": args.db.stat().st_size / (1024**3),
        "cold_start_ms": {"providers": (providers_ready - started) * 1000, "corpus_and_sparse_materialization": (corpus_ready - providers_ready) * 1000, "total": (corpus_ready - started) * 1000},
        "warm_latency_ms": warm, "first_query": warm_rows[0], "peak_rss_mb": rss_mb(), "steady_rss_mb": {"after_materialization": steady_after_materialization, "after_warm": steady_after_warm}, "sparse_runtime_bytes": sparse_memory,
        "sparse_materialization_ms": load_timing["sparse_materialization"], "runtime_contract": {"candidate_pool": 500, "dense_calls_per_query": 1, "sparse_calls_per_query": 1, "fusion_calls_per_query": 1},
        "counts": counts, "integrity_check": integrity, "legacy_tables": sorted(tables & {"dense_vectors", "sparse_terms", "dense_vectors_native_migrations"}),
        "top_objects": [{"name": row[0], "type": row[1], "bytes": row[2]} for row in objects],
    }
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Native Runtime Performance Audit", "", f"Probes: {len(probes)}", f"DB: `{args.db}`", f"Size: {payload['db_size_gib']:.3f} GiB", "", "## Warm latency (ms)", "", "| Stage | p50 | p95 | p99 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {name} | {value['p50']:.2f} | {value['p95']:.2f} | {value['p99']:.2f} |" for name, value in warm.items())
    lines += ["", f"Cold start: {payload['cold_start_ms']['total']:.2f} ms", f"Peak RSS: {payload['peak_rss_mb']:.1f} MB", f"Sparse runtime: {sparse_memory / (1024**2):.1f} MiB", f"Integrity: `{integrity}`", "", "## Top objects", "", "| Object | Type | Bytes |", "|---|---|---:|"]
    lines.extend(f"| {row['name']} | {row['type']} | {row['bytes']} |" for row in payload["top_objects"])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
