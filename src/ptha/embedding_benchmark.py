"""Local-only embedding build benchmark; never emits archive content."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from gpt_export_distillation.config import DEFAULT_CONFIG as DISTILL_CONFIG
from gpt_export_distillation.loader import load_bundle
from gpt_export_distillation.pipeline import build_documents, write_output
from kb.embeddings.bge_m3_provider import build_bge_m3_providers, embed_joint_documents
from kb.embeddings.sentence_transformer_provider import SentenceTransformerDenseProvider, SentenceTransformerSparseProvider
from kb.index.chunk_builder import build_chunk_policy
from kb.ingest.chat_md_parser import parse_chat_file
from kb.ingest.tree_walker import scan_tree
from kb.storage.native_pre_mvp import NativeBuildStore


class _TokenizerProvider:
    document_prefix = ""
    effective_max_sequence_length = 512

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    def embedding_input(self, text: str) -> str:
        return text

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def fits_token_budget(self, text: str, budget: int) -> bool:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=budget + 1)["input_ids"]) <= budget


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark PTHA embedding throughput without publishing an archive DB.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a chunk-only benchmark DB from an export.")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Benchmark one embedding backend over a fixed chunk subset.")
    run.add_argument("--chunk-db", type=Path, required=True)
    run.add_argument("--output-db", type=Path, required=True)
    run.add_argument("--backend", choices=("legacy", "unified"), required=True)
    run.add_argument("--limit", type=int, default=10_000)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--device", default="gpu")
    run.add_argument("--model", default="anfedoro/bge-m3-mlx-fp16")
    run.add_argument("--model-revision", default="58e70901dbba8de8f3df91b5a313bcefcb151bae")
    args = parser.parse_args(argv)
    report = prepare_chunks(args.source, args.output) if args.command == "prepare" else run_benchmark(
        args.chunk_db, args.output_db, backend=args.backend, limit=args.limit,
        batch_size=args.batch_size, device=args.device, model=args.model,
        model_revision=args.model_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def prepare_chunks(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ptha-embedding-benchmark-") as temporary:
        bundle = load_bundle(source.expanduser().resolve())
        documents = build_documents(bundle, DISTILL_CONFIG)
        root = write_output(bundle, documents, DISTILL_CONFIG, str(Path(temporary) / "distilled"))
        tokenizer = _TokenizerProvider()
        policy = build_chunk_policy([tokenizer], content_budget_override=506)
        with NativeBuildStore(output) as store:
            for item in scan_tree(root):
                source_id = store.insert_source_document(root, item)
                if item.detected_kind != "chat_md":
                    continue
                parsed = parse_chat_file(root / item.relative_path, source_document_id=source_id,
                                         project_id=item.project_path, folder_kind=item.folder_kind)
                store.insert_parsed_chat(parsed)
            store.commit()
            audit = store.create_chunks(policy=policy, tokenizer_provider=tokenizer, skip_low_interest=True)
            store.commit()
            store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {"schema_version": 1, "chunks": audit["total_retrieval_chunks"],
            "duration_seconds": time.perf_counter() - started, "database_size_bytes": output.stat().st_size}


def run_benchmark(
    chunk_db: Path, output_db: Path, *, backend: str, limit: int, batch_size: int, device: str,
    model: str = "anfedoro/bge-m3-mlx-fp16", model_revision: str = "58e70901dbba8de8f3df91b5a313bcefcb151bae",
) -> dict[str, Any]:
    if output_db.exists():
        raise FileExistsError(output_db)
    shutil.copy2(chunk_db, output_db)
    if backend == "unified":
        dense, sparse = build_bge_m3_providers(
            model, model_revision=model_revision, device=device,
            batch_size=batch_size, sparse_top_k=128,
        )
    else:
        resolved = None if device == "auto" else device
        dense = SentenceTransformerDenseProvider("BAAI/bge-m3", device=resolved, effective_max_seq_length=512)
        sparse = SentenceTransformerSparseProvider(
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
            device=resolved, top_k=128,
        )
    policy_id = _policy_id(output_db)
    dense_space = f"{dense.embedding_space_id};chunk_policy={policy_id}"
    sparse_space = f"{sparse.embedding_space_id};chunk_policy={policy_id}"
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    peak_mps = 0
    with NativeBuildStore(output_db, create_schema=False) as store:
        if backend == "unified":
            joint_result, peak_rss, peak_mps = _run_joint_pass(
                store, policy_id, dense, sparse, dense_space, sparse_space,
                limit, batch_size, peak_rss, peak_mps,
            )
            store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return {
                "schema_version": 1, "backend": backend, "chunks": joint_result["processed"],
                "dense": {**joint_result, "shared_joint_pass": True},
                "sparse": {**joint_result, "shared_joint_pass": True},
                "joint": joint_result,
                "total_seconds": joint_result["seconds"],
                "peak_rss_bytes": peak_rss, "peak_mps_driver_bytes": peak_mps or None,
                "database_size_bytes": output_db.stat().st_size,
            }
        dense_result, peak_rss, peak_mps = _run_pass(
            store, policy_id, dense, "dense", dense_space, limit, batch_size, peak_rss, peak_mps,
        )
        sparse_result, peak_rss, peak_mps = _run_pass(
            store, policy_id, sparse, "sparse", sparse_space, limit, batch_size, peak_rss, peak_mps,
        )
        store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {
        "schema_version": 1, "backend": backend, "chunks": min(dense_result["processed"], sparse_result["processed"]),
        "dense": dense_result, "sparse": sparse_result,
        "total_seconds": dense_result["seconds"] + sparse_result["seconds"],
        "peak_rss_bytes": peak_rss, "peak_mps_driver_bytes": peak_mps or None,
        "database_size_bytes": output_db.stat().st_size,
    }


def _run_joint_pass(
    store: NativeBuildStore, policy_id: str, dense: Any, sparse: Any,
    dense_space: str, sparse_space: str, limit: int, batch_size: int,
    peak_rss: int, peak_mps: int,
) -> tuple[dict[str, Any], int, int]:
    started = time.perf_counter()
    processed = 0
    selected = store.conn.execute(
        "SELECT id,block_id,text,token_count FROM retrieval_chunks WHERE id IN ("
        "SELECT id FROM retrieval_chunks WHERE chunk_policy_id=? ORDER BY id LIMIT ?) "
        "ORDER BY token_count,id",
        (policy_id, limit),
    ).fetchall()
    process = psutil.Process(os.getpid())
    for start in range(0, len(selected), batch_size):
        rows = selected[start:start + batch_size]
        texts = [str(row["text"]) for row in rows]
        dense_vectors, sparse_vectors = embed_joint_documents(dense, sparse, texts)
        store.write_dense_batch(rows=rows, vectors=dense_vectors, model=dense.model_name, space=dense_space)
        store.write_sparse_batch(rows=rows, vectors=sparse_vectors, model=sparse.model_name, space=sparse_space)
        store.commit()
        processed += len(rows)
        peak_rss = max(peak_rss, process.memory_info().rss)
        peak_mps = max(peak_mps, _mps_driver_memory())
        if processed % (batch_size * 25) == 0:
            _release_accelerator_cache()
    seconds = time.perf_counter() - started
    return {
        "processed": processed,
        "seconds": seconds,
        "throughput": processed / seconds if seconds else 0.0,
        "device": dense.runtime_metadata.get("device"),
    }, peak_rss, peak_mps


def _run_pass(
    store: NativeBuildStore, policy_id: str, provider: Any, kind: str, space: str,
    limit: int, batch_size: int, peak_rss: int, peak_mps: int,
) -> tuple[dict[str, Any], int, int]:
    started = time.perf_counter()
    processed = 0
    selected = store.conn.execute(
        "SELECT id,block_id,text,token_count FROM retrieval_chunks WHERE id IN ("
        "SELECT id FROM retrieval_chunks WHERE chunk_policy_id=? ORDER BY id LIMIT ?) "
        "ORDER BY token_count,id",
        (policy_id, limit),
    ).fetchall()
    for start in range(0, len(selected), batch_size):
        rows = selected[start:start + batch_size]
        texts = [str(row["text"]) for row in rows]
        vectors = provider.embed_documents(texts)
        if kind == "dense":
            store.write_dense_batch(rows=rows, vectors=vectors, model=provider.model_name, space=space)
        else:
            store.write_sparse_batch(rows=rows, vectors=vectors, model=provider.model_name, space=space)
        store.commit()
        processed += len(rows)
        peak_rss = max(peak_rss, psutil.Process(os.getpid()).memory_info().rss)
        peak_mps = max(peak_mps, _mps_driver_memory())
        if processed % (batch_size * 25) == 0:
            _release_accelerator_cache()
    seconds = time.perf_counter() - started
    result = {"processed": processed, "seconds": seconds,
              "throughput": processed / seconds if seconds else 0.0,
              "device": provider.runtime_metadata.get("device"), "batch_size": batch_size}
    return result, peak_rss, peak_mps


def _release_accelerator_cache() -> None:
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _policy_id(path: Path) -> str:
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT chunk_policy_id FROM retrieval_chunks LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("Benchmark database has no retrieval chunks.")
    return str(row[0])


def _mps_driver_memory() -> int:
    try:
        import mlx.core as mx
        return int(mx.get_peak_memory())
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
