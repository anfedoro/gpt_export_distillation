#!/usr/bin/env python3
"""Run a local message-level benchmark against the clean native retrieval DB."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from kb.embeddings.sentence_transformer_provider import (
    SentenceTransformerDenseProvider,
    SentenceTransformerSparseProvider,
)
from kb.storage.native_pre_mvp import NativePreMvpRetriever, _chunked_space


WEIGHTS = {
    "dense_only": (1.0, 0.0),
    "sparse_only": (0.0, 1.0),
    "hybrid_065_035": (0.65, 0.35),
}


def candidate_union(dense_ids: list[str], sparse_ids: list[str]) -> set[str]:
    return set(dense_ids) | set(sparse_ids)


def aggregate(hits: list[object]) -> list[object]:
    """Keep the best supporting chunk for each source message."""
    best: dict[str, object] = {}
    for hit in hits:
        message_id = str(hit.provenance["source_message_id"])
        current = best.get(message_id)
        if current is None or (-hit.final_score, hit.chunk_id) < (-current.final_score, current.chunk_id):
            best[message_id] = hit
    return sorted(best.values(), key=lambda hit: (-hit.final_score, str(hit.provenance["source_message_id"]), hit.chunk_id))


def metrics(records: list[dict[str, object]]) -> dict[str, float | int]:
    ranks = [record["rank"] for record in records]
    assert all(isinstance(rank, int) for rank in ranks)
    return {
        **{f"recall_at_{k}": sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 5, 10, 20)},
        "mrr": sum(1.0 / rank if rank <= 1000 else 0.0 for rank in ranks) / len(ranks),
        "median_rank": float(statistics.median(ranks)),
        "miss_at_20": sum(rank > 20 for rank in ranks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--probe-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-pools", default="500,1000,2000,5000")
    args = parser.parse_args()
    probes = json.loads(args.probe_file.read_text(encoding="utf-8"))["probes"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dense = SentenceTransformerDenseProvider("BAAI/bge-m3", device="mps", torch_dtype="float16")
    sparse = SentenceTransformerSparseProvider(
        "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        device="mps", torch_dtype="float16", top_k=128,
    )
    started = time.perf_counter()
    pools = tuple(sorted({int(value) for value in args.candidate_pools.split(",")}))
    if not pools or any(value <= 0 for value in pools):
        raise ValueError("candidate pools must be positive")
    max_pool = max(pools)
    records: dict[str, list[dict[str, object]]] = {f"{variant}@{pool}": [] for pool in pools for variant in WEIGHTS}
    snapshot = args.output_dir / "raw_scores.jsonl"
    with NativePreMvpRetriever(args.db) as retriever, snapshot.open("w", encoding="utf-8") as handle:
        policy_id = str(retriever.conn.execute("SELECT chunk_policy_id FROM retrieval_chunks LIMIT 1").fetchone()[0])
        if retriever.dense_space != _chunked_space(dense.embedding_space_id, policy_id):
            raise RuntimeError("Dense provider is incompatible with DB embedding space.")
        if retriever.sparse.embedding_space_id != _chunked_space(sparse.embedding_space_id, policy_id):
            raise RuntimeError("Sparse provider is incompatible with DB embedding space.")
        for number, probe in enumerate(probes, 1):
            query_dense = dense.embed_query(probe["query"])
            query_sparse = sparse.embed_query(probe["query"])
            dense_hits = retriever.dense.search(query_dense, limit=max_pool,
                                                model_name=str(retriever.dense_model), embedding_space_id=str(retriever.dense_space))
            sparse_scores = retriever.sparse.score(query_sparse)
            sparse_order = np.lexsort((retriever.sparse.chunk_ids, -sparse_scores))[:max_pool]
            dense_ids = [hit.chunk_id for hit in dense_hits]
            sparse_ids = [str(retriever.sparse.chunk_ids[index]) for index in sparse_order if sparse_scores[index] > 0]
            raw_candidate_ids = candidate_union(dense_ids, sparse_ids)
            dense_by_id = retriever.dense.scores_for_chunk_ids(query_dense, raw_candidate_ids,
                                                                 model_name=str(retriever.dense_model), embedding_space_id=str(retriever.dense_space))
            sparse_by_id = {str(retriever.sparse.chunk_ids[index]): float(sparse_scores[index]) for index in sparse_order if sparse_scores[index] > 0}
            for pool in pools:
                candidate_ids = candidate_union(dense_ids[:pool], sparse_ids[:pool])
                for variant, (alpha, beta) in WEIGHTS.items():
                    key = f"{variant}@{pool}"
                    ordered = sorted(candidate_ids, key=lambda chunk_id: (-(alpha * dense_by_id.get(chunk_id, 0.0) + beta * sparse_by_id.get(chunk_id, 0.0)), chunk_id))
                    provenance = retriever._provenance(ordered)
                    hits = [type("Hit", (), {"chunk_id": chunk_id, "dense_score": dense_by_id.get(chunk_id, 0.0), "sparse_score": sparse_by_id.get(chunk_id, 0.0), "final_score": alpha * dense_by_id.get(chunk_id, 0.0) + beta * sparse_by_id.get(chunk_id, 0.0), "provenance": provenance[chunk_id]}) for chunk_id in ordered]
                    messages = aggregate(hits)
                    expected = probe["expected_message_id"]
                    rank = next((index for index, hit in enumerate(messages, 1)
                                 if hit.provenance["source_message_id"] == expected), max_pool + 1)
                    records[key].append({"probe_id": probe["probe_id"], "category": probe["category"], "rank": rank})
                    for index, hit in enumerate(hits, 1):
                        handle.write(json.dumps({"probe_id": probe["probe_id"], "pool": pool, "variant": variant, "rank": index,
                                               "chunk_id": hit.chunk_id, "message_id": hit.provenance["source_message_id"],
                                               "conversation_id": hit.provenance["dialogue_id"], "dense_score": hit.dense_score,
                                                   "sparse_score": hit.sparse_score, "final_score": hit.final_score}, sort_keys=True) + "\n")
            if number % 10 == 0:
                print(f"[native-benchmark] completed_probes={number}/{len(probes)}", flush=True)
    report = {"status": "completed", "probe_count": len(probes), "candidate_pools": pools,
              "timing_seconds": time.perf_counter() - started,
              "categories": dict(sorted(Counter(probe["category"] for probe in probes).items())),
              "variants": {name: metrics(rows) for name, rows in records.items()}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Native Message Retrieval Benchmark", "", f"Probes: {len(probes)}", "", "| Variant / pool | R@1 | R@5 | R@10 | R@20 | MRR | Median rank | Miss@20 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["variants"].items():
        lines.append(f"| {name} | {value['recall_at_1']:.3f} | {value['recall_at_5']:.3f} | {value['recall_at_10']:.3f} | {value['recall_at_20']:.3f} | {value['mrr']:.3f} | {value['median_rank']:.1f} | {value['miss_at_20']} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
