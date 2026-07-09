"""Private snapshot-based dense/sparse fusion diagnostics.

The ``snapshot`` subcommand is the only mode allowed to load embedding models.
The ``evaluate`` subcommand is deliberately file-only: it reads a raw-score
snapshot and probe manifest, never opens the SQLite database or providers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from kb.benchmark import DirectRetrievalSession
from kb.canary.real_data import PROBES_SCHEMA, RealProbe, _source_identity_maps
from kb.cli import _build_dense_provider, _build_sparse_provider
from kb.index.chunk_builder import ChunkPolicy
from kb.storage.sqlite_store import SQLiteStore


SNAPSHOT_SCHEMA = "kb.fusion.raw_scores.v1"
SNAPSHOT_MANIFEST_SCHEMA = "kb.fusion.snapshot_manifest.v1"
EVALUATION_SCHEMA = "kb.fusion.evaluation.v1"
DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
DEFAULT_SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
CURRENT_WEIGHTS = (0.65, 0.35)
WEIGHTED_VARIANTS = ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5))


@dataclass(frozen=True)
class SnapshotRow:
    probe_id: str
    chunk_id: str
    block_id: str
    chunk_ordinal: int | None
    source_message_id: str | None
    source_conversation_id: str | None
    role: str | None
    block_type: str | None
    dense_score: float
    sparse_score: float
    dense_rank: int
    sparse_rank: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or evaluate a private dense/sparse raw-score snapshot.")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Load models once and save query-to-chunk raw scores.")
    snapshot.add_argument("--db", required=True)
    snapshot.add_argument("--probe-file", required=True)
    snapshot.add_argument("--output-dir", required=True)
    snapshot.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    snapshot.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    snapshot.add_argument("--dense-device", default="mps")
    snapshot.add_argument("--sparse-device", default="mps")
    snapshot.add_argument("--dense-torch-dtype", default="float16")
    snapshot.add_argument("--sparse-torch-dtype", default="float16")
    snapshot.add_argument("--sparse-top-k", type=int, default=128)
    evaluate = commands.add_parser("evaluate", help="Evaluate a snapshot without models or database access.")
    evaluate.add_argument("--snapshot", required=True)
    evaluate.add_argument("--probe-file", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--dense-candidate-k", default="20,50,100")
    evaluate.add_argument("--sparse-candidate-k", default="20,50,100")
    evaluate.add_argument("--rrf-k", type=int, default=60)
    evaluate.add_argument("--message-aggregation", choices=["max"], default="max")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "snapshot":
        report = create_raw_score_snapshot(
            db_path=Path(args.db).expanduser(),
            probe_path=Path(args.probe_file).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            dense_device=args.dense_device,
            sparse_device=args.sparse_device,
            dense_torch_dtype=args.dense_torch_dtype,
            sparse_torch_dtype=args.sparse_torch_dtype,
            sparse_top_k=args.sparse_top_k,
        )
    else:
        report = evaluate_raw_score_snapshot(
            snapshot_path=Path(args.snapshot).expanduser(),
            probe_path=Path(args.probe_file).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            candidate_ks=_parse_ks(args.dense_candidate_k, args.sparse_candidate_k),
            rrf_k=args.rrf_k,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "completed" else 1


def create_raw_score_snapshot(
    *,
    db_path: Path,
    probe_path: Path,
    output_dir: Path,
    dense_model: str = DEFAULT_DENSE_MODEL,
    sparse_model: str = DEFAULT_SPARSE_MODEL,
    dense_device: str | None = "mps",
    sparse_device: str | None = "mps",
    dense_torch_dtype: str = "float16",
    sparse_torch_dtype: str = "float16",
    sparse_top_k: int = 128,
) -> dict[str, Any]:
    """Create a private raw-score snapshot without mutating the DB."""
    probes = _load_probes(probe_path)
    policy = _load_chunk_policy(db_path)
    started = time.perf_counter()
    dense = _build_dense_provider("sentence-transformers", dense_model, device=dense_device, torch_dtype=dense_torch_dtype)
    sparse = _build_sparse_provider("sentence-transformers", sparse_model, sparse_top_k, device=sparse_device, torch_dtype=sparse_torch_dtype)
    providers_loaded_s = time.perf_counter() - started
    session = DirectRetrievalSession(db_path=db_path, dense_provider=dense, sparse_provider=sparse, chunk_policy=policy)
    source_messages, source_conversations = _source_identity_maps(db_path)
    source_texts = _source_message_texts(db_path)
    metadata = _chunk_metadata(db_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "raw_scores.jsonl"
    rows_written = 0
    query_timing: dict[str, dict[str, float]] = {}
    with snapshot_path.open("w", encoding="utf-8") as handle:
        for probe in probes:
            scores = session.score_query(probe.query)
            dense_ranks = _ranks(scores.dense_scores, [block.chunk_id for block in session.blocks])
            sparse_ranks = _ranks(scores.sparse_scores, [block.chunk_id for block in session.blocks])
            query_timing[probe.probe_id] = {"query_encoding_ms": scores.query_encoding_ms, "base_scoring_ms": scores.base_scoring_ms}
            for index, block in enumerate(session.blocks):
                extra = metadata[block.chunk_id]
                handle.write(json.dumps({
                    "schema_version": SNAPSHOT_SCHEMA,
                    "probe_id": probe.probe_id,
                    "chunk_id": block.chunk_id,
                    "block_id": block.block_id,
                    "chunk_ordinal": extra["chunk_ordinal"],
                    "source_message_id": source_messages.get(block.message_id),
                    "source_conversation_id": source_conversations.get(block.conversation_id),
                    "role": block.role,
                    "block_type": block.block_type,
                    "dense_score": float(scores.dense_scores[index]),
                    "sparse_score": float(scores.sparse_scores[index]),
                    "dense_rank": dense_ranks[block.chunk_id],
                    "sparse_rank": sparse_ranks[block.chunk_id],
                }, ensure_ascii=False, sort_keys=True) + "\n")
                rows_written += 1
    manifest = {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA,
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "db_path": str(db_path),
        "probe_path": str(probe_path),
        "probe_count": len(probes),
        "chunk_count": session.candidate_blocks,
        "raw_score_rows": rows_written,
        "chunk_policy_id": policy.id,
        "providers": {"dense": dense.contract_dict(), "sparse": sparse.contract_dict()},
        "devices": {"dense": dense_device, "sparse": sparse_device},
        "dtypes": {"dense": dense_torch_dtype, "sparse": sparse_torch_dtype},
        "timing_ms": {"providers_load": providers_loaded_s * 1000, "corpus_load": session.corpus_load_ms, "queries": query_timing, "total": (time.perf_counter() - started) * 1000},
        "lexical_overlap": [_lexical_overlap(probe, source_texts.get(probe.expected_message_id, "")) for probe in probes],
    }
    manifest_path = output_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"status": "completed", "snapshot": str(snapshot_path), "manifest": str(manifest_path), "probe_count": len(probes), "chunk_count": session.candidate_blocks, "rows": rows_written}


def evaluate_raw_score_snapshot(
    *,
    snapshot_path: Path,
    probe_path: Path,
    output_dir: Path,
    candidate_ks: tuple[int, ...] = (20, 50, 100),
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Evaluate variants using only local JSON files; no DB or providers."""
    probes = _load_probes(probe_path)
    rows = _load_snapshot(snapshot_path)
    grouped = _group_snapshot(rows, probes)
    snapshot_manifest = _load_snapshot_manifest(snapshot_path)
    started = time.perf_counter()
    variants = _build_variants(candidate_ks, rrf_k)
    all_results: list[dict[str, Any]] = []
    per_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in variants:
        records = [_evaluate_probe(probe, grouped[probe.probe_id], variant) for probe in probes]
        metrics = _metrics(records)
        all_results.append({"id": variant["id"], "description": variant["description"], "metrics": metrics, "latency_ms": _elapsed_variant_latency(records)})
        per_variant[variant["id"]] = {"records": records}
    baseline = next(item for item in all_results if item["id"] == "sparse_only")
    eligible = [item for item in all_results if _meets_sparse_guard(item["metrics"], baseline["metrics"])]
    best = max(eligible, key=lambda item: (item["metrics"]["message_recall_at"]["20"], item["metrics"]["message_recall_at"]["10"], item["metrics"]["message_mrr"], item["id"])) if eligible else baseline
    current_id = "current_weighted_065_035"
    diagnostics = _miss_diagnostics(
        endpoint_records={key: per_variant[key]["records"] for key in ("dense_only", "sparse_only", current_id)},
        grouped=grouped,
        candidate_ks=candidate_ks,
    )
    hybrid_improved = best["id"] != "sparse_only"
    verdict = "PASS" if hybrid_improved and _meets_sparse_guard(best["metrics"], baseline["metrics"]) else "PASS WITH WARNINGS"
    payload = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot": str(snapshot_path),
        "probe_count": len(probes),
        "chunk_count": len(next(iter(grouped.values()))),
        "current_fusion": {"formula": "0.65 * dense_cosine + 0.35 * sparse_cosine", "normalization": "none", "candidate_pool": "all chunks", "missing_score": "zero", "tie_break": "chunk_id ascending"},
        "candidate_pool_analysis": _candidate_pool_analysis(probes, grouped, candidate_ks),
        "lexical_overlap": snapshot_manifest.get("lexical_overlap", []),
        "variants": all_results,
        "best_variant": best,
        "baseline_sparse_only": baseline,
        "miss_diagnostics": diagnostics,
        "rescue_analysis": _rescue_analysis(
            dense_records=per_variant["dense_only"]["records"],
            sparse_records=per_variant["sparse_only"]["records"],
            rrf_records=per_variant["rrf_k60_union_20"]["records"],
        ),
        "verdict": verdict,
        "recommendation": _recommendation(best, baseline),
        "timing_ms": {"evaluation": (time.perf_counter() - started) * 1000},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(_markdown_report(payload), encoding="utf-8")
    payload["report_json"] = str(report_json)
    payload["report_md"] = str(report_md)
    return payload


def _load_probes(path: Path) -> list[RealProbe]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROBES_SCHEMA:
        raise ValueError("Unsupported probe manifest schema.")
    probes = []
    for item in payload["probes"]:
        normalized = dict(item)
        normalized.setdefault("probe_type", normalized.get("transformation_type", "unspecified"))
        probes.append(RealProbe(**normalized))
    return probes


def _load_chunk_policy(db_path: Path) -> ChunkPolicy:
    with SQLiteStore(db_path, read_only=True) as store:
        row = store.conn.execute("SELECT chunk_policy_id FROM retrieval_chunks GROUP BY chunk_policy_id HAVING COUNT(*) > 0 ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
    if not row:
        raise ValueError("DB has no retrieval chunks.")
    policy_id = str(row["chunk_policy_id"])
    matched = re.fullmatch(r"canonical_token_chunks:(v[12]);limit=(\d+);content=(\d+);fallback_overlap=(\d+);reserve=(\d+)", policy_id)
    if not matched:
        raise ValueError(f"Unsupported chunk policy identity: {policy_id}")
    version, limit, content, overlap, reserve = matched.groups()
    return ChunkPolicy(policy_id, int(limit), int(content), int(overlap), int(reserve), version)


def _chunk_metadata(db_path: Path) -> dict[str, dict[str, Any]]:
    with SQLiteStore(db_path, read_only=True) as store:
        rows = store.conn.execute("SELECT id, ordinal FROM retrieval_chunks").fetchall()
    return {str(row["id"]): {"chunk_ordinal": int(row["ordinal"])} for row in rows}


def _source_message_texts(db_path: Path) -> dict[str, str]:
    with SQLiteStore(db_path, read_only=True) as store:
        rows = store.conn.execute("SELECT message_id, raw_text FROM messages WHERE message_id IS NOT NULL").fetchall()
    return {str(row["message_id"]): str(row["raw_text"]) for row in rows}


def _load_snapshot_manifest(snapshot_path: Path) -> dict[str, Any]:
    path = snapshot_path.with_name("snapshot_manifest.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA:
        raise ValueError("Unsupported snapshot manifest schema.")
    return payload


def _lexical_overlap(probe: RealProbe, source_text: str) -> dict[str, Any]:
    query_terms = _normalized_terms(probe.query)
    source_terms = _normalized_terms(source_text)
    shared = sorted(query_terms & source_terms)
    union = query_terms | source_terms
    return {
        "probe_id": probe.probe_id,
        "shared_normalized_terms": shared[:20],
        "query_term_count": len(query_terms),
        "source_term_count": len(source_terms),
        "term_jaccard": len(shared) / len(union) if union else 0.0,
        "query_term_coverage": len(shared) / len(query_terms) if query_terms else 0.0,
        "suspiciously_high": len(query_terms) >= 3 and len(shared) / len(query_terms) >= 0.60,
    }


def _normalized_terms(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[^\W_]{2,}", text, flags=re.UNICODE)}


def _ranks(scores: Any, chunk_ids: list[str]) -> dict[str, int]:
    ordered = sorted(range(len(chunk_ids)), key=lambda idx: (-float(scores[idx]), chunk_ids[idx]))
    return {chunk_ids[index]: rank for rank, index in enumerate(ordered, start=1)}


def _load_snapshot(path: Path) -> list[SnapshotRow]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("Unsupported raw score snapshot schema.")
        rows.append(SnapshotRow(**{key: payload[key] for key in SnapshotRow.__dataclass_fields__}))
    return rows


def _group_snapshot(rows: list[SnapshotRow], probes: list[RealProbe]) -> dict[str, list[SnapshotRow]]:
    grouped: dict[str, list[SnapshotRow]] = defaultdict(list)
    for row in rows:
        grouped[row.probe_id].append(row)
    expected_ids = {probe.probe_id for probe in probes}
    if set(grouped) != expected_ids:
        raise ValueError("Snapshot probe IDs do not match probe manifest.")
    counts = {len(value) for value in grouped.values()}
    if len(counts) != 1 or not counts or 0 in counts:
        raise ValueError("Snapshot must contain the same non-zero chunk count for every probe.")
    return dict(grouped)


def _build_variants(candidate_ks: tuple[int, ...], rrf_k: int) -> list[dict[str, Any]]:
    variants = [
        {"id": "dense_only", "kind": "raw", "dense_weight": 1.0, "sparse_weight": 0.0, "pool": None, "description": "Dense cosine baseline over all chunks."},
        {"id": "sparse_only", "kind": "raw", "dense_weight": 0.0, "sparse_weight": 1.0, "pool": None, "description": "Sparse cosine baseline over all chunks."},
        {"id": "current_weighted_065_035", "kind": "raw", "dense_weight": 0.65, "sparse_weight": 0.35, "pool": None, "description": "Current unnormalized weighted sum over all chunks."},
    ]
    variants.extend({"id": f"normalized_weighted_{int(d*100):03d}_{int(s*100):03d}", "kind": "normalized", "dense_weight": d, "sparse_weight": s, "pool": None, "description": "Per-query max-normalized weighted sum over all chunks."} for d, s in WEIGHTED_VARIANTS)
    for pool in candidate_ks:
        variants.append({"id": f"rrf_k{rrf_k}_union_{pool}", "kind": "rrf", "rrf_k": rrf_k, "pool": pool, "description": f"RRF(k={rrf_k}) over union of dense/sparse top-{pool}."})
        variants.append({"id": f"sparse_first_bonus_union_{pool}", "kind": "sparse_first", "pool": pool, "description": f"Sparse top-{pool} reranked with a 0.10 dense max-normalized bonus."})
        variants.append({"id": f"normalized_union_{pool}", "kind": "normalized", "dense_weight": 0.2, "sparse_weight": 0.8, "pool": pool, "description": f"0.2/0.8 normalized weighted sum over dense/sparse top-{pool} union."})
    return variants


def _evaluate_probe(probe: RealProbe, rows: list[SnapshotRow], variant: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    scored = _score_variant(rows, variant)
    chunk_order = sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))
    chunk_ranks = {row.chunk_id: rank for rank, (row, _) in enumerate(chunk_order, start=1)}
    messages: dict[str, tuple[SnapshotRow, float]] = {}
    for row, score in chunk_order:
        if row.source_message_id is None:
            continue
        current = messages.get(row.source_message_id)
        if current is None or score > current[1] or (score == current[1] and row.chunk_id < current[0].chunk_id):
            messages[row.source_message_id] = (row, score)
    message_order = sorted(messages.values(), key=lambda item: (-item[1], item[0].chunk_id))
    message_ranks = {row.source_message_id: rank for rank, (row, _) in enumerate(message_order, start=1)}
    conversations: dict[str, tuple[SnapshotRow, float]] = {}
    for row, score in chunk_order:
        if row.source_conversation_id is None:
            continue
        current = conversations.get(row.source_conversation_id)
        if current is None or score > current[1] or (score == current[1] and row.chunk_id < current[0].chunk_id):
            conversations[row.source_conversation_id] = (row, score)
    conversation_order = sorted(conversations.values(), key=lambda item: (-item[1], item[0].chunk_id))
    conversation_ranks = {row.source_conversation_id: rank for rank, (row, _) in enumerate(conversation_order, start=1)}
    expected_rows = [row for row in rows if row.source_message_id == probe.expected_message_id]
    expected_chunk_rank = min((chunk_ranks.get(row.chunk_id) for row in expected_rows if row.chunk_id in chunk_ranks), default=None)
    expected_message_rank = message_ranks.get(probe.expected_message_id)
    expected_conversation_rank = conversation_ranks.get(probe.expected_conversation_id)
    return {"probe_id": probe.probe_id, "probe_type": probe.probe_type, "transformation_type": probe.transformation_type, "query_language": probe.query_language, "source_language": probe.source_language, "expected_message_id": probe.expected_message_id, "expected_conversation_id": probe.expected_conversation_id, "expected_chunk_ids": [row.chunk_id for row in expected_rows], "chunk_rank": expected_chunk_rank, "message_rank": expected_message_rank, "conversation_rank": expected_conversation_rank, "supporting_chunk_id": messages.get(probe.expected_message_id, (None, 0.0))[0].chunk_id if probe.expected_message_id in messages else None, "top_chunks": [_row_for_report(row, score, chunk_ranks[row.chunk_id]) for row, score in chunk_order[:20]], "latency_ms": (time.perf_counter() - started) * 1000}


def _score_variant(rows: list[SnapshotRow], variant: dict[str, Any]) -> list[tuple[SnapshotRow, float]]:
    dense_top = {row.chunk_id for row in sorted(rows, key=lambda row: (row.dense_rank, row.chunk_id))[:variant.get("pool") or len(rows)]}
    sparse_top = {row.chunk_id for row in sorted(rows, key=lambda row: (row.sparse_rank, row.chunk_id))[:variant.get("pool") or len(rows)]}
    kind = variant["kind"]
    if kind == "raw":
        candidates = rows
        return [(row, variant["dense_weight"] * row.dense_score + variant["sparse_weight"] * row.sparse_score) for row in candidates]
    if kind == "rrf":
        candidates = [row for row in rows if row.chunk_id in dense_top | sparse_top]
        k = variant["rrf_k"]
        return [(row, (1 / (k + row.dense_rank) if row.chunk_id in dense_top else 0.0) + (1 / (k + row.sparse_rank) if row.chunk_id in sparse_top else 0.0)) for row in candidates]
    dense_max = max((row.dense_score for row in rows), default=0.0) or 1.0
    sparse_max = max((row.sparse_score for row in rows), default=0.0) or 1.0
    if kind == "sparse_first":
        return [(row, row.sparse_score / sparse_max + 0.10 * row.dense_score / dense_max) for row in rows if row.chunk_id in sparse_top]
    candidates = [row for row in rows if variant.get("pool") is None or row.chunk_id in dense_top | sparse_top]
    return [(row, variant["dense_weight"] * row.dense_score / dense_max + variant["sparse_weight"] * row.sparse_score / sparse_max) for row in candidates]


def _row_for_report(row: SnapshotRow, score: float, rank: int) -> dict[str, Any]:
    return {"rank": rank, "chunk_id": row.chunk_id, "message_id": row.source_message_id, "conversation_id": row.source_conversation_id, "role": row.role, "block_type": row.block_type, "chunk_ordinal": row.chunk_ordinal, "dense_score": row.dense_score, "sparse_score": row.sparse_score, "final_score": score}


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    def recall(field: str, k: int) -> float:
        return sum(1 for row in records if row[field] is not None and row[field] <= k) / len(records) if records else 0.0
    def mrr(field: str) -> float:
        return sum(1.0 / row[field] for row in records if row[field]) / len(records) if records else 0.0
    return {"chunk_recall_at": {str(k): recall("chunk_rank", k) for k in (1, 5, 10, 20)}, "message_recall_at": {str(k): recall("message_rank", k) for k in (1, 5, 10, 20)}, "conversation_recall_at": {str(k): recall("conversation_rank", k) for k in (10, 20)}, "message_mrr": mrr("message_rank"), "breakdown": _breakdown(records)}


def _breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["transformation_type"]].append(row)
        groups[f"query_language:{row['query_language']}"] .append(row)
        if row["source_language"] in {"ru", "en"} and row["query_language"] in {"ru", "en"}:
            groups[f"{row['source_language'].upper()}->{row['query_language'].upper()}"] .append(row)
    return {key: {"count": len(value), "message_recall_at_10": sum(1 for row in value if row["message_rank"] and row["message_rank"] <= 10) / len(value), "message_mrr": sum(1 / row["message_rank"] for row in value if row["message_rank"]) / len(value)} for key, value in sorted(groups.items())}


def _miss_diagnostics(
    *,
    endpoint_records: dict[str, list[dict[str, Any]]],
    grouped: dict[str, list[SnapshotRow]],
    candidate_ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    diagnostics = []
    current_id = "current_weighted_065_035"
    current_records = {row["probe_id"]: row for row in endpoint_records[current_id]}
    by_mode = {mode: {row["probe_id"]: row for row in records} for mode, records in endpoint_records.items()}
    for probe_id, record in current_records.items():
        if all((records[probe_id]["message_rank"] or math.inf) <= 20 for records in by_mode.values()):
            continue
        rows = grouped[probe_id]
        expected = [row for row in rows if row.source_message_id == record["expected_message_id"]]
        dense_rank = min((row.dense_rank for row in expected), default=None)
        sparse_rank = min((row.sparse_rank for row in expected), default=None)
        dense_max = max((row.dense_score for row in rows), default=0.0) or 1.0
        sparse_max = max((row.sparse_score for row in rows), default=0.0) or 1.0
        representative = max(expected, key=lambda row: 0.65 * row.dense_score + 0.35 * row.sparse_score, default=None)
        in_union = {str(k): bool((dense_rank and dense_rank <= k) or (sparse_rank and sparse_rank <= k)) for k in candidate_ks}
        same_dialogue = any(item["conversation_id"] == record["expected_conversation_id"] and item["message_id"] != record["expected_message_id"] for item in record["top_chunks"])
        classification = "retrieval_miss" if not in_union.get(str(max(candidate_ks)), False) else "fusion_or_message_ranking_miss"
        if same_dialogue:
            classification = "dialogue_level_near_miss"
        diagnostics.append({
            "probe_id": probe_id,
            "expected_message_id": record["expected_message_id"],
            "expected_conversation_id": record["expected_conversation_id"],
            "expected_role": representative.role if representative else None,
            "expected_block_type": representative.block_type if representative else None,
            "expected_chunk_ids": [{"chunk_id": row.chunk_id, "ordinal": row.chunk_ordinal} for row in expected],
            "endpoint_message_ranks": {mode: records[probe_id]["message_rank"] for mode, records in by_mode.items()},
            "expected_dense_rank": dense_rank,
            "expected_sparse_rank": sparse_rank,
            "expected_scores": None if representative is None else {
                "dense_raw": representative.dense_score,
                "sparse_raw": representative.sparse_score,
                "dense_normalized": representative.dense_score / dense_max,
                "sparse_normalized": representative.sparse_score / sparse_max,
                "current_hybrid": 0.65 * representative.dense_score + 0.35 * representative.sparse_score,
            },
            "candidate_pool_inclusion": in_union,
            "classification": classification,
            "evaluation_mismatch": bool(record["chunk_rank"] and record["chunk_rank"] <= 20 and (record["message_rank"] or math.inf) > 20),
            "ambiguous_probe": "requires private human content review",
            "results_above_expected": record["top_chunks"],
        })
    return diagnostics


def _candidate_pool_analysis(probes: list[RealProbe], grouped: dict[str, list[SnapshotRow]], candidate_ks: tuple[int, ...]) -> list[dict[str, Any]]:
    output = []
    for probe in probes:
        expected = [row for row in grouped[probe.probe_id] if row.source_message_id == probe.expected_message_id]
        dense_rank = min((row.dense_rank for row in expected), default=None)
        sparse_rank = min((row.sparse_rank for row in expected), default=None)
        output.append({"probe_id": probe.probe_id, "earliest_dense_rank": dense_rank, "earliest_sparse_rank": sparse_rank, "in_dense_pool": {str(k): bool(dense_rank and dense_rank <= k) for k in candidate_ks}, "in_sparse_pool": {str(k): bool(sparse_rank and sparse_rank <= k) for k in candidate_ks}})
    return output


def _rescue_analysis(
    *,
    dense_records: list[dict[str, Any]],
    sparse_records: list[dict[str, Any]],
    rrf_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dense = {row["probe_id"]: row for row in dense_records}
    sparse = {row["probe_id"]: row for row in sparse_records}
    rrf = {row["probe_id"]: row for row in rrf_records}
    dense_rescues = []
    sparse_rescues = []
    rrf_rescues = []
    dense_wins = sparse_wins = both_hit = both_miss = 0
    for probe_id, dense_row in dense.items():
        sparse_row, rrf_row = sparse[probe_id], rrf[probe_id]
        dense_rank, sparse_rank, rrf_rank = dense_row["message_rank"], sparse_row["message_rank"], rrf_row["message_rank"]
        record = {"probe_id": probe_id, "transformation_type": dense_row["transformation_type"], "dense_rank": dense_rank, "sparse_rank": sparse_rank, "rrf_rank": rrf_rank}
        if dense_rank and dense_rank <= 20 and (not sparse_rank or sparse_rank > 20):
            dense_rescues.append(record)
        if sparse_rank and sparse_rank <= 20 and (not dense_rank or dense_rank > 20):
            sparse_rescues.append(record)
        if (not dense_rank or dense_rank > 10) and (not sparse_rank or sparse_rank > 10) and rrf_rank and rrf_rank <= 10:
            rrf_rescues.append(record)
        if dense_rank and dense_rank <= 10 and (not sparse_rank or sparse_rank > 20):
            dense_wins += 1
        elif sparse_rank and sparse_rank <= 10 and (not dense_rank or dense_rank > 20):
            sparse_wins += 1
        elif dense_rank and dense_rank <= 10 and sparse_rank and sparse_rank <= 10:
            both_hit += 1
        elif (not dense_rank or dense_rank > 20) and (not sparse_rank or sparse_rank > 20):
            both_miss += 1
    return {"dense_rescues": dense_rescues, "sparse_rescues": sparse_rescues, "rrf_rescues": rrf_rescues, "counts": {"dense_only_wins": dense_wins, "sparse_only_wins": sparse_wins, "both_hit_at_10": both_hit, "both_miss_at_20": both_miss}}


def _meets_sparse_guard(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return metrics["message_recall_at"]["10"] >= baseline["message_recall_at"]["10"] and metrics["message_recall_at"]["20"] >= baseline["message_recall_at"]["20"] and metrics["message_mrr"] >= baseline["message_mrr"] - 0.01


def _elapsed_variant_latency(records: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(row["latency_ms"] for row in records)
    return {"p50": _percentile(values, 50), "p95": _percentile(values, 95)}


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    return values[max(0, min(len(values) - 1, math.ceil(len(values) * percentile / 100) - 1))]


def _parse_ks(*values: str) -> tuple[int, ...]:
    parsed = {int(item.strip()) for value in values for item in value.split(",") if item.strip()}
    if not parsed or min(parsed) <= 0:
        raise ValueError("Candidate pool sizes must be positive.")
    return tuple(sorted(parsed))


def _recommendation(best: dict[str, Any], baseline: dict[str, Any]) -> str:
    if best["id"] == "sparse_only":
        return "Sparse-only is the primary retrieval mode; dense remains an optional secondary signal."
    return f"Use {best['id']} with message-level max aggregation; it meets the sparse-only guard on this probe set."


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Dense/Sparse Fusion Evaluation", "", f"Verdict: **{report['verdict']}**", "", "## Variants", "", "| Variant | Message R@10 | Message R@20 | Message MRR | Chunk R@20 | Conv R@20 | p50 ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in report["variants"]:
        metric, latency = item["metrics"], item["latency_ms"]
        lines.append(f"| {item['id']} | {metric['message_recall_at']['10']:.3f} | {metric['message_recall_at']['20']:.3f} | {metric['message_mrr']:.3f} | {metric['chunk_recall_at']['20']:.3f} | {metric['conversation_recall_at']['20']:.3f} | {latency['p50']:.3f} |")
    lexical = report.get("lexical_overlap", [])
    if lexical:
        suspicious = [item["probe_id"] for item in lexical if item.get("suspiciously_high")]
        lines.extend(["", "## Lexical-overlap Diagnostics", "", f"- probes: {len(lexical)}", f"- suspiciously high query-term overlap: {', '.join(suspicious) if suspicious else 'none'}", "- query and source text are intentionally omitted from this local report."])
    rescues = report.get("rescue_analysis", {})
    if rescues:
        counts = rescues.get("counts", {})
        lines.extend(["", "## Branch Contribution", "", f"- dense rescues@20: {len(rescues.get('dense_rescues', []))}", f"- sparse rescues@20: {len(rescues.get('sparse_rescues', []))}", f"- RRF rescues@10: {len(rescues.get('rrf_rescues', []))}", f"- dense-only wins@10: {counts.get('dense_only_wins', 0)}", f"- sparse-only wins@10: {counts.get('sparse_only_wins', 0)}"])
    key_variants = [item for item in report["variants"] if item["id"] in {"dense_only", "sparse_only", "current_weighted_065_035", "rrf_k60_union_20"}]
    categories = sorted({key for item in key_variants for key in item["metrics"]["breakdown"] if not key.startswith("query_language:")})
    if categories:
        lines.extend(["", "## Breakdown: Message R@10", "", "| Category | " + " | ".join(item["id"] for item in key_variants) + " |", "|---|" + "|".join("---:" for _ in key_variants) + "|"])
        for category in categories:
            values = [item["metrics"]["breakdown"].get(category, {}).get("message_recall_at_10", 0.0) for item in key_variants]
            lines.append("| " + category + " | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend(["", "## Recommendation", "", report["recommendation"], "", "## Miss Diagnostics", ""])
    for miss in report["miss_diagnostics"]:
        lines.append(f"- `{miss['probe_id']}`: dense rank {miss['expected_dense_rank']}, sparse rank {miss['expected_sparse_rank']}, classification `{miss['classification']}`. Content relevance requires private human review.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
