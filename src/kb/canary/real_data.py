"""Private, local real-corpus preflight for chunked multilingual retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kb.benchmark import DirectRetrievalSession, RankingConfig
from kb.canary.multilingual_dense import PROBE_MODES, _extract_audit, _percentile
from kb.cli import _build_dense_provider, _build_sparse_provider, _chunk_policy_from_audit, embed_knowledge_blocks, ingest_chats
from kb.ingest.chat_md_parser import parse_chat_file
from kb.ingest.tree_walker import scan_tree
from kb.storage.sqlite_store import SQLiteStore


MANIFEST_SCHEMA = "kb.real_data_preflight.manifest.v1"
PROBES_SCHEMA = "kb.real_data_preflight.probes.v1"
REPORT_SCHEMA = "kb.real_data_preflight.v1"
DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
DEFAULT_SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"


@dataclass(frozen=True)
class SourceDocument:
    relative_path: str
    conversation_id: str
    selection_reasons: list[str]


@dataclass(frozen=True)
class RealProbe:
    probe_id: str
    query: str
    query_language: str
    expected_conversation_id: str
    expected_message_id: str
    probe_type: str
    expected_role: str | None = None
    expected_block_type: str | None = None
    notes: str | None = None
    transformation_type: str = "unspecified"
    source_language: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated private real-data BGE-M3 preflight.")
    parser.add_argument("--input", required=True, help="Distilled ChatGPT export root; read-only.")
    parser.add_argument("--manifest", required=True, help="Private local source-document manifest.")
    parser.add_argument("--work-dir", required=True, help="Private local working directory.")
    parser.add_argument("--output-db", required=True, help="Dedicated preflight SQLite DB, never the legacy DB.")
    parser.add_argument("--output-report")
    parser.add_argument("--probe-file", help="Private local probe manifest. Generated when absent.")
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--dense-device", default="mps")
    parser.add_argument("--dense-torch-dtype", default="float16")
    parser.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--sparse-device", default="mps")
    parser.add_argument("--sparse-torch-dtype", default="float16")
    parser.add_argument("--sparse-top-k", type=int, default=128)
    parser.add_argument("--chunk-policy", choices=["canonical_token_chunks:v2"], default="canonical_token_chunks:v2")
    parser.add_argument("--chunk-content-budget", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-conversations", type=int, default=16)
    parser.add_argument("--keep-database", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_real_data_preflight(
        input_dir=Path(args.input).expanduser(),
        manifest_path=Path(args.manifest).expanduser(),
        work_dir=Path(args.work_dir).expanduser(),
        output_db=Path(args.output_db).expanduser(),
        output_report=Path(args.output_report).expanduser() if args.output_report else None,
        probe_path=Path(args.probe_file).expanduser() if args.probe_file else None,
        dense_model=args.dense_model,
        dense_device=args.dense_device,
        dense_torch_dtype=args.dense_torch_dtype,
        sparse_provider=args.sparse_provider,
        sparse_model=args.sparse_model,
        sparse_device=args.sparse_device,
        sparse_torch_dtype=args.sparse_torch_dtype,
        sparse_top_k=args.sparse_top_k,
        chunk_content_budget=args.chunk_content_budget,
        batch_size=args.batch_size,
        max_conversations=args.max_conversations,
        keep_database=args.keep_database,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["verdict"] in {"PASS", "PASS WITH WARNINGS"} else 1


def run_real_data_preflight(
    *,
    input_dir: Path,
    manifest_path: Path,
    work_dir: Path,
    output_db: Path,
    output_report: Path | None,
    probe_path: Path | None,
    dense_model: str = DEFAULT_DENSE_MODEL,
    dense_device: str | None = "mps",
    dense_torch_dtype: str = "float16",
    sparse_provider: str = "sentence-transformers",
    sparse_model: str = DEFAULT_SPARSE_MODEL,
    sparse_device: str | None = "mps",
    sparse_torch_dtype: str = "float16",
    sparse_top_k: int = 128,
    chunk_content_budget: int = 512,
    batch_size: int = 16,
    max_conversations: int = 16,
    keep_database: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_dir = input_dir.resolve()
    output_db = output_db.resolve()
    work_dir = work_dir.resolve()
    run_dir = work_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_report or run_dir / "report.md"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(input_dir),
        "output_db": str(output_db),
        "status": "failed",
        "verdict": "FAIL",
        "warnings": [],
        "errors": [],
    }
    try:
        _reject_unsafe_output_path(input_dir, output_db)
        docs = load_or_create_manifest(input_dir, manifest_path, max_conversations=max_conversations)
        filtered_root = _materialize_selected_export(input_dir, run_dir / "selected_export", docs)
        dense_started = time.perf_counter()
        dense = _build_dense_provider("sentence-transformers", dense_model, device=dense_device, torch_dtype=dense_torch_dtype)
        dense_load_s = time.perf_counter() - dense_started
        sparse_started = time.perf_counter()
        sparse = _build_sparse_provider(
            sparse_provider, sparse_model, sparse_top_k, device=sparse_device, torch_dtype=sparse_torch_dtype
        ) if sparse_provider != "none" else None
        sparse_load_s = time.perf_counter() - sparse_started
        contracts = {"dense": dense.contract_dict(), "sparse": sparse.contract_dict() if sparse else None}
        _validate_content_budget(chunk_content_budget, [dense, sparse])
        print(_resolved_configuration(input_dir, docs, contracts, chunk_content_budget, batch_size, dense_device, sparse_device, dense_torch_dtype, sparse_torch_dtype), file=sys.stderr, flush=True)

        ingest_started = time.perf_counter()
        ingest_stats = ingest_chats(filtered_root, output_db)
        ingest_s = time.perf_counter() - ingest_started
        embed_started = time.perf_counter()
        embed_stats = embed_knowledge_blocks(
            db_path=output_db,
            dense_provider="sentence-transformers",
            sparse_provider=sparse_provider,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_device=dense_device,
            sparse_device=sparse_device,
            dense_torch_dtype=dense_torch_dtype,
            sparse_torch_dtype=sparse_torch_dtype,
            sparse_top_k=sparse_top_k,
            chunk_content_budget=chunk_content_budget,
            batch_size=batch_size,
            embedding_pass_mode="joint",
            skip_low_interest_content=False,
            progress=True,
        )
        embedding_s = time.perf_counter() - embed_started
        audit = _extract_audit(embed_stats)
        offset_validation = validate_source_offsets(output_db, sample_size=30)
        probes = load_or_create_probes(output_db, probe_path or manifest_path.with_name("bge_m3_real_probes.json"))
        evaluation = evaluate_real_probes(output_db, dense, sparse, audit, probes)
        db_stats, storage_sizes, provenance = _db_observations(output_db, embed_stats)
        audit_ok = _audit_ok(audit)
        representations_ok = _representations_ok(db_stats, embed_stats, sparse is not None)
        tail_ok = bool(evaluation["manual_checks"]["tail_retrieval"]["passed"])
        provenance_ok = bool(provenance["passed"])
        mps_ok = dense_device != "mps" or not any("mps" in item.lower() and "failed" in item.lower() for item in report["errors"])
        verdict = "PASS" if all((audit_ok, representations_ok, offset_validation["passed"], tail_ok, provenance_ok, mps_ok)) else "FAIL"
        if verdict == "PASS" and evaluation["miss_count"]:
            verdict = "PASS WITH WARNINGS"
        report.update(
            {
                "status": "completed",
                "verdict": verdict,
                "contracts": contracts,
                "sample": {"manifest": str(manifest_path), "selected_documents": len(docs), "documents": [item.__dict__ for item in docs]},
                "configuration": {"chunk_policy": "canonical_token_chunks:v2", "chunk_content_budget": chunk_content_budget, "fallback_overlap": chunk_content_budget // 16, "batch_size": batch_size, "dense_device": dense_device, "sparse_device": sparse_device, "dense_torch_dtype": dense_torch_dtype, "sparse_torch_dtype": sparse_torch_dtype},
                "ingest": ingest_stats,
                "embedding": embed_stats,
                "audit": audit,
                "offset_validation": offset_validation,
                "probes": evaluation,
                "database": {"stats": db_stats, "storage_sizes": storage_sizes},
                "provenance": provenance,
                "performance": {"parse_and_ingest_seconds": ingest_s, "dense_model_load_seconds": dense_load_s, "sparse_model_load_seconds": sparse_load_s, "embedding_seconds": embedding_s, "total_seconds": time.perf_counter() - started, "dense_chunks_per_second": _per_second(embed_stats.get("dense_vectors", 0), embedding_s), "sparse_chunks_per_second": _per_second(embed_stats.get("sparse_vectors", 0), embedding_s)},
                "recommended_full_rebuild_command": _rebuild_command(input_dir, batch_size) if verdict in {"PASS", "PASS WITH WARNINGS"} else None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(str(exc))
        report["performance"] = {"total_seconds": time.perf_counter() - started}
    finally:
        report_json = report_path.with_suffix(".json")
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        report_path.write_text(_markdown_report(report), encoding="utf-8")
        report["report_json"] = str(report_json)
        report["report_md"] = str(report_path)
        if not keep_database and output_db.exists():
            output_db.unlink()
    return report


def load_or_create_manifest(input_dir: Path, manifest_path: Path, *, max_conversations: int) -> list[SourceDocument]:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported source manifest schema: {payload.get('schema_version')}")
        documents = [SourceDocument(**item) for item in payload.get("documents", [])]
        if not 12 <= len(documents) <= 20:
            raise ValueError("Real-data manifest must contain between 12 and 20 conversations.")
        return documents
    candidates: list[tuple[int, int, SourceDocument]] = []
    for item in scan_tree(input_dir):
        if item.detected_kind != "chat_md" or item.folder_kind == "common_trash":
            continue
        parsed = parse_chat_file(input_dir / item.relative_path, source_document_id="preflight", project_id=item.project_path, folder_kind=item.folder_kind)
        text = "\n".join(message.raw_text for message in parsed.messages)
        longest_message = max((len(message.raw_text) for message in parsed.messages), default=0)
        reasons = _selection_reasons(text, longest_message)
        if not reasons:
            continue
        # A preflight must stay small: prefer medium conversations while retaining
        # one verified >2k-token-message candidate below.
        candidates.append((len(text), longest_message, SourceDocument(item.relative_path, parsed.conversation.conversation_id, reasons)))
    target = max(12, min(20, max_conversations))
    medium = [item for item in candidates if 10_000 <= item[0] <= 90_000]
    medium.sort(key=lambda item: (abs(item[0] - 35_000), item[2].relative_path))
    selected_items: list[tuple[int, int, SourceDocument]] = []
    tail_candidates = sorted((item for item in candidates if item[1] >= 8_000), key=lambda item: (item[0], item[2].relative_path))
    if tail_candidates:
        selected_items.append(tail_candidates[0])
    for candidate in medium:
        if candidate[2].relative_path not in {item[2].relative_path for item in selected_items}:
            selected_items.append(candidate)
        if len(selected_items) >= target:
            break
    if len(selected_items) < target:
        for candidate in sorted(candidates, key=lambda item: (abs(item[0] - 35_000), item[2].relative_path)):
            if candidate[2].relative_path not in {item[2].relative_path for item in selected_items}:
                selected_items.append(candidate)
            if len(selected_items) >= target:
                break
    selected = [document for _, _, document in selected_items[:target]]
    if len(selected) < 12:
        raise ValueError("Corpus does not provide 12 suitable non-low-interest conversations for a preflight sample.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"schema_version": MANIFEST_SCHEMA, "source_root": str(input_dir), "documents": [item.__dict__ for item in selected]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def _selection_reasons(text: str, longest_message: int) -> list[str]:
    reasons = []
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if cyrillic > 200:
        reasons.append("russian_or_mixed")
    if latin > 300:
        reasons.append("english_or_mixed")
    if longest_message > 4_000:
        reasons.append("long_message_candidate")
    if longest_message > 8_000:
        reasons.append("very_long_message_candidate")
    if "```" in text:
        reasons.append("fenced_code")
    if "|" in text and "\n|" in text:
        reasons.append("table")
    if "\n- " in text or "\n* " in text:
        reasons.append("list")
    if any(ord(char) > 0x1F300 for char in text):
        reasons.append("emoji")
    return reasons


def _materialize_selected_export(source_root: Path, target_root: Path, documents: list[SourceDocument]) -> Path:
    for document in documents:
        source = source_root / document.relative_path
        target = target_root / document.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return target_root


def _validate_content_budget(content_budget: int, providers: list[Any]) -> None:
    failures = []
    for provider in providers:
        if provider is None:
            continue
        contract = provider.contract_dict()
        safe_budget = contract.get("computed_content_budget")
        if safe_budget is None or content_budget > int(safe_budget):
            failures.append(f"{provider.model_name}: requested={content_budget}, safe_content_budget={safe_budget}, effective_limit={provider.effective_max_sequence_length}")
    if failures:
        raise ValueError("Requested chunk content budget does not fit every active provider before embedding: " + "; ".join(failures))


def _reject_unsafe_output_path(input_dir: Path, output_db: Path) -> None:
    legacy = input_dir / "chat_memory.db"
    production = input_dir / "chat_memory_v2_bge_m3.db"
    if output_db in {legacy.resolve(), production.resolve()} or output_db.name in {"chat_memory.db", "chat_memory_v2_bge_m3.db"}:
        raise ValueError("Preflight output DB must be a distinct path, not a legacy or production rebuild target.")


def validate_source_offsets(db_path: Path, *, sample_size: int) -> dict[str, Any]:
    with SQLiteStore(db_path, read_only=True) as store:
        rows = list(store.conn.execute("""
            SELECT rc.id, rc.text, rc.source_char_start, rc.source_char_end, rc.metadata_json,
                   b.id AS block_id, b.block_type, m.raw_text, m.message_id AS source_message_id
            FROM retrieval_chunks rc
            JOIN blocks b ON b.id = rc.block_id
            JOIN messages m ON m.id = b.message_id
            ORDER BY rc.id
        """))
    if not rows:
        return {"sampled": 0, "passed": False, "mismatches": ["no chunks"]}
    ordered = sorted(rows, key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())
    selected = ordered[:sample_size]
    mismatches = []
    categories = Counter()
    for row in selected:
        actual = str(row["raw_text"])[int(row["source_char_start"]):int(row["source_char_end"])]
        if actual != row["text"]:
            mismatches.append(str(row["id"]))
        categories[str(row["block_type"])] += 1
    return {"sampled": len(selected), "passed": not mismatches, "mismatches": mismatches, "block_types": dict(categories)}


def load_or_create_probes(db_path: Path, probe_path: Path) -> list[RealProbe]:
    if probe_path.exists():
        payload = json.loads(probe_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PROBES_SCHEMA:
            raise ValueError("Unsupported preflight probe schema.")
        probes = [RealProbe(**item) for item in payload.get("probes", [])]
        if not 15 <= len(probes) <= 25:
            raise ValueError("Real-data preflight requires 15–25 probes.")
        return probes
    with SQLiteStore(db_path, read_only=True) as store:
        rows = list(store.conn.execute("""
            SELECT m.message_id, c.conversation_id, m.role, b.block_type, m.raw_text
            FROM messages m JOIN conversations c ON c.id=m.conversation_id
            JOIN blocks b ON b.message_id=m.id
            WHERE length(m.raw_text) > 300
            GROUP BY m.id ORDER BY length(m.raw_text) DESC, m.message_id
            LIMIT 20
        """))
    probes = []
    for index, row in enumerate(rows, start=1):
        query = _tail_query(str(row["raw_text"]))
        language = _language(query)
        probe_type = "long-message-tail" if len(str(row["raw_text"])) > 4_000 else "exact_identifier"
        probes.append(RealProbe(f"auto-{index:02d}", query, language, str(row["conversation_id"]), str(row["message_id"]), probe_type, str(row["role"]), str(row["block_type"]), "Auto-generated local exact-tail probe; review before benchmark use."))
    if len(probes) < 15:
        raise ValueError("Selected sample does not contain enough substantive messages to create local probes.")
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(json.dumps({"schema_version": PROBES_SCHEMA, "probes": [item.__dict__ for item in probes]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return probes


def _tail_query(text: str) -> str:
    words = [word.strip(".,:;!?()[]{}<>`*_#") for word in text[-2_000:].split()]
    words = [word for word in words if len(word) >= 3]
    return " ".join(words[-8:] or ["preflight", "tail", "query"])


def _language(text: str) -> str:
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if cyrillic and latin and min(cyrillic, latin) * 3 >= max(cyrillic, latin):
        return "mixed"
    return "ru" if cyrillic > latin else "en"


def evaluate_real_probes(db_path: Path, dense: Any, sparse: Any, audit: dict[str, Any], probes: list[RealProbe]) -> dict[str, Any]:
    session = DirectRetrievalSession(db_path=db_path, dense_provider=dense, sparse_provider=sparse, chunk_policy=_chunk_policy_from_audit(audit))
    block_by_chunk = {block.chunk_id: block for block in session.blocks}
    source_message_ids, source_conversation_ids = _source_identity_maps(db_path)
    all_results: dict[str, Any] = {}
    misses = []
    latencies: dict[str, list[float]] = defaultdict(list)
    for mode, (alpha, beta) in PROBE_MODES.items():
        records = []
        for probe in probes:
            started = time.perf_counter()
            scores = session.score_query(probe.query)
            results, ranks, _ = session.rank(scores, RankingConfig(mode, alpha, beta), top_k=20)
            latency = (time.perf_counter() - started) * 1000.0
            latencies[mode].append(latency)
            message_ranks = [rank for chunk_id, rank in ranks.items() if source_message_ids.get(block_by_chunk[chunk_id].message_id) == probe.expected_message_id]
            conversation_ranks = [rank for chunk_id, rank in ranks.items() if source_conversation_ids.get(block_by_chunk[chunk_id].conversation_id) == probe.expected_conversation_id]
            chunk_rank = min(message_ranks) if message_ranks else None
            message_rank = min(message_ranks) if message_ranks else None
            conversation_rank = min(conversation_ranks) if conversation_ranks else None
            record = {"probe_id": probe.probe_id, "probe_type": probe.probe_type, "query_language": probe.query_language, "expected_message_id": probe.expected_message_id, "expected_conversation_id": probe.expected_conversation_id, "chunk_rank": chunk_rank, "message_rank": message_rank, "conversation_rank": conversation_rank, "top_results": results, "latency_ms": latency}
            records.append(record)
            if message_rank is None or message_rank > 20:
                misses.append({"mode": mode, "probe_id": probe.probe_id, "expected_message_id": probe.expected_message_id, "top_results": [{key: item.get(key) for key in ("rank", "chunk_id", "block_id", "message_id", "conversation_id", "dense_score", "sparse_score", "final_score", "source_path")} for item in results]})
        all_results[mode] = _probe_metrics(records)
        all_results[mode]["records"] = records
        all_results[mode]["latency_p50_ms"] = _percentile(latencies[mode], 50)
        all_results[mode]["latency_p95_ms"] = _percentile(latencies[mode], 95)
    tail = [record for record in all_results["hybrid"]["records"] if record["probe_type"] == "long-message-tail"]
    return {"count": len(probes), "modes": all_results, "miss_count": len(misses), "miss_diagnostics": misses, "manual_checks": {"tail_retrieval": {"count": len(tail), "passed": bool(tail) and all((record["message_rank"] or 999999) <= 20 for record in tail)}}}


def _source_identity_maps(db_path: Path) -> tuple[dict[str | None, str | None], dict[str | None, str | None]]:
    with SQLiteStore(db_path, read_only=True) as store:
        messages = {str(row["id"]): (str(row["message_id"]) if row["message_id"] else None) for row in store.conn.execute("SELECT id, message_id FROM messages")}
        conversations = {str(row["id"]): (str(row["conversation_id"]) if row["conversation_id"] else None) for row in store.conn.execute("SELECT id, conversation_id FROM conversations")}
    return messages, conversations


def _probe_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    def recall(field: str, cutoff: int) -> float:
        return sum(1 for row in records if row[field] is not None and row[field] <= cutoff) / len(records) if records else 0.0
    def mrr(field: str) -> float:
        return sum(1 / row[field] for row in records if row[field]) / len(records) if records else 0.0
    breakdown = {}
    for key, group in _group_records(records).items():
        breakdown[key] = {"count": len(group), "message_recall_at_10": sum(1 for row in group if row["message_rank"] and row["message_rank"] <= 10) / len(group), "conversation_recall_at_10": sum(1 for row in group if row["conversation_rank"] and row["conversation_rank"] <= 10) / len(group)}
    return {"chunk_recall_at": {str(k): recall("chunk_rank", k) for k in (1, 5, 10, 20)}, "message_recall_at": {str(k): recall("message_rank", k) for k in (1, 5, 10, 20)}, "conversation_recall_at": {str(k): recall("conversation_rank", k) for k in (1, 5, 10, 20)}, "mrr_by_chunk": mrr("chunk_rank"), "mrr_by_message": mrr("message_rank"), "breakdown": breakdown}


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["probe_type"]].append(row)
        grouped[f"query_language:{row['query_language']}"] .append(row)
    return grouped


def _db_observations(db_path: Path, embed_stats: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    with SQLiteStore(db_path, read_only=True) as store:
        stats = store.stats()
        types = dict(store.conn.execute("SELECT block_type, COUNT(*) FROM blocks GROUP BY block_type").fetchall())
        chunk_counts = [row[0] for row in store.conn.execute("SELECT COUNT(*) FROM retrieval_chunks GROUP BY block_id").fetchall()]
        dense_size = int(store.conn.execute("SELECT COALESCE(SUM(length(vector_json)),0) FROM dense_vectors WHERE owner_type='retrieval_chunk'").fetchone()[0])
        sparse_size = int(store.conn.execute("SELECT COALESCE(SUM(length(token_text)+16),0) FROM sparse_terms WHERE owner_type='retrieval_chunk'").fetchone()[0])
        row = store.conn.execute("""
            SELECT rc.id, c.conversation_id, m.message_id, m.role, b.id, b.block_type, rc.ordinal
            FROM retrieval_chunks rc JOIN blocks b ON b.id=rc.block_id JOIN messages m ON m.id=b.message_id JOIN conversations c ON c.id=m.conversation_id
            LIMIT 1
        """).fetchone()
    stats["block_types"] = types
    stats["chunks_per_message"] = _distribution(chunk_counts)
    provenance = {"passed": bool(row and all(row[key] is not None for key in ("conversation_id", "message_id", "role", "id", "block_type", "ordinal"))), "example": dict(row) if row else None}
    return stats, {"database_bytes": db_path.stat().st_size, "dense_representation_bytes": dense_size, "sparse_representation_bytes_approx": sparse_size}, provenance


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {"min": min(values), "p50": _percentile(values, 50), "p95": _percentile(values, 95), "p99": _percentile(values, 99), "max": max(values)}


def _representations_ok(stats: dict[str, Any], embedding: dict[str, Any], sparse_enabled: bool) -> bool:
    chunks = int(stats.get("retrieval_chunks", 0))
    return chunks > 0 and int(embedding.get("dense_vectors", 0)) == chunks and (not sparse_enabled or int(embedding.get("sparse_vectors", 0)) == chunks)


def _audit_ok(audit: dict[str, Any]) -> bool:
    return all(audit.get(key) == 0 for key in ("uncovered_characters", "chunks_over_limit", "truncated_chunks", "blocks_with_coverage_gaps"))


def _per_second(count: object, seconds: float) -> float:
    return float(count or 0) / seconds if seconds else 0.0


def _resolved_configuration(input_dir: Path, docs: list[SourceDocument], contracts: dict[str, Any], budget: int, batch: int, dense_device: str | None, sparse_device: str | None, dense_dtype: str, sparse_dtype: str) -> str:
    dense, sparse = contracts["dense"], contracts["sparse"]
    return "\n".join([f"[kb-real-data-preflight] source={input_dir}", f"[kb-real-data-preflight] selected_conversations={len(docs)}", f"[kb-real-data-preflight] dense={dense.get('model_name')} tokenizer={dense.get('tokenizer_name')} limit={dense.get('configured_effective_max_seq_length')} device={dense_device} dtype={dense_dtype}", f"[kb-real-data-preflight] sparse={None if sparse is None else sparse.get('model_name')} tokenizer={None if sparse is None else sparse.get('tokenizer_name')} limit={None if sparse is None else sparse.get('configured_effective_max_seq_length')} device={sparse_device} dtype={sparse_dtype}", f"[kb-real-data-preflight] policy=canonical_token_chunks:v2 content_budget={budget} fallback_overlap={budget // 16} natural_overlap=0 batch_size={batch}"])


def _rebuild_command(input_dir: Path, batch_size: int) -> str:
    return " \\\n+  ".join(["kb-index import", f"--input {input_dir}", f"--db {input_dir / 'chat_memory_v2_bge_m3.db'}", "--dense-provider sentence-transformers", "--dense-model BAAI/bge-m3", "--sparse-provider sentence-transformers", "--dense-device mps", "--sparse-device mps", "--dense-torch-dtype float16", "--sparse-torch-dtype float16", "--embedding-pass-mode joint", "--chunk-policy canonical_token_chunks:v2", "--chunk-content-budget 512", f"--batch-size {batch_size}"])


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Real-data BGE-M3 Preflight", "", f"Verdict: **{report.get('verdict')}**", "", "This report is local and intentionally redacts private source text and queries.", ""]
    if report.get("errors"):
        lines.extend(["## Errors", ""] + [f"- {item}" for item in report["errors"]] + [""])
    if report.get("contracts"):
        lines.extend(["## Resolved Contracts", "", "```json", json.dumps(report["contracts"], ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    if report.get("audit"):
        lines.extend(["## Ingestion Audit", "", "```json", json.dumps(report["audit"], ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    if report.get("offset_validation"):
        lines.extend(["## Source Offsets", "", "```json", json.dumps(report["offset_validation"], ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    if report.get("probes"):
        lines.extend(["## Retrieval Metrics", ""])
        for mode, values in report["probes"]["modes"].items():
            lines.append(f"### {mode}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps({key: value for key, value in values.items() if key != "records"}, ensure_ascii=False, indent=2, sort_keys=True))
            lines.extend(["```", ""])
    if report.get("recommended_full_rebuild_command"):
        lines.extend(["## Recommended Next Command", "", "```sh", report["recommended_full_rebuild_command"], "```"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
