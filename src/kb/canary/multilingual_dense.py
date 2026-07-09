from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kb.cli import _build_dense_provider, _build_sparse_provider, _chunk_policy_from_audit, embed_knowledge_blocks, ingest_chats
from kb.benchmark import DirectRetrievalSession, RankingConfig
from kb.storage.sqlite_store import SQLiteStore


DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
]
PROBE_MODES = {
    "dense_only": (1.0, 0.0),
    "sparse_only": (0.0, 1.0),
    "hybrid": (0.65, 0.35),
}


@dataclass(frozen=True)
class Probe:
    id: str
    query: str
    query_language: str
    expected_marker: str
    probe_type: str
    direction: str


PROBES = [
    Probe("ru-ru", "маршрутизация памяти и запись решения", "ru", "RU_MEMORY_TAIL_MARKER", "long-message-tail", "RU->RU"),
    Probe("en-en", "attention cache eviction policy", "en", "EN_ATTENTION_TAIL_MARKER", "paraphrase", "EN->EN"),
    Probe("ru-en", "как работает eviction policy attention cache", "ru", "EN_ATTENTION_TAIL_MARKER", "cross-language", "EN->RU"),
    Probe("en-ru", "memory routing deterministic write policy", "en", "RU_MEMORY_TAIL_MARKER", "cross-language", "RU->EN"),
    Probe("exact-id", "SIP_INVITE_BRANCH_X9", "en", "SIP_INVITE_BRANCH_X9", "exact term", "mixed->en"),
    Probe("code", "build_context_pack token budget code", "en", "def build_context_pack", "code-related", "EN->EN"),
    Probe("boundary", "BOUNDARY_LEFT BOUNDARY_RIGHT", "en", "BOUNDARY_LEFT BOUNDARY_RIGHT", "boundary case", "mixed->en"),
    Probe("sparse-id", "VOLTE_QCI_5_IDENTIFIER", "en", "VOLTE_QCI_5_IDENTIFIER", "sparse-friendly exact identifier", "mixed->en"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multilingual dense model canary over a small safe KB fixture.")
    parser.add_argument("--input", help="Optional distilled export input. If omitted, a safe synthetic fixture is generated.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--dense-device")
    parser.add_argument("--sparse-device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-report")
    parser.add_argument("--effective-max-seq-length", type=int)
    parser.add_argument("--chunk-content-budget", type=int)
    parser.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    parser.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
    parser.add_argument("--sparse-top-k", type=int, default=128)
    parser.add_argument("--keep-databases", action="store_true")
    parser.add_argument("--dense-provider", choices=["sentence-transformers", "mock"], default="sentence-transformers")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_canary(
        input_dir=Path(args.input).expanduser() if args.input else None,
        work_dir=Path(args.work_dir).expanduser(),
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        dense_device=args.dense_device,
        sparse_device=args.sparse_device,
        batch_size=args.batch_size,
        output_report=Path(args.output_report).expanduser() if args.output_report else None,
        effective_max_seq_length=args.effective_max_seq_length,
        chunk_content_budget=args.chunk_content_budget,
        sparse_provider=args.sparse_provider,
        sparse_model=args.sparse_model,
        sparse_top_k=args.sparse_top_k,
        keep_databases=args.keep_databases,
        dense_provider=args.dense_provider,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


def run_canary(
    *,
    input_dir: Path | None,
    work_dir: Path,
    models: list[str],
    dense_device: str | None,
    sparse_device: str | None,
    batch_size: int,
    output_report: Path | None,
    effective_max_seq_length: int | None,
    chunk_content_budget: int | None,
    sparse_provider: str,
    sparse_model: str,
    sparse_top_k: int,
    keep_databases: bool,
    dense_provider: str = "sentence-transformers",
) -> dict[str, Any]:
    started = time.perf_counter()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = work_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = input_dir or _write_synthetic_export(run_dir / "synthetic_export")
    results = []
    for model in models:
        results.append(
            _run_model_canary(
                source_dir=source_dir,
                run_dir=run_dir,
                model=model,
                dense_provider=dense_provider,
                dense_device=dense_device,
                sparse_provider=sparse_provider,
                sparse_model=sparse_model,
                sparse_device=sparse_device,
                sparse_top_k=sparse_top_k,
                batch_size=batch_size,
                effective_max_seq_length=effective_max_seq_length,
                chunk_content_budget=chunk_content_budget,
                keep_database=keep_databases,
            )
        )
    report = {
        "schema_version": "kb.model_canary.v1",
        "status": "completed" if all(item.get("status") == "completed" for item in results) else "failed",
        "run_id": run_id,
        "source": str(source_dir),
        "models": results,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
    }
    report_json = run_dir / "report.json"
    report_md = output_report or (run_dir / "report.md")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(_markdown_report(report), encoding="utf-8")
    report["report_json"] = str(report_json)
    report["report_md"] = str(report_md)
    return report


def _run_model_canary(
    *,
    source_dir: Path,
    run_dir: Path,
    model: str,
    dense_provider: str,
    dense_device: str | None,
    sparse_provider: str,
    sparse_model: str,
    sparse_device: str | None,
    sparse_top_k: int,
    batch_size: int,
    effective_max_seq_length: int | None,
    chunk_content_budget: int | None,
    keep_database: bool,
) -> dict[str, Any]:
    safe_name = model.replace("/", "__").replace(":", "_")
    db_path = run_dir / f"{safe_name}.db"
    load_started = time.perf_counter()
    errors: list[str] = []
    warnings: list[str] = []
    dense = None
    try:
        dense = _build_dense_provider(
            dense_provider,
            model,
            device=dense_device,
            effective_max_seq_length=effective_max_seq_length,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model,
            "status": "failed_provider_load",
            "errors": [str(exc)],
            "mps_status": "failed" if dense_device == "mps" else "not_requested",
        }
    provider_load_seconds = time.perf_counter() - load_started
    contract = dense.contract_dict() if dense else {}
    if chunk_content_budget is not None:
        warnings.append(f"chunk content budget override: {chunk_content_budget}")
    if effective_max_seq_length is not None:
        warnings.append(f"effective max sequence length override: {effective_max_seq_length}")
    print(_resolved_configuration(model, contract, dense_device, chunk_content_budget), file=sys.stderr, flush=True)
    index_started = time.perf_counter()
    ingest_chats(source_dir, db_path)
    embed_stats = embed_knowledge_blocks(
        db_path=db_path,
        provider=dense_provider,
        dense_provider=dense_provider,
        sparse_provider=sparse_provider,
        dense_model=model,
        sparse_model=sparse_model,
        dense_device=dense_device,
        sparse_device=sparse_device,
        sparse_top_k=sparse_top_k,
        batch_size=batch_size,
        embedding_pass_mode="joint",
        dense_effective_max_seq_length=effective_max_seq_length,
        chunk_content_budget=chunk_content_budget,
        progress=False,
    )
    indexing_seconds = time.perf_counter() - index_started
    audit = _extract_audit(embed_stats)
    sparse_for_queries = _build_sparse_provider(sparse_provider, sparse_model, sparse_top_k) if sparse_provider != "none" else None
    session = DirectRetrievalSession(
        db_path=db_path,
        dense_provider=dense,
        sparse_provider=sparse_for_queries,
        chunk_policy=_chunk_policy_from_audit(audit),
    )
    metrics, query_latencies = _run_probes(
        db_path=db_path,
        model=model,
        dense_provider=dense_provider,
        dense=dense,
        sparse=sparse_for_queries,
        session=session,
        sparse_provider=sparse_provider,
        sparse_model=sparse_model,
        sparse_top_k=sparse_top_k,
    )
    with SQLiteStore(db_path, read_only=True) as store:
        stats = store.stats()
    if not keep_database:
        try:
            db_path.unlink()
        except OSError:
            warnings.append(f"failed to remove canary db: {db_path}")
    return {
        "model": model,
        "status": "completed",
        "contract": contract,
        "provider_load_seconds": provider_load_seconds,
        "indexing_seconds": indexing_seconds,
        "chunks_per_second": stats["retrieval_chunks"] / indexing_seconds if indexing_seconds else 0.0,
        "query_latency_p50_ms": _percentile(query_latencies, 50),
        "query_latency_p95_ms": _percentile(query_latencies, 95),
        "dense_index_size_bytes": _dense_index_size(db_path) if keep_database and db_path.exists() else None,
        "retrieval_chunks": stats["retrieval_chunks"],
        "audit": audit,
        "metrics": metrics,
        "mps_status": "requested" if dense_device == "mps" else "not_requested",
        "warnings": warnings,
        "errors": errors,
    }


def _run_probes(
    *,
    db_path: Path,
    model: str,
    dense_provider: str,
    dense,
    sparse,
    session: DirectRetrievalSession,
    sparse_provider: str,
    sparse_model: str,
    sparse_top_k: int,
) -> tuple[dict[str, Any], list[float]]:
    expected = _expected_chunks(db_path)
    query_latencies = []
    by_mode: dict[str, dict[str, Any]] = {}
    for mode, (alpha, beta) in PROBE_MODES.items():
        ranks = []
        breakdown: dict[str, list[int | None]] = {}
        for probe in PROBES:
            started = time.perf_counter()
            scores = session.score_query(probe.query)
            results, _, _ = session.rank(scores, RankingConfig(mode, alpha, beta), top_k=10)
            query_latencies.append((time.perf_counter() - started) * 1000.0)
            expected_chunk = expected.get(probe.expected_marker)
            rank = _rank_in_results(results, expected_chunk)
            ranks.append(rank)
            breakdown.setdefault(probe.direction, []).append(rank)
            breakdown.setdefault(probe.probe_type, []).append(rank)
        by_mode[mode] = {
            "recall_at_1": _recall_at(ranks, 1),
            "recall_at_5": _recall_at(ranks, 5),
            "recall_at_10": _recall_at(ranks, 10),
            "mrr": _mrr(ranks),
            "breakdown": {
                key: {
                    "recall_at_10": _recall_at(values, 10),
                    "mrr": _mrr(values),
                    "count": len(values),
                }
                for key, values in sorted(breakdown.items())
            },
        }
    return by_mode, query_latencies


def _write_synthetic_export(root: Path) -> Path:
    chat_dir = root / "Projects" / "Canary"
    chat_dir.mkdir(parents=True, exist_ok=True)
    ru_long = " ".join(f"русский_токен_{idx}" for idx in range(1150))
    en_long = " ".join(f"english_token_{idx}" for idx in range(2200))
    boundary = "x" * 900 + "BOUNDARY_LEFT BOUNDARY_RIGHT" + "y" * 900
    table = "| Key | Value |\n| --- | --- |\n| VOLTE_QCI_5_IDENTIFIER | SIP IMS signalling |\n"
    chat = f"""# Multilingual Canary

## Metadata

- `id`: canary-conv
- `conversation_template_id`: canary-template
- `title`: Multilingual Canary
- `create_time_utc`: 2026-07-09T10:00:00+00:00
- `update_time_utc`: 2026-07-09T10:01:00+00:00
- `message_count`: 4

## Conversation

### 1. USER
- `time_utc`: 2026-07-09T10:00:00+00:00
- `message_id`: canary-ru

{ru_long} RU_MEMORY_TAIL_MARKER 😊

### 2. ASSISTANT
- `time_utc`: 2026-07-09T10:00:10+00:00
- `message_id`: canary-en

{en_long} EN_ATTENTION_TAIL_MARKER

### 3. USER
- `time_utc`: 2026-07-09T10:00:20+00:00
- `message_id`: canary-mixed

- SIP_INVITE_BRANCH_X9 routes to IMS core
- mixed RU/EN память memory routing write policy

{table}

Before code prose explains token budget.

```python
def build_context_pack(query, token_budget):
    return query[:token_budget]
```

After code prose keeps context for retrieval.

### 4. USER
- `time_utc`: 2026-07-09T10:00:30+00:00
- `message_id`: canary-boundary

{boundary}
"""
    (chat_dir / "canary.md").write_text(chat, encoding="utf-8")
    return root


def _expected_chunks(db_path: Path) -> dict[str, str]:
    markers = {probe.expected_marker for probe in PROBES}
    found: dict[str, str] = {}
    with SQLiteStore(db_path, read_only=True) as store:
        for marker in markers:
            row = store.conn.execute(
                "SELECT id FROM retrieval_chunks WHERE text LIKE ? ORDER BY id LIMIT 1",
                (f"%{marker}%",),
            ).fetchone()
            if row:
                found[marker] = str(row["id"])
    return found


def _rank_in_results(results: list[dict[str, Any]], expected_chunk_id: str | None) -> int | None:
    if expected_chunk_id is None:
        return None
    for item in results:
        if item.get("chunk_id") == expected_chunk_id:
            return int(item["rank"])
    return None


def _recall_at(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def _mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank) / len(ranks)


def _extract_audit(stats: dict[str, Any]) -> dict[str, Any]:
    if "indexing_audit" in stats:
        return stats["indexing_audit"]
    for key in ("dense_pass", "sparse_pass"):
        value = stats.get(key)
        if isinstance(value, dict) and "indexing_audit" in value:
            return value["indexing_audit"]
    return {}


def _dense_index_size(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    return db_path.stat().st_size


def _resolved_configuration(model: str, contract: dict[str, object], device: str | None, chunk_budget: int | None) -> str:
    return "\n".join(
        [
            f"[kb-model-canary] model={model}",
            f"[kb-model-canary] tokenizer={contract.get('tokenizer_name')}",
            f"[kb-model-canary] tokenizer_limit={contract.get('tokenizer_model_max_length')} "
            f"st_limit={contract.get('sentence_transformer_max_seq_length')} "
            f"backbone_limit={contract.get('backbone_max_position_embeddings')}",
            f"[kb-model-canary] effective_limit={contract.get('configured_effective_max_seq_length')} "
            f"content_budget={chunk_budget or contract.get('computed_content_budget')} "
            f"fallback_overlap=content_budget/16",
            f"[kb-model-canary] dim={contract.get('embedding_dimension')} device={device or 'auto'}",
        ]
    )


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Multilingual Dense Canary",
        "",
        "This is a small technical canary, not a production benchmark.",
        "",
        "| Model | Dim | ST limit | Backbone limit | Effective limit | Content budget | Fallback overlap | Chunks | Indexing s | Query p50 ms | Query p95 ms | RU->RU R@10 | EN->EN R@10 | RU->EN R@10 | EN->RU R@10 | R@10 | MRR | Audit | MPS | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report["models"]:
        contract = item.get("contract", {})
        audit = item.get("audit", {})
        hybrid = item.get("metrics", {}).get("hybrid", {})
        breakdown = hybrid.get("breakdown", {})
        budget = audit.get("chunk_policy_content_token_budget", contract.get("computed_content_budget"))
        fallback = audit.get("chunk_policy_overlap_tokens")
        audit_ok = (
            audit.get("uncovered_characters") == 0
            and audit.get("chunks_over_limit") == 0
            and audit.get("truncated_chunks") == 0
            and audit.get("blocks_with_coverage_gaps") == 0
        )
        lines.append(
            "| {model} | {dim} | {st} | {backbone} | {effective} | {budget} | {fallback} | {chunks} | "
            "{indexing:.2f} | {p50:.2f} | {p95:.2f} | {ruru:.3f} | {enen:.3f} | {ruen:.3f} | {enru:.3f} | "
            "{r10:.3f} | {mrr:.3f} | {audit} | {mps} | {notes} |".format(
                model=item.get("model"),
                dim=contract.get("embedding_dimension"),
                st=contract.get("sentence_transformer_max_seq_length"),
                backbone=contract.get("backbone_max_position_embeddings"),
                effective=contract.get("configured_effective_max_seq_length"),
                budget=budget,
                fallback=fallback,
                chunks=item.get("retrieval_chunks", 0),
                indexing=float(item.get("indexing_seconds", 0.0)),
                p50=float(item.get("query_latency_p50_ms", 0.0)),
                p95=float(item.get("query_latency_p95_ms", 0.0)),
                ruru=breakdown.get("RU->RU", {}).get("recall_at_10", 0.0),
                enen=breakdown.get("EN->EN", {}).get("recall_at_10", 0.0),
                ruen=breakdown.get("RU->EN", {}).get("recall_at_10", 0.0),
                enru=breakdown.get("EN->RU", {}).get("recall_at_10", 0.0),
                r10=hybrid.get("recall_at_10", 0.0),
                mrr=hybrid.get("mrr", 0.0),
                audit="ok" if audit_ok else "failed",
                mps=item.get("mps_status", ""),
                notes="; ".join(item.get("warnings", []) + item.get("errors", [])),
            )
        )
    lines.extend(["", "## Known Limitations", "", "- Synthetic safe fixture by default; run with `--input` for a local private corpus canary.", "- BGE-M3 sparse output is not integrated here; it is tested only as a dense provider."])
    return "\n".join(lines) + "\n"


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = math.ceil((percentile / 100) * len(ordered)) - 1
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


if __name__ == "__main__":
    raise SystemExit(main())
