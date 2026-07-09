from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kb.cli import _build_dense_provider, _build_sparse_provider
from kb.retrieval.hybrid_search import _preview, _sparse_overlap
from kb.storage.sqlite_store import SQLiteStore


REQUIRED_FIELDS = {"id", "query", "query_type", "language", "topic", "expected", "notes"}
EXPECTED_REQUIRED_FIELDS = {"block_id", "relevance", "source_path", "conversation_id", "message_id"}
VALID_RELEVANCE = {1, 2, 3}
RUN_SCHEMA_VERSION = "kb.benchmark.run.v1"
QUERY_RESULT_SCHEMA_VERSION = "kb.benchmark.query_result.v1"
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
DEFAULT_RANKING_CONFIGS: tuple[tuple[str, float, float], ...] = (
    ("dense_100_sparse_000", 1.0, 0.0),
    ("dense_080_sparse_020", 0.8, 0.2),
    ("dense_065_sparse_035", 0.65, 0.35),
    ("dense_050_sparse_050", 0.5, 0.5),
    ("dense_035_sparse_065", 0.35, 0.65),
    ("dense_020_sparse_080", 0.2, 0.8),
    ("dense_000_sparse_100", 0.0, 1.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate knowledge-base benchmark datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate a direct-retrieval JSONL dataset.")
    validate.add_argument("--db", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--expected-count", type=int, default=120)
    run = sub.add_parser("run", help="Run direct-retrieval benchmark over a JSONL gold dataset.")
    run.add_argument("--db", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--dense-provider", default="sentence-transformers")
    run.add_argument("--sparse-provider", default="sentence-transformers")
    run.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    run.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    run.add_argument("--sparse-top-k", type=int, default=128)
    run.add_argument("--project")
    run.add_argument("--include-low-interest", action="store_true")
    run.add_argument("--max-queries", type=int)
    run.add_argument("--query-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        report = validate_direct_retrieval_dataset(
            db_path=Path(args.db).expanduser(),
            dataset_path=Path(args.dataset).expanduser(),
            expected_count=args.expected_count,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    if args.command == "run":
        report = run_direct_retrieval_benchmark(
            db_path=Path(args.db).expanduser(),
            dataset_path=Path(args.dataset).expanduser(),
            output_dir=Path(args.output_dir).expanduser(),
            top_k=args.top_k,
            dense_provider_name=args.dense_provider,
            sparse_provider_name=args.sparse_provider,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            sparse_top_k=args.sparse_top_k,
            project=args.project,
            include_low_interest=args.include_low_interest,
            max_queries=args.max_queries,
            query_id=args.query_id,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "completed" else 1
    raise AssertionError(f"Unsupported command: {args.command}")


@dataclass(frozen=True)
class RankingConfig:
    id: str
    alpha: float
    beta: float

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "alpha": self.alpha, "beta": self.beta}


@dataclass(frozen=True)
class CorpusBlock:
    block_id: str
    source_path: str
    project: str | None
    folder_kind: str | None
    interest_tier: str | None
    conversation_id: str | None
    conversation_title: str | None
    message_id: str | None
    role: str | None
    block_type: str | None
    preview: str
    sparse_terms: dict[str, float]
    sparse_norm: float


@dataclass
class QueryScores:
    dense_scores: np.ndarray
    sparse_scores: np.ndarray
    overlapping_terms: list[list[str]]
    query_dense_norm: float | None
    query_sparse_term_count: int
    query_encoding_ms: float
    base_scoring_ms: float
    dense_nonzero_count: int
    sparse_nonzero_count: int


class DirectRetrievalSession:
    def __init__(
        self,
        *,
        db_path: Path,
        dense_provider: Any,
        sparse_provider: Any,
        project: str | None = None,
        include_low_interest: bool = False,
    ) -> None:
        if dense_provider is None and sparse_provider is None:
            raise ValueError("At least one retrieval provider must be enabled.")
        self.db_path = db_path
        self.dense_provider = dense_provider
        self.sparse_provider = sparse_provider
        self.project = project
        self.include_low_interest = include_low_interest
        self.blocks: list[CorpusBlock] = []
        self.block_index: dict[str, int] = {}
        self.dense_matrix: np.ndarray | None = None
        self.dense_norms: np.ndarray | None = None
        self.dense_dimension: int | None = None
        self.dense_compatible_count = 0
        self.sparse_compatible_count = 0
        self.corpus_load_ms = 0.0
        self.load_corpus()

    @property
    def candidate_blocks(self) -> int:
        return len(self.blocks)

    def load_corpus(self) -> None:
        started = time.perf_counter()
        with SQLiteStore(self.db_path, read_only=True) as store:
            rows = store.searchable_knowledge_blocks(
                dense_model_name=self.dense_provider.model_name if self.dense_provider else None,
                dense_model_version=self.dense_provider.model_version if self.dense_provider else None,
                sparse_model_name=self.sparse_provider.model_name if self.sparse_provider else None,
                sparse_embedding_space_id=self.sparse_provider.embedding_space_id if self.sparse_provider else None,
                project=self.project,
                include_low_interest=self.include_low_interest,
            )

        dense_vectors: list[list[float] | None] = []
        dense_dim: int | None = None
        blocks: list[CorpusBlock] = []
        for row in rows:
            sparse_terms = {str(k): float(v) for k, v in (row["sparse_terms"] or {}).items()}
            sparse_norm = math.sqrt(sum(weight * weight for weight in sparse_terms.values()))
            vector = row["dense_vector"]
            if vector is not None:
                vector = [float(value) for value in vector]
                if dense_dim is None:
                    dense_dim = len(vector)
                elif len(vector) != dense_dim:
                    raise ValueError(
                        f"Incompatible dense document dimensions in corpus: expected {dense_dim}, got {len(vector)}"
                    )
            dense_vectors.append(vector)
            block = CorpusBlock(
                block_id=row["knowledge_block_id"],
                source_path=row["source_path"],
                project=row["project_id"],
                folder_kind=row["folder_kind"],
                interest_tier=row["interest_tier"],
                conversation_id=row["conversation_id"],
                conversation_title=row["conversation_title"],
                message_id=row["message_id"],
                role=row["role"],
                block_type=row["block_type"],
                preview=_preview(row["text_for_display"]),
                sparse_terms=sparse_terms,
                sparse_norm=sparse_norm,
            )
            blocks.append(block)

        self.blocks = blocks
        self.block_index = {block.block_id: idx for idx, block in enumerate(blocks)}
        self.dense_compatible_count = sum(1 for vector in dense_vectors if vector is not None)
        self.sparse_compatible_count = sum(1 for block in blocks if block.sparse_terms)

        if self.dense_provider is not None:
            if dense_dim is None or self.dense_compatible_count == 0:
                raise ValueError("Dense branch has no compatible document representations.")
            dense_array = np.zeros((len(blocks), dense_dim), dtype=np.float32)
            for idx, vector in enumerate(dense_vectors):
                if vector is not None:
                    dense_array[idx, :] = np.asarray(vector, dtype=np.float32)
            self.dense_matrix = dense_array
            self.dense_norms = np.linalg.norm(dense_array, axis=1)
            self.dense_dimension = dense_dim

        if self.sparse_provider is not None and self.sparse_compatible_count == 0:
            raise ValueError("Sparse branch has no compatible document representations.")

        self.corpus_load_ms = _elapsed_ms(started, time.perf_counter())

    def score_query(self, query: str) -> QueryScores:
        encoding_started = time.perf_counter()
        query_dense = self.dense_provider.embed_query(query) if self.dense_provider else None
        query_sparse = self.sparse_provider.embed_query(query) if self.sparse_provider else None
        encoded = time.perf_counter()

        scoring_started = time.perf_counter()
        dense_scores = np.zeros(len(self.blocks), dtype=np.float32)
        query_dense_norm: float | None = None
        if query_dense is not None:
            query_dense_array = np.asarray(query_dense, dtype=np.float32)
            if self.dense_matrix is None or self.dense_norms is None or self.dense_dimension is None:
                raise ValueError("Dense query was created, but dense corpus matrix is unavailable.")
            if query_dense_array.shape[0] != self.dense_dimension:
                raise ValueError(
                    f"Dense query dimension {query_dense_array.shape[0]} does not match corpus dimension {self.dense_dimension}."
                )
            query_dense_norm = float(np.linalg.norm(query_dense_array))
            if query_dense_norm:
                denom = self.dense_norms * query_dense_norm
                valid = denom != 0
                dense_scores[valid] = (self.dense_matrix[valid] @ query_dense_array) / denom[valid]

        sparse_scores = np.zeros(len(self.blocks), dtype=np.float32)
        overlapping_terms: list[list[str]] = [[] for _ in self.blocks]
        query_sparse_terms = {str(k): float(v) for k, v in (query_sparse or {}).items()}
        if query_sparse_terms:
            for idx, block in enumerate(self.blocks):
                score, terms = _sparse_overlap(query_sparse_terms, block.sparse_terms)
                sparse_scores[idx] = score
                overlapping_terms[idx] = terms[:10]
        scored = time.perf_counter()

        return QueryScores(
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            overlapping_terms=overlapping_terms,
            query_dense_norm=query_dense_norm,
            query_sparse_term_count=len(query_sparse_terms),
            query_encoding_ms=_elapsed_ms(encoding_started, encoded),
            base_scoring_ms=_elapsed_ms(scoring_started, scored),
            dense_nonzero_count=int(np.count_nonzero(dense_scores)),
            sparse_nonzero_count=int(np.count_nonzero(sparse_scores)),
        )

    def rank(self, scores: QueryScores, config: RankingConfig, *, top_k: int) -> tuple[list[dict[str, Any]], dict[str, int], float]:
        started = time.perf_counter()
        final_scores = config.alpha * scores.dense_scores + config.beta * scores.sparse_scores
        ordered = sorted(range(len(self.blocks)), key=lambda idx: (-float(final_scores[idx]), self.blocks[idx].block_id))
        ranks = {self.blocks[idx].block_id: rank for rank, idx in enumerate(ordered, start=1)}
        results = [
            self._result_item(
                idx,
                rank=rank,
                dense_score=float(scores.dense_scores[idx]),
                sparse_score=float(scores.sparse_scores[idx]),
                final_score=float(final_scores[idx]),
                overlapping_terms=scores.overlapping_terms[idx],
            )
            for rank, idx in enumerate(ordered[:top_k], start=1)
        ]
        return results, ranks, _elapsed_ms(started, time.perf_counter())

    def _result_item(
        self,
        idx: int,
        *,
        rank: int,
        dense_score: float,
        sparse_score: float,
        final_score: float,
        overlapping_terms: list[str],
    ) -> dict[str, Any]:
        block = self.blocks[idx]
        return {
            "rank": rank,
            "block_id": block.block_id,
            "source_path": block.source_path,
            "conversation_id": block.conversation_id,
            "message_id": block.message_id,
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "final_score": final_score,
            "expected": False,
            "relevance": 0,
            "overlapping_terms": overlapping_terms,
            "preview": block.preview,
        }


def validate_direct_retrieval_dataset(*, db_path: Path, dataset_path: Path, expected_count: int = 120) -> dict[str, Any]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    queries: set[str] = set()
    query_types: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    topics: Counter[str] = Counter()

    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"ok": False, "errors": [f"failed to read dataset: {exc}"]}

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_no}: record must be an object")
            continue
        records.append(record)
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            errors.append(f"line {line_no}: missing fields: {sorted(missing)}")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append(f"line {line_no}: id must be a non-empty string")
        elif record_id in ids:
            errors.append(f"line {line_no}: duplicate id: {record_id}")
        else:
            ids.add(record_id)
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"line {line_no}: query must be a non-empty string")
        elif query in queries:
            errors.append(f"line {line_no}: duplicate query: {query}")
        else:
            queries.add(query)
        if isinstance(record.get("query_type"), str):
            query_types[record["query_type"]] += 1
        if isinstance(record.get("language"), str):
            languages[record["language"]] += 1
        if isinstance(record.get("topic"), str):
            topics[record["topic"]] += 1
        expected = record.get("expected")
        if not isinstance(expected, list) or not expected:
            errors.append(f"line {line_no}: expected must be a non-empty list")
            continue
        primary_count = 0
        for expected_idx, item in enumerate(expected, start=1):
            if not isinstance(item, dict):
                errors.append(f"line {line_no}: expected[{expected_idx}] must be an object")
                continue
            missing_expected = EXPECTED_REQUIRED_FIELDS - set(item)
            if missing_expected:
                errors.append(f"line {line_no}: expected[{expected_idx}] missing fields: {sorted(missing_expected)}")
            relevance = item.get("relevance")
            if relevance not in VALID_RELEVANCE:
                errors.append(f"line {line_no}: expected[{expected_idx}] relevance must be one of {sorted(VALID_RELEVANCE)}")
            if relevance == 3:
                primary_count += 1
        if primary_count != 1:
            errors.append(f"line {line_no}: expected must contain exactly one relevance=3 block, got {primary_count}")

    if len(records) != expected_count:
        errors.append(f"dataset must contain {expected_count} records, got {len(records)}")

    if not db_path.exists():
        errors.append(f"db does not exist: {db_path}")
    else:
        errors.extend(_validate_expected_blocks(db_path, records))

    return {
        "ok": not errors,
        "errors": errors,
        "records": len(records),
        "query_type_distribution": dict(sorted(query_types.items())),
        "language_distribution": dict(sorted(languages.items())),
        "topic_distribution": dict(sorted(topics.items())),
    }


def run_direct_retrieval_benchmark(
    *,
    db_path: Path,
    dataset_path: Path,
    output_dir: Path,
    top_k: int = 20,
    dense_provider_name: str = "sentence-transformers",
    sparse_provider_name: str = "sentence-transformers",
    dense_model: str = DEFAULT_DENSE_MODEL,
    sparse_model: str = DEFAULT_SPARSE_MODEL,
    sparse_top_k: int = 128,
    project: str | None = None,
    include_low_interest: bool = False,
    max_queries: int | None = None,
    query_id: str | None = None,
    ranking_configs: list[RankingConfig] | None = None,
    dense_provider: Any | None = None,
    sparse_provider: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    configs = ranking_configs or default_ranking_configs()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_dir = output_dir / run_id
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "failed",
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "query_count": 0,
        },
        "database": {
            "path": str(db_path),
            "candidate_blocks": 0,
        },
        "providers": {
            "dense_model": None,
            "dense_embedding_space_id": None,
            "sparse_model": None,
            "sparse_embedding_space_id": None,
            "sparse_top_k": sparse_top_k,
        },
        "configurations": [config.as_dict() for config in configs],
        "timing_ms": {
            "provider_load": 0.0,
            "corpus_load": 0.0,
            "query_execution": 0.0,
            "total": 0.0,
        },
        "completed_queries": 0,
        "failed_queries": 0,
    }

    try:
        records = _load_dataset_records(dataset_path)
        if query_id is not None:
            records = [record for record in records if record.get("id") == query_id]
            if not records:
                raise ValueError(f"query id not found in dataset: {query_id}")
        if max_queries is not None:
            if max_queries <= 0:
                raise ValueError("--max-queries must be positive.")
            records = records[:max_queries]
        validation = validate_direct_retrieval_dataset(
            db_path=db_path,
            dataset_path=dataset_path,
            expected_count=len(_load_dataset_records(dataset_path)),
        )
        if not validation["ok"]:
            raise ValueError("dataset validation failed: " + "; ".join(validation["errors"]))

        provider_started = time.perf_counter()
        dense = dense_provider
        sparse = sparse_provider
        if dense is None and dense_provider_name != "none":
            dense = _build_dense_provider(dense_provider_name, dense_model)
        if sparse is None and sparse_provider_name != "none":
            sparse = _build_sparse_provider(sparse_provider_name, sparse_model, sparse_top_k)
        providers_loaded = time.perf_counter()

        session = DirectRetrievalSession(
            db_path=db_path,
            dense_provider=dense,
            sparse_provider=sparse,
            project=project,
            include_low_interest=include_low_interest,
        )
        _verify_primary_expected_candidates(records, session)
        manifest["dataset"]["query_count"] = len(records)
        manifest["database"]["candidate_blocks"] = session.candidate_blocks
        manifest["providers"] = {
            "dense_model": dense.model_name if dense else None,
            "dense_embedding_space_id": dense.embedding_space_id if dense else None,
            "sparse_model": sparse.model_name if sparse else None,
            "sparse_embedding_space_id": sparse.embedding_space_id if sparse else None,
            "sparse_top_k": sparse_top_k,
        }
        manifest["timing_ms"]["provider_load"] = _elapsed_ms(provider_started, providers_loaded)
        manifest["timing_ms"]["corpus_load"] = session.corpus_load_ms

        written = 0
        failed_queries = 0
        query_started = time.perf_counter()
        with results_path.open("w", encoding="utf-8") as handle:
            for record in records:
                try:
                    scores = session.score_query(str(record["query"]))
                    expected_by_id = {item["block_id"]: int(item["relevance"]) for item in record["expected"]}
                    for config in configs:
                        top_results, ranks, ranking_ms = session.rank(scores, config, top_k=top_k)
                        for item in top_results:
                            relevance = expected_by_id.get(item["block_id"], 0)
                            item["expected"] = relevance > 0
                            item["relevance"] = relevance
                        expected = [
                            {
                                "block_id": item["block_id"],
                                "relevance": item["relevance"],
                                "rank": ranks.get(item["block_id"]),
                            }
                            for item in record["expected"]
                        ]
                        payload = {
                            "schema_version": QUERY_RESULT_SCHEMA_VERSION,
                            "query_id": record["id"],
                            "query": record["query"],
                            "query_type": record["query_type"],
                            "language": record["language"],
                            "topic": record["topic"],
                            "configuration": config.as_dict(),
                            "expected": expected,
                            "candidate_blocks": session.candidate_blocks,
                            "top_results": top_results,
                            "diagnostics": {
                                "dense_status": "active" if dense else "disabled",
                                "sparse_status": "active" if sparse else "disabled",
                                "query_dense_norm": scores.query_dense_norm,
                                "query_sparse_term_count": scores.query_sparse_term_count,
                                "dense_nonzero_count": scores.dense_nonzero_count,
                                "sparse_nonzero_count": scores.sparse_nonzero_count,
                            },
                            "latency_ms": {
                                "query_encoding": scores.query_encoding_ms,
                                "base_scoring": scores.base_scoring_ms,
                                "ranking": ranking_ms,
                            },
                        }
                        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                        written += 1
                except Exception:
                    failed_queries += 1
                    raise
        query_finished = time.perf_counter()
        expected_records = len(records) * len(configs)
        if written != expected_records:
            raise ValueError(f"results record count mismatch: expected {expected_records}, wrote {written}")
        manifest["completed_queries"] = len(records)
        manifest["failed_queries"] = failed_queries
        manifest["timing_ms"]["query_execution"] = _elapsed_ms(query_started, query_finished)
        manifest["timing_ms"]["total"] = _elapsed_ms(started, time.perf_counter())
        manifest["status"] = "completed"
        _write_manifest(manifest_path, manifest)
        return {
            "status": "completed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": str(manifest_path),
            "results": str(results_path),
            "records_written": written,
            "completed_queries": len(records),
            "failed_queries": failed_queries,
            "timing_ms": manifest["timing_ms"],
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["timing_ms"]["total"] = _elapsed_ms(started, time.perf_counter())
        _write_manifest(manifest_path, manifest)
        return {
            "status": "failed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": str(manifest_path),
            "error": str(exc),
        }


def default_ranking_configs() -> list[RankingConfig]:
    return [RankingConfig(id=config_id, alpha=alpha, beta=beta) for config_id, alpha, beta in DEFAULT_RANKING_CONFIGS]


def _load_dataset_records(dataset_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _verify_primary_expected_candidates(records: list[dict[str, Any]], session: DirectRetrievalSession) -> None:
    missing: list[str] = []
    for record in records:
        primary = [item for item in record["expected"] if item["relevance"] == 3]
        if not primary:
            missing.append(f"{record['id']}: no primary expected block")
            continue
        block_id = primary[0]["block_id"]
        if block_id not in session.block_index:
            missing.append(f"{record['id']}: primary expected block is not in candidates: {block_id}")
    if missing:
        raise ValueError("; ".join(missing))


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_expected_blocks(db_path: Path, records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        block_ids = sorted(
            {
                item.get("block_id")
                for record in records
                for item in (record.get("expected") if isinstance(record.get("expected"), list) else [])
                if isinstance(item, dict) and isinstance(item.get("block_id"), str)
            }
        )
        if not block_ids:
            return errors
        placeholders = ",".join("?" for _ in block_ids)
        rows = conn.execute(
            f"""
            SELECT
                kb.id AS block_id,
                sd.relative_path AS source_path,
                kb.conversation_id,
                kb.message_id
            FROM knowledge_blocks kb
            JOIN source_documents sd ON sd.id = kb.source_document_id
            WHERE kb.id IN ({placeholders})
            """,
            block_ids,
        ).fetchall()
        by_id = {row["block_id"]: dict(row) for row in rows}
        for record in records:
            record_id = record.get("id", "<unknown>")
            expected = record.get("expected") if isinstance(record.get("expected"), list) else []
            for item in expected:
                if not isinstance(item, dict):
                    continue
                block_id = item.get("block_id")
                if not isinstance(block_id, str):
                    continue
                row = by_id.get(block_id)
                if row is None:
                    errors.append(f"{record_id}: block_id does not exist: {block_id}")
                    continue
                for field in ("source_path", "conversation_id", "message_id"):
                    if item.get(field) != row[field]:
                        errors.append(
                            f"{record_id}: {block_id} {field} mismatch: expected {row[field]!r}, got {item.get(field)!r}"
                        )
    finally:
        conn.close()
    return errors


if __name__ == "__main__":
    sys.exit(main())
