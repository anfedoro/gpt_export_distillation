from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kb.cli import _build_dense_provider, _build_sparse_provider, _chunked_embedding_space_id
from kb.index.chunk_builder import build_chunk_policy
from kb.retrieval.hybrid_search import _preview, _sparse_overlap
from kb.storage.sqlite_store import SQLiteStore


REQUIRED_FIELDS = {"id", "query", "query_type", "language", "source_language", "topic", "expected", "notes"}
EXPECTED_REQUIRED_FIELDS = {"block_id", "relevance", "source_path", "conversation_id", "message_id"}
VALID_RELEVANCE = {1, 2, 3}
VALID_LANGUAGES = {"ru", "en", "mixed"}
RUN_SCHEMA_VERSION = "kb.benchmark.run.v1"
QUERY_RESULT_SCHEMA_VERSION = "kb.benchmark.query_result.v1"
QUERY_METRICS_SCHEMA_VERSION = "kb.benchmark.query_metrics.v1"
EVALUATION_SCHEMA_VERSION = "kb.benchmark.evaluation.v1"
EVALUATION_MANIFEST_SCHEMA_VERSION = "kb.benchmark.evaluation_manifest.v1"
ANALYSIS_MANIFEST_SCHEMA_VERSION = "kb.benchmark.analysis_manifest.v1"
BREAKDOWNS_SCHEMA_VERSION = "kb.benchmark.breakdowns.v1"
PAIRWISE_QUERY_SCHEMA_VERSION = "kb.benchmark.pairwise_query.v1"
METRIC_K_VALUES = (1, 5, 10, 20)
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
    evaluate = sub.add_parser("evaluate", help="Evaluate an existing direct-retrieval benchmark run.")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output-dir")
    analyze = sub.add_parser("analyze", help="Analyze an existing direct-retrieval evaluation.")
    analyze.add_argument("--evaluation-dir", required=True)
    analyze.add_argument("--dataset", required=True)
    analyze.add_argument("--output-dir")
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
    if args.command == "evaluate":
        report = evaluate_direct_retrieval_run(
            run_dir=Path(args.run_dir).expanduser(),
            dataset_path=Path(args.dataset).expanduser(),
            output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "completed" else 1
    if args.command == "analyze":
        report = analyze_direct_retrieval_evaluation(
            evaluation_dir=Path(args.evaluation_dir).expanduser(),
            dataset_path=Path(args.dataset).expanduser(),
            output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
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
    chunk_id: str
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
        self.policy = build_chunk_policy([item for item in (dense_provider, sparse_provider) if item is not None])
        self.dense_embedding_space_id = (
            _chunked_embedding_space_id(dense_provider.embedding_space_id, self.policy.id) if dense_provider else None
        )
        self.sparse_embedding_space_id = (
            _chunked_embedding_space_id(sparse_provider.embedding_space_id, self.policy.id) if sparse_provider else None
        )
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
            rows = store.searchable_retrieval_chunks(
                chunk_policy_id=self.policy.id,
                dense_model_name=self.dense_provider.model_name if self.dense_provider else None,
                dense_model_version=self.dense_embedding_space_id if self.dense_provider else None,
                sparse_model_name=self.sparse_provider.model_name if self.sparse_provider else None,
                sparse_embedding_space_id=self.sparse_embedding_space_id if self.sparse_provider else None,
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
                chunk_id=row["chunk_id"],
                block_id=row["block_id"],
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
        self.block_index = {block.chunk_id: idx for idx, block in enumerate(blocks)}
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
        ordered = sorted(range(len(self.blocks)), key=lambda idx: (-float(final_scores[idx]), self.blocks[idx].chunk_id))
        ranks = {self.blocks[idx].chunk_id: rank for rank, idx in enumerate(ordered, start=1)}
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
            "chunk_id": block.chunk_id,
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
    source_languages: Counter[str] = Counter()
    language_pairs: Counter[str] = Counter()
    cross_language_pairs: Counter[str] = Counter()
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
        language = record.get("language")
        source_language = record.get("source_language")
        query_type = record.get("query_type")
        if isinstance(language, str):
            languages[language] += 1
            if language not in VALID_LANGUAGES:
                errors.append(f"line {line_no}: language must be one of {sorted(VALID_LANGUAGES)}")
        if isinstance(source_language, str):
            source_languages[source_language] += 1
            if source_language not in VALID_LANGUAGES:
                errors.append(f"line {line_no}: source_language must be one of {sorted(VALID_LANGUAGES)}")
        if isinstance(language, str) and isinstance(source_language, str):
            pair = f"{source_language}->{language}"
            language_pairs[pair] += 1
            if query_type == "cross_language":
                cross_language_pairs[pair] += 1
                if language == "mixed" or source_language == "mixed":
                    errors.append(f"line {line_no}: cross_language records cannot use mixed language fields")
                if language == source_language:
                    errors.append(f"line {line_no}: cross_language records must use different source/query languages")
        if isinstance(query, str) and isinstance(language, str):
            mismatch = _obvious_query_language_mismatch(query, language)
            if mismatch:
                errors.append(f"line {line_no}: {mismatch}")
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
        "source_language_distribution": dict(sorted(source_languages.items())),
        "language_pair_distribution": dict(sorted(language_pairs.items())),
        "cross_language_pair_distribution": dict(sorted(cross_language_pairs.items())),
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
            "dense_embedding_space_id": session.dense_embedding_space_id if dense else None,
            "sparse_model": sparse.model_name if sparse else None,
            "sparse_embedding_space_id": session.sparse_embedding_space_id if sparse else None,
            "sparse_top_k": sparse_top_k,
            "chunk_policy_id": session.policy.id,
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
                            "source_language": record["source_language"],
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


def evaluate_direct_retrieval_run(
    *,
    run_dir: Path,
    dataset_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    evaluation_dir = output_dir or (run_dir / "evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evaluation_dir / "manifest.json"
    evaluation_manifest: dict[str, Any] = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "status": "failed",
        "source_run_id": None,
        "source_run_dir": str(run_dir),
        "source_manifest_sha256": None,
        "source_results_sha256": None,
        "dataset_path": str(dataset_path),
        "dataset_sha256": None,
        "query_count": 0,
        "configuration_count": 0,
        "query_metric_records": 0,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timing_ms": {
            "input_validation": 0.0,
            "metric_calculation": 0.0,
            "artifact_writing": 0.0,
            "total": 0.0,
        },
    }
    try:
        validation_started = time.perf_counter()
        loaded = _load_and_validate_evaluation_inputs(run_dir=run_dir, dataset_path=dataset_path)
        validation_finished = time.perf_counter()
        evaluation_manifest.update(
            {
                "source_run_id": loaded["manifest"]["run_id"],
                "source_manifest_sha256": _sha256(run_dir / "manifest.json"),
                "source_results_sha256": _sha256(run_dir / "results.jsonl"),
                "dataset_sha256": loaded["dataset_sha256"],
                "query_count": len(loaded["dataset_records"]),
                "configuration_count": len(loaded["configurations"]),
            }
        )
        evaluation_manifest["timing_ms"]["input_validation"] = _elapsed_ms(validation_started, validation_finished)

        metric_started = time.perf_counter()
        query_metrics = [
            calculate_query_metrics(record, loaded["dataset_by_id"][record["query_id"]])
            for record in loaded["results"]
        ]
        summary = _aggregate_query_metrics(
            run_manifest=loaded["manifest"],
            dataset_sha256=loaded["dataset_sha256"],
            query_metrics=query_metrics,
            configurations=loaded["configurations"],
        )
        metric_finished = time.perf_counter()
        evaluation_manifest["timing_ms"]["metric_calculation"] = _elapsed_ms(metric_started, metric_finished)

        write_started = time.perf_counter()
        query_metrics_path = evaluation_dir / "query_metrics.jsonl"
        summary_path = evaluation_dir / "summary.json"
        report_path = evaluation_dir / "report.md"
        query_metrics_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in query_metrics),
            encoding="utf-8",
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path.write_text(_render_evaluation_report(summary), encoding="utf-8")
        write_finished = time.perf_counter()
        evaluation_manifest["query_metric_records"] = len(query_metrics)
        evaluation_manifest["timing_ms"]["artifact_writing"] = _elapsed_ms(write_started, write_finished)
        evaluation_manifest["timing_ms"]["total"] = _elapsed_ms(started, time.perf_counter())
        evaluation_manifest["status"] = "completed"
        _write_manifest(manifest_path, evaluation_manifest)
        return {
            "status": "completed",
            "evaluation_dir": str(evaluation_dir),
            "manifest": str(manifest_path),
            "query_metrics": str(query_metrics_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "query_metric_records": len(query_metrics),
            "query_count": len(loaded["dataset_records"]),
            "configuration_count": len(loaded["configurations"]),
            "timing_ms": evaluation_manifest["timing_ms"],
        }
    except Exception as exc:
        evaluation_manifest["error"] = str(exc)
        evaluation_manifest["timing_ms"]["total"] = _elapsed_ms(started, time.perf_counter())
        _write_manifest(manifest_path, evaluation_manifest)
        return {
            "status": "failed",
            "evaluation_dir": str(evaluation_dir),
            "manifest": str(manifest_path),
            "error": str(exc),
        }


def analyze_direct_retrieval_evaluation(
    *,
    evaluation_dir: Path,
    dataset_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    analysis_dir = output_dir or (evaluation_dir / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = analysis_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION,
        "status": "failed",
        "source_evaluation_dir": str(evaluation_dir),
        "source_evaluation_manifest_sha256": None,
        "query_metrics_sha256": None,
        "dataset_sha256": None,
        "query_count": 0,
        "configuration_count": 0,
        "breakdown_record_count": 0,
        "pairwise_record_count": 0,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timing_ms": {"input_validation": 0.0, "analysis": 0.0, "artifact_writing": 0.0, "total": 0.0},
    }
    try:
        validation_started = time.perf_counter()
        loaded = _load_and_validate_analysis_inputs(evaluation_dir=evaluation_dir, dataset_path=dataset_path)
        validation_finished = time.perf_counter()
        manifest.update(
            {
                "source_evaluation_manifest_sha256": _sha256(evaluation_dir / "manifest.json"),
                "query_metrics_sha256": _sha256(evaluation_dir / "query_metrics.jsonl"),
                "dataset_sha256": loaded["dataset_sha256"],
                "query_count": len(loaded["dataset_records"]),
                "configuration_count": len(loaded["configurations"]),
            }
        )
        manifest["timing_ms"]["input_validation"] = _elapsed_ms(validation_started, validation_finished)

        analysis_started = time.perf_counter()
        breakdowns = build_breakdowns(loaded["query_metrics"], loaded["configurations"])
        pairwise = build_pairwise_queries(loaded["query_metrics"])
        analysis_finished = time.perf_counter()
        manifest["timing_ms"]["analysis"] = _elapsed_ms(analysis_started, analysis_finished)

        write_started = time.perf_counter()
        breakdowns_path = analysis_dir / "breakdowns.json"
        pairwise_path = analysis_dir / "pairwise_queries.jsonl"
        report_path = analysis_dir / "report.md"
        breakdowns_path.write_text(json.dumps(breakdowns, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pairwise_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in pairwise), encoding="utf-8")
        report_path.write_text(_render_analysis_report(loaded["summary"], breakdowns, pairwise), encoding="utf-8")
        write_finished = time.perf_counter()
        manifest["breakdown_record_count"] = sum(
            len(slice_item["configurations"])
            for dimension in breakdowns["dimensions"].values()
            for slice_item in dimension
        )
        manifest["pairwise_record_count"] = len(pairwise)
        manifest["timing_ms"]["artifact_writing"] = _elapsed_ms(write_started, write_finished)
        manifest["timing_ms"]["total"] = _elapsed_ms(started, time.perf_counter())
        manifest["status"] = "completed"
        _write_manifest(manifest_path, manifest)
        return {
            "status": "completed",
            "analysis_dir": str(analysis_dir),
            "manifest": str(manifest_path),
            "breakdowns": str(breakdowns_path),
            "pairwise_queries": str(pairwise_path),
            "report": str(report_path),
            "query_count": manifest["query_count"],
            "configuration_count": manifest["configuration_count"],
            "breakdown_dimensions": len(breakdowns["dimensions"]),
            "pairwise_record_count": len(pairwise),
            "timing_ms": manifest["timing_ms"],
        }
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["timing_ms"]["total"] = _elapsed_ms(started, time.perf_counter())
        _write_manifest(manifest_path, manifest)
        return {"status": "failed", "analysis_dir": str(analysis_dir), "manifest": str(manifest_path), "error": str(exc)}


def calculate_query_metrics(result_record: dict[str, Any], dataset_record: dict[str, Any]) -> dict[str, Any]:
    expected_from_dataset = {item["block_id"]: int(item["relevance"]) for item in dataset_record["expected"]}
    expected_from_result = {item["block_id"]: item for item in result_record["expected"]}
    expected_ranks: dict[str, int | None] = {}
    for block_id, relevance in expected_from_dataset.items():
        result_expected = expected_from_result.get(block_id)
        rank = result_expected.get("rank") if result_expected else None
        expected_ranks[block_id] = int(rank) if isinstance(rank, int) and rank > 0 else None
        if result_expected is None or int(result_expected.get("relevance", -1)) != relevance:
            raise ValueError(f"{result_record['query_id']}: expected relevance mismatch for {block_id}")

    relevant_ranks = [rank for rank in expected_ranks.values() if rank is not None]
    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None
    primary_blocks = [block_id for block_id, relevance in expected_from_dataset.items() if relevance == 3]
    if len(primary_blocks) != 1:
        raise ValueError(f"{result_record['query_id']}: expected exactly one primary block")
    primary_rank = expected_ranks[primary_blocks[0]]
    top_results = result_record["top_results"]
    expected_source_paths = {item["source_path"] for item in dataset_record["expected"]}
    expected_conversation_ids = {
        item["conversation_id"] for item in dataset_record["expected"] if item.get("conversation_id") is not None
    }
    recall_at: dict[str, int] = {}
    primary_recall_at: dict[str, int] = {}
    ndcg_at: dict[str, float] = {}
    document_recall_at: dict[str, int] = {}
    conversation_recall_at: dict[str, int] = {}
    for k in METRIC_K_VALUES:
        recall_at[str(k)] = int(any(rank is not None and rank <= k for rank in expected_ranks.values()))
        primary_recall_at[str(k)] = int(primary_rank is not None and primary_rank <= k)
        ndcg_at[str(k)] = _ndcg_at(expected_from_dataset, expected_ranks, k)
        document_recall_at[str(k)] = int(
            any(item["rank"] <= k and item["source_path"] in expected_source_paths for item in top_results)
        )
        conversation_recall_at[str(k)] = int(
            bool(expected_conversation_ids)
            and any(
                item["rank"] <= k
                and item.get("conversation_id") is not None
                and item.get("conversation_id") in expected_conversation_ids
                for item in top_results
            )
        )

    return {
        "schema_version": QUERY_METRICS_SCHEMA_VERSION,
        "query_id": result_record["query_id"],
        "query": result_record["query"],
        "query_type": result_record["query_type"],
        "language": result_record["language"],
        "source_language": result_record["source_language"],
        "topic": result_record["topic"],
        "configuration": result_record["configuration"],
        "expected_count": len(expected_from_dataset),
        "first_relevant_rank": first_relevant_rank,
        "primary_rank": primary_rank,
        "reciprocal_rank": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "primary_reciprocal_rank": 0.0 if primary_rank is None else 1.0 / primary_rank,
        "recall_at": recall_at,
        "primary_recall_at": primary_recall_at,
        "ndcg_at": ndcg_at,
        "document_recall_at": document_recall_at,
        "conversation_recall_at": conversation_recall_at,
    }


def build_breakdowns(query_metrics: list[dict[str, Any]], configurations: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "query_type": lambda item: item["query_type"],
        "language": lambda item: item["language"],
        "source_language": lambda item: item["source_language"],
        "language_direction": lambda item: f"{item['source_language']}->{item['language']}",
        "topic": lambda item: item["topic"],
    }
    output: dict[str, list[dict[str, Any]]] = {}
    config_ids = [config["id"] for config in configurations]
    for dimension, key_fn in dimensions.items():
        query_ids_by_value: dict[str, set[str]] = {}
        metrics_by_value: dict[str, list[dict[str, Any]]] = {}
        for item in query_metrics:
            value = str(key_fn(item))
            query_ids_by_value.setdefault(value, set()).add(item["query_id"])
            metrics_by_value.setdefault(value, []).append(item)
        slices: list[dict[str, Any]] = []
        for value in sorted(metrics_by_value):
            items = metrics_by_value[value]
            query_count = len(query_ids_by_value[value])
            per_config = []
            for config in configurations:
                config_items = [item for item in items if item["configuration"]["id"] == config["id"]]
                per_config.append(_slice_metrics(config, config_items))
            slices.append(
                {
                    "value": value,
                    "query_count": query_count,
                    "small_sample": query_count < 10,
                    "best_configuration": _best_configurations(per_config),
                    "configurations": per_config,
                }
            )
        output[dimension] = slices
    return {"schema_version": BREAKDOWNS_SCHEMA_VERSION, "dimensions": output, "configuration_ids": config_ids}


def build_pairwise_queries(query_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for item in query_metrics:
        by_query.setdefault(item["query_id"], {})[item["configuration"]["id"]] = item
    records: list[dict[str, Any]] = []
    hybrid_ids = [
        "dense_080_sparse_020",
        "dense_065_sparse_035",
        "dense_050_sparse_050",
        "dense_035_sparse_065",
        "dense_020_sparse_080",
    ]
    for query_id in sorted(by_query):
        configs = by_query[query_id]
        dense = configs["dense_100_sparse_000"]
        sparse = configs["dense_000_sparse_100"]
        dense_rr = float(dense["reciprocal_rank"])
        sparse_rr = float(sparse["reciprocal_rank"])
        best_hybrid = max((configs[config_id] for config_id in hybrid_ids), key=lambda item: (float(item["reciprocal_rank"]), item["configuration"]["alpha"], item["configuration"]["id"]))
        best_hybrid_rr = float(best_hybrid["reciprocal_rank"])
        record = {
            "schema_version": PAIRWISE_QUERY_SCHEMA_VERSION,
            "query_id": query_id,
            "query": dense["query"],
            "query_type": dense["query_type"],
            "language": dense["language"],
            "source_language": dense["source_language"],
            "language_direction": f"{dense['source_language']}->{dense['language']}",
            "topic": dense["topic"],
            "dense_reciprocal_rank": dense_rr,
            "sparse_reciprocal_rank": sparse_rr,
            "rr_delta": sparse_rr - dense_rr,
            "dense_primary_rank": dense["primary_rank"],
            "sparse_primary_rank": sparse["primary_rank"],
            "dense_recall_at_10": dense["recall_at"]["10"],
            "sparse_recall_at_10": sparse["recall_at"]["10"],
            "dense_document_recall_at_10": dense["document_recall_at"]["10"],
            "sparse_document_recall_at_10": sparse["document_recall_at"]["10"],
            "dense_vs_sparse_class": _dense_sparse_class(dense, sparse),
            "best_hybrid_configuration": best_hybrid["configuration"]["id"],
            "best_hybrid_rr": best_hybrid_rr,
            "hybrid_comparison_class": _hybrid_class(best_hybrid_rr, dense_rr, sparse_rr),
        }
        records.append(record)
    return records


def _slice_metrics(config: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError(f"empty slice for configuration {config['id']}")
    ranks = [int(item["primary_rank"]) for item in items if item["primary_rank"] is not None]
    return {
        "configuration": config,
        "query_count": len(items),
        "recall_at": _average_at(items, "recall_at"),
        "mrr": _average_scalar(items, "reciprocal_rank"),
        "ndcg_at": _average_at(items, "ndcg_at"),
        "document_recall_at": _average_at(items, "document_recall_at"),
        "conversation_recall_at": _average_at(items, "conversation_recall_at"),
        "mean_primary_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "median_primary_rank": statistics.median(ranks) if ranks else None,
        "primary_miss_count": len(items) - len(ranks),
    }


def _best_configurations(config_metrics: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "mrr": _best_config(config_metrics, lambda item: item["mrr"]),
        "recall_at_10": _best_config(config_metrics, lambda item: item["recall_at"]["10"]),
        "ndcg_at_10": _best_config(config_metrics, lambda item: item["ndcg_at"]["10"]),
    }


def _best_config(config_metrics: list[dict[str, Any]], metric_fn) -> str:
    best = max(
        config_metrics,
        key=lambda item: (
            float(metric_fn(item)),
            float(item["configuration"]["alpha"]),
            _reverse_lex_key(str(item["configuration"]["id"])),
        ),
    )
    return str(best["configuration"]["id"])


def _reverse_lex_key(value: str) -> tuple[int, ...]:
    return tuple(-ord(char) for char in value)


def _dense_sparse_class(dense: dict[str, Any], sparse: dict[str, Any]) -> str:
    dense_rr = float(dense["reciprocal_rank"])
    sparse_rr = float(sparse["reciprocal_rank"])
    if dense["recall_at"]["10"] == 0 and dense["document_recall_at"]["10"] == 1:
        return "dense_block_miss_document_hit"
    if sparse["recall_at"]["10"] == 0 and sparse["document_recall_at"]["10"] == 1:
        return "sparse_block_miss_document_hit"
    if dense_rr == 0.0 and sparse_rr == 0.0:
        return "both_miss"
    if dense_rr > sparse_rr:
        base = "dense_win"
    elif sparse_rr > dense_rr:
        base = "sparse_win"
    else:
        base = "tie"
    return base


def _hybrid_class(best_hybrid_rr: float, dense_rr: float, sparse_rr: float) -> str:
    best_endpoint = max(dense_rr, sparse_rr)
    if best_hybrid_rr > dense_rr and best_hybrid_rr > sparse_rr:
        return "hybrid_beats_both"
    if best_hybrid_rr > dense_rr and best_hybrid_rr <= sparse_rr:
        return "hybrid_beats_dense_only"
    if best_hybrid_rr > sparse_rr and best_hybrid_rr <= dense_rr:
        return "hybrid_beats_sparse_only"
    if best_hybrid_rr == best_endpoint:
        return "hybrid_equals_best_endpoint"
    return "hybrid_worse_than_best_endpoint"


def _load_and_validate_evaluation_inputs(*, run_dir: Path, dataset_path: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"manifest.json does not exist: {manifest_path}")
    if not results_path.exists():
        raise ValueError(f"results.jsonl does not exist: {results_path}")
    if not dataset_path.exists():
        raise ValueError(f"dataset does not exist: {dataset_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(f"unsupported run schema_version: {manifest.get('schema_version')}")
    if manifest.get("status") != "completed":
        raise ValueError(f"run manifest status must be completed, got {manifest.get('status')}")
    if manifest.get("completed_queries") != 120:
        raise ValueError(f"completed_queries must be 120, got {manifest.get('completed_queries')}")
    if manifest.get("failed_queries") != 0:
        raise ValueError(f"failed_queries must be 0, got {manifest.get('failed_queries')}")
    dataset_sha256 = _sha256(dataset_path)
    if manifest.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("dataset SHA256 does not match run manifest")
    dataset_records = _load_dataset_records(dataset_path)
    dataset_by_id = {record["id"]: record for record in dataset_records}
    if len(dataset_by_id) != len(dataset_records):
        raise ValueError("dataset contains duplicate query ids")
    configurations = manifest.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != 7:
        raise ValueError("run manifest must contain exactly 7 ranking configurations")
    config_by_id = {config["id"]: config for config in configurations}
    if len(config_by_id) != 7:
        raise ValueError("run manifest contains duplicate ranking configuration ids")

    results = _load_result_records(results_path)
    expected_pairs = {(query_id, config_id) for query_id in dataset_by_id for config_id in config_by_id}
    seen_pairs: set[tuple[str, str]] = set()
    for record in results:
        if record.get("schema_version") != QUERY_RESULT_SCHEMA_VERSION:
            raise ValueError(f"{record.get('query_id')}: unsupported result schema_version")
        query_id = record.get("query_id")
        if query_id not in dataset_by_id:
            raise ValueError(f"result query_id is not in dataset: {query_id}")
        configuration = record.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(f"{query_id}: configuration must be an object")
        config_id = configuration.get("id")
        manifest_config = config_by_id.get(config_id)
        if manifest_config is None:
            raise ValueError(f"{query_id}: result configuration is not in manifest: {config_id}")
        if configuration.get("alpha") != manifest_config.get("alpha") or configuration.get("beta") != manifest_config.get("beta"):
            raise ValueError(f"{query_id}: result configuration weights do not match manifest")
        pair = (query_id, config_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate query/configuration result: {query_id} {config_id}")
        seen_pairs.add(pair)
        _validate_result_record_against_dataset(record, dataset_by_id[query_id])
    missing = expected_pairs - seen_pairs
    extra = seen_pairs - expected_pairs
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(f"missing query/configuration records: {sample}")
    if extra:
        sample = sorted(extra)[:3]
        raise ValueError(f"unexpected query/configuration records: {sample}")
    return {
        "manifest": manifest,
        "dataset_sha256": dataset_sha256,
        "dataset_records": dataset_records,
        "dataset_by_id": dataset_by_id,
        "configurations": configurations,
        "results": results,
    }


def _load_and_validate_analysis_inputs(*, evaluation_dir: Path, dataset_path: Path) -> dict[str, Any]:
    manifest_path = evaluation_dir / "manifest.json"
    query_metrics_path = evaluation_dir / "query_metrics.jsonl"
    summary_path = evaluation_dir / "summary.json"
    if not manifest_path.exists():
        raise ValueError(f"evaluation manifest does not exist: {manifest_path}")
    if not query_metrics_path.exists():
        raise ValueError(f"query_metrics.jsonl does not exist: {query_metrics_path}")
    if not summary_path.exists():
        raise ValueError(f"summary.json does not exist: {summary_path}")
    if not dataset_path.exists():
        raise ValueError(f"dataset does not exist: {dataset_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EVALUATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation manifest schema_version: {manifest.get('schema_version')}")
    if manifest.get("status") != "completed":
        raise ValueError(f"evaluation manifest status must be completed, got {manifest.get('status')}")
    dataset_sha256 = _sha256(dataset_path)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dataset SHA256 does not match evaluation manifest")
    if manifest.get("query_count") != 120:
        raise ValueError(f"evaluation query_count must be 120, got {manifest.get('query_count')}")
    if manifest.get("configuration_count") != 7:
        raise ValueError(f"evaluation configuration_count must be 7, got {manifest.get('configuration_count')}")
    if manifest.get("query_metric_records") != 840:
        raise ValueError(f"query_metric_records must be 840, got {manifest.get('query_metric_records')}")
    dataset_records = _load_dataset_records(dataset_path)
    dataset_by_id = {record["id"]: record for record in dataset_records}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported summary schema_version: {summary.get('schema_version')}")
    configurations = [item["configuration"] for item in summary["metrics"]]
    query_metrics = _load_query_metric_records(query_metrics_path)
    _validate_query_metrics_against_dataset(query_metrics, dataset_by_id, configurations)
    recomputed = _aggregate_query_metrics(
        run_manifest={"run_id": summary["run_id"], "completed_queries": len(dataset_records)},
        dataset_sha256=dataset_sha256,
        query_metrics=query_metrics,
        configurations=configurations,
    )
    _assert_summary_close(recomputed, summary)
    return {
        "manifest": manifest,
        "summary": summary,
        "dataset_sha256": dataset_sha256,
        "dataset_records": dataset_records,
        "dataset_by_id": dataset_by_id,
        "query_metrics": query_metrics,
        "configurations": configurations,
    }


def _load_query_metric_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"query_metrics.jsonl line {line_no}: invalid JSON: {exc}") from exc
        if record.get("schema_version") != QUERY_METRICS_SCHEMA_VERSION:
            raise ValueError(f"query_metrics.jsonl line {line_no}: unsupported schema_version")
        records.append(record)
    return records


def _validate_query_metrics_against_dataset(
    query_metrics: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> None:
    if len(query_metrics) != 840:
        raise ValueError(f"query_metrics must contain 840 records, got {len(query_metrics)}")
    config_by_id = {config["id"]: config for config in configurations}
    expected_pairs = {(query_id, config_id) for query_id in dataset_by_id for config_id in config_by_id}
    seen: set[tuple[str, str]] = set()
    for record in query_metrics:
        query_id = record["query_id"]
        config_id = record["configuration"]["id"]
        if query_id not in dataset_by_id:
            raise ValueError(f"query metric query_id not in dataset: {query_id}")
        dataset_record = dataset_by_id[query_id]
        for field in ("query_type", "language", "source_language", "topic"):
            if record.get(field) != dataset_record.get(field):
                raise ValueError(f"{query_id}: {field} mismatch between query metrics and dataset")
        if config_id not in config_by_id:
            raise ValueError(f"{query_id}: configuration not in summary: {config_id}")
        if record["configuration"] != config_by_id[config_id]:
            raise ValueError(f"{query_id}: configuration payload mismatch")
        pair = (query_id, config_id)
        if pair in seen:
            raise ValueError(f"duplicate query/configuration metric: {query_id} {config_id}")
        seen.add(pair)
    missing = expected_pairs - seen
    if missing:
        raise ValueError(f"missing query/configuration metrics: {sorted(missing)[:3]}")


def _assert_summary_close(actual: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-9) -> None:
    if actual["query_count"] != expected["query_count"] or actual["configuration_count"] != expected["configuration_count"]:
        raise ValueError("summary counts do not match recomputed metrics")
    expected_by_id = {item["configuration"]["id"]: item for item in expected["metrics"]}
    for actual_item in actual["metrics"]:
        expected_item = expected_by_id.get(actual_item["configuration"]["id"])
        if expected_item is None:
            raise ValueError(f"summary missing configuration {actual_item['configuration']['id']}")
        for field in ("recall_at", "primary_recall_at", "ndcg_at", "document_recall_at", "conversation_recall_at"):
            for key, value in actual_item[field].items():
                if abs(float(value) - float(expected_item[field][key])) > tolerance:
                    raise ValueError(f"summary mismatch for {actual_item['configuration']['id']} {field}@{key}")
        for field in ("mrr", "primary_mrr"):
            if abs(float(actual_item[field]) - float(expected_item[field])) > tolerance:
                raise ValueError(f"summary mismatch for {actual_item['configuration']['id']} {field}")


def _load_result_records(results_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"results.jsonl line {line_no}: invalid JSON: {exc}") from exc
    return records


def _validate_result_record_against_dataset(record: dict[str, Any], dataset_record: dict[str, Any]) -> None:
    query_id = record["query_id"]
    expected_from_dataset = {item["block_id"]: int(item["relevance"]) for item in dataset_record["expected"]}
    expected_from_result = {item.get("block_id"): item for item in record.get("expected", []) if isinstance(item, dict)}
    if set(expected_from_result) != set(expected_from_dataset):
        raise ValueError(f"{query_id}: result expected blocks do not match dataset")
    for block_id, relevance in expected_from_dataset.items():
        item = expected_from_result[block_id]
        if item.get("relevance") != relevance:
            raise ValueError(f"{query_id}: relevance mismatch for expected block {block_id}")
        rank = item.get("rank")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"{query_id}: expected block {block_id} rank must be a positive integer")
    top_results = record.get("top_results")
    if not isinstance(top_results, list):
        raise ValueError(f"{query_id}: top_results must be a list")
    if len(top_results) > 20:
        raise ValueError(f"{query_id}: top_results contains more than top-k records")
    ranks = [item.get("rank") for item in top_results if isinstance(item, dict)]
    if len(ranks) != len(top_results) or any(not isinstance(rank, int) or rank <= 0 for rank in ranks):
        raise ValueError(f"{query_id}: top_results ranks must be positive integers")
    if ranks != sorted(ranks):
        raise ValueError(f"{query_id}: top_results must be sorted by rank")


def _ndcg_at(expected_relevance: dict[str, int], expected_ranks: dict[str, int | None], k: int) -> float:
    dcg = 0.0
    for block_id, relevance in expected_relevance.items():
        rank = expected_ranks.get(block_id)
        if rank is not None and rank <= k:
            dcg += _dcg_gain(relevance, rank)
    ideal_relevances = sorted(expected_relevance.values(), reverse=True)
    idcg = sum(_dcg_gain(relevance, rank) for rank, relevance in enumerate(ideal_relevances[:k], start=1))
    return 0.0 if idcg == 0.0 else dcg / idcg


def _dcg_gain(relevance: int, rank: int) -> float:
    return (float(2**relevance) - 1.0) / math.log2(rank + 1)


def _aggregate_query_metrics(
    *,
    run_manifest: dict[str, Any],
    dataset_sha256: str,
    query_metrics: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics_by_config: dict[str, list[dict[str, Any]]] = {config["id"]: [] for config in configurations}
    for item in query_metrics:
        metrics_by_config[item["configuration"]["id"]].append(item)
    aggregate_metrics: list[dict[str, Any]] = []
    for config in configurations:
        items = metrics_by_config[config["id"]]
        query_count = len(items)
        if query_count == 0:
            raise ValueError(f"configuration has no query metrics: {config['id']}")
        aggregate_metrics.append(
            {
                "configuration": config,
                "query_count": query_count,
                "recall_at": _average_at(items, "recall_at"),
                "primary_recall_at": _average_at(items, "primary_recall_at"),
                "mrr": _average_scalar(items, "reciprocal_rank"),
                "primary_mrr": _average_scalar(items, "primary_reciprocal_rank"),
                "ndcg_at": _average_at(items, "ndcg_at"),
                "document_recall_at": _average_at(items, "document_recall_at"),
                "conversation_recall_at": _average_at(items, "conversation_recall_at"),
            }
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "run_id": run_manifest["run_id"],
        "dataset_sha256": dataset_sha256,
        "query_count": run_manifest["completed_queries"],
        "configuration_count": len(configurations),
        "metrics": aggregate_metrics,
    }


def _average_at(items: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        str(k): sum(float(item[field][str(k)]) for item in items) / len(items)
        for k in METRIC_K_VALUES
    }


def _average_scalar(items: list[dict[str, Any]], field: str) -> float:
    return sum(float(item[field]) for item in items) / len(items)


def _render_evaluation_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Direct Retrieval Evaluation",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Query count: {summary['query_count']}",
        f"- Configuration count: {summary['configuration_count']}",
        "",
        "## Retrieval Metrics",
        "",
        "| Configuration | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 | Doc R@10 | Conv R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["metrics"]:
        config_id = item["configuration"]["id"]
        lines.append(
            "| "
            + " | ".join(
                [
                    config_id,
                    _fmt(item["recall_at"]["1"]),
                    _fmt(item["recall_at"]["5"]),
                    _fmt(item["recall_at"]["10"]),
                    _fmt(item["recall_at"]["20"]),
                    _fmt(item["mrr"]),
                    _fmt(item["ndcg_at"]["10"]),
                    _fmt(item["document_recall_at"]["10"]),
                    _fmt(item["conversation_recall_at"]["10"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Primary Metrics",
        "",
        "| Configuration | Primary R@1 | Primary R@5 | Primary R@10 | Primary R@20 | Primary MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["metrics"]:
        config_id = item["configuration"]["id"]
        lines.append(
            "| "
            + " | ".join(
                [
                    config_id,
                    _fmt(item["primary_recall_at"]["1"]),
                    _fmt(item["primary_recall_at"]["5"]),
                    _fmt(item["primary_recall_at"]["10"]),
                    _fmt(item["primary_recall_at"]["20"]),
                    _fmt(item["primary_mrr"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _render_analysis_report(summary: dict[str, Any], breakdowns: dict[str, Any], pairwise: list[dict[str, Any]]) -> str:
    lines = [
        "# Direct Retrieval Breakdown Analysis",
        "",
        "## Overall",
        "",
        "| Configuration | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 | Doc R@10 | Conv R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["metrics"]:
        lines.append(_overall_row(item))
    lines += ["", "## By query type", ""]
    lines += _breakdown_table(breakdowns, "query_type", "Query type")
    lines += ["", "## By language", "", "### Query language", ""]
    lines += _breakdown_table(breakdowns, "language", "Query language")
    lines += ["", "### Source language", ""]
    lines += _breakdown_table(breakdowns, "source_language", "Source language")
    lines += ["", "### Language direction", ""]
    lines += _breakdown_table(breakdowns, "language_direction", "Language direction")
    lines += ["", "## By topic", ""]
    lines += _breakdown_table(breakdowns, "topic", "Topic")

    dense_counts = Counter(item["dense_vs_sparse_class"] for item in pairwise)
    lines += [
        "",
        "## Dense vs sparse",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for label in ("dense_win", "sparse_win", "tie", "both_miss", "dense_block_miss_document_hit", "sparse_block_miss_document_hit"):
        lines.append(f"| {label} | {dense_counts.get(label, 0)} |")

    hybrid_counts = Counter(item["hybrid_comparison_class"] for item in pairwise)
    lines += [
        "",
        "## Hybrid contribution",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for label in (
        "hybrid_beats_both",
        "hybrid_beats_dense_only",
        "hybrid_beats_sparse_only",
        "hybrid_equals_best_endpoint",
        "hybrid_worse_than_best_endpoint",
    ):
        lines.append(f"| {label} | {hybrid_counts.get(label, 0)} |")

    sparse_better = sorted([item for item in pairwise if item["rr_delta"] > 0], key=lambda item: item["rr_delta"], reverse=True)[:15]
    dense_better = sorted([item for item in pairwise if item["rr_delta"] < 0], key=lambda item: item["rr_delta"])[:15]
    lines += ["", "## Largest disagreements", "", "### Sparse over dense", ""]
    lines += _disagreement_table(sparse_better)
    lines += ["", "### Dense over sparse", ""]
    lines += _disagreement_table(dense_better)

    block_miss_doc_hit = [
        item
        for item in pairwise
        if (item["dense_recall_at_10"] == 0 and item["dense_document_recall_at_10"] == 1)
        or (item["sparse_recall_at_10"] == 0 and item["sparse_document_recall_at_10"] == 1)
    ][:20]
    lines += ["", "## Block miss but document hit", ""]
    lines += _block_miss_table(block_miss_doc_hit)
    return "\n".join(lines) + "\n"


def _overall_row(item: dict[str, Any]) -> str:
    return (
        f"| {item['configuration']['id']} | {_fmt(item['recall_at']['1'])} | {_fmt(item['recall_at']['5'])} | "
        f"{_fmt(item['recall_at']['10'])} | {_fmt(item['recall_at']['20'])} | {_fmt(item['mrr'])} | "
        f"{_fmt(item['ndcg_at']['10'])} | {_fmt(item['document_recall_at']['10'])} | {_fmt(item['conversation_recall_at']['10'])} |"
    )


def _breakdown_table(breakdowns: dict[str, Any], dimension: str, label: str) -> list[str]:
    lines = [
        f"| {label} | N | Configuration | R@1 | R@5 | R@10 | MRR | nDCG@10 | Doc R@10 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slice_item in breakdowns["dimensions"][dimension]:
        value = slice_item["value"] + (" (small sample)" if slice_item["small_sample"] else "")
        for config in slice_item["configurations"]:
            lines.append(
                f"| {value} | {slice_item['query_count']} | {config['configuration']['id']} | "
                f"{_fmt(config['recall_at']['1'])} | {_fmt(config['recall_at']['5'])} | {_fmt(config['recall_at']['10'])} | "
                f"{_fmt(config['mrr'])} | {_fmt(config['ndcg_at']['10'])} | {_fmt(config['document_recall_at']['10'])} |"
            )
    return lines


def _disagreement_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Query ID | Query type | Topic | Direction | Dense rank | Sparse rank | Dense RR | Sparse RR | RR delta |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        lines.append(
            f"| {item['query_id']} | {item['query_type']} | {item['topic']} | {item['language_direction']} | "
            f"{_rank_display(item['dense_primary_rank'])} | {_rank_display(item['sparse_primary_rank'])} | "
            f"{_fmt(item['dense_reciprocal_rank'])} | {_fmt(item['sparse_reciprocal_rank'])} | {_fmt(item['rr_delta'])} |"
        )
    return lines


def _block_miss_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Query ID | Query type | Topic | Direction | Dense rank | Sparse rank | Dense Doc R@10 | Sparse Doc R@10 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for item in items:
        lines.append(
            f"| {item['query_id']} | {item['query_type']} | {item['topic']} | {item['language_direction']} | "
            f"{_rank_display(item['dense_primary_rank'])} | {_rank_display(item['sparse_primary_rank'])} | "
            f"{item['dense_document_recall_at_10']} | {item['sparse_document_recall_at_10']} |"
        )
    return lines


def _rank_display(rank: Any) -> str:
    return "" if rank is None else str(rank)


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


def _obvious_query_language_mismatch(query: str, language: str) -> str | None:
    if language == "mixed":
        return None
    cyrillic_count = sum(1 for char in query if _is_cyrillic(char))
    latin_words = _latin_words(query)
    russian_words = _russian_words(query)
    if language == "ru":
        if cyrillic_count == 0 and _looks_like_english_sentence(latin_words, query):
            return "query looks like an English sentence but language=ru"
    if language == "en":
        if russian_words and len(russian_words) >= 3:
            return "query looks like a Russian sentence but language=en"
    return None


def _is_cyrillic(char: str) -> bool:
    return ("А" <= char <= "я") or char in "Ёё"


def _latin_words(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def _russian_words(text: str) -> list[str]:
    import re

    return re.findall(r"[А-Яа-яЁё]{2,}", text)


def _looks_like_english_sentence(latin_words: list[str], query: str) -> bool:
    if len(latin_words) < 5:
        return False
    lower = {word.lower() for word in latin_words}
    function_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "can",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "or",
        "should",
        "that",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "without",
    }
    if len(lower & function_words) >= 2:
        return True
    return query.rstrip().endswith("?") and bool(lower & {"what", "why", "how", "when", "where", "which"})


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
                rc.id AS block_id,
                sd.relative_path AS source_path,
                c.id AS conversation_id,
                m.id AS message_id
            FROM retrieval_chunks rc
            JOIN blocks b ON b.id = rc.block_id
            JOIN messages m ON m.id = b.message_id
            JOIN conversations c ON c.id = m.conversation_id
            JOIN source_documents sd ON sd.id = c.source_document_id
            WHERE rc.id IN ({placeholders})
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
