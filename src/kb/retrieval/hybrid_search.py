from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from kb.cli import _build_dense_provider, _build_sparse_provider
from kb.storage.sqlite_store import SQLiteStore


QUERY_RESULT_SCHEMA_VERSION = "kb.query.result.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the local ChatGPT export knowledge DB.")
    sub = parser.add_subparsers(dest="command", required=True)

    query = sub.add_parser("query", help="Run hybrid dense+sparse block search.")
    query.add_argument("query")
    query.add_argument("--db", required=True)
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--alpha", type=float, default=0.65)
    query.add_argument("--beta", type=float, default=0.35)
    query.add_argument("--project")
    query.add_argument("--dense-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    query.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    query.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    query.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
    query.add_argument("--sparse-top-k", type=int, default=128)
    query.add_argument("--include-low-interest", action="store_true")
    query.add_argument("--diagnostics", action="store_true")
    query.add_argument("--output", help="Write the full JSON retrieval result to this path.")
    query.add_argument("--json", action="store_true", dest="json_output")

    context = sub.add_parser("context", help="Build a traceable context pack.")
    context.add_argument("query")
    context.add_argument("--db", required=True)
    context.add_argument("--budget-tokens", type=int, default=4000)
    context.add_argument("--project")
    context.add_argument("--direct-limit", type=int, default=10)
    context.add_argument("--node-limit", type=int, default=5)
    context.add_argument("--node-member-limit", type=int, default=5)
    context.add_argument("--neighbor-limit", type=int, default=5)
    context.add_argument("--retrieval-strategy", choices=["auto", "basement", "semantic_groups"], default="auto")
    context.add_argument("--dense-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    context.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    context.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    context.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
    context.add_argument("--sparse-top-k", type=int, default=128)
    context.add_argument("--include-low-interest", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "query":
        results = hybrid_query(
            db_path=Path(args.db).expanduser(),
            query=args.query,
            limit=args.limit,
            alpha=args.alpha,
            beta=args.beta,
            project=args.project,
            dense_provider=args.dense_provider,
            sparse_provider=args.sparse_provider,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            sparse_top_k=args.sparse_top_k,
            include_low_interest=args.include_low_interest,
            diagnostics=args.diagnostics,
        )
        if args.output:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        if args.json_output:
            print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            _print_results(results)
    elif args.command == "context":
        from kb.retrieval.context_pack import ContextPackOptions, build_context_pack

        dense = _build_dense_provider(args.dense_provider, args.dense_model)
        sparse = _build_sparse_provider(args.sparse_provider, args.sparse_model, args.sparse_top_k)
        payload = build_context_pack(
            db_path=Path(args.db).expanduser(),
            query=args.query,
            dense=dense,
            sparse=sparse,
            dense_provider=args.dense_provider,
            sparse_provider=args.sparse_provider,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            sparse_top_k=args.sparse_top_k,
            include_low_interest=args.include_low_interest,
            project=args.project,
            options=ContextPackOptions(
                budget_tokens=args.budget_tokens,
                direct_limit=args.direct_limit,
                node_limit=args.node_limit,
                node_member_limit=args.node_member_limit,
                neighbor_limit=args.neighbor_limit,
                retrieval_strategy=args.retrieval_strategy,
            ),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def hybrid_query(
    *,
    db_path: Path,
    query: str,
    limit: int = 10,
    alpha: float = 0.65,
    beta: float = 0.35,
    project: str | None = None,
    dense_provider: str = "sentence-transformers",
    sparse_provider: str = "sentence-transformers",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    sparse_top_k: int = 128,
    include_low_interest: bool = False,
    diagnostics: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if limit <= 0:
        raise ValueError("--limit must be positive.")
    provider_started = time.perf_counter()
    dense = _build_dense_provider(dense_provider, dense_model) if dense_provider != "none" else None
    sparse = _build_sparse_provider(sparse_provider, sparse_model, sparse_top_k) if sparse_provider != "none" else None
    if dense is None and sparse is None:
        raise ValueError("At least one retrieval provider must be enabled.")
    providers_loaded = time.perf_counter()

    query_encoding_started = time.perf_counter()
    query_dense = dense.embed_query(query) if dense else None
    query_sparse = sparse.embed_query(query) if sparse else None
    query_encoded = time.perf_counter()

    db_started = time.perf_counter()
    with SQLiteStore(db_path, read_only=True) as store:
        rows = store.searchable_knowledge_blocks(
            dense_model_name=dense.model_name if dense else None,
            dense_model_version=dense.model_version if dense else None,
            sparse_model_name=sparse.model_name if sparse else None,
            sparse_embedding_space_id=sparse.embedding_space_id if sparse else None,
            project=project,
            include_low_interest=include_low_interest,
        )
        candidates_loaded = time.perf_counter()
        representation_counts = store.retrieval_representation_counts(
            dense_model_name=dense.model_name if dense else None,
            sparse_model_name=sparse.model_name if sparse else None,
            project=project,
            include_low_interest=include_low_interest,
        )
    db_finished = time.perf_counter()

    scoring_started = time.perf_counter()
    scored = []
    dense_scores: list[float] = []
    sparse_scores: list[float] = []
    dense_candidate_count = 0
    sparse_candidate_count = 0
    dense_dimension_mismatches = 0
    for row in rows:
        dense_score = 0.0
        if query_dense is not None and row["dense_vector"] is not None:
            dense_candidate_count += 1
            if len(query_dense) != len(row["dense_vector"]):
                dense_dimension_mismatches += 1
            else:
                dense_score = _cosine(query_dense, row["dense_vector"])
                dense_scores.append(dense_score)
        sparse_score = 0.0
        overlapping_terms: list[str] = []
        if query_sparse is not None and row["sparse_terms"]:
            sparse_candidate_count += 1
            sparse_score, overlapping_terms = _sparse_overlap(query_sparse, row["sparse_terms"])
            sparse_scores.append(sparse_score)
        final_score = alpha * dense_score + beta * sparse_score
        scored.append(
            {
                "block_id": row["knowledge_block_id"],
                "source_path": row["source_path"],
                "project": row["project_id"],
                "folder_kind": row["folder_kind"],
                "interest_tier": row["interest_tier"],
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "message_id": row["message_id"],
                "role": row["role"],
                "block_type": row["block_type"],
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "final_score": final_score,
                "overlapping_terms": overlapping_terms[:10],
                "preview": _preview(row["text_for_display"]),
            }
        )
    scored.sort(key=lambda item: item["final_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
    scoring_finished = time.perf_counter()
    selected = scored[:limit]
    finished = time.perf_counter()
    return {
        "schema_version": QUERY_RESULT_SCHEMA_VERSION,
        "run": {
            "retrieval_mode": "query",
            "db_path": str(db_path),
            "limit": limit,
            "project": project,
            "include_low_interest": include_low_interest,
            "dense_provider": dense_provider,
            "sparse_provider": sparse_provider,
            "sparse_top_k": sparse_top_k,
            "diagnostics_requested": diagnostics,
        },
        "query": query,
        "alpha": alpha,
        "beta": beta,
        "dense_model": dense.model_name if dense else None,
        "dense_embedding_space_id": dense.embedding_space_id if dense else None,
        "sparse_model": sparse.model_name if sparse else None,
        "sparse_embedding_space_id": sparse.embedding_space_id if sparse else None,
        "candidate_blocks": len(scored),
        "latency_ms": {
            "total": _elapsed_ms(started, finished),
            "provider_load": _elapsed_ms(provider_started, providers_loaded),
            "query_encoding": _elapsed_ms(query_encoding_started, query_encoded),
            "db_candidate_load": _elapsed_ms(db_started, candidates_loaded),
            "db_representation_counts": _elapsed_ms(candidates_loaded, db_finished),
            "scoring": _elapsed_ms(scoring_started, scoring_finished),
            "result_packaging": _elapsed_ms(scoring_finished, finished),
        },
        "results": selected,
        **(
            {
                "diagnostics": _diagnostics_payload(
                    dense_enabled=dense is not None,
                    sparse_enabled=sparse is not None,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    dense=dense,
                    sparse=sparse,
                    dense_candidate_count=dense_candidate_count,
                    sparse_candidate_count=sparse_candidate_count,
                    dense_dimension_mismatches=dense_dimension_mismatches,
                    dense_model_row_count=representation_counts["dense_model_rows"],
                    sparse_model_row_count=representation_counts["sparse_model_rows"],
                    dense_scores=dense_scores,
                    sparse_scores=sparse_scores,
                )
            }
            if diagnostics
            else {}
        ),
    }


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _sparse_overlap(query_terms: dict[str, float] | None, doc_terms: dict[str, float]) -> tuple[float, list[str]]:
    if not query_terms or not doc_terms:
        return 0.0, []
    shared = sorted(set(query_terms) & set(doc_terms))
    score = sum(float(query_terms[token]) * float(doc_terms[token]) for token in shared)
    query_norm = math.sqrt(sum(float(weight) * float(weight) for weight in query_terms.values()))
    doc_norm = math.sqrt(sum(float(weight) * float(weight) for weight in doc_terms.values()))
    if query_norm and doc_norm:
        score = score / (query_norm * doc_norm)
    ranked_terms = sorted(shared, key=lambda token: query_terms[token] * doc_terms[token], reverse=True)
    return score, ranked_terms


def _diagnostics_payload(
    *,
    dense_enabled: bool,
    sparse_enabled: bool,
    query_dense: list[float] | None,
    query_sparse: dict[str, float] | None,
    dense,
    sparse,
    dense_candidate_count: int,
    sparse_candidate_count: int,
    dense_dimension_mismatches: int,
    dense_model_row_count: int,
    sparse_model_row_count: int,
    dense_scores: list[float],
    sparse_scores: list[float],
) -> dict[str, Any]:
    return {
        "dense": {
            "enabled": dense_enabled,
            "query_embedding_created": query_dense is not None,
            "embedding_space_id": dense.embedding_space_id if dense else None,
            "query_dimension": len(query_dense) if query_dense is not None else 0,
            "query_norm": _vector_norm(query_dense) if query_dense is not None else 0.0,
            "candidate_blocks_with_vector": dense_candidate_count,
            "dimension_mismatches": dense_dimension_mismatches,
            "compatibility_mismatches": max(0, dense_model_row_count - dense_candidate_count),
            "nonzero_score_count": sum(1 for score in dense_scores if abs(score) > 1e-12),
            "min_score": min(dense_scores) if dense_scores else None,
            "max_score": max(dense_scores) if dense_scores else None,
            "status": _branch_status(dense_enabled, query_dense is not None, dense_candidate_count, dense_scores),
        },
        "sparse": {
            "enabled": sparse_enabled,
            "query_embedding_created": query_sparse is not None,
            "embedding_space_id": sparse.embedding_space_id if sparse else None,
            "model_name": sparse.model_name if sparse else None,
            "query_term_count": len(query_sparse) if query_sparse is not None else 0,
            "candidate_blocks_with_terms": sparse_candidate_count,
            "compatibility_mismatches": max(0, sparse_model_row_count - sparse_candidate_count),
            "nonzero_score_count": sum(1 for score in sparse_scores if abs(score) > 1e-12),
            "min_score": min(sparse_scores) if sparse_scores else None,
            "max_score": max(sparse_scores) if sparse_scores else None,
            "status": _branch_status(sparse_enabled, query_sparse is not None, sparse_candidate_count, sparse_scores),
        },
    }


def _branch_status(enabled: bool, query_created: bool, candidates: int, scores: list[float]) -> str:
    if not enabled:
        return "disabled"
    if not query_created:
        return "query_embedding_missing"
    if candidates == 0:
        return "no_compatible_document_representations"
    if not any(abs(score) > 1e-12 for score in scores):
        return "all_scores_zero"
    return "active"


def _vector_norm(vector: list[float] | None) -> float:
    if not vector:
        return 0.0
    return math.sqrt(sum(value * value for value in vector))


def _elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _preview(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _print_results(payload: dict[str, Any]) -> None:
    print(f"query: {payload['query']}")
    print(f"candidate_blocks: {payload['candidate_blocks']}")
    for idx, item in enumerate(payload["results"], start=1):
        print()
        print(
            f"{item.get('rank', idx)}. score={item['final_score']:.4f} dense={item['dense_score']:.4f} "
            f"sparse={item['sparse_score']:.4f}"
        )
        print(f"   source={item['source_path']}")
        print(f"   project={item['project']} conversation={item['conversation_title']} role={item['role']} block={item['block_type']}")
        if item["overlapping_terms"]:
            print(f"   overlap={', '.join(item['overlapping_terms'])}")
        print(f"   preview={item['preview']}")


if __name__ == "__main__":
    main()
