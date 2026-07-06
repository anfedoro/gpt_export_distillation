from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from kb.cli import _build_dense_provider, _build_sparse_provider
from kb.storage.sqlite_store import SQLiteStore, init_db


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
    query.add_argument("--sparse-model", default="naver/splade-cocondenser-ensembledistil")
    query.add_argument("--sparse-top-k", type=int, default=128)
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
    context.add_argument("--dense-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    context.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    context.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    context.add_argument("--sparse-model", default="naver/splade-cocondenser-ensembledistil")
    context.add_argument("--sparse-top-k", type=int, default=128)

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
        )
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
            project=args.project,
            options=ContextPackOptions(
                budget_tokens=args.budget_tokens,
                direct_limit=args.direct_limit,
                node_limit=args.node_limit,
                node_member_limit=args.node_member_limit,
                neighbor_limit=args.neighbor_limit,
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
    sparse_model: str = "naver/splade-cocondenser-ensembledistil",
    sparse_top_k: int = 128,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("--limit must be positive.")
    dense = _build_dense_provider(dense_provider, dense_model)
    sparse = _build_sparse_provider(sparse_provider, sparse_model, sparse_top_k)
    if dense is None and sparse is None:
        raise ValueError("At least one retrieval provider must be enabled.")

    query_dense = dense.embed_texts([query])[0] if dense else None
    query_sparse = sparse.embed_texts([query])[0] if sparse else None

    init_db(db_path)
    with SQLiteStore(db_path) as store:
        rows = store.searchable_knowledge_blocks(
            dense_model_name=dense.model_name if dense else None,
            dense_model_version=dense.model_version if dense else None,
            sparse_model_name=sparse.model_name if sparse else None,
            project=project,
        )

    scored = []
    for row in rows:
        dense_score = _cosine(query_dense, row["dense_vector"]) if query_dense is not None and row["dense_vector"] is not None else 0.0
        sparse_score, overlapping_terms = _sparse_overlap(query_sparse, row["sparse_terms"]) if query_sparse is not None else (0.0, [])
        final_score = alpha * dense_score + beta * sparse_score
        scored.append(
            {
                "block_id": row["knowledge_block_id"],
                "source_path": row["source_path"],
                "project": row["project_id"],
                "folder_kind": row["folder_kind"],
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
    return {
        "query": query,
        "alpha": alpha,
        "beta": beta,
        "dense_model": dense.model_name if dense else None,
        "sparse_model": sparse.model_name if sparse else None,
        "candidate_blocks": len(scored),
        "results": scored[:limit],
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


def _preview(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _print_results(payload: dict[str, Any]) -> None:
    print(f"query: {payload['query']}")
    print(f"candidate_blocks: {payload['candidate_blocks']}")
    for idx, item in enumerate(payload["results"], start=1):
        print()
        print(
            f"{idx}. score={item['final_score']:.4f} dense={item['dense_score']:.4f} "
            f"sparse={item['sparse_score']:.4f}"
        )
        print(f"   source={item['source_path']}")
        print(f"   project={item['project']} conversation={item['conversation_title']} role={item['role']} block={item['block_type']}")
        if item["overlapping_terms"]:
            print(f"   overlap={', '.join(item['overlapping_terms'])}")
        print(f"   preview={item['preview']}")


if __name__ == "__main__":
    main()
