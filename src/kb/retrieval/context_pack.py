from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.retrieval.hybrid_search import _cosine, _sparse_overlap
from kb.storage.sqlite_store import SQLiteStore, init_db


@dataclass(frozen=True)
class ContextPackOptions:
    budget_tokens: int = 4000
    direct_limit: int = 10
    node_limit: int = 5
    node_member_limit: int = 5
    neighbor_limit: int = 5
    retrieval_strategy: str = "auto"


def build_context_pack(
    *,
    db_path: Path,
    query: str,
    dense,
    sparse,
    dense_provider: str,
    sparse_provider: str,
    dense_model: str,
    sparse_model: str,
    sparse_top_k: int,
    include_low_interest: bool = False,
    project: str | None = None,
    options: ContextPackOptions = ContextPackOptions(),
    ensure_schema: bool = True,
    read_only: bool = False,
) -> dict[str, Any]:
    if options.retrieval_strategy not in {"auto", "basement", "semantic_groups"}:
        raise ValueError(f"Unsupported retrieval_strategy: {options.retrieval_strategy}")
    query_dense = dense.embed_query(query) if dense else None
    query_sparse = sparse.embed_query(query) if sparse else None

    if ensure_schema:
        init_db(db_path)
    candidates: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    with SQLiteStore(db_path, read_only=read_only) as store:
        capabilities = store.capabilities()
        strategy_used = _resolve_strategy(options.retrieval_strategy, capabilities)
        direct_hits = _score_blocks(
            store.searchable_knowledge_blocks(
                dense_model_name=dense.model_name if dense else None,
                dense_model_version=dense.model_version if dense else None,
                sparse_model_name=sparse.model_name if sparse else None,
                sparse_embedding_space_id=sparse.embedding_space_id if sparse else None,
                project=project,
                include_low_interest=include_low_interest,
            ),
            query_dense=query_dense,
            query_sparse=query_sparse,
            limit=options.direct_limit,
        )
        block_rows = store.blocks_by_ids([item["knowledge_block_id"] for item in direct_hits])
        for item in direct_hits:
            block = block_rows.get(item["knowledge_block_id"])
            if block:
                _add_candidate(candidates, block, item["score"], "query -> block direct")
                traces.append({"path": "query -> block direct", "block_id": item["knowledge_block_id"], "score": item["score"]})

        node_types = ["semantic_group"] if strategy_used == "semantic_groups" else None
        node_hits = _score_nodes(
            store.semantic_nodes_for_search(project=project, node_types=node_types),
            query_dense=query_dense,
            query_sparse=query_sparse,
            limit=options.node_limit,
        )
        for node_hit in node_hits:
            members = store.semantic_node_member_blocks(
                node_hit["node_id"],
                limit=options.node_member_limit,
                include_low_interest=include_low_interest,
            )
            for block in members:
                score = node_hit["score"] * float(block["membership_weight"])
                _add_candidate(candidates, block, score, f"query -> node:{node_hit['node_type']} -> member block")
                traces.append(
                    {
                        "path": f"query -> node:{node_hit['node_type']} -> member block",
                        "node_id": node_hit["node_id"],
                        "block_id": block["knowledge_block_id"],
                        "score": score,
                    }
                )

        seed_ids = list(candidates)
        neighbors = store.neighbor_blocks(seed_ids, limit=options.neighbor_limit, include_low_interest=include_low_interest)
        for block in neighbors:
            score = float(block["edge_weight"]) * 0.5
            _add_candidate(candidates, block, score, "query -> block -> neighbor")
            traces.append(
                {
                    "path": "query -> block -> neighbor",
                    "from_block_id": block["from_block_id"],
                    "block_id": block["knowledge_block_id"],
                    "score": score,
                }
            )

    selected = _select_with_budget(candidates.values(), options.budget_tokens)
    return {
        "query": query,
        "budget_tokens": options.budget_tokens,
        "retrieval_strategy_requested": options.retrieval_strategy,
        "retrieval_strategy_used": strategy_used,
        "db_capabilities": capabilities.as_dict(),
        "selected_blocks": selected,
        "source_trace": traces,
        "scores": [
            {"block_id": item["block_id"], "score": item["score"], "reason": item["reason"]}
            for item in selected
        ],
        "explanation": _explanation(strategy_used),
    }


def _resolve_strategy(strategy: str, capabilities) -> str:
    if strategy == "auto":
        if capabilities.has_semantic_groups and capabilities.has_group_embeddings:
            return "semantic_groups"
        return "basement"
    if strategy == "semantic_groups" and not (capabilities.has_semantic_groups and capabilities.has_group_embeddings):
        return "basement"
    return strategy


def _explanation(strategy: str) -> str:
    if strategy == "semantic_groups":
        return (
            "Semantic group nodes are searched, their members are expanded, direct block hits are kept as fallback, "
            "then graph neighbors are expanded and deduplicated within the token budget."
        )
    return (
        "Direct block hits are kept, deterministic semantic node members are added, then graph neighbors are expanded "
        "and deduplicated within the token budget."
    )


def _score_blocks(
    blocks: list[dict[str, Any]],
    *,
    query_dense: list[float] | None,
    query_sparse: dict[str, float] | None,
    limit: int,
) -> list[dict[str, Any]]:
    scored = []
    for block in blocks:
        dense_score = _cosine(query_dense, block["dense_vector"]) if query_dense is not None and block["dense_vector"] is not None else 0.0
        sparse_score, shared = _sparse_overlap(query_sparse, block["sparse_terms"]) if query_sparse is not None else (0.0, [])
        scored.append(
            {
                **block,
                "score": 0.65 * dense_score + 0.35 * sparse_score,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "shared_terms": shared[:10],
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def _score_nodes(
    nodes: list[dict[str, Any]],
    *,
    query_dense: list[float] | None,
    query_sparse: dict[str, float] | None,
    limit: int,
) -> list[dict[str, Any]]:
    scored = []
    for node in nodes:
        dense_score = _cosine(query_dense, node["dense_vector"]) if query_dense is not None and node["dense_vector"] is not None else 0.0
        sparse_score, shared = _sparse_overlap(query_sparse, node["sparse_terms"]) if query_sparse is not None else (0.0, [])
        score = 0.65 * dense_score + 0.35 * sparse_score
        scored.append({**node, "score": score, "dense_score": dense_score, "sparse_score": sparse_score, "shared_terms": shared[:10]})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def _add_candidate(candidates: dict[str, dict[str, Any]], block: dict[str, Any], score: float, reason: str) -> None:
    block_id = block["knowledge_block_id"]
    existing = candidates.get(block_id)
    if existing and existing["score"] >= score:
        return
    candidates[block_id] = {
        "block_id": block_id,
        "score": score,
        "reason": reason,
        "source_path": block["source_path"],
        "project": block["project_id"],
        "conversation_id": block["conversation_id"],
        "conversation_title": block["conversation_title"],
        "message_id": block["message_id"],
        "role": block["role"],
        "block_type": block["block_type"],
        "interest_tier": block["interest_tier"],
        "token_count_estimate": block["token_count_estimate"],
        "text": block["text_for_display"],
    }


def _select_with_budget(items, budget_tokens: int) -> list[dict[str, Any]]:
    selected = []
    used = 0
    for item in sorted(items, key=lambda value: value["score"], reverse=True):
        tokens = int(item["token_count_estimate"] or 1)
        if used + tokens > budget_tokens and selected:
            continue
        selected.append(item)
        used += tokens
        if used >= budget_tokens:
            break
    return selected
