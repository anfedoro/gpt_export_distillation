from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from kb.model.ids import stable_id
from kb.storage.sqlite_store import SQLiteStore


POLICY_VERSION = "similarity-edges-v0"


@dataclass(frozen=True)
class EdgeBuildStats:
    edges_created: int
    groups_processed: int
    candidate_pairs: int


def build_similarity_edges(
    store: SQLiteStore,
    *,
    scope: str = "project",
    top_k: int = 10,
    include_dense: bool = True,
    include_sparse: bool = True,
) -> EdgeBuildStats:
    if scope not in {"conversation", "project", "attachment"}:
        raise ValueError(f"Unsupported edge scope: {scope}")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    groups = store.edge_candidate_groups(scope=scope)
    edges: list[dict[str, Any]] = []
    candidate_pairs = 0
    for members in groups.values():
        ordered = sorted(members, key=lambda item: (item.get("conversation_id") or "", item.get("message_ordinal") or 0, item.get("block_ordinal") or 0, item["knowledge_block_id"]))
        edges.extend(_temporal_edges(ordered))
        pair_scores = []
        for left, right in combinations(ordered, 2):
            candidate_pairs += 1
            dense_score = _cosine(left["dense_vector"], right["dense_vector"]) if include_dense else None
            sparse_score, shared_terms = _sparse_similarity(left["sparse_terms"], right["sparse_terms"]) if include_sparse else (None, [])
            hybrid_score = _hybrid_score(dense_score, sparse_score)
            pair_scores.append((hybrid_score, dense_score, sparse_score, shared_terms, left, right))
        pair_scores.sort(key=lambda item: item[0], reverse=True)
        for hybrid_score, dense_score, sparse_score, shared_terms, left, right in pair_scores[:top_k]:
            if dense_score is not None:
                edges.append(_edge(left["knowledge_block_id"], right["knowledge_block_id"], "dense_sim", dense_score, dense_score, None, []))
            if sparse_score is not None:
                edges.append(_edge(left["knowledge_block_id"], right["knowledge_block_id"], "sparse_overlap", sparse_score, None, sparse_score, shared_terms))
            edges.append(_edge(left["knowledge_block_id"], right["knowledge_block_id"], "hybrid_sim", hybrid_score, dense_score, sparse_score, shared_terms))
    store.upsert_semantic_edges(edges)
    return EdgeBuildStats(edges_created=len(edges), groups_processed=len(groups), candidate_pairs=candidate_pairs)


def _temporal_edges(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        conversation_id = member.get("conversation_id")
        if conversation_id:
            by_conversation.setdefault(str(conversation_id), []).append(member)
    for conversation_members in by_conversation.values():
        for left, right in zip(conversation_members, conversation_members[1:]):
            edges.append(_edge(left["knowledge_block_id"], right["knowledge_block_id"], "temporal_neighbor", 1.0, None, None, []))
    return edges


def _edge(
    src_id: str,
    dst_id: str,
    edge_kind: str,
    weight: float,
    dense_similarity: float | None,
    sparse_similarity: float | None,
    shared_terms: list[str],
) -> dict[str, Any]:
    left, right = sorted([src_id, dst_id])
    edge_id = stable_id("edge", left, right, edge_kind, POLICY_VERSION, prefix="edge")
    return {
        "id": edge_id,
        "src_type": "block",
        "src_id": left,
        "dst_type": "block",
        "dst_id": right,
        "edge_kind": edge_kind,
        "weight": float(weight),
        "dense_similarity": dense_similarity,
        "sparse_similarity": sparse_similarity,
        "shared_terms_json": json.dumps(shared_terms[:20], ensure_ascii=False),
        "metadata_json": "{}",
        "policy_version": POLICY_VERSION,
    }


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _sparse_similarity(left: dict[str, float], right: dict[str, float]) -> tuple[float, list[str]]:
    shared = sorted(set(left) & set(right))
    if not shared:
        return 0.0, []
    numerator = sum(float(left[token]) * float(right[token]) for token in shared)
    left_norm = math.sqrt(sum(float(weight) * float(weight) for weight in left.values()))
    right_norm = math.sqrt(sum(float(weight) * float(weight) for weight in right.values()))
    score = numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
    ranked_terms = sorted(shared, key=lambda token: left[token] * right[token], reverse=True)
    return score, ranked_terms


def _hybrid_score(dense_score: float | None, sparse_score: float | None) -> float:
    dense = dense_score if dense_score is not None else 0.0
    sparse = sparse_score if sparse_score is not None else 0.0
    if dense_score is None:
        return sparse
    if sparse_score is None:
        return dense
    return 0.65 * dense + 0.35 * sparse
