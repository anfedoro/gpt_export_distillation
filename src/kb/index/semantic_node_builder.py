from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from kb.model.ids import stable_id
from kb.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class NodeBuildStats:
    nodes_created: int
    memberships_created: int
    nodes_with_dense_vectors: int
    nodes_with_sparse_terms: int


def build_deterministic_nodes(store: SQLiteStore, *, sparse_top_k: int = 50) -> NodeBuildStats:
    groups = _collect_groups(store)
    nodes_created = 0
    memberships_created = 0
    nodes_with_dense_vectors = 0
    nodes_with_sparse_terms = 0
    for group in groups:
        node_id = stable_id("semantic_node", group["node_type"], group["group_id"], prefix="node")
        dense_vector = _mean_vector([member["dense_vector"] for member in group["members"] if member["dense_vector"]])
        sparse_terms = _aggregate_sparse_terms(
            [member["sparse_terms"] for member in group["members"]],
            top_k=sparse_top_k,
        )
        top_terms = [
            {"term": term, "weight": weight}
            for term, weight in sorted(sparse_terms.items(), key=lambda item: item[1], reverse=True)[:sparse_top_k]
        ]
        dense_vector_id = None
        sparse_vector_id = None
        if dense_vector:
            dense_vector_id = store.upsert_dense_vector(
                owner_type="semantic_node",
                owner_id=node_id,
                model_name="aggregate-mean",
                model_version="v1",
                vector=dense_vector,
            )
            nodes_with_dense_vectors += 1
        if sparse_terms:
            sparse_vector_id = store.replace_sparse_terms(
                owner_type="semantic_node",
                owner_id=node_id,
                model_name="aggregate-max",
                terms=sparse_terms,
            )
            nodes_with_sparse_terms += 1
        store.upsert_semantic_node(
            node_id=node_id,
            node_type=group["node_type"],
            project_id=group["project_id"],
            dense_vector_id=dense_vector_id,
            sparse_vector_id=sparse_vector_id,
            title=group["title"],
            summary=None,
            top_terms_json=json.dumps(top_terms, ensure_ascii=False),
            metadata_json=json.dumps({"group_id": group["group_id"], "member_count": len(group["members"])}, ensure_ascii=False),
        )
        store.replace_semantic_node_members(
            node_id=node_id,
            members=[
                {
                    "knowledge_block_id": member["knowledge_block_id"],
                    "membership_weight": 1.0,
                    "membership_reason": group["membership_reason"],
                    "metadata_json": "{}",
                }
                for member in group["members"]
            ],
        )
        nodes_created += 1
        memberships_created += len(group["members"])
    return NodeBuildStats(
        nodes_created=nodes_created,
        memberships_created=memberships_created,
        nodes_with_dense_vectors=nodes_with_dense_vectors,
        nodes_with_sparse_terms=nodes_with_sparse_terms,
    )


def _collect_groups(store: SQLiteStore) -> list[dict[str, Any]]:
    blocks = store.knowledge_blocks_for_nodes()
    conversation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    project_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attachment_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if block["conversation_id"]:
            conversation_groups[str(block["conversation_id"])].append(block)
        if block["project_id"]:
            project_groups[str(block["project_id"])].append(block)
        if block["attachment_id"]:
            attachment_groups[str(block["attachment_id"])].append(block)
    groups: list[dict[str, Any]] = []
    for conversation_id, members in sorted(conversation_groups.items()):
        title = members[0]["conversation_title"] or conversation_id
        groups.append(
            _group(
                node_type="conversation",
                group_id=conversation_id,
                project_id=members[0]["project_id"],
                title=f"Conversation: {title}",
                membership_reason="same_conversation",
                members=members,
            )
        )
    for project_id, members in sorted(project_groups.items()):
        groups.append(
            _group(
                node_type="project",
                group_id=project_id,
                project_id=project_id,
                title=f"Project: {project_id}",
                membership_reason="same_project",
                members=members,
            )
        )
    for attachment_id, members in sorted(attachment_groups.items()):
        title = members[0]["attachment_path"] or attachment_id
        groups.append(
            _group(
                node_type="attachment",
                group_id=attachment_id,
                project_id=members[0]["project_id"],
                title=f"Attachment: {title}",
                membership_reason="attachment_link",
                members=members,
            )
        )
    return groups


def _group(
    *,
    node_type: str,
    group_id: str,
    project_id: str | None,
    title: str,
    membership_reason: str,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "node_type": node_type,
        "group_id": group_id,
        "project_id": project_id,
        "title": title,
        "membership_reason": membership_reason,
        "members": members,
    }


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        return []
    values = [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dim)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def _aggregate_sparse_terms(term_sets: list[dict[str, float]], *, top_k: int) -> dict[str, float]:
    weights: Counter[str] = Counter()
    for terms in term_sets:
        for token, weight in terms.items():
            if float(weight) > weights[token]:
                weights[token] = float(weight)
    return dict(weights.most_common(top_k))
