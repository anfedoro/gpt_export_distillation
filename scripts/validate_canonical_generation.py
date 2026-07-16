#!/usr/bin/env python3
"""Compare legacy and canonical native generations without mutating either database."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from kb.storage.dense_native import load_sqlite_vec


LOW_INFORMATION_RE = re.compile(
    r"^(?:```[a-z0-9_+.-]*|```|~~~|[-*_#>|:;,.!?()\[\]{}]+|"
    r"(?:or|and|или|и|далее|next|example|пример|note|примечание):?)$",
    flags=re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-db", type=Path, required=True)
    parser.add_argument("--canonical-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = analyze(args.baseline_db, canonical=False)
    canonical = analyze(args.canonical_db, canonical=True)
    comparison = {
        "schema_version": 1,
        "baseline_db": str(args.baseline_db.resolve()),
        "canonical_db": str(args.canonical_db.resolve()),
        "baseline": baseline["summary"],
        "canonical": canonical["summary"],
        "deltas": numeric_deltas(baseline["summary"], canonical["summary"]),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    write_manual_artifacts(args.output_dir, canonical)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(comparison), encoding="utf-8")
    return 0


def analyze(path: Path, *, canonical: bool) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=1")
    load_sqlite_vec(conn)
    try:
        vectors, rows = load_dense(conn)
        sparse = load_sparse(conn)
        graph = graph_metrics(vectors, rows, sparse)
        graph_detail = graph.pop("_detail")
        counts = table_counts(conn)
        texts = [str(row["text"]) for row in rows]
        low_flags = [is_low_information(text) for text in texts]
        duplicates = duplicate_fraction(texts)
        summary: dict[str, Any] = {
            **counts,
            "db_size_bytes": path.stat().st_size,
            "low_information_fraction": fraction(sum(low_flags), len(low_flags)),
            "duplicate_fraction": duplicates,
            **graph,
        }
        if canonical:
            summary.update(canonical_counts(conn))
        return {
            "summary": summary,
            "rows": rows,
            "low_flags": low_flags,
            "graph_detail": graph_detail,
            "conn_path": path,
        }
    finally:
        conn.close()


def load_dense(conn: sqlite3.Connection) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT dm.rowid,dm.chunk_id,rc.text,b.block_type,m.role,m.conversation_id,
                   c.project_id,sd.relative_path
            FROM dense_native_metadata dm
            JOIN retrieval_chunks rc ON rc.id=dm.chunk_id
            JOIN blocks b ON b.id=rc.block_id
            JOIN messages m ON m.id=b.message_id
            JOIN conversations c ON c.id=m.conversation_id
            JOIN source_documents sd ON sd.id=c.source_document_id
            ORDER BY dm.rowid
            """
        )
    ]
    payload = {
        int(row["rowid"]): bytes(row["embedding"])
        for row in conn.execute("SELECT rowid,embedding FROM dense_vectors_native ORDER BY rowid")
    }
    vectors = np.stack([np.frombuffer(payload[int(row["rowid"])], dtype="<f4") for row in rows])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    return vectors, rows


def load_sparse(conn: sqlite3.Connection) -> list[dict[int, float]]:
    result: list[dict[int, float]] = []
    for row in conn.execute(
        """
        SELECT sm.rowid,sv.indices_blob,sv.weights_blob
        FROM sparse_vector_metadata sm JOIN sparse_vectors_compact sv ON sv.rowid=sm.rowid
        ORDER BY sm.rowid
        """
    ):
        indices = np.frombuffer(bytes(row["indices_blob"]), dtype="<u4")
        weights = np.frombuffer(bytes(row["weights_blob"]), dtype="<f4")
        result.append({int(index): float(weight) for index, weight in zip(indices, weights)})
    return result


def graph_metrics(
    vectors: np.ndarray, rows: list[dict[str, Any]], sparse: list[dict[int, float]],
) -> dict[str, Any]:
    if len(rows) < 21:
        raise ValueError("Semantic-neighbourhood validation requires at least 21 indexed chunks.")
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    top20 = np.argpartition(-similarity, kth=min(19, len(rows) - 2), axis=1)[:, :20]
    top20 = np.take_along_axis(
        top20, np.argsort(-np.take_along_axis(similarity, top20, axis=1), axis=1), axis=1
    )
    ranks = {}
    for rank in (1, 5, 10, 20):
        values = similarity[np.arange(len(rows)), top20[:, rank - 1]]
        ranks[f"rank_{rank}_similarity_median"] = round(float(np.median(values)), 6)
    detail: dict[str, Any] = {"top20": top20, "similarity": similarity}
    metrics: dict[str, Any] = {**ranks}
    for k in (5, 10, 20):
        directed = {(left, int(right)) for left in range(len(rows)) for right in top20[left, :k]}
        mutual = {(left, right) for left, right in directed if (right, left) in directed}
        undirected = {(min(left, right), max(left, right)) for left, right in mutual}
        components, largest, isolated = component_stats(len(rows), undirected)
        metrics.update({
            f"mutual_link_fraction_k{k}": fraction(len(mutual), len(directed)),
            f"mutual_edges_k{k}": len(undirected),
            f"components_k{k}": components,
            f"giant_component_size_k{k}": largest,
            f"isolated_nodes_k{k}": isolated,
        })
        if k == 20:
            detail["mutual_edges"] = sorted(undirected)
    inbound = Counter(int(right) for left in range(len(rows)) for right in top20[left, :20])
    top_hubs = [node for node, _degree in inbound.most_common(100)]
    strongest = sorted(
        detail["mutual_edges"], key=lambda pair: float(similarity[pair[0], pair[1]]), reverse=True
    )[:100]
    low_flags = [is_low_information(str(row["text"])) for row in rows]
    metrics["top_100_hub_low_information_fraction"] = fraction(
        sum(low_flags[node] for node in top_hubs), len(top_hubs)
    )
    metrics["top_100_strongest_edge_low_information_fraction"] = fraction(
        sum(low_flags[left] or low_flags[right] for left, right in strongest), len(strongest)
    )
    candidate_edges = [(left, int(right)) for left in range(len(rows)) for right in top20[left, :20]]
    overlaps = [weighted_jaccard(sparse[left], sparse[right]) for left, right in candidate_edges]
    metrics["sparse_overlap_fraction"] = fraction(sum(value > 0 for value in overlaps), len(overlaps))
    metrics["sparse_overlap_median"] = round(float(np.median(overlaps)), 6)
    metrics["same_conversation_edge_fraction"] = fraction(
        sum(rows[left]["conversation_id"] == rows[right]["conversation_id"] for left, right in candidate_edges),
        len(candidate_edges),
    )
    known_cross = [
        (left, right) for left, right in candidate_edges
        if rows[left]["project_id"] and rows[right]["project_id"]
    ]
    metrics["cross_project_edge_fraction"] = fraction(
        sum(rows[left]["project_id"] != rows[right]["project_id"] for left, right in known_cross),
        len(known_cross),
    )
    detail.update({"inbound": inbound, "strongest": strongest, "sparse": sparse})
    metrics["_detail"] = detail
    return metrics


def component_stats(node_count: int, edges: set[tuple[int, int]]) -> tuple[int, int, int]:
    adjacency = [set() for _ in range(node_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited: set[int] = set()
    sizes: list[int] = []
    for node in range(node_count):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current] - visited:
                visited.add(neighbor)
                stack.append(neighbor)
        sizes.append(size)
    isolated = sum(not neighbors for neighbors in adjacency)
    return len(sizes), max(sizes, default=0), isolated


def canonical_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    block_types = {
        str(row["block_type"]): int(row["count"])
        for row in conn.execute("SELECT block_type,COUNT(*) AS count FROM blocks GROUP BY block_type")
    }
    statuses = {
        str(row["semantic_status"]): int(row["count"])
        for row in conn.execute("SELECT semantic_status,COUNT(*) AS count FROM blocks GROUP BY semantic_status")
    }
    return {
        "canonical_blocks_by_type": block_types,
        "semantic_status_counts": statuses,
        "artifact_count": int(conn.execute("SELECT COUNT(*) FROM blocks WHERE artifact_policy='store'").fetchone()[0]),
        "graph_eligible_block_count": int(conn.execute("SELECT COUNT(*) FROM blocks WHERE graph_eligibility=1").fetchone()[0]),
        "structural_relationship_count": int(conn.execute("SELECT COUNT(*) FROM block_relationships").fetchone()[0]),
    }


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("source_documents", "conversations", "messages", "blocks", "retrieval_chunks")
    }


def duplicate_fraction(texts: list[str]) -> float:
    normalized = [re.sub(r"\s+", " ", text).strip().casefold() for text in texts]
    duplicates = len(normalized) - len(set(normalized))
    return fraction(duplicates, len(normalized))


def is_low_information(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized or LOW_INFORMATION_RE.fullmatch(normalized):
        return True
    if "asset_pointer" in normalized and "content_type" in normalized:
        return True
    return len(re.findall(r"\w+", normalized, flags=re.UNICODE)) <= 1


def weighted_jaccard(left: dict[int, float], right: dict[int, float]) -> float:
    keys = left.keys() | right.keys()
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    if denominator == 0:
        return 0.0
    return sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys) / denominator


def write_manual_artifacts(output: Path, canonical: dict[str, Any]) -> None:
    rows = canonical["rows"]
    detail = canonical["graph_detail"]
    similarity = detail["similarity"]
    sparse = detail["sparse"]
    strongest = detail["strongest"]
    write_edges(output / "strongest_mutual_edges.csv", strongest[:100], rows, similarity, sparse)
    cross = [
        edge for edge in detail["mutual_edges"]
        if rows[edge[0]]["project_id"] and rows[edge[1]]["project_id"]
        and rows[edge[0]]["project_id"] != rows[edge[1]]["project_id"]
    ]
    cross.sort(key=lambda edge: float(similarity[edge[0], edge[1]]), reverse=True)
    write_edges(output / "cross_project_mutual_edges.csv", cross[:50], rows, similarity, sparse)
    high_dense_low_sparse = sorted(
        detail["mutual_edges"],
        key=lambda edge: (float(similarity[edge[0], edge[1]]), -weighted_jaccard(sparse[edge[0]], sparse[edge[1]])),
        reverse=True,
    )
    high_dense_low_sparse = [
        edge for edge in high_dense_low_sparse if weighted_jaccard(sparse[edge[0]], sparse[edge[1]]) < 0.02
    ][:50]
    write_edges(output / "high_dense_low_sparse_edges.csv", high_dense_low_sparse, rows, similarity, sparse)
    hubs = detail["inbound"].most_common(100)
    write_csv(output / "highest_inbound_hubs.csv", [
        {**rows[node], "inbound_degree": degree, "low_information": is_low_information(str(rows[node]["text"]))}
        for node, degree in hubs
    ])
    conn = sqlite3.connect(f"file:{Path(canonical['conn_path']).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        excluded = [dict(row) for row in conn.execute(
            "SELECT id,block_type,canonical_content,semantic_status,exclusion_reasons_json "
            "FROM blocks WHERE semantic_status IN ('excluded','context_only') ORDER BY id LIMIT 50"
        )]
        retained = [dict(row) for row in conn.execute(
            "SELECT id,block_type,canonical_content,semantic_status FROM blocks "
            "WHERE graph_eligibility=1 AND length(canonical_content)<=80 ORDER BY id LIMIT 50"
        )]
        relationships = [dict(row) for row in conn.execute(
            """
            SELECT br.relation_type,source.id AS source_id,source.block_type AS source_type,
                   source.canonical_content AS source_content,target.id AS target_id,
                   target.block_type AS target_type,target.canonical_content AS target_content
            FROM block_relationships br
            JOIN blocks source ON source.id=br.source_block_id
            JOIN blocks target ON target.id=br.target_block_id
            WHERE br.relation_type='has_adjacent_artifact' ORDER BY source.id,target.id LIMIT 50
            """
        )]
    finally:
        conn.close()
    write_csv(output / "excluded_or_context_only.csv", excluded)
    write_csv(output / "short_retained_meaningful.csv", retained)
    write_csv(output / "prose_to_artifact_relationships.csv", relationships)


def write_edges(
    path: Path, edges: list[tuple[int, int]], rows: list[dict[str, Any]],
    similarity: np.ndarray, sparse: list[dict[int, float]],
) -> None:
    write_csv(path, [
        {
            "left_chunk_id": rows[left]["chunk_id"],
            "right_chunk_id": rows[right]["chunk_id"],
            "dense_similarity": float(similarity[left, right]),
            "sparse_weighted_jaccard": weighted_jaccard(sparse[left], sparse[right]),
            "same_conversation": rows[left]["conversation_id"] == rows[right]["conversation_id"],
            "left_project": rows[left]["project_id"],
            "right_project": rows[right]["project_id"],
            "left_text": rows[left]["text"],
            "right_text": rows[right]["text"],
        }
        for left, right in edges
    ])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_deltas(baseline: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in canonical.items():
        old = baseline.get(key)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            result[key] = round(value - old, 6)
    return result


def fraction(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def render_report(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    canonical = comparison["canonical"]
    lines = [
        "# Canonical ingestion old-versus-new validation",
        "",
        "Both databases were opened read-only for analysis. They use the same source slice, "
        "embedding model, model revision, and 256-token content budget.",
        "",
        "## Corpus",
        "",
        f"- Source documents: {baseline['source_documents']} -> {canonical['source_documents']}.",
        f"- Messages: {baseline['messages']} -> {canonical['messages']}.",
        f"- Structural/canonical blocks: {baseline['blocks']} -> {canonical['blocks']}.",
        f"- Semantic chunks: {baseline['retrieval_chunks']} -> {canonical['retrieval_chunks']}.",
        f"- Low-information fraction: {baseline['low_information_fraction']:.1%} -> {canonical['low_information_fraction']:.1%}.",
        f"- Duplicate fraction: {baseline['duplicate_fraction']:.1%} -> {canonical['duplicate_fraction']:.1%}.",
        "",
        "## Dense neighbourhood",
        "",
    ]
    for rank in (1, 5, 10, 20):
        key = f"rank_{rank}_similarity_median"
        lines.append(f"- Rank {rank} median: {baseline[key]:.4f} -> {canonical[key]:.4f}.")
    lines.extend([
        f"- Mutual-link fraction k=20: {baseline['mutual_link_fraction_k20']:.1%} -> {canonical['mutual_link_fraction_k20']:.1%}.",
        f"- Giant component k=20: {baseline['giant_component_size_k20']} -> {canonical['giant_component_size_k20']}.",
        f"- Isolated nodes k=20: {baseline['isolated_nodes_k20']} -> {canonical['isolated_nodes_k20']}.",
        f"- Top-100 hub low-information fraction: {baseline['top_100_hub_low_information_fraction']:.1%} -> "
        f"{canonical['top_100_hub_low_information_fraction']:.1%}.",
        f"- Top-100 strongest-edge low-information fraction: "
        f"{baseline['top_100_strongest_edge_low_information_fraction']:.1%} -> "
        f"{canonical['top_100_strongest_edge_low_information_fraction']:.1%}.",
        "",
        "## Canonical representation",
        "",
        f"- Block types: `{json.dumps(canonical.get('canonical_blocks_by_type', {}), ensure_ascii=False, sort_keys=True)}`.",
        f"- Semantic statuses: `{json.dumps(canonical.get('semantic_status_counts', {}), ensure_ascii=False, sort_keys=True)}`.",
        f"- Stored artifacts: {canonical.get('artifact_count', 0)}.",
        f"- Structural relationships: {canonical.get('structural_relationship_count', 0)}.",
        "",
        "## Manual-review files",
        "",
        "- `strongest_mutual_edges.csv`",
        "- `highest_inbound_hubs.csv`",
        "- `cross_project_mutual_edges.csv`",
        "- `high_dense_low_sparse_edges.csv`",
        "- `excluded_or_context_only.csv`",
        "- `short_retained_meaningful.csv`",
        "- `prose_to_artifact_relationships.csv`",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
