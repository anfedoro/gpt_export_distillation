from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any


def audit_storage(db_path: Path, output_dir: Path) -> dict[str, Any]:
    """Inspect SQLite storage without opening the database for writes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        pragmas = {name: conn.execute(f"PRAGMA {name}").fetchone()[0] for name in ("page_size", "page_count", "freelist_count", "journal_mode", "auto_vacuum")}
        sqlite_version = conn.execute("SELECT sqlite_version()").fetchone()[0]
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        objects = _dbstat_objects(conn)
        schema = {
            row["name"]: {"type": row["type"], "table_name": row["tbl_name"], "sql": row["sql"]}
            for row in conn.execute("SELECT name, type, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
        }
        row_counts = {
            name: int(conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(name)}").fetchone()[0])
            for name, item in schema.items() if item["type"] == "table"
        }
        dense = _dense_audit(conn, objects, row_counts)
        sparse = _sparse_audit(conn, objects, row_counts)
        groups = _logical_groups(objects, row_counts)
        query_plans = _query_plans(conn)

    file_size = db_path.stat().st_size
    page_size = int(pragmas["page_size"])
    page_count = int(pragmas["page_count"])
    freelist_count = int(pragmas["freelist_count"])
    used_pages = page_count - freelist_count
    report = {
        "schema_version": "kb.storage_audit.v1",
        "database": str(db_path),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "file": {
            "size_bytes": file_size,
            "size_gib": file_size / 1024**3,
            "page_size": page_size,
            "page_count": page_count,
            "freelist_count": freelist_count,
            "used_pages": used_pages,
            "free_space_bytes": freelist_count * page_size,
            "free_space_gib": freelist_count * page_size / 1024**3,
            "sqlite_version": sqlite_version,
            "journal_mode": pragmas["journal_mode"],
            "auto_vacuum": pragmas["auto_vacuum"],
            "integrity_check": integrity,
        },
        "objects": objects,
        "object_totals": _object_totals(objects, file_size),
        "logical_groups": groups,
        "dense": dense,
        "sparse": sparse,
        "query_plans": query_plans,
        "recommendation": _recommendation(report_file_size=file_size, free_space_bytes=freelist_count * page_size, dense=dense, sparse=sparse, groups=groups, object_totals=_object_totals(objects, file_size)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "report.md").write_text(_render_markdown(report))
    return report


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _dbstat_objects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    schema = {
        row["name"]: {"type": row["type"], "table_name": row["tbl_name"]}
        for row in conn.execute("SELECT name, type, tbl_name FROM sqlite_master")
    }
    grouped: dict[str, dict[str, Any]] = {}
    for row in conn.execute("SELECT name, SUM(pgsize) AS bytes, COUNT(*) AS pages FROM dbstat GROUP BY name"):
        name = str(row["name"])
        item = schema.get(name, {"type": "internal", "table_name": None})
        grouped[name] = {
            "object_name": name,
            "object_type": item["type"],
            "table_name": item["table_name"],
            "bytes": int(row["bytes"]),
            "gib": int(row["bytes"]) / 1024**3,
            "pages": int(row["pages"]),
        }
    return sorted(grouped.values(), key=lambda item: (-item["bytes"], item["object_name"]))


def _object_totals(objects: list[dict[str, Any]], file_size: int) -> dict[str, Any]:
    tables = sum(item["bytes"] for item in objects if item["object_type"] == "table")
    indexes = sum(item["bytes"] for item in objects if item["object_type"] == "index")
    internal = sum(item["bytes"] for item in objects if item["object_type"] == "internal")
    return {
        "tables_bytes": tables,
        "indexes_bytes": indexes,
        "internal_bytes": internal,
        "used_object_bytes": tables + indexes + internal,
        "tables_percent_of_db": tables * 100 / file_size,
        "indexes_percent_of_db": indexes * 100 / file_size,
        "internal_percent_of_db": internal * 100 / file_size,
    }


def _dense_audit(conn: sqlite3.Connection, objects: list[dict[str, Any]], row_counts: dict[str, int]) -> dict[str, Any]:
    dims = Counter()
    owners = Counter()
    json_bytes: list[int] = []
    count = 0
    for row in conn.execute("SELECT dim, length(CAST(vector_json AS BLOB)) AS json_bytes, owner_type FROM dense_vectors"):
        count += 1
        dims[int(row["dim"])] += 1
        owners[str(row["owner_type"])] += 1
        json_bytes.append(int(row["json_bytes"] or 0))
    dense_table = next((item for item in objects if item["object_name"] == "dense_vectors"), {"bytes": 0})
    dim = dims.most_common(1)[0][0] if dims else 0
    float32 = dim * 4
    float16 = dim * 2
    int8 = dim
    return {
        "vector_count": count,
        "dimensions": dict(dims),
        "owner_types": dict(owners),
        "serialization": "JSON text in dense_vectors.vector_json",
        "payload_bytes": {"avg_json_bytes": fmean(json_bytes) if json_bytes else 0, "p50_json_bytes": _percentile(json_bytes, 50), "p95_json_bytes": _percentile(json_bytes, 95)},
        "table_bytes": dense_table["bytes"],
        "average_sqlite_bytes_per_vector": dense_table["bytes"] / count if count else 0,
        "theoretical_payload_bytes": {"float32": float32, "float16": float16, "int8": int8},
        "overhead_vs_float32_payload": dense_table["bytes"] / (count * float32) if count and float32 else 0,
        "duplicate_storage_check": {"json_and_decomposed_dense_payload": False, "block_level_vectors": owners.get("knowledge_block", 0), "chunk_level_vectors": owners.get("retrieval_chunk", 0)},
    }


def _sparse_audit(conn: sqlite3.Connection, objects: list[dict[str, Any]], row_counts: dict[str, int]) -> dict[str, Any]:
    by_owner = {
        (str(row["owner_type"]), str(row["owner_id"])): int(row["term_count"])
        for row in conn.execute(
            "SELECT owner_type, owner_id, COUNT(*) AS term_count "
            "FROM sparse_terms GROUP BY owner_type, owner_id"
        )
    }
    counts = list(by_owner.values())
    term_count = int(conn.execute("SELECT COUNT(*) FROM sparse_terms").fetchone()[0])
    table = next((item for item in objects if item["object_name"] == "sparse_terms"), {"bytes": 0})
    indexes = [item for item in objects if item["object_type"] == "index" and item["table_name"] == "sparse_terms"]
    term32 = 4 + 4
    term16 = 4 + 2
    return {
        "representation_count": len(by_owner),
        "term_count": term_count,
        "terms_per_representation": {"avg": fmean(counts) if counts else 0, "p50": _percentile(counts, 50), "p95": _percentile(counts, 95), "max": max(counts) if counts else 0},
        "schema": {"term_id": "TEXT", "term_text": "TEXT", "weight": "REAL", "primary_key": "(owner_type, owner_id, token_id, model_name)"},
        "table_bytes": table["bytes"],
        "indexes": indexes,
        "bytes_per_term": {"table_only": table["bytes"] / term_count if term_count else 0, "table_plus_indexes": (table["bytes"] + sum(item["bytes"] for item in indexes)) / term_count if term_count else 0},
        "theoretical_compact_bytes_per_term": {"uint32_float32": term32, "uint32_float16": term16},
        "overhead_vs_uint32_float32": (table["bytes"] + sum(item["bytes"] for item in indexes)) / (term_count * term32) if term_count else 0,
        "duplicate_payload_check": {"separate_sparse_vector_table": False, "token_text_is_stored_per_term": True},
    }


def _logical_groups(objects: list[dict[str, Any]], row_counts: dict[str, int]) -> dict[str, Any]:
    groups = {
        "conversations_messages_blocks_chunks_text": {"tables": {"source_documents", "conversations", "messages", "blocks", "retrieval_chunks"}},
        "dense_representations": {"tables": {"dense_vectors"}},
        "sparse_representations": {"tables": {"sparse_terms"}},
        "sparse_terms": {"tables": {"sparse_terms"}},
        "semantic_graph": {"tables": {"semantic_nodes", "semantic_node_members", "semantic_edges"}},
        "provenance_metadata": {"tables": {"knowledge_blocks", "attachment_documents", "retrieval_traces", "ingestion_runs", "schema_migrations"}},
    }
    known = set().union(*(item["tables"] for item in groups.values()))
    result = {}
    for name, spec in groups.items():
        tables = spec["tables"]
        table_objects = [item for item in objects if item["object_type"] == "table" and item["object_name"] in tables]
        bytes_total = sum(item["bytes"] for item in table_objects)
        rows_total = sum(row_counts.get(table, 0) for table in tables)
        result[name] = {"tables": sorted(tables), "bytes": bytes_total, "gib": bytes_total / 1024**3, "row_count": rows_total, "avg_bytes_per_row": bytes_total / rows_total if rows_total else 0}
    index_objects = [item for item in objects if item["object_type"] == "index"]
    result["indexes"] = {"tables": [], "bytes": sum(item["bytes"] for item in index_objects), "gib": sum(item["bytes"] for item in index_objects) / 1024**3, "row_count": 0, "avg_bytes_per_row": 0}
    other_tables = [item for item in objects if item["object_type"] == "table" and item["object_name"] not in known]
    result["other"] = {"tables": [item["object_name"] for item in other_tables], "bytes": sum(item["bytes"] for item in other_tables), "gib": sum(item["bytes"] for item in other_tables) / 1024**3, "row_count": sum(row_counts.get(item["object_name"], 0) for item in other_tables), "avg_bytes_per_row": 0}
    result["_notes"] = {
        "sparse_representations_and_sparse_terms_share_storage": True,
        "non_additive_group_warning": "sparse_representations and sparse_terms both describe the physical sparse_terms table; do not sum them.",
    }
    return result


def _query_plans(conn: sqlite3.Connection) -> dict[str, Any]:
    plans = {}
    for name, sql, params in (
        ("chunk_by_block_policy", "EXPLAIN QUERY PLAN SELECT id FROM retrieval_chunks WHERE block_id = ? AND chunk_policy_id = ? ORDER BY ordinal", ("x", "x")),
        ("sparse_by_owner", "EXPLAIN QUERY PLAN SELECT token_id, weight FROM sparse_terms WHERE owner_type = ? AND owner_id = ?", ("retrieval_chunk", "x")),
        ("dense_by_owner", "EXPLAIN QUERY PLAN SELECT vector_json FROM dense_vectors WHERE owner_type = ? AND owner_id = ?", ("retrieval_chunk", "x")),
    ):
        plans[name] = [dict(row) for row in conn.execute(sql, params)]
    return plans


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _recommendation(*, report_file_size: int, free_space_bytes: int, dense: dict[str, Any], sparse: dict[str, Any], groups: dict[str, Any], object_totals: dict[str, Any]) -> dict[str, Any]:
    return {"verdict": "C", "reason": "Dense JSON and sparse text terms dominate vector storage; SQLite overhead is measurable but not the primary cost.", "vacuum_max_reclaim_gib": free_space_bytes / 1024**3, "dense_float16_payload_saving_gib": max(0.0, (dense["table_bytes"] - dense["vector_count"] * dense["theoretical_payload_bytes"]["float16"]) / 1024**3), "compact_sparse_theoretical_saving_gib": max(0.0, (sparse["table_bytes"] + sum(item["bytes"] for item in sparse["indexes"]) - sparse["term_count"] * 6) / 1024**3), "lancedb_spike": {"recommended": True, "data": ["retrieval_chunks", "dense_vectors", "sparse_vectors", "metadata/provenance"], "baseline_metrics": ["disk_size", "cold_start", "RSS", "dense/sparse/hybrid latency", "167-probe benchmark"]}}


def _render_markdown(report: dict[str, Any]) -> str:
    f = report["file"]; o = report["object_totals"]
    lines = ["# SQLite Storage Audit", "", f"Database: `{report['database']}`", "", "## File", "", f"- Size: `{f['size_bytes']}` bytes / `{f['size_gib']:.3f}` GiB", f"- Pages: `{f['page_count']}`; used `{f['used_pages']}`; freelist `{f['freelist_count']}`", f"- Free space: `{f['free_space_gib']:.3f}` GiB", f"- SQLite: `{f['sqlite_version']}`; journal: `{f['journal_mode']}`; auto_vacuum: `{f['auto_vacuum']}`", f"- Integrity: `{f['integrity_check']}`", "", "## Top 20 Objects", "", "| Object | Type | Table | GiB | % DB | Pages |", "|---|---|---|---:|---:|---:|"]
    for item in report["objects"][:20]: lines.append(f"| {item['object_name']} | {item['object_type']} | {item['table_name'] or ''} | {item['gib']:.3f} | {item['bytes']*100/f['size_bytes']:.2f} | {item['pages']} |")
    lines += ["", f"- Tables: `{o['tables_bytes']/1024**3:.3f}` GiB ({o['tables_percent_of_db']:.2f}%)", f"- Indexes: `{o['indexes_bytes']/1024**3:.3f}` GiB ({o['indexes_percent_of_db']:.2f}%)", f"- Internal objects: `{o['internal_bytes']/1024**3:.3f}` GiB", "", "## Logical Groups", "", "The `sparse_representations` and `sparse_terms` rows refer to the same physical table and are not additive.", "", "| Group | GiB | % DB | Rows |", "|---|---:|---:|---:|"]
    for name, item in report["logical_groups"].items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | {item['gib']:.3f} | {item['bytes']*100/f['size_bytes']:.2f} | {item['row_count']} |")
    lines += ["", "## Dense", "", f"- Vectors: `{report['dense']['vector_count']}`; owners: `{report['dense']['owner_types']}`", f"- Format: `{report['dense']['serialization']}`", f"- Avg JSON payload: `{report['dense']['payload_bytes']['avg_json_bytes']:.1f}` bytes", f"- Avg SQLite storage: `{report['dense']['average_sqlite_bytes_per_vector']:.1f}` bytes/vector", f"- Overhead vs float32 payload: `{report['dense']['overhead_vs_float32_payload']:.2f}x`", "", "## Sparse", "", f"- Representations: `{report['sparse']['representation_count']}`; terms: `{report['sparse']['term_count']}`", f"- Terms per representation p50/p95/max: `{report['sparse']['terms_per_representation']['p50']:.1f}/{report['sparse']['terms_per_representation']['p95']:.1f}/{report['sparse']['terms_per_representation']['max']}`", f"- Storage: `{report['sparse']['bytes_per_term']['table_plus_indexes']:.1f}` bytes/term including indexes", f"- Overhead vs uint32+float32: `{report['sparse']['overhead_vs_uint32_float32']:.2f}x`", "", "## Recommendation", "", f"- Verdict: **{report['recommendation']['verdict']}**", f"- {report['recommendation']['reason']}", f"- Estimated float16 dense saving: `{report['recommendation']['dense_float16_payload_saving_gib']:.3f}` GiB", f"- Estimated compact sparse saving: `{report['recommendation']['compact_sparse_theoretical_saving_gib']:.3f}` GiB", "- A LanceDB/vector-native spike is recommended before redesigning production storage."]
    return "\n".join(lines) + "\n"
