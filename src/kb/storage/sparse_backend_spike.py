from __future__ import annotations

import json
import math
import os
import resource
import sqlite3
import statistics
import struct
import time
from pathlib import Path
from typing import Any


SPIKE_SCHEMA_VERSION = "kb.sparse_backend_spike.v1"


def _pack_uint32(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def _pack_float32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_uint32(blob: bytes) -> list[int]:
    if len(blob) % 4:
        raise ValueError("indices_blob has a non-uint32 byte length.")
    return list(struct.unpack(f"<{len(blob) // 4}I", blob))


def _unpack_float32(blob: bytes) -> list[float]:
    if len(blob) % 4:
        raise ValueError("weights_blob has a non-float32 byte length.")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def sparse_cosine(query: dict[int, float], document: dict[int, float], *, query_norm: float | None = None, document_norm: float | None = None) -> float:
    if not query or not document:
        return 0.0
    query_norm = query_norm if query_norm is not None else math.sqrt(sum(value * value for value in query.values()))
    document_norm = document_norm if document_norm is not None else math.sqrt(sum(value * value for value in document.values()))
    if not query_norm or not document_norm:
        return 0.0
    return sum(query_term * document.get(term_id, 0.0) for term_id, query_term in query.items()) / (query_norm * document_norm)


def _rank_scores(scores: dict[int, float], row_to_chunk: dict[int, str], limit: int) -> list[tuple[str, float]]:
    return [
        (row_to_chunk[rowid], score)
        for rowid, score in sorted(scores.items(), key=lambda item: (-item[1], row_to_chunk[item[0]]))[:limit]
    ]


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if os.uname().sysname != "Darwin" else 1))


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _load_subset(source_db: Path, chunk_limit: int) -> tuple[list[str], dict[str, list[tuple[str, str, float]]], str, str]:
    source = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        model_name, space_id = source.execute(
            "SELECT model_name, COALESCE(embedding_space_id, model_name) FROM sparse_terms "
            "WHERE owner_type='retrieval_chunk' ORDER BY owner_id LIMIT 1"
        ).fetchone()
        owner_rows = source.execute(
            "SELECT owner_id FROM sparse_terms WHERE owner_type='retrieval_chunk' AND model_name=? "
            "GROUP BY owner_id ORDER BY owner_id LIMIT ?",
            (model_name, chunk_limit),
        ).fetchall()
        owners = [str(row["owner_id"]) for row in owner_rows]
        terms: dict[str, list[tuple[str, str, float]]] = {owner: [] for owner in owners}
        for offset in range(0, len(owners), 900):
            batch = owners[offset : offset + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = source.execute(
                f"SELECT owner_id,token_id,token_text,weight FROM sparse_terms "
                f"WHERE owner_type='retrieval_chunk' AND model_name=? AND owner_id IN ({placeholders}) "
                "ORDER BY owner_id,token_id",
                (model_name, *batch),
            )
            for row in rows:
                terms[str(row["owner_id"])].append((str(row["token_id"]), str(row["token_text"]), float(row["weight"])))
        return owners, terms, str(model_name), str(space_id)
    finally:
        source.close()


def _build_vocab(terms: dict[str, list[tuple[str, str, float]]]) -> tuple[dict[str, int], dict[int, str], dict[int, str]]:
    external_ids = sorted({token_id for rows in terms.values() for token_id, _, _ in rows})
    term_to_id = {token_id: index for index, token_id in enumerate(external_ids, start=1)}
    id_to_external = {term_id: token_id for token_id, term_id in term_to_id.items()}
    text_by_id: dict[int, str] = {}
    for rows in terms.values():
        for token_id, token_text, _ in rows:
            text_by_id.setdefault(term_to_id[token_id], token_text)
    return term_to_id, id_to_external, text_by_id


def _create_compact_db(path: Path, *, variant: str, model_name: str, space_id: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE vocabulary (
            term_id INTEGER PRIMARY KEY,
            external_term_id TEXT NOT NULL UNIQUE,
            token_text TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE chunk_map (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            norm REAL NOT NULL
        );
        """
    )
    conn.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", [
        ("schema_version", SPIKE_SCHEMA_VERSION),
        ("variant", variant),
        ("model_name", model_name),
        ("embedding_space_id", space_id),
        ("dtype", "float32"),
    ])
    if variant == "blob":
        conn.execute(
            "CREATE TABLE compact_vectors (rowid INTEGER PRIMARY KEY, indices_blob BLOB NOT NULL, weights_blob BLOB NOT NULL)"
        )
    elif variant == "postings":
        conn.execute(
            "CREATE TABLE sparse_postings (term_id INTEGER NOT NULL, chunk_rowid INTEGER NOT NULL, weight_blob BLOB NOT NULL, PRIMARY KEY(term_id, chunk_rowid)) WITHOUT ROWID"
        )
    else:
        raise ValueError(f"Unknown sparse spike variant: {variant}")
    return conn


def _materialize_compact(
    path: Path,
    *,
    variant: str,
    owners: list[str],
    terms: dict[str, list[tuple[str, str, float]]],
    term_to_id: dict[str, int],
    id_to_external: dict[int, str],
    text_by_id: dict[int, str],
    model_name: str,
    space_id: str,
) -> dict[str, Any]:
    conn = _create_compact_db(path, variant=variant, model_name=model_name, space_id=space_id)
    conn.executemany(
        "INSERT INTO vocabulary(term_id,external_term_id,token_text) VALUES(?,?,?)",
        [(term_id, id_to_external[term_id], text_by_id[term_id]) for term_id in sorted(id_to_external)],
    )
    vector_rows: list[tuple[int, str, float]] = []
    posting_rows: list[tuple[int, int, bytes]] = []
    for rowid, owner in enumerate(owners, start=1):
        encoded = [(term_to_id[token_id], weight) for token_id, _, weight in terms[owner]]
        norm = math.sqrt(sum(weight * weight for _, weight in encoded))
        vector_rows.append((rowid, owner, norm))
        if variant == "blob":
            conn.execute(
                "INSERT INTO compact_vectors(rowid,indices_blob,weights_blob) VALUES(?,?,?)",
                (rowid, _pack_uint32([term_id for term_id, _ in encoded]), _pack_float32([weight for _, weight in encoded])),
            )
        else:
            posting_rows.extend((term_id, rowid, struct.pack("<f", weight)) for term_id, weight in encoded)
    conn.executemany("INSERT INTO chunk_map(rowid,chunk_id,norm) VALUES(?,?,?)", vector_rows)
    if variant == "postings":
        conn.executemany("INSERT INTO sparse_postings(term_id,chunk_rowid,weight_blob) VALUES(?,?,?)", posting_rows)
        conn.execute("CREATE INDEX idx_sparse_postings_chunk ON sparse_postings(chunk_rowid,term_id)")
        conn.create_function("float32_blob", 1, lambda blob: struct.unpack("<f", bytes(blob))[0])
    conn.commit()
    return {
        "chunks": len(owners),
        "terms": sum(len(rows) for rows in terms.values()),
        "vocabulary": len(id_to_external),
        "bytes": path.stat().st_size,
    }


def _load_blob_backend(path: Path) -> tuple[sqlite3.Connection, dict[int, tuple[list[int], list[float], float]], dict[int, str]]:
    conn = sqlite3.connect(path)
    records = {
        int(row[0]): (_unpack_uint32(row[1]), _unpack_float32(row[2]), float(row[3]))
        for row in conn.execute(
            "SELECT cv.rowid,cv.indices_blob,cv.weights_blob,cm.norm FROM compact_vectors cv JOIN chunk_map cm ON cm.rowid=cv.rowid ORDER BY cv.rowid"
        )
    }
    mapping = {int(row[0]): str(row[1]) for row in conn.execute("SELECT rowid,chunk_id FROM chunk_map")}
    return conn, records, mapping


def _blob_query(records: dict[int, tuple[list[int], list[float], float]], mapping: dict[int, str], query: dict[int, float], limit: int) -> list[tuple[str, float]]:
    query_norm = math.sqrt(sum(value * value for value in query.values()))
    scores: dict[int, float] = {}
    for rowid, (indices, weights, doc_norm) in records.items():
        dot = sum(query.get(term_id, 0.0) * weight for term_id, weight in zip(indices, weights, strict=True))
        scores[rowid] = dot / (query_norm * doc_norm) if query_norm and doc_norm else 0.0
    return _rank_scores(scores, mapping, limit)


def _postings_query(conn: sqlite3.Connection, query: dict[int, float], limit: int) -> list[tuple[str, float]]:
    query_norm = math.sqrt(sum(value * value for value in query.values()))
    conn.execute("DROP TABLE IF EXISTS temp_query_terms")
    conn.execute("CREATE TEMP TABLE temp_query_terms(term_id INTEGER PRIMARY KEY, weight REAL NOT NULL)")
    conn.executemany("INSERT INTO temp_query_terms VALUES(?,?)", query.items())
    rows = conn.execute(
        "SELECT cm.chunk_id, SUM(float32_blob(p.weight_blob) * q.weight) AS dot, cm.norm "
        "FROM sparse_postings p JOIN temp_query_terms q ON q.term_id=p.term_id "
        "JOIN chunk_map cm ON cm.rowid=p.chunk_rowid GROUP BY p.chunk_rowid ORDER BY dot / cm.norm DESC, cm.chunk_id LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        (str(chunk_id), float(dot) / (query_norm * float(norm)) if query_norm and norm else 0.0)
        for chunk_id, dot, norm in rows
    ]


def _baseline_query(vectors: dict[str, list[tuple[str, str, float]]], query: dict[str, float], limit: int) -> list[tuple[str, float]]:
    query_by_id = dict(query)
    query_norm = math.sqrt(sum(value * value for value in query_by_id.values()))
    scores: dict[str, float] = {}
    for owner, rows in vectors.items():
        document = {token_id: weight for token_id, _, weight in rows}
        scores[owner] = sparse_cosine(query_by_id, document, query_norm=query_norm)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def run_sparse_backend_spike(*, source_db: Path, output_dir: Path, chunk_limit: int = 15000, query_count: int = 32, top_k: int = 20) -> dict[str, Any]:
    if not source_db.exists():
        raise FileNotFoundError(source_db)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    owners, vectors, model_name, space_id = _load_subset(source_db, chunk_limit)
    legacy_load_ms = (time.perf_counter() - load_started) * 1000
    term_to_id, id_to_external, text_by_id = _build_vocab(vectors)
    binary_path = output_dir / "compact_blob.db"
    postings_path = output_dir / "compact_postings.db"
    for path in (binary_path, postings_path):
        if path.exists():
            path.unlink()
    blob_info = _materialize_compact(binary_path, variant="blob", owners=owners, terms=vectors, term_to_id=term_to_id, id_to_external=id_to_external, text_by_id=text_by_id, model_name=model_name, space_id=space_id)
    postings_info = _materialize_compact(postings_path, variant="postings", owners=owners, terms=vectors, term_to_id=term_to_id, id_to_external=id_to_external, text_by_id=text_by_id, model_name=model_name, space_id=space_id)
    query_owners = [owners[min(index * len(owners) // query_count, len(owners) - 1)] for index in range(query_count)]
    queries = [{term_to_id[token_id]: weight for token_id, _, weight in vectors[owner]} for owner in query_owners]
    baseline_queries = [{token_id: weight for token_id, _, weight in vectors[owner]} for owner in query_owners]
    blob_load_started = time.perf_counter()
    blob_conn, blob_records, blob_mapping = _load_blob_backend(binary_path)
    blob_load_ms = (time.perf_counter() - blob_load_started) * 1000
    postings_load_started = time.perf_counter()
    postings_conn = sqlite3.connect(postings_path)
    postings_conn.create_function("float32_blob", 1, lambda blob: struct.unpack("<f", bytes(blob))[0])
    postings_load_ms = (time.perf_counter() - postings_load_started) * 1000
    results: dict[str, Any] = {}
    for label, fn in (
        ("legacy", lambda query, raw: _baseline_query(vectors, raw, top_k)),
        ("compact_blob", lambda query, raw: _blob_query(blob_records, blob_mapping, query, top_k)),
        ("compact_postings", lambda query, raw: _postings_query(postings_conn, query, top_k)),
    ):
        times: list[float] = []
        parity: list[dict[str, Any]] = []
        rss_before = _rss_bytes()
        for query, raw in zip(queries, baseline_queries, strict=True):
            started = time.perf_counter()
            actual = fn(query, raw)
            times.append((time.perf_counter() - started) * 1000)
            if label != "legacy":
                expected = _baseline_query(vectors, raw, top_k)
                expected_ids = [item[0] for item in expected]
                actual_ids = [item[0] for item in actual]
                parity.append({
                    "top_k_identical": expected_ids == actual_ids,
                    "top_k_overlap": len(set(expected_ids) & set(actual_ids)) / top_k,
                    "max_score_abs_diff": max(
                        (abs(dict(expected).get(owner, 0.0) - dict(actual).get(owner, 0.0)) for owner in set(expected_ids) | set(actual_ids)),
                        default=0.0,
                    ),
                })
        results[label] = {
            "cold_load_ms": {
                "legacy": legacy_load_ms,
                "compact_blob": blob_load_ms,
                "compact_postings": postings_load_ms,
            }[label],
            "query_p50_ms": _percentile(times, 50),
            "query_p95_ms": _percentile(times, 95),
            "query_p99_ms": _percentile(times, 99),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": _rss_bytes(),
            "rss_scope": "single-process run; values include the shared subset structures",
            "parity": {
                "queries": len(parity),
                "top_k_identical_count": sum(item["top_k_identical"] for item in parity),
                "mean_top_k_overlap": statistics.fmean(item["top_k_overlap"] for item in parity) if parity else 1.0,
                "max_score_abs_diff": max((item["max_score_abs_diff"] for item in parity), default=0.0),
            },
        }
    blob_conn.close()
    postings_conn.close()
    total_terms = blob_info["terms"]
    full_term_count = 17231110
    for info in (blob_info, postings_info):
        info["bytes_per_term"] = info["bytes"] / total_terms if total_terms else 0.0
        info["estimated_full_sparse_bytes"] = info["bytes"] * full_term_count / total_terms if total_terms else 0.0
    report = {
        "schema_version": SPIKE_SCHEMA_VERSION,
        "source_db": str(source_db),
        "subset": {"chunks": len(owners), "terms": total_terms, "vocabulary": len(id_to_external), "queries": len(queries), "top_k": top_k},
        "score_semantics": {"metric": "cosine", "formula": "dot(query, document) / (||query|| * ||document||)", "fusion": "0.65 * dense + 0.35 * sparse", "tie_break": "score descending, chunk_id ascending"},
        "variants": {"compact_blob": blob_info, "compact_postings": postings_info},
        "results": results,
        "recommendation": "compact_blob_with_in_memory_index",
        "warnings": ["Queries are existing sparse document representations, not newly embedded query text.", "Full-size estimates scale the representative subset linearly."],
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Sparse Backend Spike",
        "",
        f"Source DB: {report['source_db']}",
        f"Subset: {report['subset']['chunks']} chunks, {report['subset']['terms']} terms, {report['subset']['vocabulary']} vocabulary terms",
        "",
        "| Variant | Size GiB | Bytes/term | Cold load ms | Query p50 ms | Query p95 ms | Top-20 exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, key in (("Legacy", "legacy"), ("Compact BLOB", "compact_blob"), ("Inverted postings", "compact_postings")):
        info = report["variants"].get(key, {})
        result = report["results"].get(key, {})
        parity = result.get("parity", {})
        lines.append(f"| {name} | {info.get('bytes', 0) / 1024**3:.4f} | {info.get('bytes_per_term', 0):.2f} | {result.get('cold_load_ms', 0):.3f} | {result.get('query_p50_ms', 0):.3f} | {result.get('query_p95_ms', 0):.3f} | {parity.get('top_k_identical_count', '-')}/{parity.get('queries', '-')} |")
    lines += ["", f"Recommendation: **{report['recommendation']}**", "", "The benchmark uses existing sparse document representations as deterministic query vectors; no models or production DB writes are involved.", ""]
    return "\n".join(lines)
