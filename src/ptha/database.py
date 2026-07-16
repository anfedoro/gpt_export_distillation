"""Read-only database diagnostics for the PTHA product shell."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

REQUIRED_TABLES = {
    "source_documents", "conversations", "messages", "blocks", "retrieval_chunks",
    "dense_native_metadata", "sparse_vector_metadata", "sparse_vectors_compact", "native_build_audit",
}
INCREMENTAL_TABLES = {
    "source_entity_identities", "source_entity_revisions", "conversation_source_lineage",
    "message_source_lineage", "block_source_lineage", "chunk_incremental_metadata", "generation_manifests",
}


def inspect_database(path: Path, *, integrity: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "state": "missing"}
    if not path.is_file():
        return result
    result["size_bytes"] = path.stat().st_size
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            names = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
            missing = sorted(REQUIRED_TABLES - names)
            result["missing_tables"] = missing
            if missing:
                result["state"] = "incompatible"
                return result
            counts = {}
            for table in ("conversations", "messages", "retrieval_chunks", "dense_native_metadata", "sparse_vector_metadata"):
                counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            audit = conn.execute("SELECT schema_version,status,started_at,finished_at,contracts_json FROM native_build_audit WHERE id=1").fetchone()
            result["counts"] = counts
            result["sqlite_user_version"] = int(conn.execute("PRAGMA user_version").fetchone()[0])
            incremental_available = INCREMENTAL_TABLES <= names
            result["incremental_metadata"] = {"available": incremental_available}
            if incremental_available:
                manifest_columns = {
                    str(row[1]) for row in conn.execute("PRAGMA table_info(generation_manifests)").fetchall()
                }
                canonical_generation = {
                    "parser_contract", "canonical_representation_version", "source_transform_version",
                } <= manifest_columns
                selected = (
                    "generation_id,manifest_schema_version,created_at,parser_contract,"
                    "canonical_representation_version,canonicalizer_version,source_transform_version,"
                    "block_builder_version,chunker_version,embedding_contract_fingerprint,database_content_fingerprint"
                    if canonical_generation
                    else
                    "generation_id,manifest_schema_version,created_at,canonicalizer_version,"
                    "block_builder_version,chunker_version,embedding_contract_fingerprint,database_content_fingerprint"
                )
                manifest = conn.execute(
                    f"SELECT {selected} FROM generation_manifests "
                    "ORDER BY created_at DESC,generation_id DESC LIMIT 1"
                ).fetchone()
                if manifest:
                    if canonical_generation:
                        generation = {
                            "id": manifest[0], "manifest_schema_version": manifest[1], "created_at": manifest[2],
                            "parser_contract": manifest[3], "canonical_representation_version": manifest[4],
                            "canonicalizer_version": manifest[5], "source_transform_version": manifest[6],
                            "block_builder_version": manifest[7], "chunker_version": manifest[8],
                            "embedding_contract_fingerprint": manifest[9],
                            "database_content_fingerprint": manifest[10],
                            "representation": "canonical",
                        }
                    else:
                        generation = {
                            "id": manifest[0], "manifest_schema_version": manifest[1], "created_at": manifest[2],
                            "canonicalizer_version": manifest[3], "block_builder_version": manifest[4],
                            "chunker_version": manifest[5], "embedding_contract_fingerprint": manifest[6],
                            "database_content_fingerprint": manifest[7],
                            "representation": "legacy_precanonical",
                        }
                    result["incremental_metadata"]["generation"] = generation
            if audit:
                result.update({"schema_version": audit[0], "build_status": audit[1], "imported": audit[3]})
                try:
                    contracts = json.loads(audit[4])
                except (TypeError, json.JSONDecodeError):
                    contracts = {}
                result["models"] = {
                    "dense": contracts.get("dense", {}).get("model"),
                    "sparse": contracts.get("sparse", {}).get("model"),
                }
                result["chunk_policy"] = contracts.get("chunk_policy")
            complete = counts["retrieval_chunks"] == counts["dense_native_metadata"] == counts["sparse_vector_metadata"]
            result["state"] = "ready" if complete and (not audit or audit[1] == "completed") else "incomplete"
            if integrity:
                result["integrity_check"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
                result["foreign_key_errors"] = len(conn.execute("PRAGMA foreign_key_check").fetchall())
    except sqlite3.Error as exc:
        result.update({"state": "corrupt", "error": type(exc).__name__})
    return result
