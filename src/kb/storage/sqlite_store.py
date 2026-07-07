from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.ingest.tree_walker import InventoryItem
from kb.ingest.attachment_parser import ParsedAttachment
from kb.model.entities import Block, Conversation, Message, ParsedChat
from kb.model.ids import stable_id


SCHEMA_PATH = Path(__file__).with_name("sqlite_schema.sql")


@dataclass(frozen=True)
class DbCapabilities:
    has_block_embeddings: bool
    has_sparse_terms: bool
    has_deterministic_nodes: bool
    has_similarity_edges: bool
    has_semantic_groups: bool
    has_group_embeddings: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "has_block_embeddings": self.has_block_embeddings,
            "has_sparse_terms": self.has_sparse_terms,
            "has_deterministic_nodes": self.has_deterministic_nodes,
            "has_similarity_edges": self.has_similarity_edges,
            "has_semantic_groups": self.has_semantic_groups,
            "has_group_embeddings": self.has_group_embeddings,
        }


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _ensure_interest_tier_columns(conn)
        conn.commit()
    finally:
        conn.close()


def _ensure_interest_tier_columns(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "source_documents", "interest_tier", "TEXT NOT NULL DEFAULT 'normal'")
    _add_column_if_missing(conn, "knowledge_blocks", "interest_tier", "TEXT NOT NULL DEFAULT 'normal'")
    conn.execute("UPDATE source_documents SET interest_tier = 'low' WHERE folder_kind = 'common_trash'")
    conn.execute("UPDATE knowledge_blocks SET interest_tier = 'low' WHERE folder_kind = 'common_trash'")


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


class SQLiteStore:
    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        self.db_path = db_path
        self.conn = connect(db_path, read_only=read_only)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def capabilities(self) -> DbCapabilities:
        block_embeddings = self.conn.execute(
            "SELECT COUNT(*) FROM dense_vectors WHERE owner_type = 'knowledge_block'"
        ).fetchone()[0]
        sparse_terms = self.conn.execute(
            "SELECT COUNT(*) FROM sparse_terms WHERE owner_type = 'knowledge_block'"
        ).fetchone()[0]
        deterministic_nodes = self.conn.execute(
            "SELECT COUNT(*) FROM semantic_nodes WHERE node_type IN ('conversation', 'project', 'attachment')"
        ).fetchone()[0]
        similarity_edges = self.conn.execute(
            "SELECT COUNT(*) FROM semantic_edges"
        ).fetchone()[0]
        semantic_groups = self.conn.execute(
            "SELECT COUNT(*) FROM semantic_nodes WHERE node_type = 'semantic_group'"
        ).fetchone()[0]
        group_embeddings = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM semantic_nodes sn
            WHERE sn.node_type = 'semantic_group'
              AND (
                sn.dense_vector_id IS NOT NULL
                OR EXISTS (
                    SELECT 1
                    FROM sparse_terms st
                    WHERE st.owner_type = 'semantic_node'
                      AND st.owner_id = sn.id
                )
              )
            """
        ).fetchone()[0]
        return DbCapabilities(
            has_block_embeddings=block_embeddings > 0,
            has_sparse_terms=sparse_terms > 0,
            has_deterministic_nodes=deterministic_nodes > 0,
            has_similarity_edges=similarity_edges > 0,
            has_semantic_groups=semantic_groups > 0,
            has_group_embeddings=group_embeddings > 0,
        )

    def upsert_source_document(self, input_root: Path, item: InventoryItem) -> str:
        source_id = stable_id(item.relative_path, item.sha256, prefix="src")
        path = input_root / item.relative_path
        self.conn.execute(
            """
            INSERT INTO source_documents (
                id, path, relative_path, source_kind, folder_kind, interest_tier, project_id, project_name,
                file_name, extension, sha256, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path, sha256) DO UPDATE SET
                path=excluded.path,
                source_kind=excluded.source_kind,
                folder_kind=excluded.folder_kind,
                interest_tier=excluded.interest_tier,
                project_id=excluded.project_id,
                project_name=excluded.project_name,
                file_name=excluded.file_name,
                extension=excluded.extension,
                metadata_json=excluded.metadata_json
            """,
            (
                source_id,
                str(path),
                item.relative_path,
                item.detected_kind,
                item.folder_kind,
                item.interest_tier,
                item.project_path,
                item.project_path,
                item.file_name,
                item.extension,
                item.sha256,
                _json({"size": item.size, "is_attachment": item.is_attachment}),
            ),
        )
        return source_id

    def insert_parsed_chat(self, parsed: ParsedChat) -> None:
        conversation = parsed.conversation
        self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation.id,))
        self._insert_conversation(conversation)
        for message in parsed.messages:
            self._insert_message(message)
        for block in parsed.blocks:
            self._insert_block(block)
        for block in parsed.blocks:
            self._insert_knowledge_block(conversation, parsed.messages, block)

    def insert_parsed_attachment(self, input_root: Path, item: InventoryItem, source_document_id: str, parsed: ParsedAttachment) -> None:
        attachment_id = stable_id(source_document_id, "attachment", prefix="att")
        path = input_root / item.relative_path
        self.conn.execute("DELETE FROM attachment_documents WHERE id = ?", (attachment_id,))
        self.conn.execute(
            """
            INSERT INTO attachment_documents (
                id, source_document_id, linked_conversation_id, linked_message_id, path,
                relative_path, file_name, extension, mime_type, sha256, extraction_status,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attachment_id,
                source_document_id,
                None,
                None,
                str(path),
                item.relative_path,
                item.file_name,
                item.extension,
                parsed.mime_type,
                item.sha256,
                parsed.extraction_status,
                _json(parsed.metadata_json),
            ),
        )
        self.conn.execute(
            "DELETE FROM knowledge_blocks WHERE source_type = 'attachment_block' AND attachment_id = ?",
            (attachment_id,),
        )
        for block in parsed.blocks:
            text_for_display = block.text
            text_for_embedding = "\n".join(
                part
                for part in [
                    f"Project: {item.project_path}" if item.project_path else None,
                    f"Attachment: {item.relative_path}",
                    f"Block type: {block.block_type}",
                    f"Content: {block.text}",
                ]
                if part
            )
            self.conn.execute(
                """
                INSERT INTO knowledge_blocks (
                    id, source_type, source_document_id, conversation_id, message_id, block_id,
                    attachment_id, project_id, folder_kind, interest_tier, role, block_type, text_for_embedding,
                    text_for_display, token_count_estimate, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_id("attachment_block", attachment_id, block.ordinal, prefix="kb"),
                    "attachment_block",
                    source_document_id,
                    None,
                    None,
                    None,
                    attachment_id,
                    item.project_path,
                    item.folder_kind,
                    item.interest_tier,
                    None,
                    block.block_type,
                    text_for_embedding,
                    text_for_display,
                    max(1, len(text_for_embedding.split())),
                    _json(block.metadata_json),
                ),
            )

    def commit(self) -> None:
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        names = [
            "source_documents",
            "conversations",
            "messages",
            "blocks",
            "knowledge_blocks",
            "attachment_documents",
            "dense_vectors",
            "sparse_terms",
            "semantic_nodes",
            "semantic_node_members",
            "semantic_edges",
        ]
        result = {name: int(self.conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in names}
        result["attachments_seen"] = int(
            self.conn.execute("SELECT COUNT(*) FROM source_documents WHERE source_kind = 'attachment'").fetchone()[0]
        )
        return result

    def knowledge_blocks_for_embedding(
        self,
        *,
        limit: int | None = None,
        dense_model_name: str | None = None,
        dense_model_version: str | None = None,
        sparse_model_name: str | None = None,
        force: bool = False,
        skip_low_interest_content: bool = True,
    ) -> list[sqlite3.Row]:
        query, params = self._knowledge_blocks_for_embedding_query(
            limit=limit,
            dense_model_name=dense_model_name,
            dense_model_version=dense_model_version,
            sparse_model_name=sparse_model_name,
            force=force,
            skip_low_interest_content=skip_low_interest_content,
        )
        return list(self.conn.execute(query, params).fetchall())

    def count_knowledge_blocks_for_embedding(
        self,
        *,
        limit: int | None = None,
        dense_model_name: str | None = None,
        dense_model_version: str | None = None,
        sparse_model_name: str | None = None,
        force: bool = False,
        skip_low_interest_content: bool = True,
    ) -> int:
        query, params = self._knowledge_blocks_for_embedding_query(
            limit=limit,
            dense_model_name=dense_model_name,
            dense_model_version=dense_model_version,
            sparse_model_name=sparse_model_name,
            force=force,
            skip_low_interest_content=skip_low_interest_content,
            count_only=True,
        )
        return int(self.conn.execute(query, params).fetchone()[0])

    def knowledge_blocks_for_embedding_batch(
        self,
        *,
        after_id: str | None,
        batch_size: int,
        dense_model_name: str | None = None,
        dense_model_version: str | None = None,
        sparse_model_name: str | None = None,
        force: bool = False,
        skip_low_interest_content: bool = True,
    ) -> list[sqlite3.Row]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        query, params = self._knowledge_blocks_for_embedding_query(
            limit=batch_size,
            dense_model_name=dense_model_name,
            dense_model_version=dense_model_version,
            sparse_model_name=sparse_model_name,
            force=force,
            skip_low_interest_content=skip_low_interest_content,
            after_id=after_id,
        )
        return list(self.conn.execute(query, params).fetchall())

    def _knowledge_blocks_for_embedding_query(
        self,
        *,
        limit: int | None,
        dense_model_name: str | None,
        dense_model_version: str | None,
        sparse_model_name: str | None,
        force: bool,
        skip_low_interest_content: bool,
        after_id: str | None = None,
        count_only: bool = False,
    ) -> tuple[str, list[Any]]:
        where: list[str] = []
        pending: list[str] = []
        params: list[Any] = []
        if skip_low_interest_content:
            where.append("interest_tier NOT IN ('low', 'quarantine')")
        if not force and dense_model_name is not None:
            pending.append(
                """
                NOT EXISTS (
                    SELECT 1 FROM dense_vectors dv
                    WHERE dv.owner_type = 'knowledge_block'
                      AND dv.owner_id = knowledge_blocks.id
                      AND dv.model_name = ?
                      AND COALESCE(dv.model_version, '') = COALESCE(?, '')
                )
                """
            )
            params.extend([dense_model_name, dense_model_version])
        if not force and sparse_model_name is not None:
            pending.append(
                """
                NOT EXISTS (
                    SELECT 1 FROM sparse_terms st
                    WHERE st.owner_type = 'knowledge_block'
                      AND st.owner_id = knowledge_blocks.id
                      AND st.model_name = ?
                )
                """
            )
            params.append(sparse_model_name)
        if pending:
            where.append("(" + " OR ".join(f"({clause})" for clause in pending) + ")")
        if after_id is not None:
            where.append("id > ?")
            params.append(after_id)
        query = "SELECT COUNT(*) AS cnt FROM knowledge_blocks" if count_only else "SELECT id, text_for_embedding FROM knowledge_blocks"
        if where:
            query += " WHERE " + " AND ".join(f"({clause})" for clause in where)
        if not count_only:
            query += " ORDER BY id"
        if limit is not None and not count_only:
            query += " LIMIT ?"
            params.append(limit)
        if limit is not None and count_only:
            query = f"SELECT MIN(cnt, ?) FROM ({query})"
            params = [limit, *params]
        return query, params

    def upsert_dense_vector(
        self,
        *,
        owner_type: str,
        owner_id: str,
        model_name: str,
        model_version: str | None,
        vector: list[float],
    ) -> str:
        vector_id = stable_id(owner_type, owner_id, model_name, model_version, "dense", prefix="vec")
        self.conn.execute(
            """
            INSERT INTO dense_vectors (
                id, owner_type, owner_id, model_name, model_version, dim, vector_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_type, owner_id, model_name, model_version) DO UPDATE SET
                dim=excluded.dim,
                vector_json=excluded.vector_json
            """,
            (
                vector_id,
                owner_type,
                owner_id,
                model_name,
                model_version,
                len(vector),
                json.dumps(vector, separators=(",", ":")),
            ),
        )
        return vector_id

    def replace_sparse_terms(
        self,
        *,
        owner_type: str,
        owner_id: str,
        model_name: str,
        terms: dict[str, float],
    ) -> str:
        sparse_id = stable_id(owner_type, owner_id, model_name, "sparse", prefix="sparse")
        self.conn.execute(
            "DELETE FROM sparse_terms WHERE owner_type = ? AND owner_id = ? AND model_name = ?",
            (owner_type, owner_id, model_name),
        )
        self.conn.executemany(
            """
            INSERT INTO sparse_terms (
                owner_type, owner_id, token_id, token_text, weight, model_name
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    owner_type,
                    owner_id,
                    stable_id(model_name, token),
                    token,
                    float(weight),
                    model_name,
                )
                for token, weight in terms.items()
            ),
        )
        return sparse_id

    def set_knowledge_block_vector_ids(
        self,
        *,
        knowledge_block_id: str,
        dense_vector_id: str | None,
        sparse_vector_id: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE knowledge_blocks
            SET dense_vector_id = COALESCE(?, dense_vector_id),
                sparse_vector_id = COALESCE(?, sparse_vector_id)
            WHERE id = ?
            """,
            (dense_vector_id, sparse_vector_id, knowledge_block_id),
        )

    def searchable_knowledge_blocks(
        self,
        *,
        dense_model_name: str | None,
        dense_model_version: str | None,
        sparse_model_name: str | None,
        project: str | None = None,
        include_low_interest: bool = False,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        dense_join = ""
        sparse_join = ""
        if dense_model_name is not None:
            dense_join = """
            LEFT JOIN dense_vectors dv
              ON dv.owner_type = 'knowledge_block'
             AND dv.owner_id = kb.id
             AND dv.model_name = ?
             AND COALESCE(dv.model_version, '') = COALESCE(?, '')
            """
            params.extend([dense_model_name, dense_model_version])
        if sparse_model_name is not None:
            sparse_join = """
            LEFT JOIN sparse_terms st
              ON st.owner_type = 'knowledge_block'
             AND st.owner_id = kb.id
             AND st.model_name = ?
            """
            params.append(sparse_model_name)
        where = []
        embedding_filters = []
        if dense_model_name is not None:
            embedding_filters.append("dv.id IS NOT NULL")
        if sparse_model_name is not None:
            embedding_filters.append("st.owner_id IS NOT NULL")
        if embedding_filters:
            where.append("(" + " OR ".join(embedding_filters) + ")")
        if not include_low_interest:
            where.append("kb.interest_tier NOT IN ('low', 'quarantine')")
        if project is not None:
            where.append("kb.project_id = ?")
            params.append(project)
        query = f"""
            SELECT
                kb.id AS knowledge_block_id,
                kb.project_id,
                kb.folder_kind,
                kb.interest_tier,
                kb.conversation_id,
                kb.message_id,
                kb.role,
                kb.block_type,
                kb.interest_tier,
                kb.text_for_display,
                sd.relative_path AS source_path,
                c.title AS conversation_title,
                dv.vector_json AS dense_vector_json,
                st.token_text,
                st.weight
            FROM knowledge_blocks kb
            JOIN source_documents sd ON sd.id = kb.source_document_id
            LEFT JOIN conversations c ON c.id = kb.conversation_id
            {dense_join}
            {sparse_join}
        """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY kb.id"
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.conn.execute(query, params).fetchall():
            block_id = str(row["knowledge_block_id"])
            item = grouped.setdefault(
                block_id,
                {
                    "knowledge_block_id": block_id,
                    "project_id": row["project_id"],
                    "folder_kind": row["folder_kind"],
                    "interest_tier": row["interest_tier"],
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                    "role": row["role"],
                    "block_type": row["block_type"],
                    "text_for_display": row["text_for_display"],
                    "source_path": row["source_path"],
                    "conversation_title": row["conversation_title"],
                    "dense_vector": json.loads(row["dense_vector_json"]) if row["dense_vector_json"] else None,
                    "sparse_terms": {},
                },
            )
            if row["token_text"] is not None:
                item["sparse_terms"][row["token_text"]] = float(row["weight"])
        return list(grouped.values())

    def knowledge_blocks_for_nodes(self) -> list[dict[str, Any]]:
        query = """
            SELECT
                kb.id AS knowledge_block_id,
                kb.project_id,
                kb.conversation_id,
                kb.attachment_id,
                c.title AS conversation_title,
                ad.relative_path AS attachment_path,
                dv.vector_json AS dense_vector_json,
                st.token_text,
                st.weight
            FROM knowledge_blocks kb
            LEFT JOIN conversations c ON c.id = kb.conversation_id
            LEFT JOIN attachment_documents ad ON ad.id = kb.attachment_id
            LEFT JOIN dense_vectors dv ON dv.id = kb.dense_vector_id
            LEFT JOIN sparse_terms st
              ON st.owner_type = 'knowledge_block'
             AND st.owner_id = kb.id
            ORDER BY kb.id
        """
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.conn.execute(query).fetchall():
            block_id = str(row["knowledge_block_id"])
            item = grouped.setdefault(
                block_id,
                {
                    "knowledge_block_id": block_id,
                    "project_id": row["project_id"],
                    "conversation_id": row["conversation_id"],
                    "attachment_id": row["attachment_id"],
                    "conversation_title": row["conversation_title"],
                    "attachment_path": row["attachment_path"],
                    "dense_vector": json.loads(row["dense_vector_json"]) if row["dense_vector_json"] else None,
                    "sparse_terms": {},
                },
            )
            if row["token_text"] is not None:
                item["sparse_terms"][row["token_text"]] = float(row["weight"])
        return list(grouped.values())

    def upsert_semantic_node(
        self,
        *,
        node_id: str,
        node_type: str,
        project_id: str | None,
        dense_vector_id: str | None,
        sparse_vector_id: str | None,
        title: str,
        summary: str | None,
        top_terms_json: str,
        metadata_json: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO semantic_nodes (
                id, node_type, project_id, dense_vector_id, sparse_vector_id,
                title, summary, top_terms_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                node_type=excluded.node_type,
                project_id=excluded.project_id,
                dense_vector_id=excluded.dense_vector_id,
                sparse_vector_id=excluded.sparse_vector_id,
                title=excluded.title,
                summary=excluded.summary,
                top_terms_json=excluded.top_terms_json,
                metadata_json=excluded.metadata_json
            """,
            (
                node_id,
                node_type,
                project_id,
                dense_vector_id,
                sparse_vector_id,
                title,
                summary,
                top_terms_json,
                metadata_json,
            ),
        )

    def replace_semantic_node_members(self, *, node_id: str, members: list[dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM semantic_node_members WHERE node_id = ?", (node_id,))
        self.conn.executemany(
            """
            INSERT INTO semantic_node_members (
                node_id, knowledge_block_id, membership_weight, membership_reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    node_id,
                    member["knowledge_block_id"],
                    member["membership_weight"],
                    member["membership_reason"],
                    member["metadata_json"],
                )
                for member in members
            ],
        )

    def edge_candidate_groups(self, *, scope: str) -> dict[str, list[dict[str, Any]]]:
        if scope == "conversation":
            group_expr = "kb.conversation_id"
            where = "kb.conversation_id IS NOT NULL"
        elif scope == "project":
            group_expr = "kb.project_id"
            where = "kb.project_id IS NOT NULL"
        elif scope == "attachment":
            group_expr = "kb.attachment_id"
            where = "kb.attachment_id IS NOT NULL"
        else:
            raise ValueError(f"Unsupported edge scope: {scope}")
        query = f"""
            SELECT
                {group_expr} AS group_id,
                kb.id AS knowledge_block_id,
                kb.conversation_id,
                m.ordinal AS message_ordinal,
                b.ordinal AS block_ordinal,
                dv.vector_json AS dense_vector_json,
                st.token_text,
                st.weight
            FROM knowledge_blocks kb
            LEFT JOIN messages m ON m.id = kb.message_id
            LEFT JOIN blocks b ON b.id = kb.block_id
            LEFT JOIN dense_vectors dv ON dv.id = kb.dense_vector_id
            LEFT JOIN sparse_terms st
              ON st.owner_type = 'knowledge_block'
             AND st.owner_id = kb.id
            WHERE {where}
            ORDER BY group_id, kb.id
        """
        grouped_blocks: dict[str, dict[str, dict[str, Any]]] = {}
        for row in self.conn.execute(query).fetchall():
            group_id = str(row["group_id"])
            block_id = str(row["knowledge_block_id"])
            group = grouped_blocks.setdefault(group_id, {})
            item = group.setdefault(
                block_id,
                {
                    "knowledge_block_id": block_id,
                    "conversation_id": row["conversation_id"],
                    "message_ordinal": row["message_ordinal"],
                    "block_ordinal": row["block_ordinal"],
                    "dense_vector": json.loads(row["dense_vector_json"]) if row["dense_vector_json"] else None,
                    "sparse_terms": {},
                },
            )
            if row["token_text"] is not None:
                item["sparse_terms"][row["token_text"]] = float(row["weight"])
        return {
            group_id: list(blocks.values())
            for group_id, blocks in grouped_blocks.items()
            if len(blocks) >= 2
        }

    def upsert_semantic_edges(self, edges: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO semantic_edges (
                id, src_type, src_id, dst_type, dst_id, edge_kind, weight,
                dense_similarity, sparse_similarity, shared_terms_json, metadata_json,
                policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(src_type, src_id, dst_type, dst_id, edge_kind, policy_version)
            DO UPDATE SET
                weight=excluded.weight,
                dense_similarity=excluded.dense_similarity,
                sparse_similarity=excluded.sparse_similarity,
                shared_terms_json=excluded.shared_terms_json,
                metadata_json=excluded.metadata_json
            """,
            [
                (
                    edge["id"],
                    edge["src_type"],
                    edge["src_id"],
                    edge["dst_type"],
                    edge["dst_id"],
                    edge["edge_kind"],
                    edge["weight"],
                    edge["dense_similarity"],
                    edge["sparse_similarity"],
                    edge["shared_terms_json"],
                    edge["metadata_json"],
                    edge["policy_version"],
                )
                for edge in edges
            ],
        )

    def blocks_by_ids(self, block_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not block_ids:
            return {}
        placeholders = ",".join("?" for _ in block_ids)
        query = f"""
            SELECT
                kb.id AS knowledge_block_id,
                kb.project_id,
                kb.conversation_id,
                kb.message_id,
                kb.role,
                kb.block_type,
                kb.interest_tier,
                kb.text_for_display,
                kb.token_count_estimate,
                sd.relative_path AS source_path,
                c.title AS conversation_title
            FROM knowledge_blocks kb
            JOIN source_documents sd ON sd.id = kb.source_document_id
            LEFT JOIN conversations c ON c.id = kb.conversation_id
            WHERE kb.id IN ({placeholders})
        """
        return {
            str(row["knowledge_block_id"]): dict(row)
            for row in self.conn.execute(query, block_ids).fetchall()
        }

    def semantic_nodes_for_search(self, *, project: str | None = None, node_types: list[str] | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        where_parts: list[str] = []
        if project is not None:
            where_parts.append("sn.project_id = ?")
            params.append(project)
        if node_types:
            placeholders = ", ".join("?" for _ in node_types)
            where_parts.append(f"sn.node_type IN ({placeholders})")
            params.extend(node_types)
        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        query = f"""
            SELECT
                sn.id AS node_id,
                sn.node_type,
                sn.project_id,
                sn.title,
                sn.top_terms_json,
                dv.vector_json AS dense_vector_json,
                st.token_text,
                st.weight
            FROM semantic_nodes sn
            LEFT JOIN dense_vectors dv ON dv.id = sn.dense_vector_id
            LEFT JOIN sparse_terms st
              ON st.owner_type = 'semantic_node'
             AND st.owner_id = sn.id
            {where}
            ORDER BY sn.id
        """
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.conn.execute(query, params).fetchall():
            node_id = str(row["node_id"])
            item = grouped.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "node_type": row["node_type"],
                    "project_id": row["project_id"],
                    "title": row["title"],
                    "top_terms_json": row["top_terms_json"],
                    "dense_vector": json.loads(row["dense_vector_json"]) if row["dense_vector_json"] else None,
                    "sparse_terms": {},
                },
            )
            if row["token_text"] is not None:
                item["sparse_terms"][row["token_text"]] = float(row["weight"])
        return list(grouped.values())

    def semantic_node_member_blocks(self, node_id: str, *, limit: int, include_low_interest: bool = False) -> list[dict[str, Any]]:
        interest_filter = "" if include_low_interest else "AND kb.interest_tier NOT IN ('low', 'quarantine')"
        query = """
            SELECT
                kb.id AS knowledge_block_id,
                kb.project_id,
                kb.conversation_id,
                kb.message_id,
                kb.role,
                kb.block_type,
                kb.interest_tier,
                kb.text_for_display,
                kb.token_count_estimate,
                sd.relative_path AS source_path,
                c.title AS conversation_title,
                snm.membership_weight
            FROM semantic_node_members snm
            JOIN knowledge_blocks kb ON kb.id = snm.knowledge_block_id
            JOIN source_documents sd ON sd.id = kb.source_document_id
            LEFT JOIN conversations c ON c.id = kb.conversation_id
            WHERE snm.node_id = ?
              {interest_filter}
            ORDER BY snm.membership_weight DESC, kb.token_count_estimate DESC, kb.id
            LIMIT ?
        """.format(interest_filter=interest_filter)
        return [dict(row) for row in self.conn.execute(query, (node_id, limit)).fetchall()]

    def neighbor_blocks(self, block_ids: list[str], *, limit: int, include_low_interest: bool = False) -> list[dict[str, Any]]:
        if not block_ids:
            return []
        placeholders = ",".join("?" for _ in block_ids)
        query = """
            WITH edge_hits AS (
                SELECT
                    CASE WHEN src_id IN ({placeholders}) THEN dst_id ELSE src_id END AS neighbor_id,
                    CASE WHEN src_id IN ({placeholders}) THEN src_id ELSE dst_id END AS from_block_id,
                    weight,
                    edge_kind
                FROM semantic_edges
                WHERE src_type = 'block'
                  AND dst_type = 'block'
                  AND (src_id IN ({placeholders}) OR dst_id IN ({placeholders}))
            )
            SELECT
                kb.id AS knowledge_block_id,
                kb.project_id,
                kb.conversation_id,
                kb.message_id,
                kb.role,
                kb.block_type,
                kb.interest_tier,
                kb.text_for_display,
                kb.token_count_estimate,
                sd.relative_path AS source_path,
                c.title AS conversation_title,
                eh.from_block_id,
                eh.weight AS edge_weight,
                eh.edge_kind
            FROM edge_hits eh
            JOIN knowledge_blocks kb ON kb.id = eh.neighbor_id
            JOIN source_documents sd ON sd.id = kb.source_document_id
            LEFT JOIN conversations c ON c.id = kb.conversation_id
            {interest_filter}
            ORDER BY eh.weight DESC, kb.id
            LIMIT ?
        """.format(
            placeholders=placeholders,
            interest_filter="" if include_low_interest else "WHERE kb.interest_tier NOT IN ('low', 'quarantine')",
        )
        params = block_ids + block_ids + block_ids + block_ids + [limit]
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]

    def _insert_conversation(self, conversation: Conversation) -> None:
        self.conn.execute(
            """
            INSERT INTO conversations (
                id, source_document_id, conversation_id, conversation_template_id, title,
                create_time_utc, update_time_utc, message_count, assistant_messages,
                user_messages, text_chars, estimated_code_blocks, project_id, folder_kind,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation.id,
                conversation.source_document_id,
                conversation.conversation_id,
                conversation.conversation_template_id,
                conversation.title,
                conversation.create_time_utc,
                conversation.update_time_utc,
                conversation.message_count,
                conversation.assistant_messages,
                conversation.user_messages,
                conversation.text_chars,
                conversation.estimated_code_blocks,
                conversation.project_id,
                conversation.folder_kind,
                _json(conversation.metadata_json),
            ),
        )

    def _insert_message(self, message: Message) -> None:
        self.conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, ordinal, role, message_id, time_utc, raw_text, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.conversation_id,
                message.ordinal,
                message.role,
                message.message_id,
                message.time_utc,
                message.raw_text,
                _json(message.metadata_json),
            ),
        )

    def _insert_block(self, block: Block) -> None:
        self.conn.execute(
            """
            INSERT INTO blocks (
                id, message_id, conversation_id, ordinal, block_type, language, raw_text,
                normalized_text, char_start, char_end, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block.id,
                block.message_id,
                block.conversation_id,
                block.ordinal,
                block.block_type,
                block.language,
                block.raw_text,
                block.normalized_text,
                block.char_start,
                block.char_end,
                _json(block.metadata_json),
            ),
        )

    def _insert_knowledge_block(self, conversation: Conversation, messages: list[Message], block: Block) -> None:
        message_by_id = {message.id: message for message in messages}
        message = message_by_id[block.message_id]
        interest_tier = self._source_interest_tier(conversation.source_document_id)
        text_for_display = block.raw_text
        text_for_embedding = "\n".join(
            part
            for part in [
                f"Project: {conversation.project_id}" if conversation.project_id else None,
                f"Conversation: {conversation.title}" if conversation.title else None,
                f"Role: {message.role.upper()}",
                f"Block type: {block.block_type}",
                f"Content: {block.normalized_text or block.raw_text}",
            ]
            if part
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO knowledge_blocks (
                id, source_type, source_document_id, conversation_id, message_id, block_id,
                attachment_id, project_id, folder_kind, interest_tier, role, block_type, text_for_embedding,
                text_for_display, token_count_estimate, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_id("chat_block", block.id, prefix="kb"),
                "chat_block",
                conversation.source_document_id,
                conversation.id,
                message.id,
                block.id,
                None,
                conversation.project_id,
                conversation.folder_kind,
                interest_tier,
                message.role,
                block.block_type,
                text_for_embedding,
                text_for_display,
                max(1, len(text_for_embedding.split())),
                _json({}),
            ),
        )

    def _source_interest_tier(self, source_document_id: str) -> str:
        row = self.conn.execute(
            "SELECT interest_tier FROM source_documents WHERE id = ?",
            (source_document_id,),
        ).fetchone()
        return str(row["interest_tier"]) if row and row["interest_tier"] else "normal"


def _json(value: dict[str, Any] | list[Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
