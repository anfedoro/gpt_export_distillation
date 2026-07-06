from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from kb.ingest.tree_walker import InventoryItem
from kb.ingest.attachment_parser import ParsedAttachment
from kb.model.entities import Block, Conversation, Message, ParsedChat
from kb.model.ids import stable_id


SCHEMA_PATH = Path(__file__).with_name("sqlite_schema.sql")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = connect(db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def upsert_source_document(self, input_root: Path, item: InventoryItem) -> str:
        source_id = stable_id(item.relative_path, item.sha256, prefix="src")
        path = input_root / item.relative_path
        self.conn.execute(
            """
            INSERT INTO source_documents (
                id, path, relative_path, source_kind, folder_kind, project_id, project_name,
                file_name, extension, sha256, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path, sha256) DO UPDATE SET
                path=excluded.path,
                source_kind=excluded.source_kind,
                folder_kind=excluded.folder_kind,
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
                    attachment_id, project_id, folder_kind, role, block_type, text_for_embedding,
                    text_for_display, token_count_estimate, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> list[sqlite3.Row]:
        where: list[str] = []
        params: list[Any] = []
        if not force and dense_model_name is not None:
            where.append(
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
            where.append(
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
        query = "SELECT id, text_for_embedding FROM knowledge_blocks"
        if where:
            query += " WHERE " + " OR ".join(f"({clause})" for clause in where)
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return list(self.conn.execute(query, params).fetchall())

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
            [
                (
                    owner_type,
                    owner_id,
                    stable_id(model_name, token),
                    token,
                    float(weight),
                    model_name,
                )
                for token, weight in terms.items()
            ],
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
        if project is not None:
            where.append("kb.project_id = ?")
            params.append(project)
        query = f"""
            SELECT
                kb.id AS knowledge_block_id,
                kb.project_id,
                kb.folder_kind,
                kb.conversation_id,
                kb.message_id,
                kb.role,
                kb.block_type,
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
                attachment_id, project_id, folder_kind, role, block_type, text_for_embedding,
                text_for_display, token_count_estimate, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                message.role,
                block.block_type,
                text_for_embedding,
                text_for_display,
                max(1, len(text_for_embedding.split())),
                _json({}),
            ),
        )


def _json(value: dict[str, Any] | list[Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
