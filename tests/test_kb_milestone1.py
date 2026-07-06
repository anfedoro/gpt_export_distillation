from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kb.cli import build_edges_command, build_nodes_command, embed_knowledge_blocks, ingest_attachments, ingest_chats  # noqa: E402
from kb.ingest.chat_md_parser import parse_chat_file  # noqa: E402
from kb.ingest.tree_walker import scan_tree  # noqa: E402
from kb.retrieval.hybrid_search import hybrid_query  # noqa: E402
from kb.storage.sqlite_store import SQLiteStore, init_db  # noqa: E402


SAMPLE_CHAT = """# Memory Routing

## Metadata

- `id`: conv-1
- `conversation_template_id`: tmpl-1
- `title`: Memory Routing
- `create_time_utc`: 2026-06-30T10:00:00+00:00
- `update_time_utc`: 2026-06-30T10:05:00+00:00
- `message_count`: 2

## Conversation

### 1. USER
- `time_utc`: 2026-06-30T10:00:00+00:00
- `message_id`: msg-user-1

How should memory write routing work?

```python
print("route")
```

### 2. ASSISTANT
- `time_utc`: 2026-06-30T10:01:00+00:00
- `message_id`: msg-assistant-1

Use a deterministic policy.

```mermaid
flowchart TD
  A --> B
```
"""


class KBMilestone1Tests(unittest.TestCase):
    def test_scan_detects_export_tree_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Common" / "useful").mkdir(parents=True)
            (root / "Common" / "potential_trash").mkdir(parents=True)
            (root / "Pinned").mkdir()
            (root / "Projects" / "Project_17" / "attachments").mkdir(parents=True)
            (root / "Common" / "useful" / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            (root / "Common" / "useful" / "INDEX.md").write_text("# Index", encoding="utf-8")
            (root / "Projects" / "Project_17" / "attachments" / "note.txt").write_text("note", encoding="utf-8")

            items = list(scan_tree(root))
            by_name = {item.file_name: item for item in items}

            self.assertEqual(by_name["chat.md"].detected_kind, "chat_md")
            self.assertEqual(by_name["chat.md"].folder_kind, "common_useful")
            self.assertEqual(by_name["INDEX.md"].detected_kind, "index_md")
            self.assertTrue(by_name["note.txt"].is_attachment)
            self.assertEqual(by_name["note.txt"].project_path, "Project_17")

    def test_parse_chat_preserves_messages_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.md"
            path.write_text(SAMPLE_CHAT, encoding="utf-8")

            parsed = parse_chat_file(path, source_document_id="src-1")

            self.assertEqual(parsed.conversation.conversation_id, "conv-1")
            self.assertEqual(parsed.conversation.message_count, 2)
            self.assertEqual([message.role for message in parsed.messages], ["user", "assistant"])
            self.assertEqual(parsed.messages[0].message_id, "msg-user-1")
            self.assertIn("How should memory", parsed.messages[0].raw_text)
            self.assertIn("code", {block.block_type for block in parsed.blocks})
            self.assertIn("mermaid", {block.block_type for block in parsed.blocks})

    def test_parse_chat_ignores_numbered_content_headings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.md"
            path.write_text(
                SAMPLE_CHAT
                + "\n### 3. ASSISTANT\n"
                + "- `message_id`: msg-assistant-2\n\n"
                + "### 1. Architecture\n\n"
                + "This is a content heading, not a message.\n",
                encoding="utf-8",
            )

            parsed = parse_chat_file(path, source_document_id="src-1")

            self.assertEqual(parsed.conversation.message_count, 3)
            self.assertEqual(parsed.messages[-1].message_id, "msg-assistant-2")
            self.assertIn("### 1. Architecture", parsed.messages[-1].raw_text)

    def test_init_and_ingest_chats_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            init_db(db)
            first = ingest_chats(root, db, limit=10)
            second = ingest_chats(root, db, limit=10)

            self.assertEqual(first["parsed_chats"], 1)
            self.assertEqual(second["parsed_chats"], 1)
            with SQLiteStore(db) as store:
                stats = store.stats()
            self.assertEqual(stats["source_documents"], 1)
            self.assertEqual(stats["conversations"], 1)
            self.assertEqual(stats["messages"], 2)
            self.assertGreaterEqual(stats["blocks"], 4)
            self.assertEqual(stats["knowledge_blocks"], stats["blocks"])

    def test_ingest_attachments_extracts_text_and_tracks_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            attachments = root / "Projects" / "Project_17" / "attachments"
            attachments.mkdir(parents=True)
            (attachments / "note.txt").write_text("Attachment memory text", encoding="utf-8")
            (attachments / "image.png").write_bytes(b"not really an image")
            db = Path(tmp) / "chat_memory.db"

            stats = ingest_attachments(root, db)
            again = ingest_attachments(root, db)

            self.assertEqual(stats["attempted"], 2)
            self.assertEqual(stats["extracted"], 1)
            self.assertEqual(stats["unsupported"], 1)
            self.assertEqual(again["attempted"], 2)
            with SQLiteStore(db) as store:
                db_stats = store.stats()
                rows = store.conn.execute(
                    "SELECT extraction_status FROM attachment_documents ORDER BY file_name"
                ).fetchall()
                kb_rows = store.conn.execute(
                    "SELECT block_type, text_for_display FROM knowledge_blocks WHERE source_type = 'attachment_block'"
                ).fetchall()
            self.assertEqual(db_stats["attachment_documents"], 2)
            self.assertEqual(db_stats["knowledge_blocks"], 1)
            self.assertEqual([row["extraction_status"] for row in rows], ["unsupported", "extracted"])
            self.assertEqual(kb_rows[0]["block_type"], "text")
            self.assertIn("Attachment memory text", kb_rows[0]["text_for_display"])

    def test_embed_mock_providers_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            first = embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                batch_size=2,
            )
            second = embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                batch_size=2,
            )

            self.assertGreater(first["dense_vectors"], 0)
            self.assertGreater(first["sparse_vectors"], 0)
            self.assertEqual(second["candidate_blocks"], 0)
            with SQLiteStore(db) as store:
                stats = store.stats()
                linked = store.conn.execute(
                    """
                    SELECT COUNT(*) FROM knowledge_blocks
                    WHERE dense_vector_id IS NOT NULL AND sparse_vector_id IS NOT NULL
                    """
                ).fetchone()[0]
            self.assertEqual(stats["dense_vectors"], stats["knowledge_blocks"])
            self.assertGreater(stats["sparse_terms"], 0)
            self.assertEqual(linked, stats["knowledge_blocks"])

    def test_hybrid_query_returns_traceable_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
            )
            payload = hybrid_query(
                db_path=db,
                query="memory routing",
                dense_provider="mock",
                sparse_provider="mock",
                limit=3,
            )

            self.assertGreater(payload["candidate_blocks"], 0)
            self.assertGreaterEqual(len(payload["results"]), 1)
            top = payload["results"][0]
            self.assertIn("Projects/Project_17/chat.md", top["source_path"])
            self.assertIn("final_score", top)
            self.assertIn("dense_score", top)
            self.assertIn("sparse_score", top)
            self.assertIn("preview", top)

    def test_build_nodes_creates_deterministic_memberships(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
            )
            first = build_nodes_command(db_path=db, mode="deterministic", sparse_top_k=10)
            second = build_nodes_command(db_path=db, mode="deterministic", sparse_top_k=10)

            self.assertEqual(first["nodes_created"], 2)
            self.assertEqual(second["nodes_created"], 2)
            self.assertGreater(first["memberships_created"], 0)
            with SQLiteStore(db) as store:
                node_types = [
                    row["node_type"]
                    for row in store.conn.execute("SELECT node_type FROM semantic_nodes ORDER BY node_type").fetchall()
                ]
                member_count = store.conn.execute("SELECT COUNT(*) FROM semantic_node_members").fetchone()[0]
                node_vector_count = store.conn.execute(
                    "SELECT COUNT(*) FROM dense_vectors WHERE owner_type = 'semantic_node'"
                ).fetchone()[0]
            self.assertEqual(node_types, ["conversation", "project"])
            self.assertEqual(member_count, first["memberships_created"])
            self.assertEqual(node_vector_count, 2)

    def test_build_edges_creates_idempotent_similarity_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
            )
            first = build_edges_command(db_path=db, scope="project", top_k=2)
            second = build_edges_command(db_path=db, scope="project", top_k=2)

            self.assertGreater(first["edges_created"], 0)
            self.assertEqual(first["semantic_edges"], second["semantic_edges"])
            with SQLiteStore(db) as store:
                edge_kinds = [
                    row["edge_kind"]
                    for row in store.conn.execute("SELECT DISTINCT edge_kind FROM semantic_edges ORDER BY edge_kind").fetchall()
                ]
                policy_versions = [
                    row["policy_version"]
                    for row in store.conn.execute("SELECT DISTINCT policy_version FROM semantic_edges").fetchall()
                ]
                shared_terms_rows = store.conn.execute(
                    "SELECT COUNT(*) FROM semantic_edges WHERE edge_kind = 'sparse_overlap' AND shared_terms_json IS NOT NULL"
                ).fetchone()[0]
            self.assertIn("temporal_neighbor", edge_kinds)
            self.assertIn("dense_sim", edge_kinds)
            self.assertIn("sparse_overlap", edge_kinds)
            self.assertIn("hybrid_sim", edge_kinds)
            self.assertEqual(policy_versions, ["similarity-edges-v0"])
            self.assertGreater(shared_terms_rows, 0)


if __name__ == "__main__":
    unittest.main()
