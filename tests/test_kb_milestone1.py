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

from kb.cli import build_edges_command, build_nodes_command, embed_knowledge_blocks, import_knowledge_base, ingest_attachments, ingest_chats  # noqa: E402
from kb.ingest.chat_md_parser import parse_chat_file  # noqa: E402
from kb.ingest.tree_walker import scan_tree  # noqa: E402
from kb.embeddings.mock_provider import MockDenseProvider, MockSparseProvider  # noqa: E402
from kb.mcp.server import ServerConfig, handle_request  # noqa: E402
from kb.retrieval.context_pack import ContextPackOptions, build_context_pack  # noqa: E402
from kb.retrieval.hybrid_search import build_parser as build_search_parser, hybrid_query  # noqa: E402
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
            self.assertEqual(by_name["chat.md"].interest_tier, "normal")
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

    def test_parse_chat_ignores_message_headings_inside_code_fences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.md"
            path.write_text(
                SAMPLE_CHAT
                + "\n### 3. USER\n"
                + "- `message_id`: msg-user-2\n\n"
                + "```markdown\n"
                + "### 1. USER\n"
                + "- `message_id`: fake-message\n\n"
                + "This is an example, not a real message.\n"
                + "```\n",
                encoding="utf-8",
            )

            parsed = parse_chat_file(path, source_document_id="src-1")

            self.assertEqual(parsed.conversation.message_count, 3)
            self.assertEqual([message.ordinal for message in parsed.messages], [1, 2, 3])
            self.assertEqual(parsed.messages[-1].message_id, "msg-user-2")
            self.assertIn("fake-message", parsed.messages[-1].raw_text)

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

    def test_embed_limit_does_not_round_up_to_full_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            stats = embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                batch_size=5,
                limit=3,
            )

            self.assertEqual(stats["candidate_blocks"], 3)
            self.assertEqual(stats["blocks_embedded"], 3)
            self.assertEqual(stats["dense_vectors"], 3)
            self.assertEqual(stats["sparse_vectors"], 3)

    def test_embed_separate_pass_mode_preserves_dense_and_sparse_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            stats = embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                embedding_pass_mode="separate",
                batch_size=5,
                limit=3,
            )

            self.assertEqual(stats["embedding_pass_mode"], "separate")
            self.assertEqual(stats["dense_vectors"], 3)
            self.assertEqual(stats["sparse_vectors"], 3)
            with SQLiteStore(db) as store:
                linked = store.conn.execute(
                    """
                    SELECT COUNT(*) FROM knowledge_blocks
                    WHERE dense_vector_id IS NOT NULL AND sparse_vector_id IS NOT NULL
                    """
                ).fetchone()[0]
            self.assertEqual(linked, 3)

    def test_low_interest_content_is_skipped_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            low_dir = root / "Common" / "potential_trash"
            useful_dir = root / "Common" / "useful"
            low_dir.mkdir(parents=True)
            useful_dir.mkdir(parents=True)
            (low_dir / "low.md").write_text(SAMPLE_CHAT.replace("conv-1", "conv-low"), encoding="utf-8")
            (useful_dir / "useful.md").write_text(SAMPLE_CHAT.replace("conv-1", "conv-useful"), encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db)
            with SQLiteStore(db) as store:
                tiers = {
                    row["relative_path"]: row["interest_tier"]
                    for row in store.conn.execute(
                        "SELECT relative_path, interest_tier FROM source_documents ORDER BY relative_path"
                    ).fetchall()
                }
                kb_tiers = {
                    row["interest_tier"]: row["count"]
                    for row in store.conn.execute(
                        "SELECT interest_tier, COUNT(*) AS count FROM knowledge_blocks GROUP BY interest_tier"
                    ).fetchall()
                }
            self.assertEqual(tiers["Common/potential_trash/low.md"], "low")
            self.assertEqual(tiers["Common/useful/useful.md"], "normal")
            self.assertGreater(kb_tiers["low"], 0)
            self.assertGreater(kb_tiers["normal"], 0)

            embedded = embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
            )
            self.assertEqual(embedded["dense_vectors"], kb_tiers["normal"])
            embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                skip_low_interest_content=False,
            )
            excluded = hybrid_query(
                db_path=db,
                query="memory routing",
                dense_provider="mock",
                sparse_provider="mock",
                limit=20,
            )
            included = hybrid_query(
                db_path=db,
                query="memory routing",
                dense_provider="mock",
                sparse_provider="mock",
                include_low_interest=True,
                limit=20,
            )
            self.assertTrue(all(item["interest_tier"] == "normal" for item in excluded["results"]))
            self.assertTrue(any(item["interest_tier"] == "low" for item in included["results"]))

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

    def test_search_defaults_match_import_sparse_model(self) -> None:
        parser = build_search_parser()

        query_args = parser.parse_args(["query", "memory routing", "--db", "chat_memory.db"])
        context_args = parser.parse_args(["context", "memory routing", "--db", "chat_memory.db"])

        self.assertEqual(
            query_args.sparse_model,
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        )
        self.assertEqual(
            context_args.sparse_model,
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        )

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

    def test_build_edges_without_embeddings_skips_similarity_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            stats = build_edges_command(db_path=db, scope="project", top_k=2)

            self.assertEqual(stats["candidate_pairs"], 0)
            with SQLiteStore(db) as store:
                edge_kinds = [
                    row["edge_kind"]
                    for row in store.conn.execute("SELECT DISTINCT edge_kind FROM semantic_edges ORDER BY edge_kind").fetchall()
                ]
            self.assertEqual(edge_kinds, ["temporal_neighbor"])

    def test_import_knowledge_base_runs_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            attachment_dir = chat_dir / "attachments"
            attachment_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            (attachment_dir / "note.txt").write_text("memory routing attachment", encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            stats = import_knowledge_base(
                input_dir=root,
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                edge_top_k=2,
            )

            self.assertEqual(stats["stages"]["ingest_chats"]["failed_chats"], 0)
            self.assertGreater(stats["stages"]["ingest_attachments"]["extracted"], 0)
            self.assertGreater(stats["stages"]["embed"]["blocks_embedded"], 0)
            self.assertGreater(stats["stages"]["build_nodes"]["nodes_created"], 0)
            self.assertGreater(stats["stages"]["build_edges"]["edges_created"], 0)
            self.assertGreater(stats["final"]["knowledge_blocks"], 0)

    def test_context_pack_combines_direct_node_and_neighbor_paths(self) -> None:
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
            build_nodes_command(db_path=db, mode="deterministic", sparse_top_k=10)
            build_edges_command(db_path=db, scope="project", top_k=2)

            payload = build_context_pack(
                db_path=db,
                query="memory routing",
                dense=MockDenseProvider(),
                sparse=MockSparseProvider(),
                dense_provider="mock",
                sparse_provider="mock",
                dense_model="",
                sparse_model="",
                sparse_top_k=10,
                options=ContextPackOptions(
                    budget_tokens=200,
                    direct_limit=2,
                    node_limit=2,
                    node_member_limit=2,
                    neighbor_limit=2,
                ),
            )

            self.assertGreaterEqual(len(payload["selected_blocks"]), 1)
            self.assertEqual(payload["retrieval_strategy_requested"], "auto")
            self.assertEqual(payload["retrieval_strategy_used"], "basement")
            self.assertFalse(payload["db_capabilities"]["has_semantic_groups"])
            paths = {trace["path"] for trace in payload["source_trace"]}
            self.assertIn("query -> block direct", paths)
            self.assertTrue(any(path.startswith("query -> node:") for path in paths))
            self.assertIn("query -> block -> neighbor", paths)
            self.assertLessEqual(
                sum(item["token_count_estimate"] for item in payload["selected_blocks"]),
                200,
            )

    def test_context_pack_auto_uses_semantic_groups_when_available(self) -> None:
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
            with SQLiteStore(db) as store:
                block_ids = [
                    row["id"]
                    for row in store.conn.execute(
                        "SELECT id FROM knowledge_blocks ORDER BY id LIMIT 2"
                    ).fetchall()
                ]
                dense_vector_id = store.upsert_dense_vector(
                    owner_type="semantic_node",
                    owner_id="node-semantic-group-1",
                    model_name="mock-dense",
                    model_version="v1",
                    vector=MockDenseProvider().embed_texts(["memory routing"])[0],
                )
                store.replace_sparse_terms(
                    owner_type="semantic_node",
                    owner_id="node-semantic-group-1",
                    model_name="mock-sparse",
                    terms=MockSparseProvider().embed_texts(["memory routing"])[0],
                )
                store.upsert_semantic_node(
                    node_id="node-semantic-group-1",
                    node_type="semantic_group",
                    project_id="Project_17",
                    dense_vector_id=dense_vector_id,
                    sparse_vector_id=None,
                    title="memory routing group",
                    summary=None,
                    top_terms_json="[]",
                    metadata_json="{}",
                )
                store.replace_semantic_node_members(
                    node_id="node-semantic-group-1",
                    members=[
                        {
                            "knowledge_block_id": block_id,
                            "membership_weight": 1.0,
                            "membership_reason": "dense_similarity",
                            "metadata_json": "{}",
                        }
                        for block_id in block_ids
                    ],
                )
                store.commit()

            payload = build_context_pack(
                db_path=db,
                query="memory routing",
                dense=MockDenseProvider(),
                sparse=MockSparseProvider(),
                dense_provider="mock",
                sparse_provider="mock",
                dense_model="",
                sparse_model="",
                sparse_top_k=10,
                options=ContextPackOptions(
                    budget_tokens=200,
                    direct_limit=2,
                    node_limit=2,
                    node_member_limit=2,
                    neighbor_limit=0,
                    retrieval_strategy="auto",
                ),
            )

            self.assertEqual(payload["retrieval_strategy_used"], "semantic_groups")
            self.assertTrue(payload["db_capabilities"]["has_semantic_groups"])
            self.assertTrue(payload["db_capabilities"]["has_group_embeddings"])
            self.assertTrue(
                any(trace["path"] == "query -> node:semantic_group -> member block" for trace in payload["source_trace"])
            )

    def test_mcp_server_exposes_only_context_pack_tool(self) -> None:
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
            build_nodes_command(db_path=db, mode="deterministic", sparse_top_k=10)
            build_edges_command(db_path=db, scope="project", top_k=2)
            config = ServerConfig(
                db_path=db,
                dense_provider="mock",
                sparse_provider="mock",
            )

            tools = handle_request(config, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual([tool["name"] for tool in tools["result"]["tools"]], ["build_context_pack"])

            response = handle_request(
                config,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "build_context_pack",
                        "arguments": {"query": "memory routing", "token_budget": 200},
                    },
                },
            )

            self.assertFalse(response["result"]["isError"])
            payload = json.loads(response["result"]["content"][0]["text"])
            self.assertIn("context_text", payload)
            self.assertIn("source_references", payload)
            self.assertEqual(payload["retrieval_strategy_used"], "basement")
            self.assertIn("db_capabilities", payload)
            self.assertGreaterEqual(len(payload["selected_blocks"]), 1)


if __name__ == "__main__":
    unittest.main()
