from __future__ import annotations

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kb.cli import build_edges_command, build_nodes_command, build_parser as build_index_parser, embed_knowledge_blocks, import_knowledge_base, ingest_attachments, ingest_chats  # noqa: E402
from kb.benchmark import DirectRetrievalSession, RankingConfig, default_ranking_configs, run_direct_retrieval_benchmark, validate_direct_retrieval_dataset  # noqa: E402
from kb.ingest.chat_md_parser import parse_chat_file  # noqa: E402
from kb.ingest.tree_walker import scan_tree  # noqa: E402
from kb.embeddings.mock_provider import MockDenseProvider, MockSparseProvider  # noqa: E402
from kb.mcp.server import ServerConfig, build_parser as build_mcp_parser, handle_request  # noqa: E402
from kb.retrieval.context_pack import ContextPackOptions, build_context_pack  # noqa: E402
from kb.retrieval.hybrid_search import QUERY_RESULT_SCHEMA_VERSION, build_parser as build_search_parser, hybrid_query, main as kb_search_main  # noqa: E402
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


class StaticDenseProvider:
    model_name = "static-dense"
    embedding_space_id = "static-dense;dim=2;normalize=false;symmetric=true"
    runtime_metadata = {"backend": "test"}

    @property
    def model_version(self) -> str:
        return self.embedding_space_id

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [_static_dense_vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0]


class RuntimeVariantDenseProvider(StaticDenseProvider):
    runtime_metadata = {"backend": "test", "device": "cpu", "torch_dtype": "float32"}


class IncompatibleDenseProvider(StaticDenseProvider):
    embedding_space_id = "static-dense;dim=3;normalize=false;symmetric=true"

    @property
    def model_version(self) -> str:
        return self.embedding_space_id

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0, 0.0]


class StaticSparseProvider:
    model_name = "static-sparse"
    embedding_space_id = "static-sparse;document_encoder=documents;query_encoder=query;top_k=all"
    runtime_metadata = {"backend": "test"}

    @property
    def model_version(self) -> str:
        return self.embedding_space_id

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        self.document_calls += 1
        return [_static_sparse_terms(text) for text in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        self.query_calls += 1
        return {"sparse": 1.0}


class SpySparseEncoder:
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def encode_document(self, texts, **kwargs):
        self.document_calls += 1
        return [["document", text] for text in texts]

    def encode_query(self, texts, **kwargs):
        self.query_calls += 1
        return [["query", text] for text in texts]

    def decode(self, embeddings, *, top_k):
        return [[(str(item[0]), 1.0)] for item in embeddings]


def _static_dense_vector(text: str) -> list[float]:
    if "dense-only" in text:
        return [1.0, 0.0]
    if "sparse-only" in text:
        return [0.0, 1.0]
    if "hybrid-both" in text:
        return [0.8, 0.6]
    return [0.0, 1.0]


def _static_sparse_terms(text: str) -> dict[str, float]:
    if "dense-only" in text:
        return {"dense": 1.0}
    if "sparse-only" in text:
        return {"sparse": 1.0}
    if "hybrid-both" in text:
        return {"sparse": 0.8, "dense": 0.2}
    return {"other": 1.0}


class KBMilestone1Tests(unittest.TestCase):
    def _static_retrieval_db(self, tmp: str) -> Path:
        root = Path(tmp) / "export"
        chat_dir = root / "Projects" / "Project_17"
        chat_dir.mkdir(parents=True)
        for idx, marker in enumerate(["dense-only", "sparse-only", "hybrid-both"], start=1):
            chat = (
                SAMPLE_CHAT.replace("conv-1", f"conv-static-{idx}")
                .replace("msg-user-1", f"msg-static-user-{idx}")
                .replace("msg-assistant-1", f"msg-static-assistant-{idx}")
                .replace("Memory Routing", f"Static {idx}")
                .replace("How should memory write routing work?", f"{marker} block")
            )
            (chat_dir / f"{marker}.md").write_text(chat, encoding="utf-8")
        db = Path(tmp) / "chat_memory.db"
        ingest_chats(root, db, limit=10)
        dense = StaticDenseProvider()
        sparse = StaticSparseProvider()
        with SQLiteStore(db) as store:
            rows = store.conn.execute(
                "SELECT id, text_for_embedding FROM knowledge_blocks ORDER BY text_for_embedding"
            ).fetchall()
            for row in rows:
                text = str(row["text_for_embedding"])
                dense_vector_id = store.upsert_dense_vector(
                    owner_type="knowledge_block",
                    owner_id=str(row["id"]),
                    model_name=dense.model_name,
                    model_version=dense.model_version,
                    runtime_metadata_json=json.dumps(dense.runtime_metadata, sort_keys=True),
                    vector=dense.embed_documents([text])[0],
                )
                sparse_vector_id = store.replace_sparse_terms(
                    owner_type="knowledge_block",
                    owner_id=str(row["id"]),
                    model_name=sparse.model_name,
                    embedding_space_id=sparse.embedding_space_id,
                    terms=sparse.embed_documents([text])[0],
                )
                store.set_knowledge_block_vector_ids(
                    knowledge_block_id=str(row["id"]),
                    dense_vector_id=dense_vector_id,
                    sparse_vector_id=sparse_vector_id,
                )
            store.commit()
        return db

    def _single_block_dataset_record(
        self,
        db: Path,
        *,
        record_id: str = "query-1",
        query: str = "memory routing",
        query_type: str = "exact_terms",
        language: str = "en",
        source_language: str = "en",
    ) -> dict:
        with SQLiteStore(db) as store:
            row = store.conn.execute(
                """
                SELECT kb.id, sd.relative_path, kb.conversation_id, kb.message_id
                FROM knowledge_blocks kb
                JOIN source_documents sd ON sd.id = kb.source_document_id
                ORDER BY kb.id
                LIMIT 1
                """
            ).fetchone()
        return {
            "id": record_id,
            "query": query,
            "query_type": query_type,
            "language": language,
            "source_language": source_language,
            "topic": "llm_memory",
            "expected": [
                {
                    "block_id": row["id"],
                    "relevance": 3,
                    "source_path": row["relative_path"],
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                }
            ],
            "notes": "Synthetic validator fixture.",
        }

    def _write_dataset(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

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
            self.assertEqual(payload["schema_version"], QUERY_RESULT_SCHEMA_VERSION)
            self.assertEqual(top["rank"], 1)
            self.assertIn("latency_ms", payload)
            self.assertIn("provider_load", payload["latency_ms"])
            self.assertIn("query_encoding", payload["latency_ms"])
            self.assertIn("db_candidate_load", payload["latency_ms"])
            self.assertIn("scoring", payload["latency_ms"])
            self.assertEqual(payload["run"]["retrieval_mode"], "query")

    def test_search_defaults_match_import_sparse_model(self) -> None:
        parser = build_search_parser()
        index_parser = build_index_parser()
        mcp_parser = build_mcp_parser()

        query_args = parser.parse_args(["query", "memory routing", "--db", "chat_memory.db"])
        context_args = parser.parse_args(["context", "memory routing", "--db", "chat_memory.db"])
        import_args = index_parser.parse_args(["import", "--input", "export", "--db", "chat_memory.db"])
        mcp_args = mcp_parser.parse_args(["--db", "chat_memory.db"])

        self.assertEqual(
            query_args.sparse_model,
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        )
        self.assertEqual(
            context_args.sparse_model,
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
        )
        self.assertEqual(query_args.sparse_model, import_args.sparse_model)
        self.assertEqual(query_args.sparse_model, mcp_args.sparse_model)

    def test_dense_runtime_metadata_mismatch_does_not_block_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)

            provider = RuntimeVariantDenseProvider()
            with patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=provider):
                payload = hybrid_query(
                    db_path=db,
                    query="memory routing",
                    dense_provider="mock",
                    sparse_provider="none",
                    limit=3,
                    diagnostics=True,
                )

            self.assertGreater(payload["candidate_blocks"], 0)
            self.assertGreater(payload["diagnostics"]["dense"]["candidate_blocks_with_vector"], 0)
            self.assertEqual(payload["diagnostics"]["dense"]["status"], "active")

    def test_incompatible_dense_space_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)

            provider = IncompatibleDenseProvider()
            with patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=provider):
                payload = hybrid_query(
                    db_path=db,
                    query="memory routing",
                    dense_provider="mock",
                    sparse_provider="none",
                    limit=3,
                    diagnostics=True,
                )

            self.assertEqual(payload["diagnostics"]["dense"]["candidate_blocks_with_vector"], 0)
            self.assertGreater(payload["diagnostics"]["dense"]["compatibility_mismatches"], 0)
            self.assertEqual(payload["diagnostics"]["dense"]["status"], "no_compatible_document_representations")

    def test_dense_only_retrieval_uses_dense_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dense = StaticDenseProvider()

            with (
                patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=dense),
                patch("kb.retrieval.hybrid_search._build_sparse_provider", side_effect=AssertionError("sparse should not be built")),
            ):
                payload = hybrid_query(
                    db_path=db,
                    query="memory routing",
                    dense_provider="mock",
                    sparse_provider="none",
                    limit=5,
                    diagnostics=True,
                )

            self.assertEqual(dense.query_calls, 1)
            self.assertGreater(payload["diagnostics"]["dense"]["nonzero_score_count"], 0)
            self.assertEqual(payload["diagnostics"]["sparse"]["status"], "disabled")
            self.assertGreaterEqual(payload["results"][0]["dense_score"], payload["results"][-1]["dense_score"])
            self.assertIn("dense-only", payload["results"][0]["preview"])

    def test_sparse_only_retrieval_uses_sparse_query_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            sparse = StaticSparseProvider()

            with (
                patch("kb.retrieval.hybrid_search._build_dense_provider", side_effect=AssertionError("dense should not be built")),
                patch("kb.retrieval.hybrid_search._build_sparse_provider", return_value=sparse),
            ):
                payload = hybrid_query(
                    db_path=db,
                    query="memory routing",
                    dense_provider="none",
                    sparse_provider="mock",
                    limit=5,
                    diagnostics=True,
                )

            self.assertEqual(sparse.query_calls, 1)
            self.assertGreater(payload["diagnostics"]["sparse"]["nonzero_score_count"], 0)
            self.assertEqual(payload["diagnostics"]["dense"]["status"], "disabled")
            self.assertGreaterEqual(payload["results"][0]["sparse_score"], payload["results"][-1]["sparse_score"])
            self.assertIn("sparse-only", payload["results"][0]["preview"])

    def test_hybrid_score_uses_dense_and_sparse_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dense = StaticDenseProvider()
            sparse = StaticSparseProvider()

            with (
                patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=dense),
                patch("kb.retrieval.hybrid_search._build_sparse_provider", return_value=sparse),
            ):
                payload = hybrid_query(
                    db_path=db,
                    query="memory routing",
                    dense_provider="mock",
                    sparse_provider="mock",
                    limit=10,
                    diagnostics=True,
                )

            for item in payload["results"]:
                expected = 0.65 * item["dense_score"] + 0.35 * item["sparse_score"]
                self.assertAlmostEqual(item["final_score"], expected)
                self.assertGreaterEqual(item["rank"], 1)
            self.assertEqual(payload["diagnostics"]["dense"]["status"], "active")
            self.assertEqual(payload["diagnostics"]["sparse"]["status"], "active")

    def test_query_output_writes_full_json_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            output = Path(tmp) / "retrieval-result.json"
            dense = StaticDenseProvider()
            sparse = StaticSparseProvider()
            argv = [
                "kb-search",
                "query",
                "memory routing",
                "--db",
                str(db),
                "--dense-provider",
                "mock",
                "--sparse-provider",
                "mock",
                "--limit",
                "3",
                "--json",
                "--diagnostics",
                "--output",
                str(output),
            ]

            with (
                patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=dense),
                patch("kb.retrieval.hybrid_search._build_sparse_provider", return_value=sparse),
                patch.object(sys, "argv", argv),
                patch("sys.stdout", new_callable=StringIO),
            ):
                kb_search_main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], QUERY_RESULT_SCHEMA_VERSION)
            self.assertEqual(len(payload["results"]), 3)
            self.assertEqual([item["rank"] for item in payload["results"]], [1, 2, 3])
            self.assertIn("latency_ms", payload)
            self.assertIn("diagnostics", payload)

    def test_benchmark_validator_accepts_valid_direct_retrieval_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"
            ingest_chats(root, db, limit=10)
            with SQLiteStore(db) as store:
                row = store.conn.execute(
                    """
                    SELECT kb.id, sd.relative_path, kb.conversation_id, kb.message_id
                    FROM knowledge_blocks kb
                    JOIN source_documents sd ON sd.id = kb.source_document_id
                    ORDER BY kb.id
                    LIMIT 1
                    """
                ).fetchone()
            dataset = Path(tmp) / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "id": "memory-001-exact",
                        "query": "memory write routing",
                        "query_type": "exact_terms",
                        "language": "en",
                        "source_language": "en",
                        "topic": "llm_memory",
                        "expected": [
                            {
                                "block_id": row["id"],
                                "relevance": 3,
                                "source_path": row["relative_path"],
                                "conversation_id": row["conversation_id"],
                                "message_id": row["message_id"],
                            }
                        ],
                        "notes": "Synthetic validator fixture.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["query_type_distribution"], {"exact_terms": 1})

    def test_benchmark_validator_rejects_bad_block_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"
            ingest_chats(root, db, limit=10)
            with SQLiteStore(db) as store:
                row = store.conn.execute("SELECT id, conversation_id, message_id FROM knowledge_blocks LIMIT 1").fetchone()
            dataset = Path(tmp) / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "id": "bad-001",
                        "query": "bad metadata",
                        "query_type": "exact_terms",
                        "language": "en",
                        "source_language": "en",
                        "topic": "validator",
                        "expected": [
                            {
                                "block_id": row["id"],
                                "relevance": 3,
                                "source_path": "wrong.md",
                                "conversation_id": row["conversation_id"],
                                "message_id": row["message_id"],
                            }
                        ],
                        "notes": "Should fail.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertFalse(report["ok"])
            self.assertTrue(any("source_path mismatch" in error for error in report["errors"]))

    def test_benchmark_validator_rejects_english_query_labeled_ru(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Why should memory routing use an explicit policy?",
                        language="ru",
                        source_language="ru",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertFalse(report["ok"])
            self.assertTrue(any("English sentence" in error for error in report["errors"]))

    def test_benchmark_validator_rejects_russian_query_labeled_en(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Почему память должна использовать явную политику маршрутизации?",
                        language="en",
                        source_language="ru",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertFalse(report["ok"])
            self.assertTrue(any("Russian sentence" in error for error in report["errors"]))

    def test_benchmark_validator_accepts_russian_query_with_english_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Почему SparseEncoder encode_query должен отличаться от encode_document?",
                        language="ru",
                        source_language="ru",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertTrue(report["ok"], report["errors"])

    def test_benchmark_validator_accepts_cross_language_ru_to_en(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Why should memory routing use an explicit policy?",
                        query_type="cross_language",
                        language="en",
                        source_language="ru",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["cross_language_pair_distribution"], {"ru->en": 1})

    def test_benchmark_validator_rejects_cross_language_with_same_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Why should memory routing use an explicit policy?",
                        query_type="cross_language",
                        language="en",
                        source_language="en",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertFalse(report["ok"])
            self.assertTrue(any("must use different" in error for error in report["errors"]))

    def test_benchmark_validator_rejects_cross_language_with_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Why should memory routing use an explicit policy?",
                        query_type="cross_language",
                        language="en",
                        source_language="mixed",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertFalse(report["ok"])
            self.assertTrue(any("cannot use mixed" in error for error in report["errors"]))

    def test_benchmark_validator_allows_regular_query_same_source_and_query_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dataset = Path(tmp) / "dataset.jsonl"
            self._write_dataset(
                dataset,
                [
                    self._single_block_dataset_record(
                        db,
                        query="Why should memory routing use an explicit policy?",
                        query_type="answer_question",
                        language="en",
                        source_language="en",
                    )
                ],
            )

            report = validate_direct_retrieval_dataset(db_path=db, dataset_path=dataset, expected_count=1)

            self.assertTrue(report["ok"], report["errors"])

    def test_direct_retrieval_session_matches_hybrid_query_for_all_benchmark_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dense = StaticDenseProvider()
            sparse = StaticSparseProvider()
            session = DirectRetrievalSession(db_path=db, dense_provider=dense, sparse_provider=sparse)

            scores = session.score_query("memory routing")

            for config in default_ranking_configs():
                results, _, _ = session.rank(scores, config, top_k=10)
                with (
                    patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=StaticDenseProvider()),
                    patch("kb.retrieval.hybrid_search._build_sparse_provider", return_value=StaticSparseProvider()),
                ):
                    payload = hybrid_query(
                        db_path=db,
                        query="memory routing",
                        dense_provider="mock",
                        sparse_provider="mock",
                        alpha=config.alpha,
                        beta=config.beta,
                        limit=10,
                    )

                self.assertEqual(
                    [item["block_id"] for item in results],
                    [item["block_id"] for item in payload["results"]],
                )
                for actual, expected in zip(results, payload["results"], strict=True):
                    self.assertAlmostEqual(actual["dense_score"], expected["dense_score"], places=6)
                    self.assertAlmostEqual(actual["sparse_score"], expected["sparse_score"], places=6)
                    self.assertAlmostEqual(actual["final_score"], expected["final_score"], places=6)

    def test_direct_retrieval_session_uses_stable_block_id_tiebreak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            session = DirectRetrievalSession(
                db_path=db,
                dense_provider=StaticDenseProvider(),
                sparse_provider=None,
            )

            scores = session.score_query("zero tie")
            scores.dense_scores[:] = 0.0
            results, _, _ = session.rank(scores, RankingConfig("dense_100_sparse_000", 1.0, 0.0), top_k=10)

            block_ids = [item["block_id"] for item in results]
            self.assertEqual(block_ids, sorted(block_ids))

    def test_benchmark_run_writes_manifest_and_results_from_single_raw_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            with SQLiteStore(db) as store:
                rows = store.conn.execute(
                    """
                    SELECT kb.id, sd.relative_path, kb.conversation_id, kb.message_id, kb.text_for_display
                    FROM knowledge_blocks kb
                    JOIN source_documents sd ON sd.id = kb.source_document_id
                    ORDER BY kb.id
                    LIMIT 2
                    """
                ).fetchall()
            dataset = Path(tmp) / "dataset.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"query-{idx}",
                            "query": f"memory routing {idx}",
                            "query_type": "exact_terms",
                            "language": "en",
                            "source_language": "en",
                            "topic": "llm_memory",
                            "expected": [
                                {
                                    "block_id": row["id"],
                                    "relevance": 3,
                                    "source_path": row["relative_path"],
                                    "conversation_id": row["conversation_id"],
                                    "message_id": row["message_id"],
                                }
                            ],
                            "notes": "Synthetic benchmark fixture.",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for idx, row in enumerate(rows, start=1)
                ),
                encoding="utf-8",
            )
            dense = StaticDenseProvider()
            sparse = StaticSparseProvider()
            output_dir = Path(tmp) / "runs"

            report = run_direct_retrieval_benchmark(
                db_path=db,
                dataset_path=dataset,
                output_dir=output_dir,
                top_k=10,
                dense_provider_name="mock",
                sparse_provider_name="mock",
                dense_provider=dense,
                sparse_provider=sparse,
            )

            self.assertEqual(report["status"], "completed")
            self.assertEqual(dense.query_calls, 2)
            self.assertEqual(sparse.query_calls, 2)
            manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
            results = [json.loads(line) for line in Path(report["results"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["completed_queries"], 2)
            self.assertEqual(manifest["failed_queries"], 0)
            self.assertEqual(len(results), 2 * len(default_ranking_configs()))
            self.assertEqual(report["records_written"], len(results))
            per_query = [row for row in results if row["query_id"] == "query-1"]
            self.assertEqual(len(per_query), len(default_ranking_configs()))
            first_scores = {
                item["block_id"]: (item["dense_score"], item["sparse_score"]) for item in per_query[0]["top_results"]
            }
            last_scores = {
                item["block_id"]: (item["dense_score"], item["sparse_score"]) for item in per_query[-1]["top_results"]
            }
            self.assertEqual(first_scores, last_scores)
            for row in results:
                self.assertIn("language", row)
                self.assertIn("source_language", row)
                self.assertEqual(row["language"], "en")
                self.assertEqual(row["source_language"], "en")
                self.assertIn("rank", row["expected"][0])
                self.assertGreaterEqual(row["expected"][0]["rank"], 1)
                self.assertEqual(row["candidate_blocks"], manifest["database"]["candidate_blocks"])
                self.assertIn("query_encoding", row["latency_ms"])
                self.assertIn("base_scoring", row["latency_ms"])
                self.assertIn("ranking", row["latency_ms"])

    def test_benchmark_run_keeps_expected_rank_outside_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            with SQLiteStore(db) as store:
                row = store.conn.execute(
                    """
                    SELECT kb.id, sd.relative_path, kb.conversation_id, kb.message_id
                    FROM knowledge_blocks kb
                    JOIN source_documents sd ON sd.id = kb.source_document_id
                    ORDER BY kb.id DESC
                    LIMIT 1
                    """
                ).fetchone()
            dataset = Path(tmp) / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "id": "rank-outside-top-k",
                        "query": "memory routing",
                        "query_type": "exact_terms",
                        "language": "en",
                        "source_language": "en",
                        "topic": "llm_memory",
                        "expected": [
                            {
                                "block_id": row["id"],
                                "relevance": 3,
                                "source_path": row["relative_path"],
                                "conversation_id": row["conversation_id"],
                                "message_id": row["message_id"],
                            }
                        ],
                        "notes": "Expected block rank is still recorded with top_k=1.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = run_direct_retrieval_benchmark(
                db_path=db,
                dataset_path=dataset,
                output_dir=Path(tmp) / "runs",
                top_k=1,
                dense_provider_name="mock",
                sparse_provider_name="none",
                dense_provider=StaticDenseProvider(),
                sparse_provider=None,
                ranking_configs=[RankingConfig("dense_100_sparse_000", 1.0, 0.0)],
            )

            result = json.loads(Path(report["results"]).read_text(encoding="utf-8").strip())
            self.assertEqual(len(result["top_results"]), 1)
            self.assertIsNotNone(result["expected"][0]["rank"])
            self.assertGreaterEqual(result["expected"][0]["rank"], 1)

    def test_benchmark_run_builds_providers_and_loads_corpus_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            with SQLiteStore(db) as store:
                row = store.conn.execute(
                    """
                    SELECT kb.id, sd.relative_path, kb.conversation_id, kb.message_id
                    FROM knowledge_blocks kb
                    JOIN source_documents sd ON sd.id = kb.source_document_id
                    ORDER BY kb.id
                    LIMIT 1
                    """
                ).fetchone()
            dataset = Path(tmp) / "dataset.jsonl"
            dataset.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"query-{idx}",
                            "query": f"memory routing {idx}",
                            "query_type": "exact_terms",
                            "language": "en",
                            "source_language": "en",
                            "topic": "llm_memory",
                            "expected": [
                                {
                                    "block_id": row["id"],
                                    "relevance": 3,
                                    "source_path": row["relative_path"],
                                    "conversation_id": row["conversation_id"],
                                    "message_id": row["message_id"],
                                }
                            ],
                            "notes": "Synthetic benchmark fixture.",
                        },
                        ensure_ascii=False,
                    )
                    for idx in range(1, 4)
                )
                + "\n",
                encoding="utf-8",
            )
            dense = StaticDenseProvider()
            sparse = StaticSparseProvider()
            original_load_corpus = DirectRetrievalSession.load_corpus
            load_calls = 0

            def counted_load_corpus(session):
                nonlocal load_calls
                load_calls += 1
                return original_load_corpus(session)

            with (
                patch("kb.benchmark._build_dense_provider", return_value=dense) as build_dense,
                patch("kb.benchmark._build_sparse_provider", return_value=sparse) as build_sparse,
                patch.object(DirectRetrievalSession, "load_corpus", counted_load_corpus),
            ):
                report = run_direct_retrieval_benchmark(
                    db_path=db,
                    dataset_path=dataset,
                    output_dir=Path(tmp) / "runs",
                    top_k=3,
                    dense_provider_name="mock",
                    sparse_provider_name="mock",
                )

            self.assertEqual(report["status"], "completed")
            self.assertEqual(build_dense.call_count, 1)
            self.assertEqual(build_sparse.call_count, 1)
            self.assertEqual(load_calls, 1)
            self.assertEqual(dense.query_calls, 3)
            self.assertEqual(sparse.query_calls, 3)

    def test_query_and_context_direct_scores_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            dense = StaticDenseProvider()
            sparse = StaticSparseProvider()
            with (
                patch("kb.retrieval.hybrid_search._build_dense_provider", return_value=dense),
                patch("kb.retrieval.hybrid_search._build_sparse_provider", return_value=sparse),
            ):
                query_payload = hybrid_query(
                    db_path=db,
                    query="memory routing",
                    dense_provider="mock",
                    sparse_provider="mock",
                    limit=3,
                )
            context_payload = build_context_pack(
                db_path=db,
                query="memory routing",
                dense=StaticDenseProvider(),
                sparse=StaticSparseProvider(),
                dense_provider="mock",
                sparse_provider="mock",
                dense_model="",
                sparse_model="",
                sparse_top_k=10,
                options=ContextPackOptions(
                    budget_tokens=1000,
                    direct_limit=3,
                    node_limit=0,
                    node_member_limit=0,
                    neighbor_limit=0,
                ),
            )

            query_scores = {item["block_id"]: item["final_score"] for item in query_payload["results"]}
            context_scores = {
                trace["block_id"]: trace["score"]
                for trace in context_payload["source_trace"]
                if trace["path"] == "query -> block direct"
            }
            self.assertEqual(query_scores.keys(), context_scores.keys())
            for block_id, score in query_scores.items():
                self.assertAlmostEqual(score, context_scores[block_id])

    def test_sentence_transformer_sparse_uses_document_and_query_methods(self) -> None:
        from kb.embeddings.sentence_transformer_provider import SentenceTransformerSparseProvider

        provider = object.__new__(SentenceTransformerSparseProvider)
        provider.model_name = "spy-sparse"
        provider.embedding_space_id = "spy-sparse-space"
        provider.runtime_metadata = {"backend": "spy"}
        provider.top_k = 10
        provider._model = SpySparseEncoder()

        document_terms = provider.embed_documents(["document text"])
        query_terms = provider.embed_query("query text")

        self.assertEqual(provider._model.document_calls, 1)
        self.assertEqual(provider._model.query_calls, 1)
        self.assertEqual(document_terms, [{"document": 1.0}])
        self.assertEqual(query_terms, {"query": 1.0})

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
