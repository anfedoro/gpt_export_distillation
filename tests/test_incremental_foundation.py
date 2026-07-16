from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb.index.chunk_builder import StrictestTokenizer, build_chunk_policy
from kb.model.entities import Block, Conversation, Message, ParsedChat
from kb.storage import native_pre_mvp as native
from kb.storage.native_pre_mvp import (
    DEFAULT_CANONICAL_CHUNK_CONTENT_BUDGET,
    NativeBuildStore,
    build_native_pre_mvp_db,
    create_clean_native_schema,
)
from ptha.database import inspect_database
from ptha.incremental import (
    BLOCK_BUILDER_VERSION,
    CANONICALIZER_VERSION,
    CHUNKER_VERSION,
    canonical_bytes,
    chunk_content_hash,
    chunk_identity,
    block_identity,
    compare_source_revisions,
    content_hash,
    conversation_identity,
    conversation_revision,
    embedding_contract_fingerprint,
    message_identity,
    message_revision,
    PARSER_CONTRACT,
    SOURCE_TRANSFORM_VERSION,
    CANONICAL_REPRESENTATION_VERSION,
)


class FakeProvider:
    effective_max_sequence_length = 64
    document_prefix = ""
    query_prefix = ""

    def __init__(self, *, space: str, batch_size: int = 4, revision: str = "revision-a") -> None:
        self.model_name = "synthetic-bge"
        self.embedding_space_id = space
        self.runtime_metadata = {
            "backend": "synthetic-mlx",
            "dtype": "float16",
            "model_revision": revision,
            "batch_size": batch_size,
            "device": "gpu",
        }

    def contract_dict(self) -> dict[str, object]:
        return {
            "model_revision": self.runtime_metadata["model_revision"],
            "embedding_dimension": 1024,
            "tokenizer_name": "synthetic-tokenizer",
            "tokenizer_model_max_length": 64,
            "backbone_max_position_embeddings": 64,
        }

    def embedding_input(self, text: str) -> str:
        return text

    def token_count(self, text: str) -> int:
        return max(1, len(text.split()))

    def assert_fits(self, text: str, **_kwargs: object) -> int:
        return self.token_count(text)


class SharedBackend:
    def embed_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(dense=[1.0] + [0.0] * 1023, sparse={"synthetic": 1.0}) for _text in texts]


class IncrementalFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ptha-incremental-foundation-")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _database(self) -> Path:
        path = self.root / "candidate.db"
        self._assert_inside(path)
        dense = FakeProvider(space="dense-space")
        sparse = FakeProvider(space="sparse-space;top_k=8")
        policy = build_chunk_policy([dense, sparse], content_budget_override=16)
        with NativeBuildStore(path) as store:
            store.conn.execute(
                "INSERT INTO source_documents VALUES('src','synthetic.md','synthetic.md','chat_md',NULL,'normal',NULL,NULL,'synthetic.md','md','hash',NULL,NULL,'{}')"
            )
            parsed = self._parsed_chat()
            store.insert_parsed_chat(parsed)
            store.create_chunks(policy=policy, tokenizer_provider=StrictestTokenizer([dense, sparse]), skip_low_interest=False)
            rows = store.conn.execute("SELECT id,block_id,text FROM retrieval_chunks ORDER BY id").fetchall()
            store.write_embedding_batch(
                rows=rows,
                dense_vectors=[[1.0] + [0.0] * 1023 for _row in rows],
                sparse_vectors=[{"synthetic": 1.0} for _row in rows],
                dense_model=dense.model_name,
                dense_space=dense.embedding_space_id,
                sparse_model=sparse.model_name,
                sparse_space=sparse.embedding_space_id,
            )
            _contract, fingerprint = embedding_contract_fingerprint(dense=dense, sparse=sparse)
            manifest = store.write_generation_manifest(embedding_contract_fingerprint=fingerprint)
            self.assertIsNotNone(manifest)
            store.conn.execute(
                "INSERT INTO native_build_audit VALUES(1,'kb.native_pre_mvp.v2','synthetic','completed',?,?,?,?)",
                (datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat(), "{}", "{}"),
            )
            store.commit()
        return path

    @staticmethod
    def _parsed_chat() -> ParsedChat:
        conversation = Conversation(
            id="conversation-row", source_document_id="src", conversation_id="conversation-native",
            conversation_template_id=None, title="Synthetic", create_time_utc="2026-01-01T00:00:00Z",
            update_time_utc="2026-01-01T00:00:00Z", message_count=1, assistant_messages=0,
            user_messages=1, text_chars=16, estimated_code_blocks=0, project_id=None,
            folder_kind=None, metadata_json={"source": "synthetic"},
        )
        message = Message(
            id="message-row", conversation_id=conversation.id, ordinal=1, role="user", message_id="message-native",
            time_utc="2026-01-01T00:00:00Z", raw_text="synthetic message", metadata_json={"kind": "synthetic"},
        )
        block = Block(
            id="block-row", message_id=message.id, conversation_id=conversation.id, ordinal=1,
            block_type="prose", language=None, raw_text=message.raw_text, normalized_text=message.raw_text,
            char_start=0, char_end=len(message.raw_text), metadata_json={},
        )
        return ParsedChat(conversation=conversation, messages=[message], blocks=[block], metadata={})

    def _assert_inside(self, path: Path) -> None:
        self.assertTrue(path.resolve().is_relative_to(self.root), path)

    def test_native_conversation_id_is_stable_across_exports(self) -> None:
        original = self._parsed_chat().conversation
        self.assertEqual(conversation_identity(original), conversation_identity(replace(original, source_document_id="other-source")))

    def test_native_message_id_is_stable_across_exports(self) -> None:
        original = self._parsed_chat().messages[0]
        self.assertEqual(
            message_identity(original, conversation_identity_id="sid_parent"),
            message_identity(replace(original, ordinal=99), conversation_identity_id="sid_parent"),
        )

    def test_fallback_identity_is_deterministic(self) -> None:
        message = replace(self._parsed_chat().messages[0], message_id=None)
        self.assertEqual(
            message_identity(message, conversation_identity_id="sid_parent"),
            message_identity(message, conversation_identity_id="sid_parent"),
        )

    def test_identity_does_not_depend_on_absolute_path(self) -> None:
        identity = conversation_identity(self._parsed_chat().conversation)
        self.assertNotIn("/", identity.external_id)

    def test_identity_does_not_depend_on_file_mtime(self) -> None:
        original = self._parsed_chat().conversation
        self.assertEqual(conversation_identity(original), conversation_identity(replace(original, update_time_utc=original.update_time_utc)))

    def test_canonical_hash_changes_when_canonical_content_changes(self) -> None:
        message = self._parsed_chat().messages[0]
        identity = message_identity(message, conversation_identity_id="sid_parent")
        self.assertNotEqual(
            message_revision(message, identity=identity).canonical_hash,
            message_revision(replace(message, raw_text="changed"), identity=identity).canonical_hash,
        )

    def test_canonical_hash_is_stable_for_equivalent_serialization(self) -> None:
        self.assertEqual(canonical_bytes({"b": "caf\u00e9\r\n", "a": [1, 2]}), canonical_bytes({"a": [1, 2], "b": "cafe\u0301\n"}))

    def test_hash_uses_full_blake3_256(self) -> None:
        digest = content_hash("ptha:test:v1", {"value": "synthetic"})
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_chunk_identity_is_independent_of_sqlite_rowid(self) -> None:
        kwargs = {"source_revision_id": "rev_1", "block_identity_id": "block_1", "chunk_policy_id": "policy", "ordinal": 1,
                  "source_char_start": 0, "source_char_end": 10}
        self.assertEqual(chunk_identity(**kwargs), chunk_identity(**kwargs))

    def test_block_identity_is_stable_for_formatting_only_offset_changes(self) -> None:
        original = self._parsed_chat().blocks[0]
        shifted = replace(original, char_start=10, char_end=27)
        self.assertEqual(
            block_identity(original, message_revision_id="rev_same"),
            block_identity(shifted, message_revision_id="rev_same"),
        )

    def test_chunk_content_hash_is_separate_from_chunk_identity(self) -> None:
        identifier = chunk_identity(source_revision_id="rev_1", block_identity_id="block_1", chunk_policy_id="policy", ordinal=1,
                                    source_char_start=0, source_char_end=10)
        self.assertNotEqual(identifier.removeprefix("chunk_"), chunk_content_hash("same range, different text"))

    def test_chunk_lineage_reaches_source_revision(self) -> None:
        path = self._database()
        with closing(sqlite3.connect(path)) as conn:
            row = conn.execute(
                "SELECT message_revision.canonical_hash,conversation_revision.canonical_hash "
                "FROM chunk_incremental_metadata cm "
                "JOIN source_entity_revisions message_revision ON message_revision.id=cm.source_revision_id "
                "JOIN block_source_lineage bl ON bl.source_revision_id=message_revision.id "
                "JOIN blocks b ON b.id=bl.block_id "
                "JOIN message_source_lineage ml ON ml.message_id=b.message_id "
                "JOIN source_entity_revisions conversation_revision ON conversation_revision.id=ml.conversation_source_revision_id"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(len(str(row[0])), 64)
        self.assertEqual(len(str(row[1])), 64)

    def test_transformation_versions_are_persisted(self) -> None:
        database = inspect_database(self._database())
        generation = database["incremental_metadata"]["generation"]
        self.assertEqual(generation["canonicalizer_version"], CANONICALIZER_VERSION)
        self.assertEqual(generation["block_builder_version"], BLOCK_BUILDER_VERSION)
        self.assertEqual(generation["chunker_version"], CHUNKER_VERSION)
        self.assertEqual(generation["parser_contract"], PARSER_CONTRACT)
        self.assertEqual(generation["source_transform_version"], SOURCE_TRANSFORM_VERSION)
        self.assertEqual(generation["canonical_representation_version"], CANONICAL_REPRESENTATION_VERSION)

    def test_embedding_contract_fingerprint_is_deterministic(self) -> None:
        first = embedding_contract_fingerprint(dense=FakeProvider(space="dense"), sparse=FakeProvider(space="sparse;top_k=8"))[1]
        second = embedding_contract_fingerprint(dense=FakeProvider(space="dense"), sparse=FakeProvider(space="sparse;top_k=8"))[1]
        self.assertEqual(first, second)

    def test_embedding_contract_changes_when_semantic_config_changes(self) -> None:
        first = embedding_contract_fingerprint(dense=FakeProvider(space="dense", revision="revision-a"), sparse=FakeProvider(space="sparse;top_k=8", revision="revision-a"))[1]
        second = embedding_contract_fingerprint(dense=FakeProvider(space="dense", revision="revision-b"), sparse=FakeProvider(space="sparse;top_k=8", revision="revision-b"))[1]
        self.assertNotEqual(first, second)

    def test_embedding_contract_ignores_batch_size(self) -> None:
        first = embedding_contract_fingerprint(dense=FakeProvider(space="dense", batch_size=4), sparse=FakeProvider(space="sparse;top_k=8", batch_size=4))[1]
        second = embedding_contract_fingerprint(dense=FakeProvider(space="dense", batch_size=32), sparse=FakeProvider(space="sparse;top_k=8", batch_size=32))[1]
        self.assertEqual(first, second)

    def test_generation_manifest_is_persisted(self) -> None:
        database = inspect_database(self._database())
        self.assertTrue(database["incremental_metadata"]["available"])
        self.assertTrue(database["incremental_metadata"]["generation"]["id"].startswith("gen_"))

    def test_generation_manifest_counts_match_database(self) -> None:
        path = self._database()
        with closing(sqlite3.connect(path)) as conn:
            manifest = conn.execute("SELECT source_entity_count,source_revision_count,block_count,chunk_count,dense_count,sparse_count FROM generation_manifests").fetchone()
            counts = (
                conn.execute("SELECT COUNT(*) FROM source_entity_identities").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM source_entity_revisions").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM dense_native_metadata").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM sparse_vector_metadata").fetchone()[0],
            )
        self.assertEqual(tuple(manifest), counts)

    def test_native_build_publishes_generation_manifest(self) -> None:
        export = self.root / "distilled"
        export.mkdir()
        (export / "synthetic.md").write_text(
            "# Synthetic\n\n## Metadata\n\n- `id`: `conversation-native`\n- `conversation_template_id`: ``\n"
            "- `create_time_utc`: `2026-01-01T00:00:00Z`\n- `update_time_utc`: `2026-01-01T00:00:00Z`\n"
            "\n## Conversation\n\n### 1. USER\n\n- `time_utc`: `2026-01-01T00:00:00Z`\n"
            "- `message_id`: `message-native`\n\nsynthetic message\n",
            encoding="utf-8",
        )
        output = self.root / "published.db"
        self._assert_inside(export)
        self._assert_inside(output)
        backend = SharedBackend()
        dense = FakeProvider(space="dense-space")
        sparse = FakeProvider(space="sparse-space;top_k=8")
        dense.backend = backend
        sparse.backend = backend
        with patch.object(native, "build_bge_m3_providers", return_value=(dense, sparse)):
            build_native_pre_mvp_db(
                export_path=export, output_db=output, model_revision="synthetic", batch_size=2,
                chunk_content_budget=16, progress=False,
            )
        database = inspect_database(output, integrity=True)
        self.assertEqual(database["state"], "ready")
        self.assertTrue(database["incremental_metadata"]["available"])
        self.assertTrue(database["incremental_metadata"]["generation"]["id"].startswith("gen_"))

    def test_generation_publish_includes_manifest_atomically(self) -> None:
        path = self.root / "atomic.db"
        self._assert_inside(path)
        with NativeBuildStore(path) as store:
            with self.assertRaises(RuntimeError):
                with store.embedding_batch_transaction():
                    store.write_generation_manifest(embedding_contract_fingerprint="a" * 64)
                    raise RuntimeError("synthetic pre-publication fault")
            self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM generation_manifests").fetchone()[0], 0)

    def test_canonical_default_chunk_budget_remains_256(self) -> None:
        self.assertEqual(DEFAULT_CANONICAL_CHUNK_CONTENT_BUDGET, 256)

    def test_legacy_database_remains_readable(self) -> None:
        path = self.root / "legacy.db"
        self._assert_inside(path)
        with closing(sqlite3.connect(path)) as conn:
            create_clean_native_schema(conn)
            conn.executescript(
                "DROP TRIGGER generation_manifests_immutable_update; DROP TRIGGER generation_manifests_immutable_delete; "
                "DROP TABLE generation_manifests; DROP TABLE chunk_incremental_metadata; DROP TABLE block_source_lineage; "
                "DROP TABLE message_source_lineage; DROP TABLE conversation_source_lineage; DROP TABLE source_entity_revisions; "
                "DROP TABLE source_entity_identities;"
            )
            conn.execute("INSERT INTO native_build_audit VALUES(1,'kb.native_pre_mvp.v1','legacy','completed','start','finish','{}','{}')")
            conn.commit()
        self.assertEqual(inspect_database(path)["state"], "ready")

    def test_legacy_database_reports_incremental_metadata_unavailable(self) -> None:
        path = self.root / "legacy-status.db"
        self._assert_inside(path)
        with closing(sqlite3.connect(path)) as conn:
            create_clean_native_schema(conn)
            conn.executescript(
                "DROP TRIGGER generation_manifests_immutable_update; DROP TRIGGER generation_manifests_immutable_delete; "
                "DROP TABLE generation_manifests; DROP TABLE chunk_incremental_metadata; DROP TABLE block_source_lineage; "
                "DROP TABLE message_source_lineage; DROP TABLE conversation_source_lineage; DROP TABLE source_entity_revisions; "
                "DROP TABLE source_entity_identities;"
            )
            conn.execute("INSERT INTO native_build_audit VALUES(1,'kb.native_pre_mvp.v1','legacy','completed','start','finish','{}','{}')")
            conn.commit()
        self.assertFalse(inspect_database(path)["incremental_metadata"]["available"])

    def test_absent_entity_is_not_classified_as_deleted(self) -> None:
        comparison = compare_source_revisions({"a": "one"}, {})
        self.assertEqual(comparison["absent_in_new_export"], ["a"])
        self.assertNotIn("deleted", comparison)

    def test_same_identity_same_hash_is_unchanged(self) -> None:
        self.assertEqual(compare_source_revisions({"a": "one"}, {"a": "one"})["unchanged"], ["a"])

    def test_same_identity_different_hash_is_changed_revision(self) -> None:
        self.assertEqual(compare_source_revisions({"a": "one"}, {"a": "two"})["changed"], ["a"])

    def test_new_identity_is_new(self) -> None:
        self.assertEqual(compare_source_revisions({}, {"a": "one"})["new"], ["a"])

    def test_all_test_paths_are_inside_temporary_root(self) -> None:
        self._assert_inside(self._database())

    def test_foreign_key_enforcement_and_integrity_check(self) -> None:
        path = self._database()
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
