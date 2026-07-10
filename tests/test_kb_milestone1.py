from __future__ import annotations

import json
import math
import sqlite3
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

from kb.cli import _chunked_embedding_space_id, build_edges_command, build_nodes_command, build_parser as build_index_parser, embed_knowledge_blocks, import_knowledge_base, ingest_attachments, ingest_chats  # noqa: E402
from kb.benchmark import DirectRetrievalSession, RankingConfig, analyze_direct_retrieval_evaluation, build_breakdowns, build_pairwise_queries, calculate_query_metrics, default_ranking_configs, evaluate_direct_retrieval_run, run_direct_retrieval_benchmark, validate_direct_retrieval_dataset  # noqa: E402
from kb.block_chunk_audit import audit_block_chunks  # noqa: E402
from kb.storage_audit import audit_storage  # noqa: E402
from kb.canary.multilingual_dense import run_canary  # noqa: E402
from kb.canary.real_data import _probe_metrics, _reject_unsafe_output_path, _validate_content_budget, load_or_create_manifest, validate_source_offsets  # noqa: E402
from kb.fusion_eval import SNAPSHOT_SCHEMA, SnapshotRow, _load_probes, _lexical_overlap, _rescue_analysis, _score_variant, evaluate_raw_score_snapshot  # noqa: E402
from kb.ingest.chat_md_parser import parse_chat_file  # noqa: E402
from kb.ingest.tree_walker import scan_tree  # noqa: E402
from kb.embeddings.mock_provider import MockDenseProvider, MockSparseProvider  # noqa: E402
from kb.embeddings.sentence_transformer_provider import _dense_embedding_space_id  # noqa: E402
from kb.index.chunk_builder import build_chunk_policy, build_retrieval_chunks  # noqa: E402
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
    effective_max_sequence_length = 64
    embedding_space_id = "static-dense;dim=2;normalize=false;symmetric=true;max_seq=64"
    runtime_metadata = {"backend": "test"}
    document_prefix = ""

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

    def embedding_input(self, text: str) -> str:
        return text

    def token_count(self, text: str) -> int:
        return len(text.split()) or (1 if text else 0)

    def assert_fits(self, text: str, *, chunk_id: str, block_id: str, source_identity: str) -> int:
        count = self.token_count(self.embedding_input(text))
        if count > self.effective_max_sequence_length:
            raise ValueError("over limit")
        return count


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
    effective_max_sequence_length = 64
    embedding_space_id = "static-sparse;document_encoder=documents;query_encoder=query;top_k=all;max_seq=64"
    runtime_metadata = {"backend": "test"}
    document_prefix = ""

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

    def embedding_input(self, text: str) -> str:
        return text

    def token_count(self, text: str) -> int:
        return len(text.split()) or (1 if text else 0)

    def assert_fits(self, text: str, *, chunk_id: str, block_id: str, source_identity: str) -> int:
        count = self.token_count(self.embedding_input(text))
        if count > self.effective_max_sequence_length:
            raise ValueError("over limit")
        return count


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
    def test_block_chunk_audit_reports_distribution_and_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "audit.db"
            output_dir = Path(tmp) / "report"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE source_documents (id TEXT PRIMARY KEY, interest_tier TEXT NOT NULL);
                CREATE TABLE conversations (id TEXT PRIMARY KEY, source_document_id TEXT NOT NULL);
                CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL);
                CREATE TABLE blocks (
                    id TEXT PRIMARY KEY, message_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    block_type TEXT NOT NULL, raw_text TEXT NOT NULL, char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL
                );
                CREATE TABLE retrieval_chunks (
                    id TEXT PRIMARY KEY, block_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    source_char_start INTEGER NOT NULL, source_char_end INTEGER NOT NULL,
                    token_count INTEGER NOT NULL
                );
                INSERT INTO source_documents VALUES ('sd-1', 'normal');
                INSERT INTO conversations VALUES ('conv-1', 'sd-1');
                INSERT INTO messages VALUES ('msg-1', 'conv-1', 'user');
                INSERT INTO blocks VALUES ('block-1', 'msg-1', 'conv-1', 'prose', 'abcdefghij', 0, 10);
                INSERT INTO blocks VALUES ('block-2', 'msg-1', 'conv-1', 'code', '   ', 10, 13);
                INSERT INTO retrieval_chunks VALUES ('chunk-1', 'block-1', 1, 0, 5, 4);
                INSERT INTO retrieval_chunks VALUES ('chunk-2', 'block-1', 2, 5, 10, 5);
                """
            )
            conn.commit()
            conn.close()

            report = audit_block_chunks(db_path, output_dir)

            self.assertEqual(report["summary"]["total_structural_blocks"], 2)
            self.assertEqual(report["summary"]["total_retrieval_chunks"], 2)
            self.assertEqual(report["distribution"][0]["chunks_per_block"], 0)
            self.assertEqual(report["distribution"][1]["chunks_per_block"], 2)
            self.assertEqual(report["blocks_without_chunks"]["by_reason"], {"empty_or_whitespace": 1})
            self.assertTrue(report["consistency"]["all_checks_passed"])
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "report.json").exists())

    def test_storage_audit_is_read_only_and_reports_vector_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "audit.db"
            output_dir = Path(tmp) / "report"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE source_documents (id TEXT PRIMARY KEY);
                CREATE TABLE conversations (id TEXT PRIMARY KEY, source_document_id TEXT);
                CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT);
                CREATE TABLE blocks (id TEXT PRIMARY KEY, message_id TEXT, block_type TEXT, raw_text TEXT);
                CREATE TABLE retrieval_chunks (id TEXT PRIMARY KEY, block_id TEXT, ordinal INTEGER, chunk_policy_id TEXT, text TEXT);
                CREATE TABLE dense_vectors (id TEXT PRIMARY KEY, owner_type TEXT, owner_id TEXT, dim INTEGER, vector_json TEXT);
                CREATE TABLE sparse_terms (owner_type TEXT, owner_id TEXT, token_id TEXT, token_text TEXT, weight REAL, model_name TEXT, PRIMARY KEY (owner_type, owner_id, token_id, model_name));
                CREATE INDEX idx_chunks_block ON retrieval_chunks(block_id);
                INSERT INTO source_documents VALUES ('sd');
                INSERT INTO conversations VALUES ('c', 'sd');
                INSERT INTO messages VALUES ('m', 'c', 'user');
                INSERT INTO blocks VALUES ('b', 'm', 'prose', 'hello');
                INSERT INTO retrieval_chunks VALUES ('ch', 'b', 0, 'v2', 'hello');
                INSERT INTO dense_vectors VALUES ('dv', 'retrieval_chunk', 'ch', 2, '[0.1, 0.2]');
                INSERT INTO sparse_terms VALUES ('retrieval_chunk', 'ch', '1', 'hello', 0.5, 'sparse');
                """
            )
            conn.commit()
            before = db_path.stat().st_size
            conn.close()

            report = audit_storage(db_path, output_dir)

            self.assertEqual(report["file"]["integrity_check"], "ok")
            self.assertEqual(report["dense"]["vector_count"], 1)
            self.assertEqual(report["sparse"]["term_count"], 1)
            self.assertEqual(report["sparse"]["representation_count"], 1)
            self.assertEqual(db_path.stat().st_size, before)
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "report.json").exists())

    def test_fusion_lexical_overlap_and_rescue_counts(self) -> None:
        from kb.canary.real_data import RealProbe

        probe = RealProbe("p", "different conceptual wording", "en", "c", "m", "strong_paraphrase", transformation_type="strong_paraphrase", source_language="en")
        lexical = _lexical_overlap(probe, "unrelated source vocabulary")
        self.assertEqual(lexical["shared_normalized_terms"], [])
        records = lambda rank: [{"probe_id": "p", "transformation_type": "strong_paraphrase", "message_rank": rank}]
        rescue = _rescue_analysis(dense_records=records(4), sparse_records=records(31), rrf_records=records(3))
        self.assertEqual(len(rescue["dense_rescues"]), 1)
        self.assertEqual(rescue["counts"]["dense_only_wins"], 1)
    def test_rrf_uses_branch_ranks_and_candidate_union(self) -> None:
        rows = [
            SnapshotRow("p", "dense", "b", 1, "m1", "c", "assistant", "prose", 0.9, 0.0, 1, 3),
            SnapshotRow("p", "sparse", "b", 2, "m2", "c", "assistant", "prose", 0.0, 0.9, 3, 1),
            SnapshotRow("p", "both", "b", 3, "m3", "c", "assistant", "prose", 0.8, 0.8, 2, 2),
        ]
        scores = dict((row.chunk_id, score) for row, score in _score_variant(rows, {"kind": "rrf", "pool": 2, "rrf_k": 60}))
        self.assertEqual(set(scores), {"dense", "sparse", "both"})
        self.assertAlmostEqual(scores["both"], 2 / 62)
        self.assertGreater(scores["both"], scores["dense"])

    def test_fusion_missing_branch_score_is_zero_not_penalty(self) -> None:
        rows = [
            SnapshotRow("p", "sparse-hit", "b", 1, "m1", "c", "assistant", "prose", 0.0, 0.9, 2, 1),
            SnapshotRow("p", "dense-hit", "b", 2, "m2", "c", "assistant", "prose", 0.9, 0.0, 1, 2),
        ]
        scores = dict((row.chunk_id, score) for row, score in _score_variant(rows, {"kind": "raw", "dense_weight": 0.1, "sparse_weight": 0.9, "pool": None}))
        self.assertAlmostEqual(scores["sparse-hit"], 0.81)
        self.assertGreater(scores["sparse-hit"], scores["dense-hit"])

    def test_fusion_evaluation_is_file_only_and_message_max_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = {
                "schema_version": "kb.real_data_preflight.probes.v1",
                "probes": [{"probe_id": "p", "query": "private", "query_language": "en", "source_language": "ru", "transformation_type": "RU->EN", "expected_conversation_id": "c1", "expected_message_id": "m1"}],
            }
            probe_path = root / "probes.json"
            probe_path.write_text(json.dumps(probes), encoding="utf-8")
            rows = [
                {"schema_version": SNAPSHOT_SCHEMA, "probe_id": "p", "chunk_id": "a", "block_id": "b", "chunk_ordinal": 1, "source_message_id": "m1", "source_conversation_id": "c1", "role": "assistant", "block_type": "code", "dense_score": 0.2, "sparse_score": 0.9, "dense_rank": 3, "sparse_rank": 1},
                {"schema_version": SNAPSHOT_SCHEMA, "probe_id": "p", "chunk_id": "b", "block_id": "b", "chunk_ordinal": 2, "source_message_id": "m1", "source_conversation_id": "c1", "role": "assistant", "block_type": "code", "dense_score": 0.1, "sparse_score": 0.8, "dense_rank": 4, "sparse_rank": 2},
                {"schema_version": SNAPSHOT_SCHEMA, "probe_id": "p", "chunk_id": "c", "block_id": "b", "chunk_ordinal": 1, "source_message_id": "m2", "source_conversation_id": "c2", "role": "assistant", "block_type": "prose", "dense_score": 0.9, "sparse_score": 0.1, "dense_rank": 1, "sparse_rank": 3},
                {"schema_version": SNAPSHOT_SCHEMA, "probe_id": "p", "chunk_id": "d", "block_id": "b", "chunk_ordinal": 1, "source_message_id": "m3", "source_conversation_id": "c3", "role": "assistant", "block_type": "prose", "dense_score": 0.8, "sparse_score": 0.0, "dense_rank": 2, "sparse_rank": 4},
            ]
            snapshot = root / "raw_scores.jsonl"
            snapshot.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with patch("kb.fusion_eval._build_dense_provider", side_effect=AssertionError("providers must not be loaded")), patch("kb.fusion_eval.SQLiteStore", side_effect=AssertionError("DB must not be opened")):
                report = evaluate_raw_score_snapshot(snapshot_path=snapshot, probe_path=probe_path, output_dir=root / "out")
                repeated = evaluate_raw_score_snapshot(snapshot_path=snapshot, probe_path=probe_path, output_dir=root / "out-repeated")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(
                [(item["id"], item["metrics"]) for item in report["variants"]],
                [(item["id"], item["metrics"]) for item in repeated["variants"]],
            )
            sparse = next(item for item in report["variants"] if item["id"] == "sparse_only")
            self.assertEqual(sparse["metrics"]["message_recall_at"]["1"], 1.0)
            self.assertTrue(Path(report["report_json"]).exists())

    def test_fusion_loader_accepts_unified_probe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unified.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "kb.unified_retrieval_probes.v1",
                        "probes": [
                            {
                                "probe_id": "p",
                                "query": "private",
                                "category": "exact_identifier",
                                "query_language": "en",
                                "expected_conversation_id": "c1",
                                "expected_message_id": "m1",
                                "expected_role": "user",
                                "source_dataset": "gold",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            probe = _load_probes(path)[0]
            self.assertEqual(probe.category, "exact_identifier")
            self.assertEqual(probe.source_dataset, "gold")
            self.assertEqual(probe.probe_type, "exact_identifier")
    def test_real_preflight_manifest_selection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Preflight"
            chat_dir.mkdir(parents=True)
            for index in range(12):
                chat = SAMPLE_CHAT.replace("conv-1", f"conv-{index}").replace("msg-user-1", f"msg-user-{index}")
                chat = chat.replace("How should memory write routing work?", "русский English mixed text " + "tail " * 300)
                (chat_dir / f"chat-{index:02d}.md").write_text(chat, encoding="utf-8")
            first = load_or_create_manifest(root, Path(tmp) / "first.json", max_conversations=16)
            second = load_or_create_manifest(root, Path(tmp) / "second.json", max_conversations=16)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 12)

    def test_real_preflight_rejects_production_and_legacy_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "distinct path"):
                _reject_unsafe_output_path(root, root / "chat_memory.db")
            with self.assertRaisesRegex(ValueError, "distinct path"):
                _reject_unsafe_output_path(root, root / "chat_memory_v2_bge_m3.db")

    def test_real_preflight_checks_budget_against_every_provider_contract(self) -> None:
        class Provider:
            model_name = "strict-provider"
            effective_max_sequence_length = 512

            def contract_dict(self):
                return {"computed_content_budget": 506}

        with self.assertRaisesRegex(ValueError, "requested=512, safe_content_budget=506"):
            _validate_content_budget(512, [Provider()])
        _validate_content_budget(506, [Provider()])

    def test_real_preflight_offset_validation_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project"
            chat_dir.mkdir(parents=True)
            (chat_dir / "chat.md").write_text(SAMPLE_CHAT, encoding="utf-8")
            db = Path(tmp) / "preflight.db"
            ingest_chats(root, db)
            embed_knowledge_blocks(db_path=db, provider="mock", dense_provider="mock", sparse_provider="mock", batch_size=4)
            self.assertTrue(validate_source_offsets(db, sample_size=30)["passed"])
            with SQLiteStore(db) as store:
                store.conn.execute("UPDATE retrieval_chunks SET text='offset mismatch' WHERE id=(SELECT id FROM retrieval_chunks LIMIT 1)")
                store.commit()
            self.assertFalse(validate_source_offsets(db, sample_size=30)["passed"])

    def test_real_preflight_probe_metrics_cover_chunk_message_and_conversation(self) -> None:
        metrics = _probe_metrics([
            {"probe_type": "tail", "query_language": "en", "chunk_rank": 2, "message_rank": 2, "conversation_rank": 1},
            {"probe_type": "code", "query_language": "ru", "chunk_rank": None, "message_rank": 9, "conversation_rank": 4},
        ])
        self.assertEqual(metrics["chunk_recall_at"]["1"], 0.0)
        self.assertEqual(metrics["message_recall_at"]["5"], 0.5)
        self.assertEqual(metrics["conversation_recall_at"]["5"], 1.0)
    def test_long_prose_is_split_into_tokenizer_aware_chunks_with_full_coverage(self) -> None:
        provider = MockDenseProvider(max_sequence_length=16)
        policy = build_chunk_policy([provider], version="v1")
        text = " ".join(f"token{i}" for i in range(80))
        chunks = build_retrieval_chunks(
            block_id="block-long",
            block_text=text,
            block_char_start=10,
            policy=policy,
            tokenizer_provider=provider,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= policy.content_token_budget for chunk in chunks))
        ranges = sorted((chunk.source_char_start, chunk.source_char_end) for chunk in chunks)
        covered = set()
        for start, end in ranges:
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(10, 10 + len(text))))
        self.assertLess(chunks[1].source_char_start, chunks[0].source_char_end)

    def test_chunk_policy_v2_natural_boundary_does_not_overlap(self) -> None:
        provider = MockDenseProvider(max_sequence_length=16)
        policy = build_chunk_policy([provider], version="v2")
        text = " ".join(f"token{i}" for i in range(80))
        chunks = build_retrieval_chunks(
            block_id="block-v2-natural",
            block_text=text,
            block_char_start=0,
            policy=policy,
            tokenizer_provider=provider,
        )

        self.assertGreater(len(chunks), 1)
        self.assertNotEqual(build_chunk_policy([provider], version="v1").id, policy.id)
        self.assertEqual(sum(chunk.overlap_token_count for chunk in chunks), 0)
        self.assertTrue(all(chunk.split_reason in {"natural_boundary", "complete"} for chunk in chunks))

    def test_chunk_policy_v2_token_window_fallback_uses_small_overlap(self) -> None:
        class CharTokenProvider(MockDenseProvider):
            def token_count(self, text: str) -> int:
                return len(text)

        provider = CharTokenProvider(max_sequence_length=20)
        policy = build_chunk_policy([provider], version="v2")
        text = "x" * 500
        chunks = build_retrieval_chunks(
            block_id="block-v2-fallback",
            block_text=text,
            block_char_start=0,
            policy=policy,
            tokenizer_provider=provider,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(chunk.split_reason == "token_window_fallback" for chunk in chunks))
        self.assertTrue(any(chunk.overlap_token_count > 0 for chunk in chunks))
        self.assertEqual(policy.overlap_tokens, max(1, policy.content_token_budget // 16))

    def test_provider_contract_exposes_distinct_limits(self) -> None:
        provider = MockDenseProvider(max_sequence_length=32)

        contract = provider.contract_dict()

        self.assertEqual(contract["tokenizer_model_max_length"], 32)
        self.assertEqual(contract["backbone_max_position_embeddings"], 32)
        self.assertEqual(contract["sentence_transformer_max_seq_length"], 32)
        self.assertEqual(contract["configured_effective_max_seq_length"], 32)
        self.assertEqual(contract["computed_content_budget"], 28)

    def test_max_sequence_override_is_explicit_in_embedding_space_identity(self) -> None:
        default_id = _dense_embedding_space_id(
            "example/model",
            normalize_embeddings=True,
            output_dim=384,
            max_seq_length=128,
        )
        override_id = _dense_embedding_space_id(
            "example/model",
            normalize_embeddings=True,
            output_dim=384,
            max_seq_length=256,
            max_seq_override=256,
        )

        self.assertNotIn("max_seq_override", default_id)
        self.assertIn("max_seq_override=256", override_id)
        self.assertNotEqual(default_id, override_id)

    def test_mock_canary_report_is_created_and_counts_cross_language_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_canary(
                input_dir=None,
                work_dir=Path(tmp),
                models=["mock"],
                dense_device=None,
                sparse_device=None,
                batch_size=8,
                output_report=None,
                effective_max_seq_length=None,
                chunk_content_budget=None,
                sparse_provider="mock",
                sparse_model="mock-sparse",
                sparse_top_k=16,
                keep_databases=False,
                dense_provider="mock",
            )

            self.assertEqual(report["status"], "completed")
            self.assertTrue(Path(report["report_json"]).exists())
            self.assertTrue(Path(report["report_md"]).exists())
            model = report["models"][0]
            self.assertEqual(model["status"], "completed")
            self.assertGreater(model["retrieval_chunks"], 0)
            self.assertIn("RU->EN", model["metrics"]["hybrid"]["breakdown"])
            self.assertIn("EN->RU", model["metrics"]["hybrid"]["breakdown"])

    def test_canary_mps_provider_failure_is_not_reported_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("kb.canary.multilingual_dense._build_dense_provider", side_effect=RuntimeError("mps unavailable")):
                report = run_canary(
                    input_dir=None,
                    work_dir=Path(tmp),
                    models=["mock"],
                    dense_device="mps",
                    sparse_device=None,
                    batch_size=8,
                    output_report=None,
                    effective_max_seq_length=None,
                    chunk_content_budget=None,
                    sparse_provider="none",
                    sparse_model="mock-sparse",
                    sparse_top_k=16,
                    keep_databases=False,
                    dense_provider="mock",
                )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["models"][0]["status"], "failed_provider_load")
        self.assertEqual(report["models"][0]["mps_status"], "failed")

    def test_chunk_offsets_preserve_cyrillic_emoji_and_mixed_text(self) -> None:
        provider = MockDenseProvider(max_sequence_length=12)
        policy = build_chunk_policy([provider])
        text = "Привет 😀 memory routing\nСледующая строка с SIP и VoLTE"
        chunks = build_retrieval_chunks(
            block_id="block-unicode",
            block_text=text,
            block_char_start=3,
            policy=policy,
            tokenizer_provider=provider,
        )

        for chunk in chunks:
            local_start = chunk.source_char_start - 3
            local_end = chunk.source_char_end - 3
            self.assertEqual(chunk.text, text[local_start:local_end])

    def test_provider_assert_fits_rejects_over_limit_embedding_input(self) -> None:
        provider = MockDenseProvider(max_sequence_length=4)
        with self.assertRaisesRegex(ValueError, "Embedding input exceeds provider limit"):
            provider.assert_fits(
                "one two three four five",
                chunk_id="chunk-over",
                block_id="block-over",
                source_identity="message-over",
            )

    def test_embedding_pipeline_writes_chunk_representations_and_searches_late_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            chat_dir = root / "Projects" / "Project_17"
            chat_dir.mkdir(parents=True)
            long_prefix = " ".join(f"filler{i}" for i in range(90))
            late_terms = "late searchable sentinel protocol aware tokenization"
            chat = SAMPLE_CHAT.replace("How should memory write routing work?", f"{long_prefix} {late_terms}")
            (chat_dir / "chat.md").write_text(chat, encoding="utf-8")
            db = Path(tmp) / "chat_memory.db"

            ingest_chats(root, db, limit=10)
            embed_knowledge_blocks(
                db_path=db,
                provider="mock",
                dense_provider="mock",
                sparse_provider="mock",
                batch_size=4,
            )
            payload = hybrid_query(
                db_path=db,
                query="sentinel protocol aware tokenization",
                dense_provider="mock",
                sparse_provider="mock",
                alpha=0.0,
                beta=1.0,
                limit=5,
            )

            self.assertGreater(payload["candidate_blocks"], 0)
            self.assertGreater(payload["results"][0]["sparse_score"], 0.0)
            self.assertGreater(payload["results"][0]["source_char_start"], 0)
            with SQLiteStore(db) as store:
                table_stats = store.stats()
                self.assertGreater(table_stats["retrieval_chunks"], table_stats["blocks"])
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) FROM dense_vectors WHERE owner_type != 'retrieval_chunk'").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) FROM sparse_terms WHERE owner_type != 'retrieval_chunk'").fetchone()[0],
                    0,
                )

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
        policy = build_chunk_policy([dense, sparse])
        with SQLiteStore(db) as store:
            store.rebuild_retrieval_chunks(policy=policy, tokenizer_provider=dense, skip_low_interest_content=True)
            dense_space_id = _chunked_embedding_space_id(dense.embedding_space_id, policy.id)
            sparse_space_id = _chunked_embedding_space_id(sparse.embedding_space_id, policy.id)
            rows = store.conn.execute("SELECT id, text FROM retrieval_chunks ORDER BY text").fetchall()
            for row in rows:
                text = str(row["text"])
                store.upsert_dense_vector(
                    owner_type="retrieval_chunk",
                    owner_id=str(row["id"]),
                    model_name=dense.model_name,
                    model_version=dense_space_id,
                    runtime_metadata_json=json.dumps(dense.runtime_metadata, sort_keys=True),
                    vector=dense.embed_documents([text])[0],
                )
                store.replace_sparse_terms(
                    owner_type="retrieval_chunk",
                    owner_id=str(row["id"]),
                    model_name=sparse.model_name,
                    embedding_space_id=sparse_space_id,
                    terms=sparse.embed_documents([text])[0],
                )
            store.commit()
        return db

    def _first_chunk_row(self, db: Path, *, descending: bool = False):
        order = "DESC" if descending else "ASC"
        with SQLiteStore(db) as store:
            return store.conn.execute(
                f"""
                SELECT
                    rc.id,
                    sd.relative_path,
                    c.id AS conversation_id,
                    m.id AS message_id,
                    rc.text AS text_for_display
                FROM retrieval_chunks rc
                JOIN blocks b ON b.id = rc.block_id
                JOIN messages m ON m.id = b.message_id
                JOIN conversations c ON c.id = m.conversation_id
                JOIN source_documents sd ON sd.id = c.source_document_id
                ORDER BY rc.id {order}
                LIMIT 1
                """
            ).fetchone()

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
        row = self._first_chunk_row(db)
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

    def _metric_dataset_record(self, query_id: str, expected: list[dict]) -> dict:
        return {
            "id": query_id,
            "query": f"query {query_id}",
            "query_type": "exact_terms",
            "language": "en",
            "source_language": "en",
            "topic": "metrics",
            "expected": expected,
            "notes": "Synthetic metric fixture.",
        }

    def _metric_result_record(self, dataset_record: dict, *, config: dict | None = None, ranks: dict[str, int | None], top_results: list[dict] | None = None) -> dict:
        config = config or {"id": "dense_100_sparse_000", "alpha": 1.0, "beta": 0.0}
        return {
            "schema_version": "kb.benchmark.query_result.v1",
            "query_id": dataset_record["id"],
            "query": dataset_record["query"],
            "query_type": dataset_record["query_type"],
            "language": dataset_record["language"],
            "source_language": dataset_record["source_language"],
            "topic": dataset_record["topic"],
            "configuration": config,
            "expected": [
                {"block_id": item["block_id"], "relevance": item["relevance"], "rank": ranks.get(item["block_id"])}
                for item in dataset_record["expected"]
            ],
            "candidate_blocks": 10,
            "top_results": top_results if top_results is not None else [],
            "diagnostics": {},
            "latency_ms": {},
        }

    def _write_synthetic_evaluation_run(self, root: Path, *, query_count: int = 120, duplicate: bool = False, status: str = "completed", hash_mismatch: bool = False) -> tuple[Path, Path]:
        run_dir = root / "run"
        run_dir.mkdir()
        dataset = root / "dataset.jsonl"
        configs = [config.as_dict() for config in default_ranking_configs()]
        dataset_records = [
            self._metric_dataset_record(
                f"query-{idx:03d}",
                [
                    {
                        "block_id": f"kb-{idx:03d}",
                        "relevance": 3,
                        "source_path": f"doc-{idx:03d}.md",
                        "conversation_id": f"conv-{idx:03d}",
                        "message_id": f"msg-{idx:03d}",
                    }
                ],
            )
            for idx in range(query_count)
        ]
        self._write_dataset(dataset, dataset_records)
        dataset_hash = "bad" if hash_mismatch else self._file_sha256(dataset)
        manifest = {
            "schema_version": "kb.benchmark.run.v1",
            "run_id": "synthetic-run",
            "status": status,
            "dataset": {"path": str(dataset), "sha256": dataset_hash, "query_count": query_count},
            "database": {"path": "db", "candidate_blocks": 10},
            "providers": {},
            "configurations": configs,
            "timing_ms": {},
            "completed_queries": query_count,
            "failed_queries": 0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        records = []
        for dataset_record in dataset_records:
            for config in configs:
                records.append(
                    self._metric_result_record(
                        dataset_record,
                        config=config,
                        ranks={dataset_record["expected"][0]["block_id"]: 1},
                        top_results=[
                            {
                                "rank": 1,
                                "block_id": dataset_record["expected"][0]["block_id"],
                                "source_path": dataset_record["expected"][0]["source_path"],
                                "conversation_id": dataset_record["expected"][0]["conversation_id"],
                                "message_id": dataset_record["expected"][0]["message_id"],
                            }
                        ],
                    )
                )
        if duplicate:
            records.append(records[0])
        (run_dir / "results.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return run_dir, dataset

    def _file_sha256(self, path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

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
                dense_chunk_vectors = store.conn.execute(
                    "SELECT COUNT(*) FROM dense_vectors WHERE owner_type = 'retrieval_chunk'"
                ).fetchone()[0]
                sparse_chunk_vectors = store.conn.execute(
                    "SELECT COUNT(DISTINCT owner_id) FROM sparse_terms WHERE owner_type = 'retrieval_chunk'"
                ).fetchone()[0]
            self.assertEqual(stats["dense_vectors"], stats["retrieval_chunks"])
            self.assertEqual(dense_chunk_vectors, stats["retrieval_chunks"])
            self.assertEqual(sparse_chunk_vectors, stats["retrieval_chunks"])
            self.assertGreater(stats["sparse_terms"], 0)

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
                dense_chunks = store.conn.execute(
                    "SELECT COUNT(*) FROM dense_vectors WHERE owner_type = 'retrieval_chunk'"
                ).fetchone()[0]
                sparse_chunks = store.conn.execute(
                    "SELECT COUNT(DISTINCT owner_id) FROM sparse_terms WHERE owner_type = 'retrieval_chunk'"
                ).fetchone()[0]
            self.assertEqual(dense_chunks, 3)
            self.assertEqual(sparse_chunks, 3)

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
        explicit_policy_args = index_parser.parse_args(
            [
                "import",
                "--input",
                "export",
                "--db",
                "chat_memory.db",
                "--chunk-policy",
                "canonical_token_chunks:v2",
            ]
        )
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
        self.assertEqual(explicit_policy_args.chunk_policy, "canonical_token_chunks:v2")

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
            embed_knowledge_blocks(db_path=db, provider="mock", dense_provider="mock", sparse_provider="mock")
            row = self._first_chunk_row(db)
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
            embed_knowledge_blocks(db_path=db, provider="mock", dense_provider="mock", sparse_provider="mock")
            row = self._first_chunk_row(db)
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

            chunk_ids = [item["chunk_id"] for item in results]
            self.assertEqual(chunk_ids, sorted(chunk_ids))

    def test_benchmark_run_writes_manifest_and_results_from_single_raw_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._static_retrieval_db(tmp)
            rows = [self._first_chunk_row(db), self._first_chunk_row(db, descending=True)]
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
                top_k=50,
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
                item["chunk_id"]: (item["dense_score"], item["sparse_score"]) for item in per_query[0]["top_results"]
            }
            last_scores = {
                item["chunk_id"]: (item["dense_score"], item["sparse_score"]) for item in per_query[-1]["top_results"]
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
            row = self._first_chunk_row(db, descending=True)
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
            row = self._first_chunk_row(db)
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

    def test_evaluation_metrics_recall_mrr_and_primary_are_rank_based(self) -> None:
        dataset = self._metric_dataset_record(
            "q",
            [
                {"block_id": "primary", "relevance": 3, "source_path": "a.md", "conversation_id": "conv-a", "message_id": "m1"},
                {"block_id": "secondary", "relevance": 2, "source_path": "b.md", "conversation_id": "conv-b", "message_id": "m2"},
            ],
        )
        result = self._metric_result_record(
            dataset,
            ranks={"primary": 5, "secondary": 1},
            top_results=[
                {"rank": 1, "block_id": "secondary", "source_path": "b.md", "conversation_id": "conv-b", "message_id": "m2"},
                {"rank": 5, "block_id": "primary", "source_path": "a.md", "conversation_id": "conv-a", "message_id": "m1"},
            ],
        )

        metrics = calculate_query_metrics(result, dataset)

        self.assertEqual(metrics["first_relevant_rank"], 1)
        self.assertEqual(metrics["primary_rank"], 5)
        self.assertEqual(metrics["recall_at"], {"1": 1, "5": 1, "10": 1, "20": 1})
        self.assertEqual(metrics["primary_recall_at"], {"1": 0, "5": 1, "10": 1, "20": 1})
        self.assertAlmostEqual(metrics["reciprocal_rank"], 1.0)
        self.assertAlmostEqual(metrics["primary_reciprocal_rank"], 0.2)

    def test_evaluation_metrics_handle_rank_beyond_20(self) -> None:
        dataset = self._metric_dataset_record(
            "q",
            [{"block_id": "primary", "relevance": 3, "source_path": "a.md", "conversation_id": "conv-a", "message_id": "m1"}],
        )
        result = self._metric_result_record(dataset, ranks={"primary": 25})

        metrics = calculate_query_metrics(result, dataset)

        self.assertEqual(metrics["recall_at"], {"1": 0, "5": 0, "10": 0, "20": 0})
        self.assertAlmostEqual(metrics["reciprocal_rank"], 1 / 25)

    def test_evaluation_metrics_ndcg_uses_graded_relevance_and_truncation(self) -> None:
        dataset = self._metric_dataset_record(
            "q",
            [
                {"block_id": "rel3", "relevance": 3, "source_path": "a.md", "conversation_id": "conv-a", "message_id": "m1"},
                {"block_id": "rel2", "relevance": 2, "source_path": "b.md", "conversation_id": "conv-b", "message_id": "m2"},
                {"block_id": "rel1", "relevance": 1, "source_path": "c.md", "conversation_id": "conv-c", "message_id": "m3"},
            ],
        )
        result = self._metric_result_record(dataset, ranks={"rel3": 4, "rel2": 2, "rel1": 7})

        metrics = calculate_query_metrics(result, dataset)

        self.assertEqual(metrics["ndcg_at"]["1"], 0.0)
        dcg5 = (3 / math.log2(3)) + (7 / math.log2(5))
        idcg5 = 7 + (3 / math.log2(3)) + (1 / math.log2(4))
        self.assertAlmostEqual(metrics["ndcg_at"]["5"], dcg5 / idcg5)
        dcg10 = dcg5 + (1 / math.log2(8))
        self.assertAlmostEqual(metrics["ndcg_at"]["10"], dcg10 / idcg5)

    def test_evaluation_metrics_document_and_conversation_recall(self) -> None:
        dataset = self._metric_dataset_record(
            "q",
            [{"block_id": "primary", "relevance": 3, "source_path": "doc.md", "conversation_id": "conv-a", "message_id": "m1"}],
        )
        result = self._metric_result_record(
            dataset,
            ranks={"primary": 50},
            top_results=[
                {"rank": 1, "block_id": "other-doc", "source_path": "doc.md", "conversation_id": "other", "message_id": "m2"},
                {"rank": 2, "block_id": "other-conv", "source_path": "other.md", "conversation_id": "conv-a", "message_id": "m3"},
            ],
        )

        metrics = calculate_query_metrics(result, dataset)

        self.assertEqual(metrics["document_recall_at"]["1"], 1)
        self.assertEqual(metrics["conversation_recall_at"]["1"], 0)
        self.assertEqual(metrics["conversation_recall_at"]["5"], 1)

    def test_evaluation_metrics_ignore_null_conversation_matches(self) -> None:
        dataset = self._metric_dataset_record(
            "q",
            [{"block_id": "primary", "relevance": 3, "source_path": "doc.md", "conversation_id": None, "message_id": "m1"}],
        )
        result = self._metric_result_record(
            dataset,
            ranks={"primary": 10},
            top_results=[{"rank": 1, "block_id": "other", "source_path": "other.md", "conversation_id": None, "message_id": "m2"}],
        )

        metrics = calculate_query_metrics(result, dataset)

        self.assertEqual(metrics["conversation_recall_at"], {"1": 0, "5": 0, "10": 0, "20": 0})

    def test_evaluate_run_writes_completed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp))

            report = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["query_count"], 120)
            self.assertEqual(report["configuration_count"], 7)
            self.assertEqual(report["query_metric_records"], 840)
            evaluation_dir = Path(report["evaluation_dir"])
            self.assertEqual(len((evaluation_dir / "query_metrics.jsonl").read_text().splitlines()), 840)
            summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["configuration_count"], 7)
            self.assertIn("| Configuration | R@1 | R@5 |", (evaluation_dir / "report.md").read_text(encoding="utf-8"))
            manifest = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["query_metric_records"], 840)
            self.assertIn("source_manifest_sha256", manifest)

    def test_evaluate_run_rejects_duplicate_query_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp), duplicate=True)

            report = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)

            self.assertEqual(report["status"], "failed")
            self.assertIn("duplicate", report["error"])

    def test_evaluate_run_rejects_hash_mismatch_and_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp), hash_mismatch=True)
            hash_report = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)
            self.assertEqual(hash_report["status"], "failed")
            self.assertIn("SHA256", hash_report["error"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp), status="failed")
            status_report = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)
            self.assertEqual(status_report["status"], "failed")
            self.assertIn("completed", status_report["error"])

    def test_analysis_breakdowns_group_dimensions_and_metrics(self) -> None:
        configs = [
            {"id": "dense_100_sparse_000", "alpha": 1.0, "beta": 0.0},
            {"id": "dense_000_sparse_100", "alpha": 0.0, "beta": 1.0},
        ]
        items = []
        for query_id, query_type, language, source_language, topic, rr in [
            ("q1", "exact_terms", "en", "ru", "llm_memory", 1.0),
            ("q2", "paraphrase", "ru", "ru", "career", 0.5),
        ]:
            for config in configs:
                items.append(
                    {
                        "query_id": query_id,
                        "query": query_id,
                        "query_type": query_type,
                        "language": language,
                        "source_language": source_language,
                        "topic": topic,
                        "configuration": config,
                        "recall_at": {"1": int(rr == 1.0), "5": 1, "10": 1, "20": 1},
                        "primary_recall_at": {"1": int(rr == 1.0), "5": 1, "10": 1, "20": 1},
                        "reciprocal_rank": rr,
                        "primary_reciprocal_rank": rr,
                        "ndcg_at": {"1": rr, "5": rr, "10": rr, "20": rr},
                        "document_recall_at": {"1": 0, "5": 1, "10": 1, "20": 1},
                        "conversation_recall_at": {"1": 0, "5": 1, "10": 1, "20": 1},
                        "primary_rank": int(1 / rr),
                    }
                )

        breakdowns = build_breakdowns(items, configs)

        self.assertEqual(set(breakdowns["dimensions"]), {"query_type", "language", "source_language", "language_direction", "topic"})
        exact_slice = next(item for item in breakdowns["dimensions"]["query_type"] if item["value"] == "exact_terms")
        self.assertEqual(exact_slice["query_count"], 1)
        self.assertTrue(exact_slice["small_sample"])
        self.assertEqual(exact_slice["configurations"][0]["mean_primary_rank"], 1.0)
        self.assertEqual(exact_slice["configurations"][0]["primary_miss_count"], 0)

    def test_analysis_best_configuration_uses_metrics_and_tiebreak(self) -> None:
        configs = [
            {"id": "dense_100_sparse_000", "alpha": 1.0, "beta": 0.0},
            {"id": "dense_000_sparse_100", "alpha": 0.0, "beta": 1.0},
        ]
        items = []
        for config in configs:
            items.append(
                {
                    "query_id": "q1",
                    "query": "q1",
                    "query_type": "exact_terms",
                    "language": "en",
                    "source_language": "en",
                    "topic": "topic",
                    "configuration": config,
                    "recall_at": {"1": 1, "5": 1, "10": 1, "20": 1},
                    "primary_recall_at": {"1": 1, "5": 1, "10": 1, "20": 1},
                    "reciprocal_rank": 1.0,
                    "primary_reciprocal_rank": 1.0,
                    "ndcg_at": {"1": 1, "5": 1, "10": 1, "20": 1},
                    "document_recall_at": {"1": 1, "5": 1, "10": 1, "20": 1},
                    "conversation_recall_at": {"1": 1, "5": 1, "10": 1, "20": 1},
                    "primary_rank": 1,
                }
            )

        breakdowns = build_breakdowns(items, configs)
        slice_item = breakdowns["dimensions"]["query_type"][0]

        self.assertEqual(slice_item["best_configuration"]["mrr"], "dense_100_sparse_000")
        self.assertEqual(slice_item["best_configuration"]["recall_at_10"], "dense_100_sparse_000")
        self.assertEqual(slice_item["best_configuration"]["ndcg_at_10"], "dense_100_sparse_000")

    def test_analysis_pairwise_and_hybrid_classes(self) -> None:
        def metric(query_id, config_id, rr, recall10=1, doc10=1, rank=1):
            alpha = 1.0 if config_id == "dense_100_sparse_000" else 0.0
            return {
                "query_id": query_id,
                "query": query_id,
                "query_type": "exact_terms",
                "language": "en",
                "source_language": "en",
                "topic": "topic",
                "configuration": {"id": config_id, "alpha": alpha, "beta": 1.0 - alpha},
                "reciprocal_rank": rr,
                "primary_rank": rank,
                "recall_at": {"10": recall10},
                "document_recall_at": {"10": doc10},
            }

        config_ids = [c[0] for c in [
            ("dense_100_sparse_000",),
            ("dense_080_sparse_020",),
            ("dense_065_sparse_035",),
            ("dense_050_sparse_050",),
            ("dense_035_sparse_065",),
            ("dense_020_sparse_080",),
            ("dense_000_sparse_100",),
        ]]
        items = []
        cases = {
            "dense_win": (1.0, 0.5, 0.75),
            "sparse_win": (0.5, 1.0, 0.75),
            "tie": (1.0, 1.0, 1.0),
            "both_miss": (0.0, 0.0, 0.0),
            "hybrid_beats_both": (0.5, 0.5, 1.0),
        }
        for query_id, (dense_rr, sparse_rr, hybrid_rr) in cases.items():
            for config_id in config_ids:
                rr = dense_rr if config_id == "dense_100_sparse_000" else sparse_rr if config_id == "dense_000_sparse_100" else hybrid_rr
                items.append(metric(query_id, config_id, rr, recall10=int(rr > 0), doc10=int(rr > 0), rank=1 if rr else None))
        items.append(metric("doc_hit", "dense_100_sparse_000", 0.0, recall10=0, doc10=1, rank=None))
        for config_id in config_ids[1:]:
            items.append(metric("doc_hit", config_id, 0.0, recall10=0, doc10=0, rank=None))

        pairwise = {item["query_id"]: item for item in build_pairwise_queries(items)}

        self.assertEqual(pairwise["dense_win"]["dense_vs_sparse_class"], "dense_win")
        self.assertEqual(pairwise["sparse_win"]["dense_vs_sparse_class"], "sparse_win")
        self.assertEqual(pairwise["tie"]["dense_vs_sparse_class"], "tie")
        self.assertEqual(pairwise["both_miss"]["dense_vs_sparse_class"], "both_miss")
        self.assertEqual(pairwise["doc_hit"]["dense_vs_sparse_class"], "dense_block_miss_document_hit")
        self.assertEqual(pairwise["hybrid_beats_both"]["hybrid_comparison_class"], "hybrid_beats_both")

    def test_analyze_run_writes_completed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp))
            evaluation = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)

            report = analyze_direct_retrieval_evaluation(evaluation_dir=Path(evaluation["evaluation_dir"]), dataset_path=dataset)

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["query_count"], 120)
            self.assertEqual(report["configuration_count"], 7)
            self.assertEqual(report["breakdown_dimensions"], 5)
            self.assertEqual(report["pairwise_record_count"], 120)
            analysis_dir = Path(report["analysis_dir"])
            self.assertEqual(len((analysis_dir / "pairwise_queries.jsonl").read_text().splitlines()), 120)
            breakdowns = json.loads((analysis_dir / "breakdowns.json").read_text(encoding="utf-8"))
            self.assertEqual(set(breakdowns["dimensions"]), {"query_type", "language", "source_language", "language_direction", "topic"})
            self.assertIn("Dense vs sparse", (analysis_dir / "report.md").read_text(encoding="utf-8"))
            manifest = json.loads((analysis_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["pairwise_record_count"], 120)

    def test_analyze_run_rejects_hash_and_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp))
            evaluation = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)
            dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            report = analyze_direct_retrieval_evaluation(evaluation_dir=Path(evaluation["evaluation_dir"]), dataset_path=dataset)
            self.assertEqual(report["status"], "failed")
            self.assertIn("SHA256", report["error"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset = self._write_synthetic_evaluation_run(Path(tmp))
            evaluation = evaluate_direct_retrieval_run(run_dir=run_dir, dataset_path=dataset)
            summary_path = Path(evaluation["evaluation_dir"]) / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metrics"][0]["mrr"] = 0.123
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            report = analyze_direct_retrieval_evaluation(evaluation_dir=Path(evaluation["evaluation_dir"]), dataset_path=dataset)
            self.assertEqual(report["status"], "failed")
            self.assertIn("summary mismatch", report["error"])

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

            query_scores = {item["chunk_id"]: item["final_score"] for item in query_payload["results"]}
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
            self.assertEqual(node_vector_count, 0)

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
            self.assertNotIn("dense_sim", edge_kinds)
            self.assertNotIn("sparse_overlap", edge_kinds)
            self.assertNotIn("hybrid_sim", edge_kinds)
            self.assertEqual(policy_versions, ["similarity-edges-v0"])
            self.assertEqual(shared_terms_rows, 0)

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
