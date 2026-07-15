from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kb.mcp.archive import ArchiveConfig, ArchiveSession  # noqa: E402
from kb.mcp.result_assembly import (  # noqa: E402
    MAX_GROUPS_PER_CONVERSATION,
    MAX_POST_RETRIEVAL_CANDIDATES,
    build_compact_items,
    diversify_groups,
    filter_low_information_groups,
    group_overlapping_hits,
    rerank_hits,
)
from kb.mcp.tools import search_archive_tool  # noqa: E402


def _row(
    *,
    chunk: str,
    message: str,
    conversation: str,
    ordinal: int,
    score: float,
    text: str,
    block: str | None = None,
    start: int = 0,
    end: int = 100,
    project: str = "project-a",
    timestamp: str = "2026-01-01T00:00:00Z",
) -> dict[str, object]:
    return {
        "chunk_id": chunk,
        "message_id": message,
        "source_message_id": message,
        "conversation_id": conversation,
        "message_ordinal": ordinal,
        "block_id": block or f"block-{message}",
        "source_char_start": start,
        "source_char_end": end,
        "text": text,
        "role": "assistant",
        "project_id": project,
        "time_utc": timestamp,
        "conversation_title": f"Conversation {conversation}",
        "source_path": "synthetic/chat.md",
        "scores": {"dense": score, "sparse": score, "fused": score},
        "reason": "Matched by both semantic and lexical retrieval.",
    }


class _Retriever:
    def __init__(self, messages: dict[str, list[dict[str, object]]]) -> None:
        self.messages = messages

    def messages_for_windows(self, windows: dict[str, tuple[int, int]]) -> dict[str, list[dict[str, object]]]:
        return {
            conversation: [message for message in self.messages.get(conversation, []) if start <= int(message["ordinal"]) <= end]
            for conversation, (start, end) in windows.items()
        }


def _message(identifier: str, ordinal: int, text: str) -> dict[str, object]:
    return {"id": identifier, "ordinal": ordinal, "role": "assistant", "time_utc": "2026-01-01T00:00:00Z", "raw_text": text}


class CompactRetrievalResultTests(unittest.TestCase):
    def _session(self, rows: list[dict[str, object]], messages: dict[str, list[dict[str, object]]]) -> ArchiveSession:
        session = ArchiveSession.__new__(ArchiveSession)
        session.config = ArchiveConfig(db_path=Path("synthetic.db"), candidate_pool=20)
        session.calls = 1
        session.retriever = _Retriever(messages)
        session.search = lambda query, **_: list(rows)  # type: ignore[method-assign]
        return session

    def _groups(self, rows: list[dict[str, object]], query: str = "research status progress") -> list[dict[str, object]]:
        groups, _ = group_overlapping_hits(rerank_hits(rows, query))
        return groups

    def test_exact_duplicate_hits_are_collapsed(self) -> None:
        rows = [_row(chunk="x", message="m1", conversation="c1", ordinal=1, score=0.9, text="Research progress is measured."),
                _row(chunk="x", message="m1", conversation="c1", ordinal=1, score=0.8, text="Research progress is measured.")]
        groups, dropped = group_overlapping_hits(rerank_hits(rows, "research progress"))
        self.assertEqual(len(groups), 1)
        self.assertEqual(dropped, 1)

    def test_overlapping_chunks_are_merged(self) -> None:
        rows = [_row(chunk="x1", message="m1", conversation="c1", ordinal=1, score=0.9, text="Measured research status evidence.", block="b", start=0, end=90),
                _row(chunk="x2", message="m1", conversation="c1", ordinal=1, score=0.8, text="Measured research status evidence continued.", block="b", start=70, end=140)]
        groups, _ = group_overlapping_hits(rerank_hits(rows, "research status"))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["contributing_chunk_ids"], ["x1", "x2"])

    def test_adjacent_hits_with_same_context_are_merged(self) -> None:
        rows = [_row(chunk="x1", message="m1", conversation="c1", ordinal=4, score=0.8, text="Encoder decoder migration plan and evidence."),
                _row(chunk="x2", message="m1", conversation="c1", ordinal=4, score=0.79, text="Encoder decoder migration plan and evidence extended.")]
        self.assertEqual(len(self._groups(rows)), 1)

    def test_same_source_message_is_not_returned_twice(self) -> None:
        rows = [_row(chunk="x1", message="m1", conversation="c1", ordinal=1, score=0.9, text="First relevant implementation evidence."),
                _row(chunk="x2", message="m1", conversation="c1", ordinal=1, score=0.7, text="Second chunk from same source message.")]
        self.assertEqual(len(self._groups(rows)), 1)

    def test_low_information_anchor_is_dropped(self) -> None:
        groups = self._groups([_row(chunk="x", message="m", conversation="c", ordinal=1, score=0.5, text="BGP:")])
        retained, dropped = filter_low_information_groups(groups)
        self.assertEqual(retained, [])
        self.assertEqual(dropped, 1)

    def test_short_anchor_with_unique_context_is_kept(self) -> None:
        groups = self._groups([_row(chunk="x", message="m", conversation="c", ordinal=1, score=0.5, text="BGP:")])
        retained, dropped = filter_low_information_groups(groups, lambda _: "The research status is measured and the implementation completed its validation.")
        self.assertEqual(len(retained), 1)
        self.assertEqual(dropped, 0)

    def test_results_are_capped_per_conversation(self) -> None:
        aspects = ["architecture decision", "runtime throughput", "validation protocol", "release packaging"]
        rows = [_row(chunk=f"x{index}", message=f"m{index}", conversation="c1", ordinal=index, score=0.95 - index / 100, text=f"{aspect} provides separate measured evidence.") for index, aspect in enumerate(aspects)]
        selected = diversify_groups(self._groups(rows), limit=4)
        self.assertEqual(len(selected), MAX_GROUPS_PER_CONVERSATION)

    def test_diversification_prefers_distinct_conversation(self) -> None:
        rows = [_row(chunk="a1", message="a1", conversation="a", ordinal=1, score=0.9, text="Research status validation completed."),
                _row(chunk="a2", message="a2", conversation="a", ordinal=2, score=0.89, text="Research implementation measurements are available."),
                _row(chunk="b1", message="b1", conversation="b", ordinal=1, score=0.86, text="Separate project status evidence was recorded.")]
        selected = diversify_groups(self._groups(rows), limit=2)
        self.assertEqual({item["representative"]["conversation_id"] for item in selected}, {"a", "b"})

    def test_higher_score_still_wins_first_position(self) -> None:
        rows = [_row(chunk="high", message="high", conversation="a", ordinal=1, score=0.92, text="Generic implementation evidence."),
                _row(chunk="low", message="low", conversation="b", ordinal=1, score=0.70, text="Research status progress exact evidence.")]
        selected = diversify_groups(self._groups(rows), limit=2)
        self.assertEqual(selected[0]["representative"]["chunk_id"], "high")

    def test_near_duplicate_text_receives_penalty(self) -> None:
        rows = [_row(chunk="x1", message="m1", conversation="c", ordinal=1, score=0.8, text="Encoder decoder research status is measured."),
                _row(chunk="x2", message="m2", conversation="c", ordinal=2, score=0.79, text="Encoder decoder research status is measured."),
                _row(chunk="x3", message="m3", conversation="c", ordinal=3, score=0.78, text="Separate runtime validation completed successfully.")]
        groups = self._groups(rows)
        self.assertEqual(len(groups), 2)
        self.assertEqual(diversify_groups(groups, limit=3)[1]["representative"]["chunk_id"], "x3")

    def test_distinct_aspects_from_same_conversation_can_survive(self) -> None:
        rows = [_row(chunk="architecture", message="architecture", conversation="c", ordinal=1, score=0.9, text="Architecture decision selected a compact payload."),
                _row(chunk="measurement", message="measurement", conversation="c", ordinal=8, score=0.88, text="Runtime measurement recorded throughput and memory usage.")]
        self.assertEqual(len(diversify_groups(self._groups(rows), limit=2)), 2)

    def test_neighbor_context_respects_include_neighbors(self) -> None:
        groups = self._groups([_row(chunk="x", message="m2", conversation="c", ordinal=2, score=0.9, text="Useful evidence with implementation status.")])
        messages = {"c": [_message("m1", 1, "Before evidence."), _message("m2", 2, "Anchor."), _message("m3", 3, "After evidence.")]}
        items, _ = build_compact_items(groups, neighbors=0, budget_tokens=300, messages_by_conversation=messages)
        self.assertEqual(items[0]["supporting_context"], [])
        self.assertEqual(items[0]["context_before"], [])
        self.assertEqual(items[0]["context_after"], [])

    def test_neighbor_context_respects_char_budget(self) -> None:
        groups = self._groups([_row(chunk="x", message="m2", conversation="c", ordinal=2, score=0.9, text="Useful evidence " * 30)])
        messages = {"c": [_message("m1", 1, "Before context " * 100), _message("m2", 2, "Anchor."), _message("m3", 3, "After context " * 100)]}
        items, _ = build_compact_items(groups, neighbors=1, budget_tokens=100, messages_by_conversation=messages)
        total_chars = len(items[0]["text"]) + sum(len(item["text"]) for item in items[0]["supporting_context"])
        self.assertLessEqual(total_chars, 400)

    def test_neighbor_message_is_not_repeated_across_items(self) -> None:
        groups = self._groups([_row(chunk="x1", message="m2", conversation="c", ordinal=2, score=0.9, text="First distinct evidence."),
                               _row(chunk="x2", message="m3", conversation="c", ordinal=3, score=0.8, text="Second distinct evidence.")])
        messages = {"c": [_message("m1", 1, "Shared prior message."), _message("m2", 2, "First."), _message("m3", 3, "Second."), _message("m4", 4, "Shared later message.")]}
        items, _ = build_compact_items(groups, neighbors=1, budget_tokens=400, messages_by_conversation=messages)
        contexts = [entry["message_id"] for item in items for entry in item["supporting_context"]]
        self.assertEqual(len(contexts), len(set(contexts)))

    def test_summary_counts_evidence_groups_not_raw_hits(self) -> None:
        rows = [_row(chunk="x1", message="m", conversation="c", ordinal=1, score=0.9, text="Research status evidence is complete."),
                _row(chunk="x2", message="m", conversation="c", ordinal=1, score=0.8, text="Research status evidence is complete.")]
        payload = self._session(rows, {"c": [_message("m", 1, "Anchor")]}).search_archive({"query": "research status", "limit": 8, "include_neighbors": 0})
        self.assertIn("1 distinct evidence group", payload["summary"])
        self.assertEqual(payload["coverage"]["raw_hit_count"], 2)

    def test_coverage_reports_dropped_counts(self) -> None:
        rows = [_row(chunk="x1", message="m", conversation="c", ordinal=1, score=0.9, text="Research status evidence is complete."),
                _row(chunk="x2", message="m", conversation="c", ordinal=1, score=0.8, text="Research status evidence is complete."),
                _row(chunk="short", message="short", conversation="d", ordinal=1, score=0.3, text="Да.")]
        payload = self._session(rows, {"c": [_message("m", 1, "Anchor")], "d": [_message("short", 1, "Anchor")]}).search_archive({"query": "research status", "include_neighbors": 0})
        self.assertEqual(payload["coverage"]["dropped_duplicate_count"], 1)
        self.assertEqual(payload["coverage"]["dropped_low_information_count"], 1)

    def test_runtime_fields_remain_available(self) -> None:
        row = _row(chunk="x", message="m", conversation="c", ordinal=1, score=0.9, text="Research status evidence is complete.")
        payload = self._session([row], {"c": [_message("m", 1, "Anchor")]}).search_archive({"query": "research status", "include_neighbors": 0})
        self.assertIn("candidate_pool", payload["runtime"])
        self.assertIn("session_calls", payload["runtime"])

    def test_search_archive_input_schema_is_unchanged(self) -> None:
        schema = search_archive_tool()["inputSchema"]
        self.assertEqual(schema["required"], ["query"])
        self.assertEqual(schema["properties"]["include_neighbors"]["maximum"], 4)
        self.assertEqual(schema["properties"]["retrieval_mode"]["enum"], ["hybrid"])

    def test_construct_archive_context_behavior_is_unchanged(self) -> None:
        rows = [_row(chunk="x", message="m", conversation="c", ordinal=1, score=0.9, text="Research status evidence is complete.")]
        session = self._session(rows, {"c": [_message("m", 1, "Anchor")]})
        payload = session.construct_archive_context({"current_context": "research status", "include_preferences": False, "include_decisions": False})
        self.assertEqual(payload["mode"], "broad")
        self.assertIn("item_count", payload["coverage"])

    def test_legacy_response_consumers_do_not_crash(self) -> None:
        row = _row(chunk="x", message="m", conversation="c", ordinal=1, score=0.9, text="Research status evidence is complete.")
        payload = self._session([row], {"c": [_message("m", 1, "Anchor")]}).search_archive({"query": "research status", "include_neighbors": 0})
        item = payload["items"][0]
        self.assertTrue(item["text"])
        self.assertIsInstance(item["context_before"], list)
        self.assertIsInstance(item["context_after"], list)
        self.assertEqual(item["provenance"]["chunk_id"], "x")

    def test_rerank_prefers_status_evidence_for_status_query(self) -> None:
        rows = [_row(chunk="stale", message="stale", conversation="a", ordinal=1, score=0.80, text="There is access to the old memory."),
                _row(chunk="status", message="status", conversation="b", ordinal=1, score=0.79, text="Research status progress was measured and validation completed.")]
        self.assertEqual(rerank_hits(rows, "research status progress")[0]["chunk_id"], "status")

    def test_rerank_keeps_fused_score_as_primary_signal(self) -> None:
        rows = [_row(chunk="high", message="high", conversation="a", ordinal=1, score=0.91, text="Generic archive memory."),
                _row(chunk="low", message="low", conversation="b", ordinal=1, score=0.70, text="Research status progress exact evidence.")]
        self.assertEqual(rerank_hits(rows, "research status progress")[0]["chunk_id"], "high")

    def test_rerank_penalizes_short_low_information_anchor(self) -> None:
        rows = [_row(chunk="short", message="short", conversation="a", ordinal=1, score=0.80, text="BGP:"),
                _row(chunk="full", message="full", conversation="b", ordinal=1, score=0.80, text="Research status validation is completed.")]
        self.assertEqual(rerank_hits(rows, "research status")[0]["chunk_id"], "full")

    def test_rerank_does_not_require_embeddings_or_external_model(self) -> None:
        rows = [_row(chunk="x", message="m", conversation="c", ordinal=1, score=0.9, text="Research status evidence is complete.")]
        self.assertEqual(len(rerank_hits(rows, "research status")), 1)

    def test_duplicate_heavy_payload_is_compact(self) -> None:
        rows = [_row(chunk=f"x{index}", message="m", conversation="c", ordinal=1, score=0.9 - index / 100, text="Encoder decoder research evidence is completed and measured. " * 8) for index in range(5)]
        raw_chars = sum(len(str(row["text"])) for row in rows)
        payload = self._session(rows, {"c": [_message("m", 1, "Anchor")]}).search_archive({"query": "encoder decoder research", "include_neighbors": 0})
        final_chars = sum(len(str(item["text"])) for item in payload["items"])
        self.assertLessEqual(final_chars, raw_chars * 0.70)

    def test_post_processing_is_bounded_to_final_candidate_list(self) -> None:
        rows = [
            _row(chunk=f"x{index}", message=f"m{index}", conversation=f"c{index}", ordinal=1, score=1.0 - index / 1000,
                 text=f"Distinct synthetic evidence aspect {index}.")
            for index in range(MAX_POST_RETRIEVAL_CANDIDATES + 40)
        ]
        messages = {str(row["conversation_id"]): [_message(str(row["message_id"]), 1, str(row["text"]))] for row in rows}
        payload = self._session(rows, messages).search_archive({"query": "synthetic evidence", "limit": 30, "include_neighbors": 0})
        self.assertEqual(payload["coverage"]["raw_hit_count"], len(rows))
        self.assertEqual(payload["coverage"]["post_rerank_candidate_count"], MAX_POST_RETRIEVAL_CANDIDATES)

    def test_all_test_paths_are_inside_temporary_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ptha-compact-results-") as temporary:
            root = Path(temporary).resolve()
            old_home = os.environ.get("PTHA_HOME")
            try:
                os.environ["PTHA_HOME"] = str(root / "ptha-home")
                self.assertTrue(Path(os.environ["PTHA_HOME"]).resolve().is_relative_to(root))
            finally:
                if old_home is None:
                    os.environ.pop("PTHA_HOME", None)
                else:
                    os.environ["PTHA_HOME"] = old_home


if __name__ == "__main__":
    unittest.main()
