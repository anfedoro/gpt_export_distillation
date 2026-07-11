from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kb.mcp.archive import _dedupe_messages, _diverse_messages  # noqa: E402
from kb.mcp.server import _context_tool, _search_tool  # noqa: E402


def _row(message: str, conversation: str, score: float) -> dict:
    return {"message_id": message, "conversation_id": conversation, "scores": {"fused": score}}


class MCPArchiveContractTests(unittest.TestCase):
    def test_tool_schemas_and_routing_are_explicit(self) -> None:
        context, search = _context_tool(), _search_tool()
        self.assertEqual(context["name"], "construct_archive_context")
        self.assertEqual(search["name"], "search_archive")
        self.assertEqual(context["inputSchema"]["required"], ["current_context"])
        self.assertEqual(search["inputSchema"]["required"], ["query"])
        self.assertEqual(search["inputSchema"]["properties"]["retrieval_mode"]["enum"], ["hybrid"])
        self.assertIn("Never claim", context["description"])

    def test_multiple_queries_return_one_message_once(self) -> None:
        rows = [_row("m1", "c1", 0.4), _row("m1", "c1", 0.9), _row("m2", "c1", 0.7)]
        selected = _dedupe_messages(rows, limit=10, max_per_conversation=3)
        self.assertEqual([item["message_id"] for item in selected], ["m1", "m2"])
        self.assertEqual(selected[0]["scores"]["fused"], 0.9)

    def test_broad_diversity_caps_conversations_and_messages(self) -> None:
        rows = [_row("a1", "a", 0.9), _row("a2", "a", 0.8), _row("b1", "b", 0.7), _row("c1", "c", 0.6)]
        selected = _diverse_messages(rows, limit=4, max_conversations=2, max_per_conversation=1)
        self.assertEqual({item["conversation_id"] for item in selected}, {"a", "b"})
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
