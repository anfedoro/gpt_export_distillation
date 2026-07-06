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

from kb.cli import ingest_attachments, ingest_chats  # noqa: E402
from kb.ingest.chat_md_parser import parse_chat_file  # noqa: E402
from kb.ingest.tree_walker import scan_tree  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
