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

from gpt_export_distillation.config import DEFAULT_CONFIG  # noqa: E402
from gpt_export_distillation.loader import load_bundle  # noqa: E402
from gpt_export_distillation.pipeline import build_documents, write_output  # noqa: E402


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


class LibraryFilesTests(unittest.TestCase):
    def test_load_bundle_parses_library_file_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "conversations-000.json",
                [
                    {
                        "id": "thread-1",
                        "conversation_id": "thread-1",
                        "conversation_template_id": "tmpl-1",
                        "title": "Project Alpha",
                        "create_time": 1,
                        "update_time": 1,
                        "mapping": {},
                    }
                ],
            )
            write_json(
                root / "library_files.json",
                [
                    {
                        "file_id": "file-1",
                        "file_name": "alpha.md",
                        "normalized_name": "alpha.md",
                        "mime_type": "text/markdown",
                        "library_file_category": "text",
                        "directory_id": "libdir-1",
                        "knowledge_store_id": {"id": "iks-1", "kind": "id"},
                        "origination_thread_id": "thread-1",
                        "origination_message_id": "msg-1",
                        "pinned_at": "2026-06-29T11:21:40+00:00",
                        "is_project": True,
                        "context_scopes": ["project"],
                    }
                ],
            )

            bundle = load_bundle(root)

            self.assertEqual(len(bundle.library_files), 1)
            record = bundle.library_files[0]
            self.assertEqual(record.file_id, "file-1")
            self.assertEqual(record.knowledge_store_id, "iks-1")
            self.assertEqual(record.directory_id, "libdir-1")
            self.assertEqual(record.origination_thread_id, "thread-1")
            self.assertEqual(record.pinned_at, "2026-06-29T11:21:40+00:00")
            self.assertEqual(record.context_scopes, ("project",))
            self.assertTrue(record.is_project)

    def test_write_output_writes_library_summary_and_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "conversations-000.json",
                [
                    {
                        "id": "thread-1",
                        "conversation_id": "thread-1",
                        "conversation_template_id": "tmpl-1",
                        "title": "Project Alpha",
                        "create_time": 1,
                        "update_time": 1,
                        "mapping": {},
                    }
                ],
            )
            write_json(
                root / "library_files.json",
                [
                    {
                        "file_id": "file-1",
                        "file_name": "alpha.md",
                        "normalized_name": "alpha.md",
                        "mime_type": "text/markdown",
                        "library_file_category": "text",
                        "directory_id": "libdir-1",
                        "knowledge_store_id": {"id": "iks-1", "kind": "id"},
                        "origination_thread_id": "thread-1",
                        "origination_message_id": "msg-1",
                        "pinned_at": "2026-06-29T11:21:40+00:00",
                        "is_project": True,
                        "context_scopes": ["project"],
                    }
                ],
            )

            bundle = load_bundle(root)
            documents = build_documents(bundle, DEFAULT_CONFIG)
            output_root = write_output(
                bundle,
                documents,
                DEFAULT_CONFIG,
                explicit_output_dir=str(root / "out"),
            )

            files_md = (output_root / "FILES.md").read_text(encoding="utf-8")
            library_md = (output_root / "LIBRARY_FILES.md").read_text(encoding="utf-8")
            self.assertEqual(files_md, library_md)
            self.assertIn("project_like_entries", library_md)
            self.assertIn("unique_knowledge_store_id", library_md)
            self.assertIn("unique_directory_id", library_md)
            self.assertIn("By library_file_category", library_md)
            self.assertIn("By mime_type", library_md)
            self.assertIn("By directory_id", library_md)
            self.assertIn("By knowledge_store_id", library_md)
            self.assertIn("Project Hints", library_md)
            self.assertIn("tmpl-1", library_md)
            self.assertIn("alpha.md", library_md)


if __name__ == "__main__":
    unittest.main()
