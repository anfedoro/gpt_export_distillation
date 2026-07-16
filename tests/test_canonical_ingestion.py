from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kb.ingest.canonical_markdown import CANONICAL_REPRESENTATION_VERSION, canonicalize_message
from kb.model.entities import Message
from kb.storage.native_pre_mvp import NativeBuildStore


def _message(text: str, *, role: str = "assistant") -> Message:
    return Message(
        id="message", conversation_id="conversation", ordinal=1, role=role,
        message_id="native-message", time_utc=None, raw_text=text, metadata_json={},
    )


class CanonicalIngestionTests(unittest.TestCase):
    def test_heading_and_list_become_one_clean_prose_block(self) -> None:
        blocks, _ = canonicalize_message(_message("## Reasons\n\n- high **cost**\n- poor *scalability*\n- complex ~~maintenance~~"))
        self.assertEqual([(block.block_type, block.normalized_text) for block in blocks], [
            ("prose", "Reasons: high cost; poor scalability; complex maintenance"),
        ])
        self.assertNotRegex(blocks[0].normalized_text, r"(^|\s)[#*_~-]+")

    def test_fenced_code_is_exact_without_fences_and_is_not_indexed(self) -> None:
        blocks, relationships = canonicalize_message(_message("Before.\n\n```python\r\nprint('x')\r\n```\n\nAfter."))
        code = next(block for block in blocks if block.block_type == "code")
        self.assertEqual(code.normalized_text, "print('x')")
        self.assertEqual(code.language, "python")
        self.assertEqual(code.dense_index_policy, "exclude")
        self.assertFalse(code.graph_eligibility)
        self.assertTrue(any(item.relation_type == "has_adjacent_artifact" for item in relationships))

    def test_code_languages_and_unknown_fence_are_preserved(self) -> None:
        cases = {
            "bash": "echo ok",
            "powershell": "Get-ChildItem",
            "c": "int main(void) { return 0; }",
            "cpp": "std::vector<int> values;",
            "unknown-lang": "opaque syntax",
        }
        for language, content in cases.items():
            with self.subTest(language=language):
                blocks, _ = canonicalize_message(_message(f"```{language}\n{content}\n```"))
                self.assertEqual(blocks[0].block_type, "code")
                self.assertEqual(blocks[0].language, language)
                self.assertEqual(blocks[0].normalized_text, content)

    def test_empty_fence_does_not_create_artifact(self) -> None:
        blocks, _ = canonicalize_message(_message("```python\n```"))
        self.assertEqual(blocks, [])

    def test_structured_json_is_deterministic_and_excluded(self) -> None:
        blocks, _ = canonicalize_message(_message('```json\n{"b": 2, "a": 1}\n```'))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "structured_data")
        self.assertEqual(blocks[0].normalized_text, '{"a":1,"b":2}')
        self.assertEqual(blocks[0].metadata_json["parse_status"], "parsed")
        self.assertEqual(blocks[0].semantic_status, "artifact")

    def test_malformed_json_and_other_structured_formats_are_preserved(self) -> None:
        cases = {
            "json": ('{"broken":', "malformed"),
            "xml": ("<root><value>1</value></root>", "preserved_unparsed"),
            "yaml": ("key: value", "preserved_unparsed"),
            "toml": ('key = "value"', "preserved_unparsed"),
        }
        for format_name, (content, status) in cases.items():
            with self.subTest(format=format_name):
                blocks, _ = canonicalize_message(_message(f"```{format_name}\n{content}\n```"))
                self.assertEqual(blocks[0].block_type, "structured_data")
                self.assertEqual(blocks[0].metadata_json["parse_status"], status)
                self.assertEqual(blocks[0].normalized_text, content)

    def test_audio_transport_extracts_text_without_embedding_wrapper(self) -> None:
        source = {
            "content_type": "audio_transcription",
            "text": "The actual user statement",
            "audio_asset_pointer": {"asset_pointer": "asset://one"},
        }
        blocks, _ = canonicalize_message(_message(f"```json\n{json.dumps(source)}\n```"))
        self.assertEqual([block.block_type for block in blocks], ["prose", "media_reference"])
        self.assertEqual(blocks[0].normalized_text, "The actual user statement")
        self.assertNotIn("audio_asset_pointer", blocks[0].normalized_text)

    def test_asset_pointer_is_media_not_prose(self) -> None:
        blocks, _ = canonicalize_message(_message('```json\n{"content_type":"image_asset_pointer","asset_pointer":"asset://one"}\n```'))
        self.assertEqual([block.block_type for block in blocks], ["media_reference"])

    def test_table_is_structured_and_pipe_syntax_is_absent(self) -> None:
        blocks, _ = canonicalize_message(_message("| A | B |\n|:--|--:|\n| один | two |"))
        self.assertEqual(blocks[0].block_type, "table")
        self.assertEqual(blocks[0].metadata_json["columns"], ["A", "B"])
        self.assertEqual(blocks[0].metadata_json["rows"], [["один", "two"]])
        self.assertNotIn("|", blocks[0].normalized_text)

    def test_table_empty_cells_and_unicode_are_preserved(self) -> None:
        blocks, _ = canonicalize_message(_message("| A | Б |\n|---|---|\n| | значение |"))
        self.assertEqual(blocks[0].metadata_json["rows"], [["", "значение"]])

    def test_mermaid_is_diagram_without_fences(self) -> None:
        blocks, _ = canonicalize_message(_message("```mermaid\ngraph TD\n A-->B\n```"))
        self.assertEqual(blocks[0].block_type, "diagram")
        self.assertEqual(blocks[0].normalized_text, "graph TD\n A-->B")
        self.assertEqual(blocks[0].metadata_json["format"], "mermaid")

    def test_malformed_mermaid_is_preserved_without_interpretation(self) -> None:
        blocks, _ = canonicalize_message(_message("```mermaid\nthis is not valid mermaid\n```"))
        self.assertEqual(blocks[0].block_type, "diagram")
        self.assertEqual(blocks[0].normalized_text, "this is not valid mermaid")

    def test_quote_remains_distinct_from_authored_prose(self) -> None:
        blocks, _ = canonicalize_message(_message("> Third-party claim\n\nMy response."))
        self.assertEqual([block.block_type for block in blocks], ["quote_or_external_content", "prose"])
        self.assertFalse(blocks[0].graph_eligibility)

    def test_inline_media_preserves_pointer_and_alt_text(self) -> None:
        blocks, _ = canonicalize_message(_message("See ![network diagram](asset.png \"Architecture\")."))
        media = next(block for block in blocks if block.block_type == "media_reference")
        self.assertEqual(media.metadata_json["pointer"], "asset.png")
        self.assertEqual(media.metadata_json["alt_text"], "network diagram")

    def test_punctuation_only_and_empty_content_are_not_blocks(self) -> None:
        blocks, _ = canonicalize_message(_message("---\n\n!!!\n\n** **"))
        self.assertEqual(blocks, [])

    def test_unicode_crlf_whitespace_and_inline_code_normalization(self) -> None:
        blocks, _ = canonicalize_message(_message("Cafe\u0301\r\n\r\nUse `AT+COPS?`   now.\u200b"))
        self.assertEqual([block.normalized_text for block in blocks], ["Café", "Use AT+COPS? now."])

    def test_short_meaningful_statement_is_retained(self) -> None:
        blocks, _ = canonicalize_message(_message("Do not merge this branch."))
        self.assertEqual(blocks[0].semantic_status, "graph_eligible")
        self.assertTrue(blocks[0].graph_eligibility)

    def test_standalone_language_label_is_context_only(self) -> None:
        blocks, _ = canonicalize_message(_message("python"))
        self.assertEqual(blocks[0].semantic_status, "context_only")
        self.assertIn("standalone_language_label", blocks[0].exclusion_reasons)

    def test_unstructured_user_content_is_preserved_conservatively(self) -> None:
        text = "Please inspect this.\n\nINVITE sip:user@example.com SIP/2.0\nVia: SIP/2.0/UDP host\n\n{\"mixed\": true}"
        blocks, _ = canonicalize_message(_message(text, role="user"))
        self.assertEqual([block.block_type for block in blocks], ["prose", "prose", "prose"])
        self.assertIn("INVITE sip:user@example.com SIP/2.0", blocks[1].normalized_text)
        self.assertIn('{"mixed": true}', blocks[2].normalized_text)

    def test_formatting_only_change_keeps_canonical_content_hash(self) -> None:
        plain, _ = canonicalize_message(_message("Important decision"))
        formatted, _ = canonicalize_message(_message("## Important **decision**"))
        self.assertEqual(plain[0].canonical_content_hash, formatted[0].canonical_content_hash)

    def test_relationships_are_deterministic(self) -> None:
        source = "Explanation.\n\n```bash\necho ok\n```\n\nDone."
        first = canonicalize_message(_message(source))
        second = canonicalize_message(_message(source))
        self.assertEqual(first, second)
        relation_types = {relationship.relation_type for relationship in first[1]}
        self.assertIn("same_document", relation_types)
        self.assertIn("same_section", relation_types)

    def test_contract_version_is_persisted_in_metadata(self) -> None:
        blocks, _ = canonicalize_message(_message("Meaningful."))
        self.assertEqual(blocks[0].metadata_json["canonical_representation_version"], CANONICAL_REPRESENTATION_VERSION)

    def test_generic_transition_is_context_only_with_reason(self) -> None:
        blocks, _ = canonicalize_message(_message("Лучше так:"))
        self.assertEqual(blocks[0].semantic_status, "context_only")
        self.assertEqual(blocks[0].dense_index_policy, "exclude")
        self.assertIn("generic_transition", blocks[0].exclusion_reasons)

    def test_generic_contextual_intro_is_not_a_semantic_node(self) -> None:
        blocks, _ = canonicalize_message(_message("Хороший вариант:"))
        self.assertEqual(blocks[0].semantic_status, "context_only")
        self.assertIn("generic_contextual_intro", blocks[0].exclusion_reasons)

    def test_writing_transport_wrapper_is_removed(self) -> None:
        blocks, _ = canonicalize_message(_message(
            ':::writing{variant="standard" id="42862"}\nActual draft text.\n:::'
        ))
        self.assertEqual(blocks[0].normalized_text, "Actual draft text.")
        self.assertNotIn("writing", blocks[0].normalized_text)

    def test_normalized_source_offsets_are_explicit(self) -> None:
        blocks, _ = canonicalize_message(_message("First.\r\n\r\nSecond.\u200b"))
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(block.metadata_json["source_offsets_basis"] == "normalized_message" for block in blocks))

    def test_exact_duplicates_are_preserved_but_only_one_is_indexable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.db"
            with NativeBuildStore(path) as store:
                store.conn.execute(
                    "INSERT INTO source_documents VALUES("
                    "'src','x','x','chat_md',NULL,'normal',NULL,NULL,'x','md','hash',NULL,NULL,'{}')"
                )
                for ordinal in (1, 2):
                    conversation_id = f"conversation-{ordinal}"
                    message_id = f"message-{ordinal}"
                    store.conn.execute(
                        "INSERT INTO conversations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (conversation_id, "src", f"native-{ordinal}", None, None, None, None, 1, 0, 1, 10, 0, None, None, "{}"),
                    )
                    store.conn.execute(
                        "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)",
                        (message_id, conversation_id, 1, "user", f"native-message-{ordinal}", None, "Same thought.", "{}"),
                    )
                    blocks, _ = canonicalize_message(_message("Same thought."))
                    block = blocks[0]
                    store.conn.execute(
                        "INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            f"block-{ordinal}", message_id, None, 1, block.block_type, block.language,
                            block.char_start, block.char_end, block.raw_text, block.normalized_text,
                            block.canonical_content_hash, block.parser_version, block.canonicalizer_version,
                            block.semantic_status, block.dense_index_policy, block.sparse_index_policy,
                            int(block.graph_eligibility), block.artifact_policy, block.context_policy,
                            json.dumps(list(block.exclusion_reasons)), json.dumps(block.metadata_json),
                        ),
                    )
                audit = store.finalize_semantic_eligibility()
                statuses = store.conn.execute(
                    "SELECT semantic_status FROM blocks ORDER BY id"
                ).fetchall()
                self.assertEqual(audit["duplicate_blocks_downgraded"], 1)
                self.assertEqual([row[0] for row in statuses], ["graph_eligible", "context_only"])


if __name__ == "__main__":
    unittest.main()
