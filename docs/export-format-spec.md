# ChatGPT Export Format Spec

This repository uses a mixed source of truth for the export format:

- Official OpenAI documentation for the export workflow.
- A locally inferred schema based on a real ChatGPT export bundle inspected during development.

Important limitation:

- The local sample is known to be incomplete because some export-side files were removed before this repository was created.
- This document should therefore be treated as a partial working spec, not a complete consumer-export schema.

## Official status

As of 2026-06-13, OpenAI documents how to request a ChatGPT export, but does not publish a detailed schema for the consumer export bundle.

Reference:

- OpenAI Help Center: `How do I export my ChatGPT history and data?`
- URL: `https://help.openai.com/en/articles/7260999-how-do-i-export-my-chatgpt-history-and-data`

Practical consequence:

- The tool should treat field presence as semi-stable.
- Parsers should be permissive and tolerate missing or extra keys.
- New top-level export files should be ignored safely until explicit support is added.

## Bundle-level files observed in the partial local sample

### `conversations-*.json`

- Type: `list[conversation]`
- Purpose: primary conversation payloads split across multiple files.

Conversation top-level keys observed:

- `conversation_id`
- `conversation_template_id`
- `create_time`
- `current_node`
- `default_model_slug`
- `id`
- `is_archived`
- `is_do_not_remember`
- `is_read_only`
- `is_starred`
- `is_study_mode`
- `mapping`
- `memory_scope`
- `pinned_time`
- `plugin_ids`
- `title`
- `update_time`
- `voice`

`mapping` node shape observed:

- node keys:
  - `id`
  - `message`
  - `parent`
- `message` keys:
  - `author`
  - `content`
  - `create_time`
  - `id`
  - `metadata`
- `author` keys:
  - `name`
  - `role`
- `content` keys:
  - `content_type`
  - `parts`
- sample `metadata` keys:
  - `content_references`
  - `model_slug`
  - `parent_id`

Notes:

- `parts` is usually a list of strings, but structured dict items can also appear.
- Some conversations contain `tool` and `system` roles in addition to `user` and `assistant`.
- `conversation_template_id` is useful for project grouping.
- `pinned_time` is a reliable signal for pinned chats in the observed export.
- In the full ZIP inspected on 2026-06-13, project names were not found as a separate structured field in the top-level JSON files.
- Practical implication: human-readable project names should be treated as external metadata and supplied via local config overrides when needed.

### `export_manifest.json`

- Type: `dict`
- Observed keys:
  - `export_files`
  - `logical_files`
  - `manifest_file`
  - `version`

### `conversation_asset_file_names.json`

- Type: `dict[str, str]`
- Purpose: maps internal asset identifiers / exported `.dat` names to original filenames.
- Useful for attachment summaries and extension-level categorization.

### `library_files.json`

- Type: `list`
- Observed local sample: empty list.
- Expectation: may contain file metadata for user library items in other exports.

### `message_feedback.json`

- Type: `list[dict]`
- Observed item keys:
  - `content`
  - `conversation_id`
  - `create_time`
  - `id`
  - `rating`
  - `update_time`
  - `user_id`
  - `workspace_id`

### `shared_conversations.json`

- Type: `list[dict]`
- Observed item keys:
  - `conversation_id`
  - `id`
  - `is_anonymous`
  - `title`

### `user.json`

- Type: `dict`
- Observed keys:
  - `birth_year`
  - `chatgpt_plus_user`
  - `email`
  - `id`
  - `phone_number`

### `user_settings.json`

- Type: `list[dict]`
- Observed item keys:
  - `announcements`
  - `beta_settings`
  - `habitat_object_version`
  - `settings`
  - `user_id`

## Parser guidance

- Do not hard-fail on unknown top-level files.
- Do not require every message node to have all fields.
- Preserve raw text order by sorting message rows on `message.create_time` and then `message.id`.
- Assume filenames and project grouping conventions may change across exports.
- Treat attachment and library metadata as optional.
