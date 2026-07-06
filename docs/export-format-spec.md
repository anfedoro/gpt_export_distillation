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
- Project-related signals may also be absent from `conversations-*.json` and instead appear in library metadata. This repository therefore treats `conversations-*.json` as a chat payload source, not as the only project signal source.

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

- Type: `list[library_file]`
- Observed in the newer export inspected on 2026-06-29: 391 entries.
- This file appears to carry much richer library/project metadata than `conversations-*.json`.

Observed keys in the newer sample:

- `app_id`
- `client_sha256_digest`
- `content_backing_kind`
- `context_scopes`
- `context_scopes_v2`
- `created_at`
- `current_version_number`
- `deleted_at`
- `deletion_origin`
- `deletion_reason`
- `directory_id`
- `error_msg`
- `etag`
- `expires_at`
- `file_extension`
- `file_failure_time`
- `file_id`
- `file_name`
- `file_processed_time`
- `file_size_bytes`
- `file_upload_time`
- `gizmo_id`
- `hide_from_file_search`
- `id`
- `image_gen_generation_id`
- `initiating_conversation_id`
- `is_project`
- `is_visible`
- `knowledge_store_id`
- `library_artifact_type`
- `library_file_category`
- `metadata_updated_at`
- `mime_type`
- `normalized_name`
- `object_version`
- `origination_message_id`
- `origination_thread_id`
- `pinned_at`
- `record_creation_time`
- `request_read_version`
- `root_directory_id`
- `sha256_digest`
- `source_version_number`
- `state`
- `thumbnail_sources`
- `trash_original_directory_id`
- `trashed_at`
- `ttl`
- `updated_at`
- `uploading_account_user_id`
- `version_created_at`
- `version_provenance`

Practical observations from the newer export:

- `is_project` is present but may be unset for all entries in a given export.
- `pinned_at` is present but may be unset for all entries in a given export.
- `knowledge_store_id` can be present on only a subset of entries.
- `directory_id` may be shared by all library entries in a given export.
- `origination_thread_id` and `origination_message_id` can provide thread-level linkage back to conversation payloads.
- `library_file_category`, `mime_type`, and `normalized_name` are useful for inspection and grouping.
- `context_scopes` may be sparse or null, but it is still worth preserving as an inferred project/library signal.
- `id` is a structured object in the observed sample, so parsers should not assume it is a plain string.

Practical implication:

- Treat `library_files.json` as the primary source for library/project hints when it is available.
- Do not infer a project name as a fact from this file alone.
- Use it only as a hint source for summaries, grouping assistance, and manual review.

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
- Observed settings flags in the newer export include project/UI-related hints such as `project_pins_backfilled`, `first_party_project_pins_unmigrated`, and `study_mode_project_pins_unmigrated`.
- Practical role: auxiliary signal only, not a primary project source of truth.

## Parser guidance

- Do not hard-fail on unknown top-level files.
- Do not require every message node to have all fields.
- Preserve raw text order by sorting message rows on `message.create_time` and then `message.id`.
- Assume filenames and project grouping conventions may change across exports.
- Treat attachment and library metadata as optional.
- Parsers should stay permissive if `library_files.json` is missing, empty, or partially populated.
