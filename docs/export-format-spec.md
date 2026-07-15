# Supported ChatGPT export inputs

PTHA accepts either the original ChatGPT export ZIP or an extracted directory.
The importer is deliberately permissive because consumer export fields can
change over time.

The primary conversation payload is normally provided in one or more
`conversations-*.json` files. PTHA reads conversation metadata, ordered mapping
nodes, message authors and roles, timestamps, text parts, and supported project
signals. Missing or additional fields are tolerated where they do not affect
canonical identity.

Some exports also contain manifest, asset, library, settings, or feedback files.
PTHA uses only the files and fields required by the current import contract;
unknown files are ignored safely. Attachment recovery depends on the files
present in the export and does not imply attachment indexing.

The export format is not a stable public schema. Keep parsers permissive and
add synthetic fixtures when supporting a new observed shape. Do not add a real
user export, identifiers, message text, or generated database to the repository.

For the supported user workflow, see [docs/first-run.md](first-run.md).
