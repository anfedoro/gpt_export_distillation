# Product radar

This document records intentionally deferred product directions. Items here
are not implemented features and do not change the current PTHA contract.

## Planned: attachment artifacts

PTHA should treat chat attachments as accompanying artifacts, not as ordinary
conversation memory. Memory remains the default retrieval corpus: discussions,
decisions, preferences, and conclusions from conversations. Attachments are
discoverable only as references and require an explicit read before a model or
client uses their contents.

The intended progressive-disclosure flow is:

```text
memory search
  -> attachment reference
  -> explicit attachment read
  -> deeper answer
```

This prevents a filename or sparse keyword match from being presented as proof
of a document's contents. `construct_archive_context` must not automatically
include attachment text.

### First implementation boundary

The first attachment phase is deliberately narrow:

- Register every discovered attachment in an inventory with provenance.
- Preserve an original only when the export actually provides its bytes.
- Index Markdown attachments (`.md`) as `source_kind=attachment` using sparse
  document- or heading-section representations only.
- Keep PDF, DOCX, PPTX, XLSX, XML, JSON, CSV, TXT, images, and other formats
  inventory-only until a separate format policy is approved.
- Do not create dense embeddings for attachments in this phase.

Each attachment will have a stable identity and provenance including its parent
conversation/message when available, filename, MIME type, content hash when
bytes are present, managed-object reference, indexing policy, and indexing
status. Missing source bytes must be represented as unavailable content, not
as a fabricated stored artifact.

### Retrieval and access contract

Ordinary archive search may return a compact attachment reference: identifier,
filename, MIME type, matched sparse terms, a bounded preview, provenance, and
whether content can be read or downloaded. That response indicates potential
relevance only.

A future explicit attachment access operation will support bounded metadata or
text reads, and eventually a safe download/resource reference. It must use
opaque IDs, avoid disclosing filesystem paths, validate filenames and MIME
types, and prevent path traversal. Large originals must not be returned as
base64 in an MCP result.

### Explicitly deferred

This direction does not yet include origin classification (model-generated
versus third-party), PDF/DOCX/PPTX/XLSX parsing, OCR, vision analysis,
attachment dense retrieval, automatic attachment context injection, a permanent
download server, or attachment-management UI.
