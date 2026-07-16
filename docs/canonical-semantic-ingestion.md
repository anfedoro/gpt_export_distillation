# Canonical semantic ingestion

PTHA canonical generation version 1 parses source content into ordered,
policy-bearing blocks before chunking or embedding. Raw Markdown is a rendering
format, not a stable memory representation: fences, table borders, list
delimiters, transport wrappers, asset pointers, and decorative fragments must
not become independent semantic evidence.

```text
raw source
→ source parsing
→ ordered canonical blocks
→ type-aware normalization
→ semantic eligibility
→ prose-only chunking
→ dense and sparse representations
→ retrieval indexes
```

Layer 2 graph persistence is not part of this generation.

## Parser choice

PTHA uses `markdown-it-py` 4.x with CommonMark plus table and strikethrough
rules. It was selected for its predictable ordered tokens, mature fenced-block
handling, language extraction, nested-list and blockquote support, table
tokens, inline formatting, line maps, maintenance quality, and small dependency
footprint. `mistune` and `marko` were also viable, but `markdown-it-py` exposes
the clearest rule-configurable token stream for a deliberately small persisted
type system.

Line maps are converted to character offsets. When normalization changes the
parser input, a block records `source_offsets_basis=normalized_message` and a
stable raw-content reference rather than claiming raw-message offsets.

## Persisted block model

The compact type system is `prose`, `code`, `structured_data`, `table`,
`diagram`, `media_reference`, `attachment_reference`,
`quote_or_external_content`, and `unknown`.

Headings, paragraphs, and list items map to prose. A heading followed by list
items is assembled into coherent prose. Formatting markers, fences, horizontal
rules, and table borders are not canonical content.

Each block stores source/message lineage, order, offsets or their explicit
basis, raw content/reference, canonical content and hash, parser and
canonicalizer versions, language/format, downstream policies, reason codes,
and metadata.

## Prose and artifacts

Prose is normalized without summarizing, translating, stemming, or rewriting.
NFC, LF line endings, whitespace cleanup, zero-width/control cleanup, and known
serialization-wrapper removal are deterministic.

Fenced code is stored exactly after line-ending normalization, without fences.
JSON is deterministically serialized when parsing succeeds. XML, YAML, and TOML
are preserved with `preserved_unparsed` status in version 1. Tables are stored
as columns and rows. Mermaid is stored as a diagram. Quotes remain separate
from authored prose.

Audio-transcription envelopes extract meaningful text into prose and preserve
the asset as a media reference. Asset-pointer envelopes do not become prose.
Attachments remain a separate artifact layer; inventory presence does not
imply that attachment content has been read.

## Eligibility policy

| Block type | Dense | Sparse | Graph | Artifact | Context |
|---|---|---|---|---|---|
| information-bearing prose | include | include | eligible | no | include |
| context-only prose | exclude | exclude | no | no | include |
| code | exclude | exclude | no | store | structurally reachable |
| structured data | exclude | exclude | no | store | structurally reachable |
| table | exclude | exclude | no | store | structurally reachable |
| diagram | exclude | exclude | no | store | structurally reachable |
| media/attachment reference | exclude by default | exclude by default | no | store | structurally reachable |
| quote/external content | scoped | scoped | no | store | structurally reachable |
| unknown | exclude | exclude | no | store | structurally reachable |

Empty and punctuation-only content, language labels, short standalone headings,
known generic transitions, contextual introductions, and exact duplicate prose
are context-only or excluded with explicit reason codes. Exact duplicates are
preserved and linked to the retained occurrence with `exact_duplicate_of`.
Short commands, decisions, negations, identifiers, paths, versions, and
protocol values remain eligible when informative.

## Structural relationships

SQLite stores `previous_block`, `next_block`, `adjacent_block`, `same_message`,
`same_document`, `same_section`, `has_adjacent_artifact`, and
`exact_duplicate_of`. Foreign keys materialize `belongs_to_source` and
`belongs_to_message`. These are provenance/navigation relations, not semantic
claims.

## Unstructured user messages

Unfenced user code, logs, SIP messages, shell commands, and mixed technical
content remain conservative prose in version 1. Paragraph boundaries are
preserved; content is not discarded or aggressively classified. A future span
classifier can refine these blocks without redesigning source identity,
revision identity, or canonical storage.

## Identity and generation contract

Raw identity, source revision, canonical block identity, canonical content
hash, semantic chunk identity, embedding identity, and generation contract
remain separate. Formatting-only changes that produce identical canonical
blocks retain canonical content hashes. Message revision hashing uses canonical
blocks rather than Markdown serialization. Canonical block identity uses the
message revision, ordinal, type/language, and canonical content hash; raw
offsets remain provenance and do not destabilize formatting-only revisions.

Native schema `kb.native_pre_mvp.v3` and manifest schema 2 persist the parser,
`canonical_representation_version=1`, canonicalizer, source-transform,
block-builder, chunker, dense, and sparse contracts. The canonical default
content budget remains 256 tokens. Parser/canonicalization semantic changes require a
full staged rebuild; old and new embeddings must never be mixed.

## Rebuild and compatibility

A candidate is built and validated before atomic publication by the existing
import lifecycle. The active generation is never reinterpreted in place.
Legacy databases remain readable for status and diagnostics and are labelled
`legacy_precanonical`, not silently treated as equivalent.
