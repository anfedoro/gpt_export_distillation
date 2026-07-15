# Incremental import foundation

PTHA generations built with schema `kb.native_pre_mvp.v2` record durable metadata for a future incremental importer. This document describes that metadata contract; it does not introduce delta import or change retrieval behavior.

## Source identities and revisions

ChatGPT-native conversation and message IDs are preferred as stable source identities. They are not derived from an export path, file modification time, SQLite row ID, or message text.

When a native ID is unavailable, PTHA records a labelled deterministic fallback. A conversation fallback uses stable exported metadata. A message fallback uses its parent conversation identity, role, timestamp, and ordinal. The fallback method is persisted, so its lower stability is explicit rather than hidden.

Each identity can have immutable source revisions. A revision has a full BLAKE3-256 digest of a versioned canonical serialization. Strings are NFC-normalized, use LF line endings, remove trailing horizontal whitespace per line, and trim leading/trailing blank lines. Mapping keys are serialized deterministically; list order is retained. Conversation and message hashes use different domain-separation prefixes.

An entity missing from a later export is reported only as `absent_in_new_export`. It is not deleted, deactivated, or removed from retrieval. Future imports will classify the same identity and hash as `unchanged`, the same identity with another hash as `changed`, and a new identity as `new`.

## Transformations and lineage

The metadata persists versioned canonicalizer, block-builder, and chunker contracts. A source message records its specific parent conversation revision, then links to a stable block identity. A retrieval chunk links to that message revision and block identity through `chunk_incremental_metadata`.

Chunk identity describes its lineage, chunker contract, ordinal, and source range. Chunk content hash describes only the canonical chunk text. They are intentionally separate: future embedding reuse keys will use chunk content hash plus embedding contract, never a SQLite row ID.

## Embedding contract

Each generation stores a deterministic BLAKE3 fingerprint of semantic embedding behavior: provider type, model repository and pinned revision, FP16 precision, tokenizer/config limits, dense dimension, CLS pooling and normalization, sparse representation version/top-k behavior, and query/document prefixes. Runtime-only batch size and device selection are excluded.

## Generation manifests

Before a candidate database is atomically published, PTHA appends an immutable `generation_manifests` row. It contains a random immutable generation ID, schema and transformation versions, embedding contract fingerprint, canonical/derived counts, and a BLAKE3 database-content fingerprint computed from stable lineage metadata rather than SQLite file bytes. The manifest is written in the candidate database before publication, so it is published atomically with that database.

`ptha status` and `ptha doctor` report whether this metadata is available and, when it is, the generation ID. Legacy databases remain readable and report that incremental metadata is unavailable. PTHA does not mutate an active legacy database during service start or status inspection.

## Not implemented yet

This foundation does not compare exports, copy embeddings between generations, create changed revisions during import, rebuild only affected descendants, alter retrieval policy for multiple revisions, or add Layer 2 clustering/memory features.
