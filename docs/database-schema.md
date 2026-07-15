# Storage overview

PTHA stores one local SQLite database. The schema separates canonical archive
content from derived retrieval data so that an interrupted import or reindex
cannot silently replace source content.

## Canonical entities

- source documents identify the imported archive inputs;
- conversations contain conversation-level metadata;
- messages preserve ordered roles, timestamps, and source text;
- structural blocks preserve deterministic source ranges and block types;
- retrieval chunks contain the derived chunk text and versioned chunk policy.

Canonical IDs, ordering, content hashes, and source coordinates are stable
inputs to validation and reindexing.

## Derived retrieval entities

- dense vector metadata and native dense vectors;
- sparse term weights and lookup structures;
- indexes and runtime metadata needed by the active provider.

Derived data can be rebuilt from canonical rows. Reindex builds a clone,
validates canonical equivalence and vector counts, then atomically publishes the
replacement database. The active database is not rebuilt in place.

## Compatibility rules

Schema version, model identity, embedding dtype, chunk policy, and vector
dimensions are part of database compatibility. `ptha doctor` checks these
invariants. A database created by an incompatible or legacy storage path must be
rebuilt or re-imported; PTHA does not silently reinterpret old vectors.

The full implementation schema is intentionally not a public migration
contract. Changes to canonical identity, derived table ownership, or retrieval
semantics require tests and an explicit compatibility decision.
