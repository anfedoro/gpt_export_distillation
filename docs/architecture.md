# PTHA architecture

PTHA is a local pipeline with explicit boundaries between import, storage,
retrieval service, and MCP transport.

```text
ChatGPT export
    -> importer and canonical content
    -> deterministic retrieval chunks
    -> dense + sparse native SQLite indexes
    -> foreground/background local retrieval service
    -> versioned Unix IPC
    -> stdio MCP adapter
    -> MCP client
```

## Import and storage

The importer reads a ZIP or extracted export, creates canonical conversations,
messages, roles, timestamps, projects, and structural blocks, then derives
retrieval chunks without changing the source content. Derived indexes are built
in a staging database and published atomically after validation. An interrupted
import leaves the last active database unchanged.

SQLite is the local source of truth. Canonical rows are preserved separately
from dense vectors, sparse terms, and other derived retrieval structures.

## Embedding provider

On supported Apple Silicon, one pinned MLX FP16 BGE-M3 artifact produces both
normalized dense vectors and sparse lexical weights in one backbone forward.
Length-aware batches default to four chunks. The provider owns tokenization,
padding, model loading, sparse aggregation, and conversion to the storage
format; indexing and retrieval remain backend-neutral.

## Service and IPC

The service owns one database session and one model lifetime. It listens only on
a user-owned Unix-domain socket with restrictive permissions. IPC uses a
versioned length-prefixed JSON protocol and sanitized stable error codes. The
background lifecycle starts the same foreground service entry point, waits for
real readiness, records process identity, and removes state on clean shutdown.

## MCP boundary

The stdio adapter reads and writes MCP JSON-RPC, but does not open SQLite, load
models, or implement retrieval. It allowlists `search_archive` and
`construct_archive_context` and forwards calls to the service over IPC.
