# PTHA Reindex

`ptha reindex` rebuilds every derived retrieval structure without the original
ZIP or another distillation pass. The service must be stopped.

Canonical data consists of source documents and provenance, conversations,
projects, timestamps and ordering, complete messages and roles, and structural
blocks with source coordinates. Derived data consists of retrieval chunks,
sqlite-vec dense rows, compact sparse data, and runtime build contracts/audits.

The algorithm is clone, reset derived tables in the clone, deterministically
reconstruct chunks, build dense and sparse representations, validate, compare
canonical SHA-256/counts, smoke-test native retrieval, and atomically replace
the active DB. The clone is in the active DB directory so publication stays on
one filesystem.

```bash
ptha service stop
ptha reindex
ptha service start
```

An interrupted run preserves `ptha.db.reindexing` and
`maintenance-state.json`; the active DB remains unchanged. After inspection,
`ptha reindex --force` safely removes only that exact owned non-symlink clone
and restarts the complete rebuild. Partial reindex is not supported because the
dense and sparse representations form one consistency contract.
