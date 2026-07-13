# PTHA v1.0 Implementation Assessment

Date: 2026-07-13

## Executive decision

PTHA will have one authoritative product import path:

```text
ChatGPT export ZIP or directory
  -> existing distillation library in a private temporary workspace
  -> clean native SQLite builder
  -> validation
  -> atomic publication as ptha.db
```

The existing direct native builder is not a raw ChatGPT-export importer. Its
`export_path` is scanned for distilled `chat_md` files, so presenting it as a
ZIP-to-database path would be incorrect. The product layer must call shared
Python APIs and must not orchestrate the legacy console commands.

## 1. Current raw-export to native-DB paths

The Markdown path starts in `gpt_export_distillation.loader`, loads a ZIP,
directory, or conversations JSON, builds `ChatDocument` objects in
`gpt_export_distillation.pipeline`, and writes a structured Markdown archive.
`kb.ingest.tree_walker` scans that archive; `kb.ingest.chat_md_parser` restores
conversations, messages, and typed blocks. The legacy `kb-index import` then
creates chunks, embeddings, optional semantic nodes, and optional edges.

The clean native path is `kb.storage.native_pre_mvp.build_native_pre_mvp_db`.

The production embedding path now uses a single `BAAI/bge-m3` backbone with
its official dense and lexical heads. Dense and sparse publication are two
sequential batched passes over the same chunks and the same resolved device.
See `docs/ptha-embedding-pipeline.md` for the benchmark contract and commands.
It scans a distilled directory, parses `chat_md`, creates canonical rows and
retrieval chunks, writes sqlite-vec dense vectors plus compact sparse vectors,
audits the result, and renames `<output>.building` to the requested output.

## 2. Markdown pipeline versus clean native build

The Markdown/legacy pipeline supports attachments, compatibility tables,
semantic nodes, and semantic edges. It can be run as several developer-facing
stages and therefore permits partially built databases.

The clean native builder intentionally excludes legacy tables, semantic graph
tables, and attachment entities. It owns a complete, fixed native retrieval
layout and publishes only after its audit succeeds. Native retrieval and the
current two-tool archive contract use this clean layout.

## 3. Feature parity

The two paths do not have full feature parity. Both preserve the core chat
hierarchy and retrieval text. Only the Markdown/legacy database has a first-
class attachment catalog and optional graph structures. The clean builder
records non-chat source files in `source_documents`, but does not extract or
link their content as attachment entities. PTHA v1 must document attachments as
unsupported until the clean schema and importer preserve them without adding a
legacy fallback.

## 4. Canonical data in the clean native DB

| Data | State | Storage |
|---|---|---|
| Conversations | Preserved | `conversations` |
| Messages and order | Preserved | `messages` |
| Roles | Preserved | `messages.role` |
| Timestamps | Preserved when present | conversation/message UTC columns |
| Projects | Preserved from distilled layout | project columns |
| Attachments | Source files catalogued only | `source_documents`; no attachment entity/content |
| Source IDs | Preserved | source, conversation, message IDs |
| Raw message text | Preserved | `messages.raw_text` |
| Structural blocks | Preserved by coordinates/type | `blocks` |
| Retrieval chunks | Preserved with text and coordinates | `retrieval_chunks` |

The clean build audit currently stores the absolute distilled workspace path.
PTHA must sanitize this before product release because that path is operational
metadata, not required archive provenance.

## 5. Reindex feasibility

The DB contains sufficient canonical material to recreate chunks and both
representations: full message text, typed block coordinates, chunk text,
policy identifiers, and model contracts are present. There is no existing safe
clean-native reindex API, however. The current builder couples canonical import
and embedding writes, and blocks omit their own text column. A v1 reindex should
clone the active DB, rebuild derived tables in the clone from canonical message
text plus block coordinates, validate it, and atomically replace the DB while
the service is stopped. Reindex must not be advertised as implemented before
that API and its equivalence tests exist.

## 6. Reusable modules

- `gpt_export_distillation.loader` and `.pipeline`: raw export loading and
  deterministic distillation.
- `kb.ingest.chat_md_parser` and `.tree_walker`: structured Markdown parsing.
- `kb.index.chunk_builder`: unchanged chunk policy and chunking semantics.
- `kb.embeddings.sentence_transformer_provider`: dense and sparse providers.
- `kb.storage.native_pre_mvp`: clean schema, builder, audit, and retriever.
- `kb.mcp.archive.ArchiveSession`: one process-lifetime retrieval session and
  the two public archive operations.
- MCP tool schemas in `kb.mcp.server`, after separating protocol handling from
  model-owning runtime construction.

## 7. Command classification

Product-facing commands are only `ptha init`, `import`, `reindex`, `status`,
`doctor`, `service ...`, and `mcp ...`.

`gpt-export-distillation`, most `kb-index` subcommands, `kb-search`, and
`kb-mcp test-call` are developer/debug surfaces. Benchmark, canary, fusion-eval,
storage-audit, and scripts under `tools/` are benchmark/research surfaces.
Dense migration commands are migration-only. Legacy entry points may remain for
compatibility but must not appear in the normal PTHA workflow.

## 8. Internal service API

Protocol version 1 will expose `ping`, `status`, `search_archive`,
`construct_archive_context`, and `shutdown`. Each JSON object carries a UUID
request ID, operation, plain JSON arguments, and timeout. Responses echo the ID
and contain either `result` or a stable sanitized error object. Query and result
content is never logged.

## 9. IPC protocol

Unix/macOS uses a user-only Unix domain socket at the platform runtime path.
Frames are a four-byte unsigned big-endian length followed by UTF-8 JSON. Both
directions enforce configurable hard limits before allocation. Exact reads
handle fragmentation; malformed frames affect only their connection. The
transport sits behind a client/server abstraction so a Windows named-pipe
implementation can be added without changing operations.

## 10. Process lifecycle

`service run` validates the DB, binds a mode-0600 socket, loads providers,
constructs one `ArchiveSession`, and becomes ready only after an IPC ping can
succeed. Background start records PID plus process identity metadata and waits
for readiness. Stop first requests authenticated local graceful shutdown and
uses signals only after validating process identity. PID or socket existence
alone never means ready. SIGINT/SIGTERM closes the session and removes the
socket.

## 11. File-by-file implementation plan

- `pyproject.toml`: platformdirs dependency, `ptha` entry point, package build.
- `src/ptha/config.py`, `paths.py`: versioned config and precedence.
- `src/ptha/cli.py`, `errors.py`, `output.py`: product CLI and stable errors.
- `src/ptha/importer.py`: temporary distillation, native build, validation,
  metadata, atomic publication.
- `src/ptha/database.py`, `doctor.py`, `reindex.py`: read-only inspection and
  safe replacement workflows.
- `src/ptha/ipc.py`, `service.py`, `lifecycle.py`: framed protocol and daemon.
- `src/ptha/mcp.py`: lightweight stdio-to-IPC adapter.
- `tests/test_ptha_*.py`: unit, failure, and synthetic integration coverage.
- README and `docs/ptha-*.md`: product documentation and explicit maturity.

## 12. Risks and incompatibilities

1. The requested attachment count/content is not available in the clean schema.
2. Reindex is feasible but not currently factored into a safe API.
3. Existing provider helpers default devices differently from the proposed
   `auto` product configuration and need one normalization point.
4. `ArchiveSession` serializes calls; multiple adapters share models but not
   concurrent retrieval execution. This is acceptable for local v1 if exposed
   as an explicit capacity limit.
5. Existing MCP errors include exception text and therefore do not meet the
   sanitized-error requirement.
6. Existing native build progress writes to stdout and must be redirected
   through a product progress callback before MCP/JSON purity can be claimed.
7. Current native build failure retains `.building`; PTHA needs a documented
   recovery marker or deterministic cleanup policy.
8. The current repository requires Python 3.13, reducing install portability.

No retrieval weights, candidate union behavior, message aggregation, or tie
ordering should change as part of PTHA orchestration.

## Runtime vertical-slice decisions

The following decisions are fixed for PTHA v1.0:

1. `ptha import --replace` and the future `ptha reindex` must refuse to run
   while a healthy retrieval service answers on the configured socket. This is
   the v1 maintenance boundary; live database switching is out of scope.
2. Local IPC protocol v1 uses a Unix domain socket and has no TCP listener.
3. The runtime directory is created for the current user with mode `0700`; the
   socket is mode `0600`. The service verifies peer credentials where the host
   exposes a practical Unix credential API and rejects a peer whose effective
   UID differs from the service UID.
4. There is no separate application token for local IPC. Filesystem ownership,
   directory/socket permissions, and peer credentials form the local security
   boundary.
5. Attachment content is not part of the clean PTHA v1 retrieval database.
6. Physical attachments copied into temporary distilled Markdown are deleted
   with that workspace after a normal import. They are retained only when the
   user explicitly requests `--keep-distilled`; the clean DB otherwise keeps
   only source-document catalogue metadata and cannot retrieve attachment
   content.
7. Retrieval calls remain serialized by `ArchiveSession.RLock`. This is an
   explicit PTHA v1 capacity limit, not a promise of parallel model inference.
8. `timeout_ms` is a cooperative deadline: it is forwarded into archive
   operations and checked at supported boundaries, but it does not forcibly
   cancel an embedding or SQLite call already executing.

## Background lifecycle decisions

Background mode always spawns the installed `ptha --config ... service run`
entry point (with `python -m ptha.cli` only as an installation-development
fallback). It does not have a second runtime implementation.

Process ownership is established by a versioned state record containing PID,
process create time, resolved executable, and command line. Lifecycle code uses
`psutil` because PID-only checks and platform-specific parsing of `ps` output do
not protect against PID reuse. A process is signalled only when PID, create time,
and executable still match. Command-line data remains diagnostic rather than an
exact identity key because launchers may legitimately normalize it.

Lifecycle mutations are serialized with an advisory OS `flock` on
`service.lock`; file existence alone is not a lock. `start` writes provisional
metadata, waits for live IPC `status`, `state=ready`, and `models_loaded=true`,
then marks the state ready. `stop` prefers internal IPC shutdown, falls back to
SIGTERM only after identity validation, and permits SIGKILL only with `--force`
after both earlier paths time out. `restart` completes this validated stop
before spawning its replacement.

The service uses a bounded rotating log. Requests are logged only by ID,
operation, duration, counts, and exception class; query arguments and results
are excluded. An unmanaged foreground service can still be reported ready from
live IPC, but signal fallback is unavailable without matching process metadata.

### IPC v1 wire contract

Each connection carries one request and one response. A frame is a four-byte
unsigned big-endian payload length followed by exactly that many UTF-8 bytes.
The payload must be a JSON object. Default limits are 1 MiB for requests and
16 MiB for responses and are configurable as `service.max_request_bytes` and
`service.max_response_bytes`. Length is rejected before the payload is read.

Requests contain `protocol_version=1`, a non-empty `request_id`, one of `ping`,
`status`, `search_archive`, `construct_archive_context`, or `shutdown`, a JSON
object `arguments`, and a positive integer `timeout_ms`. Responses echo the
protocol and request ID and contain either `ok=true` plus `result`, or
`ok=false` plus a stable `{code,message}` error. The implemented wire error
codes are `invalid_request`, `unsupported_protocol`, `unsupported_operation`,
`invalid_arguments`, `request_too_large`, `response_too_large`,
`retrieval_timeout`, `service_shutting_down`, and `internal_error`. Startup
diagnostics additionally use `database_not_ready` and `model_load_failed`.

## Operational completeness decisions

Canonical archive data is `source_documents`, `conversations`, `messages`, and
`blocks`. Retrieval chunks and every dense/sparse representation are derived.
Chunks are reproducible because each block stores message ownership and source
character coordinates while `messages.raw_text` stores the complete text.

Reindex never mutates the active DB. It uses SQLite backup into a same-directory
`.reindexing` clone, resets only derived tables, reconstructs chunks through the
existing chunk policy API, embeds them through the existing providers, audits
the clone, compares canonical IDs/content with a stable SHA-256, runs a native
retrieval smoke, and then publishes with `os.replace`.

Import and reindex share an OS advisory `maintenance.lock`. Service start and
restart acquire the same lock while holding the lifecycle lock; import and
reindex never acquire the lifecycle lock. This fixed ordering prevents a
service from starting during mutation without introducing a lock cycle.

Background instances carry a protected random `instance_id` in both the child
environment and mode-0600 lifecycle state. Live correlation requires PID,
create time, executable, and matching IPC/environment instance identity. The ID
is internal and is not exposed through MCP.
