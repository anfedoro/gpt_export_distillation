# MCP memory read-layer (local pre-MVP)

## Decision

The server exposes two tools, not a mode switch:

- `construct_archive_context` is for broad personal context: prior work on a topic, preferences, decisions, or project continuation.
- `search_archive` is for a focused historical lookup: a named conversation, issue, decision, person, or exact evidence.

This is the smallest interface that gives the model an unambiguous broad/focused choice. A single `mode` parameter moves that choice into arguments and makes accidental focused retrieval for a broad continuation more likely. Both tools use the same `ArchiveSession`, native retriever, result assembler, filtering, and provenance contract.

The model must call a tool only when personal archive context could affect its answer: references to prior discussion or decisions, continuation of an old project, preferences, historical comparison, or a named project/entity likely present in the archive. It must not call one for general questions, when the current conversation is sufficient, or when the user opts out. Tool descriptions explicitly prohibit claims about archive content before a successful call.

## Contract

`construct_archive_context(current_context, max_tokens?, max_chars?, project_hint?, time_range?, include_preferences?, include_decisions?, include_recent_related?, timeout_ms?)` returns a broad package. It sends the original context plus deterministic term, decision, and preference variants through hybrid retrieval. There is no server-side LLM query expansion.

`search_archive(query, limit?, project?, date_from?, date_to?, roles?, conversation_id?, retrieval_mode="hybrid", include_neighbors?, max_tokens?, max_chars?, timeout_ms?)` returns a focused package. The native pre-MVP deliberately exposes only `hybrid`; dense/sparse-only modes would change the verified retrieval semantics.

Each response is `kb.mcp.memory.v1` JSON with `summary`, bounded `items`, `coverage`, `warnings`, and a non-sensitive runtime summary. Every item contains message/conversation IDs, source path, chunk/block IDs, source offsets, role, timestamp, dense/sparse/fused scores, and a deterministic match reason. The summary is a factual count/title synopsis only; it does not invent semantic claims.

## Retrieval and assembly

The underlying path remains unchanged: query encoding -> dense top-500 and sparse top-500 -> ID union -> 0.65/0.35 fusion -> chunk provenance. The MCP layer adds only presentation logic:

1. Deduplicate all multi-query results by message ID, retaining the best fused score.
2. Limit a focused response to 30 items and three messages per conversation. Broad context selects at most six conversations and three messages each for diversity.
3. Fetch zero to four chronological neighbours on either side; neighbours are labelled and bounded to 800 characters.
4. Trim hit text and select items under a default 1,800-token budget (hard maximum 6,000). Long code blocks are therefore naturally truncated with an ellipsis rather than copied wholesale.
5. Order selected hits by relevance; each neighbour window preserves message ordinal. This lets callers see updates/corrections in chronology without pretending that an old high-score excerpt is the final decision.

Project/date/role/conversation constraints are post-filters over the 500-candidate native pool. This is intentionally explicit: the compact clean-native index does not have metadata prefilter semantics. If a filter produces no matches, the response says so rather than falling back to unfiltered data. A later index change may add prefiltering only after recall is measured.

## Lifecycle and transport

`kb-mcp serve --db /path/to/chat_memory_native_pre_mvp.db --transport stdio` loads dense and sparse providers, sqlite-vec, and compact sparse arrays once before it is ready. A shared re-entrant lock serializes calls over the reusable SQLite/native session; no legacy backend is opened. `kb-mcp test-call --db ... --tool search_archive --query "..."` is a local smoke call.

The server uses JSON-RPC stdio today. Retrieval is separate from the transport, so a future HTTP adapter can place authentication/authorization middleware before `MCPServer` without exposing raw SQL or the database file. Logs record only event type, tool name, and exception class: neither queries nor result text are logged.

## Limits and evaluation

The prior runtime audit is the performance baseline: model cold start is about 12 seconds and warm native hybrid p50 is about 467 ms. MCP-specific warm latency must be measured against the real 1.248 GiB database before a production claim; the local report records this as pending, not as a substituted number.

The local-only `benchmarks/mcp_pre_mvp/` directory is reserved for a manually labelled 20--30 query evaluation set (factual recall, decisions, project continuation, preferences, broad synthesis, ambiguity, temporal updates, and no-result cases). It must contain expected conversation/message IDs, tool/mode correctness, provenance completeness, output size, and latency; no ChatGPT-export text or identifiers may be committed. This is pre-MVP read-only software, not a remote deployment or an authoritative memory source.
