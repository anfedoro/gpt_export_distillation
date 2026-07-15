# MCP integration

PTHA exposes two read-only tools through a local stdio MCP adapter:

- `search_archive` for focused archive lookup;
- `construct_archive_context` for bounded context assembly with source metadata.

Start the retrieval service first:

```bash
ptha service start
ptha service status
ptha mcp config --absolute
```

Paste the generated JSON into the MCP client's configuration. The generated
command includes the installed executable and active PTHA configuration path.
The client starts `ptha mcp serve` itself; do not run a second daemon for the
adapter.

The adapter does not load models, open the database, or auto-start the service.
If the service is unavailable, the client receives a sanitized error and the
adapter writes startup guidance to stderr. Internal IPC operations such as
`shutdown` are not exposed as MCP tools.

## Focused search result assembly

`search_archive` does not return its raw dense+sparse candidate list. After
the existing hybrid fusion step, PTHA performs a deterministic, local-only
post-retrieval pass over that bounded list: intent-aware reranking, overlap and
near-duplicate grouping, low-information anchor filtering, conversation-aware
diversification, and bounded neighbour assembly. It does not call another
model, alter index scores, or require a reindex.

The post-retrieval pass is capped at 180 fused candidates (or fewer for small
requests), so its duplicate and novelty comparisons never scan the archive.

The response remains a `kb.mcp.memory.v1` envelope. Each focused item keeps the
legacy `text`, `context_before`, `context_after`, and provenance fields, and
also provides `excerpt`, `supporting_context`, and merged provenance such as
`contributing_chunk_ids`. `coverage` reports raw hit count, evidence-group
count, duplicate and low-information drops, estimated tokens, and the output
budget.

By default, at most two evidence groups from a conversation are selected. A
neighbour is included only when it is near the anchor, fits the response
budget, and has not already been used as neighbour context for another item.
Set `include_neighbors` to `0` to return no neighbour context.
