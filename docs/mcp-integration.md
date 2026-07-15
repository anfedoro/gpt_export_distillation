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
