# PTHA — Personal Thought Archive

PTHA is a local-first tool that imports a ChatGPT export into a private SQLite
knowledge archive and exposes bounded archive retrieval through MCP. The data,
embedding model cache, database, and service remain on the user's machine.

## Status and supported runtime

PTHA is an early release candidate for macOS on Apple Silicon. The production
embedding path uses MLX FP16 BGE-M3 with fused scaled dot-product attention.
The public MCP surface currently contains two read-only tools:

- `search_archive`
- `construct_archive_context`

Attachments are preserved only where the import pipeline can recover them;
attachment content is not indexed by the PTHA v1 retrieval database. There is
no supported CUDA or remote-service path in this release.

The planned attachment-artifact direction keeps documents separate from
conversation memory and requires an explicit read before their content is
used. See [docs/product-radar.md](docs/product-radar.md).

## Install

Requirements: macOS on Apple Silicon, Python 3.13+, and [uv](https://docs.astral.sh/uv/).

Install directly from GitHub:

```bash
uv tool install git+https://github.com/anfedoro/ptha
```

Check the installed tool:

```bash
ptha --help
```

## First run

```bash
ptha init
time ptha import /absolute/path/to/chatgpt-export.zip
ptha doctor
ptha service start
ptha service status
ptha mcp config --absolute
```

Import is explicit and publishes the database only after validation. The first
large import can take roughly 20–50 minutes after the model is cached; the
terminal displays phase-labelled progress bars, speed, and ETA. The exact
paths, database location, and logs are printed by `ptha init` and service
commands.

Read the complete copy-paste workflow in [docs/first-run.md](docs/first-run.md).

## MCP clients

Keep the retrieval service running, then give the JSON printed by
`ptha mcp config --absolute` to an MCP client such as LM Studio. The client
starts the stdio adapter; the adapter connects to the local service and does
not load models or open the database itself. See
[docs/mcp-integration.md](docs/mcp-integration.md).

## Architecture

The runtime has four boundaries:

1. Import reads the export, distills canonical content, builds deterministic
   retrieval chunks, and atomically publishes a validated SQLite database.
2. One local service process opens the database, loads the MLX model once, and
   serves versioned Unix-socket IPC.
3. Dense and sparse representations are produced by one MLX BGE-M3 backbone
   forward and stored in the existing native indexes.
4. The stdio MCP adapter translates JSON-RPC and allowlists only the two public
   archive tools.

See [docs/architecture.md](docs/architecture.md) for the stable runtime
overview and [docs/ptha-embedding-pipeline.md](docs/ptha-embedding-pipeline.md)
for model provenance and the provider contract.

## Development

```bash
uv sync
uv run python -m unittest discover -s tests
uv build
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing storage, MCP schemas,
or model artifacts.

## Operational documentation

- [Installation and first run](docs/first-run.md)
- [Configuration](docs/configuration.md)
- [MCP integration](docs/mcp-integration.md)
- [Service lifecycle](docs/ptha-service.md)
- [Doctor checks](docs/ptha-doctor.md)
- [Reindex and recovery](docs/ptha-reindex.md), [docs/ptha-recovery.md](docs/ptha-recovery.md)

## License

PTHA is distributed under the license in [LICENSE](LICENSE).
