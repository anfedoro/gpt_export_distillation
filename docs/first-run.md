# PTHA: first run

This is the practical local path for macOS on Apple Silicon. It does not run an
import implicitly: the user chooses the export, starts import, and sees its
progress in the terminal.

`ptha init` currently has no workspace positional argument. With the default
configuration on macOS, PTHA uses:

| Item | Default location |
| --- | --- |
| Configuration and database | `~/Library/Application Support/ptha/` |
| Model cache | `~/Library/Caches/ptha/` |
| Runtime socket | `~/Library/Caches/TemporaryItems/ptha/` |
| Service log | `~/Library/Application Support/ptha/logs/service.log` |

Every `ptha init` invocation prints the actual paths selected on that machine.

## 1. Install

Install from a built wheel:

```bash
uv tool install /absolute/path/to/gpt_export_distillation-0.2.15-py3-none-any.whl
```

Or install the current repository version:

```bash
uv tool install git+https://github.com/anfedoro/gpt_export_distillation
```

Confirm that the installed console command is available:

```bash
ptha --help
```

## 2. Initialize local storage

```bash
ptha init
```

The command prints the configuration, database, and log locations. It is safe
to run again; an existing configuration is not overwritten.

### Put the database in a chosen location

Use an explicit config file, then change its `[paths]` section before import:

```bash
WORKSPACE="$HOME/ptha-workspace"
mkdir -p "$WORKSPACE"
ptha --config "$WORKSPACE/config.toml" init
```

Edit `$WORKSPACE/config.toml` and set absolute paths:

```toml
[paths]
database = "/absolute/path/to/ptha-workspace/ptha.db"
working_dir = "/absolute/path/to/ptha-workspace/tmp"
model_cache = "/absolute/path/to/ptha-workspace/model-cache"
```

Pass the same `--config "$WORKSPACE/config.toml"` before every subsequent
PTHA command. The service and MCP adapter receive that path too. An environment
override, `PTHA_DB_PATH=/absolute/path/to/ptha.db`, is available for one-off
commands, but a config file is the persistent and MCP-safe choice.

## 3. Import a ChatGPT export

PTHA accepts either the original ChatGPT export ZIP or an extracted export
directory.

```bash
ptha import /absolute/path/to/chatgpt-export.zip
```

Use `--replace` only to rebuild an existing PTHA database. The service must be
stopped first:

```bash
ptha service stop
ptha import /absolute/path/to/chatgpt-export.zip --replace
```

The first import downloads the pinned MLX FP16 BGE-M3 model if it is not in the
local Hugging Face cache. The terminal prints stages such as:

```text
[1/7] Reading export
[2/7] Distilling conversations
[3/7] Importing canonical content
[native-build] joint_processed=400
[native-build] joint_processed=800
[7/7] Validating database

Import completed.

Database:
  /.../ptha.db

Content:
  Conversations: ...
  Messages: ...
  Retrieval chunks: ...
```

The CPU/GPU time depends on archive size, local cache state, and Apple Silicon
hardware. PTHA publishes the active database only after dense and sparse
indexes pass validation; an interrupted build leaves the previous active DB
unchanged. Run `ptha doctor` after completion for the database check.

`joint_processed` is the number of retrieval chunks whose dense and sparse
representations have both been written. With the default batch size of four it
is reported every 400 chunks, so the terminal remains visibly active during a
long build. To record your own wall-clock duration, prefix the command with
`time`:

```bash
time ptha import /absolute/path/to/chatgpt-export.zip
```

For a large personal archive, plan for roughly **20–50 minutes** after the
model is cached. This is an operating range, not a guarantee: the archive's
number and length of messages, available memory, model download, and local
storage speed all affect it. Do not start a second import while the first is
running.

## 4. Check the database and start the service

```bash
ptha status
ptha doctor
ptha service start
ptha service status
```

`service start` waits until the model and database are ready. It runs in the
background. The service log path is shown by `ptha init`, `ptha service start`,
and `ptha service status`.

## 5. Check retrieval without an MCP client

`ptha doctor --full` loads the local runtime and performs a non-content smoke
check. A supplied query is never printed in the report:

```bash
ptha doctor --full --query "personal knowledge base"
```

For the full MCP protocol check, keep the service running and use the stdio
adapter in the next step. The adapter itself does not load a model or open the
database.

PTHA intentionally has no separate `ptha search` command in this release
candidate. `doctor --full` is the direct CLI compatibility check; the two
public archive queries are invoked through MCP so the same path is used by LM
Studio and any other MCP client.

## 6. Start the MCP adapter

The external MCP client starts this process. Do not run it as a background
daemon yourself:

```bash
ptha mcp serve
```

The retrieval service must already be running:

```bash
ptha service start
```

## 7. Add the MCP configuration to a client

Print a portable generic snippet:

```bash
ptha mcp config
```

For a copy-paste configuration tied to the installed executable and the active
PTHA configuration, use:

```bash
ptha mcp config --absolute
```

It prints this MCP-client-shaped JSON (with real absolute paths):

```json
{
  "mcpServers": {
    "ptha": {
      "command": "/absolute/path/to/ptha",
      "args": ["--config", "/absolute/path/to/config.toml", "mcp", "serve"]
    }
  }
}
```

The public tools are `search_archive` and `construct_archive_context`.

With a custom workspace config, generate the snippet from that same config:

```bash
ptha --config "$WORKSPACE/config.toml" mcp config --absolute
```

### LM Studio

1. Start PTHA first: `ptha service start`.
2. In LM Studio, open the right sidebar's **Program** tab, choose
   **Install > Edit mcp.json**, and paste the JSON printed by
   `ptha mcp config --absolute`.
3. Select a model that supports tool calling, enable the PTHA tools, and start
   a chat.
4. Ask for a focused lookup, for example: *"Use `search_archive` to find
   `personal knowledge base`. Return the source metadata and a short answer."*
5. Ask for a bounded historical context, for example: *"Use
   `construct_archive_context` for my current work on a personal knowledge
   base. Keep the context under 600 tokens and preserve source metadata."*

LM Studio starts the stdio adapter itself from `mcp.json`; do not separately
start `ptha mcp serve` in another terminal. Its current MCP setup instructions
are available at <https://lmstudio.ai/docs/app/mcp>.

## 8. Stop the service

```bash
ptha service stop
```

## 9. Common errors

- **`PTHA database is not ready`**: run `ptha import /path/to/export.zip`.
- **`PTHA service is not running` from MCP**: run `ptha service start` before
  connecting the MCP client.
- **Existing database**: either keep it, or stop the service and rerun import
  with `--replace`.
- **Service state is stale**: inspect `ptha service status`; use
  `ptha service cleanup` only for proven stale state.
- **Model download or MLX problem**: run `ptha doctor` for a local diagnostic;
  on Apple Silicon, install with a supported Python 3.13+ runtime.
