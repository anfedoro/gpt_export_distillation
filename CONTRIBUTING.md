# Contributing to PTHA

## Development setup

```bash
uv sync
uv run python -m unittest discover -s tests
uv build
```

The default runtime is MLX FP16 on Apple Silicon. PyTorch is optional and is
used only by the offline model-conversion extra and reference benchmarks.

## Versioning

PTHA is pre-1.0. Every change that affects runtime behavior, the CLI, storage,
configuration, or the user workflow must increment the minor version at least.
For example, the current `0.3.x` line moves to `0.4.0` after such a change.
Do not leave a new user-visible build under the previous version number. Update
`pyproject.toml`, rebuild the wheel, and verify `ptha --version` before handoff.

## Repository boundaries

- `src/ptha/` contains the product CLI, service lifecycle, IPC, MCP adapter,
  operational checks, and configuration.
- `src/kb/` contains the storage and retrieval implementation used by PTHA.
- `src/gpt_export_distillation/` is the import-time legacy distillation library;
  it is an implementation dependency, not a second public product CLI.
- `tests/` must use synthetic data or temporary fixtures.
- `scripts/convert_bge_m3_to_mlx_fp16.py` is the reproducible offline model
  conversion tool.

## Public contracts

Treat the CLI commands, MCP tool schemas, IPC protocol, SQLite schema, canonical
content identity, and retrieval ranking behavior as compatibility boundaries.
Discuss changes to those contracts before implementing them. Do not add a new
MCP tool or change chunking/fusion semantics as part of an unrelated cleanup.

## Never commit

Never commit credentials, model caches or weights, user archives, generated
databases, logs, absolute local paths, private benchmark results, Hugging Face
tokens, exported ChatGPT data, or evaluation artifacts derived from a personal
archive. Use ignored local paths and synthetic fixtures instead.

Before opening a change, run the test suite, `uv build`, and `git diff --check`.
