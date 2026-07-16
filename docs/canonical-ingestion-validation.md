# Canonical ingestion validation

Validation used a local ignored representative slice of 16 conversations and
352 messages. Both generations used the same BGE-M3 MLX revision, sparse
contract, 256-token content budget, source files, and machine. The baseline was
built from clean `main`; the candidate was built from
`feat/canonical-semantic-ingestion`. Databases and detailed CSVs remain under
ignored `.local/validation/`.

## Build and corpus

| Metric | Baseline | Canonical |
|---|---:|---:|
| source documents | 16 | 16 |
| messages | 352 | 352 |
| blocks | 1,042 | 3,524 |
| semantic chunks | 1,449 | 3,054 |
| stored artifacts | not explicit | 201 |
| context-only blocks | not explicit | 347 |
| exact duplicates downgraded | 0 | 145 |
| duplicate chunk-text fraction | 3.2% | 0.3% |
| low-information chunk fraction | 1.1% | 1.7% |
| DB size | 16.4 MiB | 38.8 MiB |
| total build time | 21.23 s | 16.87 s |
| embedding throughput | 107.2 chunks/s | 232.2 chunks/s |
| parser throughput | 295.7 chats/s | 38.4 chats/s |
| chunking throughput | 369.5 blocks/s | 2,961.3 blocks/s |
| peak resident memory | not remeasured | 1.72 GiB |

The candidate has more paragraph-level prose blocks, but embeds fewer tokens
(133,214 versus 159,838). Block distribution was 3,323 prose, 151 code, 32
quote/external, 16 table, and 2 structured-data blocks. The slice contained no
explicit Mermaid or media-pointer examples; deterministic tests cover them.

## Semantic neighbourhood

| Metric | Baseline | Canonical |
|---|---:|---:|
| rank-1 median | 0.7863 | 0.7420 |
| rank-5 median | 0.7084 | 0.6707 |
| rank-10 median | 0.6712 | 0.6426 |
| rank-20 median | 0.6339 | 0.6157 |
| mutual-link fraction, k=20 | 63.1% | 55.1% |
| giant component, k=20 | 1,372 | 3,024 |
| isolated nodes, k=20 | 18 | 28 |
| low-information among top-100 hubs | 1.0% | 0.0% |
| low-information among top-100 strongest edges | 11.0% | 2.0% |

Lower similarity and mutuality are expected because repeated wrappers and
duplicate fragments no longer create artificial near-identical neighbours.
Component sizes are not directly normalized because node counts differ.

## Manual review

Review covered 100 strongest mutual edges, 100 highest-inbound hubs, 50
cross-project mutual edges, 50 high-dense/low-sparse edges, 50
excluded/context-only blocks, 50 short retained blocks, and 50
prose-to-artifact relations.

The first review found `:::writing{...}` wrappers and generic introductory
phrases still eligible; both were corrected before the final candidate. The
final review confirmed that fences/table borders are absent, artifacts remain
reachable but unembedded, exact repeated prose is downgraded, and strongest
edges predominantly connect related technical observations, command families,
edited prose variants, and closely related statements.

Known limitations:

- unfenced user commands, hashes, package output, and protocol logs stay prose;
- near-duplicate paraphrases are not globally downgraded;
- conservative rules retain some generic introductions to avoid dropping short
  meaningful statements;
- XML/YAML/TOML are preserved but not parsed into trees;
- detailed review CSVs contain private-derived content and remain local-only.

No active database, archive, installed tool, service runtime, or published
generation was modified. No merge or push was performed.

## Reproduction

Set `SOURCE_SLICE` to a read-only distilled archive directory. Keep every
output under ignored `.local/`.

```bash
export SOURCE_SLICE=/absolute/path/to/read-only/selected_export
export RUN_DIR="$PWD/.local/validation/canonical-semantic-ingestion"
mkdir -p "$RUN_DIR"

git worktree add --detach "$RUN_DIR/main-baseline" main

(cd "$RUN_DIR/main-baseline" && uv run python - <<'PY'
import os
from pathlib import Path
from kb.storage.native_pre_mvp import build_native_pre_mvp_db
build_native_pre_mvp_db(
    export_path=Path(os.environ["SOURCE_SLICE"]),
    output_db=Path(os.environ["RUN_DIR"]) / "baseline.db",
    batch_size=8,
    chunk_content_budget=256,
)
PY
)

/usr/bin/time -l uv run python - <<'PY'
import os
from pathlib import Path
from kb.storage.native_pre_mvp import build_native_pre_mvp_db
build_native_pre_mvp_db(
    export_path=Path(os.environ["SOURCE_SLICE"]),
    output_db=Path(os.environ["RUN_DIR"]) / "canonical.db",
    batch_size=8,
    chunk_content_budget=256,
)
PY

uv run python scripts/validate_canonical_generation.py \
  --baseline-db "$RUN_DIR/baseline.db" \
  --canonical-db "$RUN_DIR/canonical.db" \
  --output-dir "$RUN_DIR/report"

uv run python -m unittest discover -s tests -q
uv build --wheel --out-dir "$RUN_DIR/wheel"
uv venv --python 3.13 "$RUN_DIR/smoke-venv"
uv pip install \
  --python "$RUN_DIR/smoke-venv/bin/python" \
  "$RUN_DIR/wheel/ptha-0.5.0-py3-none-any.whl"
"$RUN_DIR/smoke-venv/bin/python" -c \
  "from ptha.incremental import PARSER_CONTRACT; print(PARSER_CONTRACT)"
```

Final regression result: 235 tests passed; one optional hardware-artifact test
was skipped because its explicit environment variable was not configured.
