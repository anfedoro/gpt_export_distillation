# Canonical generation migration note

Canonical representation version 1 changes the semantic input to dense and
sparse encoders. Existing embeddings are incompatible with it.

Do not alter an active database in place. Build a new candidate through the
staged import path, validate it, and use the existing atomic publication
mechanism only after review. Keep the old database available until acceptance.

Required checks:

- native schema `kb.native_pre_mvp.v3`;
- manifest schema 2;
- parser and canonical representation contracts present;
- matching canonicalizer, source-transform, block-builder, and chunker;
- matching dense and sparse embedding contract fingerprint;
- equal chunk, dense, and sparse counts;
- passing integrity, foreign keys, lineage, and chunk coverage.

A v1/v2 database without canonical manifest fields is
`legacy_precanonical`. Status, doctor, service start, and reindex do not mutate
or silently upgrade it.

Isolated rebuild:

```bash
uv run python -c \
'from pathlib import Path; from kb.storage.native_pre_mvp import build_native_pre_mvp_db; build_native_pre_mvp_db(export_path=Path("/read-only/distilled"), output_db=Path("/tmp/ptha-canonical.db"), chunk_content_budget=256)'
```

Old-versus-new analysis:

```bash
uv run python scripts/validate_canonical_generation.py \
  --baseline-db /tmp/ptha-baseline.db \
  --canonical-db /tmp/ptha-canonical.db \
  --output-dir /tmp/ptha-canonical-validation
```

Generated databases, CSV reviews, and reports may contain private content.
Keep them in ignored local paths and never commit or publish them.
