# Repository handling rules

This repository is public. Treat any benchmark gold scenarios, fixtures,
probes, expected outputs, or evaluation artifacts derived from personal
ChatGPT exports or other private sources as local-only data.

- Keep private test data outside tracked files, preferably under ignored paths
  such as `benchmarks/gold/` or `.local/`.
- Do not commit private conversation text, identifiers, metadata, embeddings,
  raw retrieval scores, or generated reports that can reveal private content.
- Public test fixtures must be synthetic or demonstrably anonymized. Check
  this before staging benchmark changes.
