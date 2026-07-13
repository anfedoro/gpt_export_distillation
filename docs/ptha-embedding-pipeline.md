# PTHA embedding pipeline

PTHA uses one `BAAI/bge-m3` backbone for both retrieval representations:

```text
BAAI/bge-m3
  -> normalized 1024-dimensional CLS dense vectors
  -> lexical weights from the official sparse_linear.pt head
```

The sparse head belongs to the same Hugging Face model repository. PTHA does
not load or download a second sparse encoder in normal import, reindex,
service, or doctor workflows. The legacy OpenSearch `SparseEncoder` is retained
only by the opt-in local comparison benchmark.

## Build order

Index construction is deliberately sequential at the batch level:

1. create canonical retrieval chunks;
2. run one BGE-M3 forward for a batch;
3. derive dense and sparse representations from that hidden state;
4. write the separate dense and sparse tables for that batch;
5. validate counts and publish the database atomically.

There are no concurrent dense/sparse workers and no second backbone forward.
Both representations resolve one shared device and retain one model backbone.
The database schema, dense/sparse candidate union, fusion weights, and MCP
tools are unchanged. Dense and sparse rows remain separate and the complete
database is still published atomically.

The effective input limit is 512 tokens. Chunk construction checks the input
against every active representation contract before inference. This avoids
quadratic 8K-token attention allocations on Apple Silicon.

## Hardware and measurements

Apple Silicon with MPS is the preferred local runtime. CPU remains a supported
fallback but is expected to be materially slower. A cold two-text M4 Max smoke
run used approximately 4 GiB peak RSS and completed in 22.4 seconds, including
model load. This is a smoke measurement, not an initial-import estimate.

The real 2026-07-13 M4 Max benchmark prepared 131,717 chunks from the authorized
export in 573.7 seconds. On a fixed 10,000-chunk subset with batch size 32:

- hash-ID batching: dense 17.11 chunks/s, sparse 17.62 chunks/s, 1,152.1 seconds total;
- length-bucketed batching: dense 68.30 chunks/s, sparse 56.49 chunks/s, 323.4 seconds total.

After joint-forward inference, the same fixed 10,000-chunk length-bucketed
diagnostic completed in 10.13 seconds (987.6 chunks/s) on MPS. The shared
backbone forward took 7.22 seconds; dense pooling, sparse projection/transfer,
decoding, and SQLite writes accounted for the remainder. The old measurements
above are retained as the pre-joint baseline.

Length bucketing is therefore part of the production build path. It changes
only scheduling, not chunk identity or vector values. Batch size 128 improved a
1,000-chunk probe by only about 12 percent while raising MPS driver allocation
substantially, so 32 remains the conservative default recommendation. The
length-bucketed result extrapolates to roughly 70 minutes for embeddings over
the complete real archive, plus about 10 minutes for distillation/chunking. It
does not reproduce the historical 20-minute claim; that older run is not
comparable until its exact chunk count, sequence-length distribution, model
heads, and publication work are recovered.

Every native build reports the following privacy-safe metrics:

```text
Embedding build
  Chunks
  Joint: processed, seconds, throughput, device
  Dense/Sparse: processed, shared joint-pass seconds, throughput, device
  Total seconds
```

No source or retrieved text is included.

## Runtime precision diagnostic

The diagnostic-only harness checks the effective PyTorch runtime and separates
tokenization, backbone forward, pooling/normalization, sparse projection,
host transfer, decoding, and SQLite insertion. It does not modify the product
import path and does not print chunk or retrieval content:

```bash
uv run python -m ptha.embedding_diagnostics \
  --chunk-db .local/ptha-benchmark/chunks.db \
  --output-db .local/ptha-benchmark/diagnostic-10k.db \
  --limit 10000 --batch-size 32 --device auto
```

The report includes the effective `from_pretrained` dtype argument, model and
layer dtypes, model/input/sparse-head devices, autocast state, and privacy-safe
timing fields. The current BGE-M3 path loads `float16` weights on MPS, uses
`model.eval()` plus `torch.inference_mode()`, and does not enable autocast.
The XLM-R pooler module is present but unused: dense output is CLS pooling
followed by L2 normalization. The sparse checkpoint is loaded as `float16`;
conversion to `float32` occurs only when materializing CPU lexical weights for
the existing storage contract.

## Fixed 10,000-chunk benchmark

Prepare one deterministic chunk database from the explicitly authorized export:

```bash
uv run python -m ptha.embedding_benchmark prepare \
  --source /path/to/chatgpt-export.zip \
  --output .local/ptha-benchmark/chunks.db
```

Run the unified backend:

```bash
uv run python -m ptha.embedding_benchmark run \
  --chunk-db .local/ptha-benchmark/chunks.db \
  --output-db .local/ptha-benchmark/unified.db \
  --backend unified \
  --limit 10000 \
  --batch-size 32 \
  --device mps
```

For a one-time historical comparison, run `--backend legacy` into a separate
output database. That mode intentionally uses the old second sparse model and
must not be treated as the PTHA product configuration. Keep all benchmark DBs
and reports under ignored `.local/`; real archive derivatives must never be
committed.

## Full import

After selecting the benchmarked batch size:

```bash
ptha import /path/to/chatgpt-export.zip --replace --batch-size 32
```

PTHA removes staging database, WAL, and SHM files from dead import processes
before a new import. It never removes files owned by a live PID or follows a
symlink during cleanup.
