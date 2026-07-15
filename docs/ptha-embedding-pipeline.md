# PTHA embedding pipeline

PTHA uses one project-owned MLX FP16 BGE-M3 artifact for both retrieval
representations:

```text
BAAI/bge-m3 (pinned source revision)
  -> anfedoro/bge-m3-mlx-fp16 (pinned converted revision)
  -> normalized 1024-dimensional CLS dense vectors
  -> lexical weights from sparse_linear.safetensors
```

The artifact also contains `colbert_linear.safetensors` for complete BGE-M3
provenance, although PTHA v1 does not expose a ColBERT retrieval path. No
additional training is performed. The conversion metadata and model card record
the upstream revision, conversion date, script, FP16 dtype, head provenance,
and upstream MIT license.

## Runtime contract

Apple Silicon with the MLX GPU backend is the supported production runtime.
There is no silent CPU or CUDA fallback in this release. Startup validates the
model revision, required files, 1024 hidden size, vocabulary, position limit,
FP16 backbone and sparse head, and availability of fused scaled dot-product
attention.

The pinned `mlx-embeddings` fork revision is
`4a8277aa523eb34ff29a5a832fa3f3f654336b54`. It provides fused
`mx.fast.scaled_dot_product_attention` for XLM-RoBERTa and exposes
`last_hidden_state` for the official BGE-M3 sparse projection.

For every length-aware batch PTHA tokenizes without global padding, pads only to
the local maximum, executes the backbone once, derives normalized dense vectors
and sparse token weights, materializes both with `mx.eval()`, restores original
input order, and writes the existing dense and compact sparse storage formats.
Repeated sparse token IDs use their maximum weight; padding, special tokens,
zeroes, and non-finite weights are excluded. Chunking, schema, candidate union,
fusion weights, ranking, and MCP contracts are unchanged.

Default configuration:

```toml
[models]
embedding_backend = "mlx"
embedding_model = "anfedoro/bge-m3-mlx-fp16"
embedding_model_revision = "58e70901dbba8de8f3df91b5a313bcefcb151bae"
embedding_dtype = "float16"
embedding_device = "gpu"
embedding_batch_size = 4
embedding_max_padded_tokens = 0
embedding_sparse_head = "sparse_linear.safetensors"
embedding_colbert_head = "colbert_linear.safetensors"
```

## Conversion

PyTorch is not a default dependency and is not imported by production indexing
or service startup. It is used only by the offline converter and the reference
benchmark:

```bash
uv run --extra model-conversion python scripts/convert_bge_m3_to_mlx_fp16.py \
  --source-revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --output .local/bge-m3-mlx-fp16 \
  --upload-repo anfedoro/bge-m3-mlx-fp16
```

The tool accepts a local source snapshot or downloads the requested revision,
converts the backbone and both heads to FP16 safetensors, copies tokenizer
assets, validates saved shapes and dtypes, and optionally uploads the directory.
Runtime never converts `.pt` files.

## Measurements

The framework-isolated Apple M4 Max synthetic comparison measured approximately:

| Runtime | Precision | Batch | Throughput |
|---|---|---:|---:|
| MLX fused SDPA | FP16 | 4 | 43.5 chunks/s |
| PyTorch MPS SDPA | FP16 | 4 | 34 chunks/s |
| MLX | 8-bit | 4 | 37 chunks/s |

These values are not a full-import estimate. The committed synthetic harnesses
report median wall time, chunks/s, real tokens/s, padded tokens, and padding
efficiency for fixed and mixed lengths. Peak GPU memory is omitted because the
frameworks do not expose equivalent measurements.

The migrated provider was remeasured on 2026-07-14 with MLX 0.32.0,
`mlx-embeddings` 0.1.1 at the pinned fork commit, and Transformers 5.12.1.
For 1,000 fixed 200-word synthetic chunks, batch 4, joint dense+sparse, and
tokenization included, the three-run median was 22.982 seconds: 43.51 chunks/s,
15,186 real tokens/s, and padding efficiency 1.0. Model load took 0.710 seconds.

## Numerical regression

A synthetic multilingual/code fixture compared the official Torch FP16 source
with the converted MLX FP16 artifact. The observed minimum dense cosine was
`0.9999955`, sparse token-ID Jaccard was `1.0`, and maximum common sparse-weight
absolute delta was `0.00211`. The accepted safety bounds are dense cosine
`>= 0.999`, sparse token-ID Jaccard `>= 0.99`, and sparse common-weight absolute
delta `<= 0.01`.

A separate synthetic six-topic hybrid fixture produced top-5 and top-10 overlap
of `1.0`; Recall@5, Recall@10, and MRR were all `1.0` for both backends. Five of
six complete rank lists were identical. The regression failure thresholds are
top-5/top-10 overlap below `0.95`, any Recall@5/Recall@10 decrease, or MRR loss
greater than `0.01`. Bitwise equality is not required.

## Import

Production chunking remains unchanged. Run the full import explicitly after the
model repository and revision are configured:

```bash
ptha import /path/to/chatgpt-export.zip --replace --batch-size 4
```

PTHA removes staging DB/WAL/SHM files from proven dead import processes before a
new import. It never follows symlinks or removes files owned by a live process.
