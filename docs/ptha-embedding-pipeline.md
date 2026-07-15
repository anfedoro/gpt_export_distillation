# Embedding provider

On supported Apple Silicon, PTHA uses one pinned MLX FP16 BGE-M3 artifact for
both retrieval representations:

```text
anfedoro/bge-m3-mlx-fp16
    -> normalized 1024-dimensional dense vectors
    -> sparse lexical weights from sparse_linear.safetensors
```

The artifact also contains `colbert_linear.safetensors` for provenance, but
PTHA v1 does not expose a ColBERT retrieval path. No additional model training
is performed.

## Runtime contract

The pinned model revision is:

```text
58e70901dbba8de8f3df91b5a313bcefcb151bae
```

The pinned `mlx-embeddings` revision is:

```text
4a8277aa523eb34ff29a5a832fa3f3f654336b54
```

For each length-aware batch, PTHA tokenizes without global padding, pads only
to the local maximum, executes the backbone once, derives dense and sparse
outputs, materializes the computation, restores input order, and writes the
existing storage formats. Default batch size is four. Chunking, schema,
candidate union, fusion weights, ranking, and MCP contracts are unchanged.

PyTorch is not a default inference dependency. It is used only by the offline
conversion extra and reference development tools. CUDA is not a supported
production path in this release.

## Conversion

The reproducible converter is
`scripts/convert_bge_m3_to_mlx_fp16.py`. It accepts a source BGE-M3 snapshot,
converts the backbone and heads to FP16 safetensors, copies tokenizer assets,
validates shapes and dtypes, and can upload a prepared model repository. The
normal PTHA runtime never converts `.pt` files.

Do not commit downloaded model files, conversion outputs, or benchmark results.
