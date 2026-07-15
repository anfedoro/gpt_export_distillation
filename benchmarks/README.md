# BGE-M3 runtime benchmarks

These scripts use synthetic text only. They do not read PTHA databases or ChatGPT exports.

The MLX benchmark supports fixed or mixed lengths, batch sizes 2 through 32,
dense-only and joint dense+sparse execution, and timing with tokenization included or
precomputed. It reports median wall time, real token throughput, chunk throughput, and
padding efficiency. Always compare the same text construction, precision, and mode.

```bash
uv run python benchmarks/benchmark_bge_m3_mlx.py \
  --model .local/bge-m3-mlx-fp16 --mixed --mode dense-sparse

uv run --extra model-conversion python benchmarks/benchmark_bge_m3_torch.py \
  --model BAAI/bge-m3 --sparse-head /path/to/sparse_linear.pt \
  --device mps --dtype float16 --sparse-bias-sign plus --batch-sizes 4 8
```

Observed on Apple M4 Max for the earlier 1000-chunk, approximately 200-word
synthetic comparison:

| Runtime | Precision | Batch | Throughput |
|---|---|---:|---:|
| MLX fused SDPA | FP16 | 4 | about 43 chunks/s |
| PyTorch MPS SDPA | FP16 | 4 | about 34 chunks/s |
| MLX | 8-bit | 4 | about 37 chunks/s |

Peak GPU memory is intentionally omitted: neither runtime currently exposes a
cross-framework measurement with equivalent semantics. Process RSS may be measured
externally, but it is not GPU peak memory.
