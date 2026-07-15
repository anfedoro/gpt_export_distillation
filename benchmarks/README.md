# BGE-M3 regression benchmark

These scripts use synthetic text only. They do not read PTHA databases or ChatGPT exports.

The public harness uses synthetic text only. It supports fixed or mixed lengths,
batch sizes 2 through 32, dense-only and joint dense+sparse execution, and
timing with tokenization included or precomputed. It reports median wall time,
real token throughput, chunk throughput, and padding efficiency. Always compare
the same text construction, precision, and mode.

```bash
uv run python benchmarks/benchmark_bge_m3_mlx.py \
  --model /path/to/bge-m3-mlx-fp16 --mixed --mode dense-sparse
```

Peak GPU memory is intentionally omitted: neither runtime currently exposes a
cross-framework measurement with equivalent semantics. Process RSS may be measured
externally, but it is not GPU peak memory.
