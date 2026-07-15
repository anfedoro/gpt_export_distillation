"""Compatibility entry point for the retired production Torch diagnostic."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The production embedding runtime is MLX; use the standalone synthetic benchmarks."
    )
    parser.parse_args(argv)
    parser.error(
        "The PyTorch production diagnostic was retired. Run "
        "benchmarks/benchmark_bge_m3_mlx.py, or install the model-conversion extra "
        "and run benchmarks/benchmark_bge_m3_torch.py for a reference comparison."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
