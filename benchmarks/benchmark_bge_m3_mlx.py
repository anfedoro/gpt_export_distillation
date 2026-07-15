#!/usr/bin/env python3
"""Synthetic MLX BGE-M3 benchmark independent of PTHA storage and retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_embeddings.sparse import load_sparse_linear, sparse_token_weights
from mlx_embeddings.utils import load_model, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 320, 512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--chunks", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--exclude-tokenization", action="store_true")
    parser.add_argument("--mode", choices=("dense-only", "dense-sparse"), default="dense-sparse")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    load_started = time.perf_counter()
    model = load_model(args.model, path_to_repo=str(args.model))
    tokenizer = load_tokenizer(args.model)
    weight, bias = load_sparse_linear(args.model / "sparse_linear.safetensors")
    mx.eval(model.parameters(), weight, bias)
    load_seconds = time.perf_counter() - load_started
    reports = []
    for batch_size in args.batch_sizes:
        selected_lengths = args.lengths if args.mixed else [args.lengths[-1]]
        texts = make_texts(args.chunks, selected_lengths)
        prepared = tokenize(tokenizer, texts) if args.exclude_tokenization else None
        run_times = []
        metrics = None
        run_once(model, tokenizer, weight, bias, texts[:batch_size], batch_size, args.mode, prepared=None)
        for _ in range(args.runs):
            started = time.perf_counter()
            metrics = run_once(
                model, tokenizer, weight, bias, texts, batch_size, args.mode,
                prepared=prepared,
            )
            run_times.append(time.perf_counter() - started)
        median = statistics.median(run_times)
        reports.append({
            **metrics,
            "batch_size": batch_size,
            "mode": args.mode,
            "tokenization_included": not args.exclude_tokenization,
            "median_seconds": median,
            "chunks_per_second": len(texts) / median,
            "real_tokens_per_second": metrics["real_tokens"] / median,
        })
    print(json.dumps({"framework": "mlx", "precision": "float16", "load_seconds": load_seconds, "runs": reports}, indent=2))
    return 0


def make_texts(count: int, lengths: list[int]) -> list[str]:
    words = "multilingual archive technical architecture retrieval decisions preferences code database service indexing".split()
    return [" ".join((words * ((lengths[index % len(lengths)] // len(words)) + 1))[: lengths[index % len(lengths)]]) for index in range(count)]


def tokenize(tokenizer, texts: list[str]) -> list[list[int]]:
    return [tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512) for text in texts]


def run_once(model, tokenizer, weight, bias, texts, batch_size, mode, *, prepared):
    encoded = prepared if prepared is not None else tokenize(tokenizer, texts)
    real_tokens = padded_tokens = 0
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        width = max(map(len, batch))
        pad = int(tokenizer.pad_token_id)
        ids = mx.array([row + [pad] * (width - len(row)) for row in batch], dtype=mx.int32)
        mask = mx.array([[1] * len(row) + [0] * (width - len(row)) for row in batch], dtype=mx.int32)
        hidden = model(ids, attention_mask=mask).last_hidden_state
        dense = hidden[:, 0]
        dense = dense / mx.sqrt(mx.sum(dense * dense, axis=-1, keepdims=True))
        if mode == "dense-sparse":
            sparse = sparse_token_weights(hidden, weight, bias, mask)
            mx.eval(dense, sparse)
        else:
            mx.eval(dense)
        real_tokens += sum(map(len, batch))
        padded_tokens += width * len(batch)
    return {
        "chunks": len(encoded),
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "padding_efficiency": real_tokens / padded_tokens,
    }


if __name__ == "__main__":
    raise SystemExit(main())
