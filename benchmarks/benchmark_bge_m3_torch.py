"""Standalone Torch equivalent of the MLX BGE-M3 throughput benchmark.

This script intentionally does not import PTHA or write a database. It measures
only tokenization, one BGE-M3 backbone forward, dense pooling, and the sparse
linear head on synthetic repeated chunks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


WORDS_PER_CHUNK = 200
NUM_CHUNKS = 1_000
BATCH_SIZES = (4, 8)


def make_chunks(words_per_chunk: int, num_chunks: int) -> list[str]:
    base = """
    This is a realistic knowledge archive chunk generated from a long
    conversational history. The content contains technical discussions,
    architecture decisions, implementation details, experiments, corrections,
    and conclusions. The purpose is to simulate a personal knowledge base
    retrieval workload where every chunk represents a meaningful semantic unit.

    The system includes distributed services, databases, machine learning
    models, software architecture, security analysis, and engineering notes.
    """
    words = base.split()
    chunk = " ".join((words * ((words_per_chunk // len(words)) + 1))[:words_per_chunk])
    return [chunk] * num_chunks


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> Any:
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    return torch.float16 if device in {"mps", "cuda"} else torch.float32


def sync(device: str) -> None:
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_sparse_head(path: Path, *, device: str, dtype: Any) -> tuple[Any, Any]:
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    weight = state["weight"]
    bias = state["bias"]
    if tuple(weight.shape) == (1, 1024):
        weight = weight[0]
    elif tuple(weight.shape) == (1024, 1):
        weight = weight[:, 0]
    elif tuple(weight.shape) != (1024,):
        raise ValueError(f"Unexpected sparse weight shape: {tuple(weight.shape)}")
    if tuple(bias.shape) == (1,):
        bias = bias[0]
    elif tuple(bias.shape) != ():
        raise ValueError(f"Unexpected sparse bias shape: {tuple(bias.shape)}")
    weight = weight.to(device=device, dtype=dtype)
    bias = bias.to(device=device, dtype=dtype)
    print(f"Sparse head weight: {tuple(weight.shape)}, {weight.dtype}")
    print(f"Sparse head bias:   {tuple(bias.shape)}, {bias.dtype}")
    return weight, bias


def encode_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    sparse_weight: Any,
    sparse_bias: Any,
    *,
    device: str,
    sparse_bias_sign: str,
) -> tuple[Any, Any]:
    import torch

    # Match the supplied MLX benchmark: encode each item independently and
    # form one dense tensor for the equal-length synthetic batch.
    tokens = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    if len({len(item) for item in tokens}) != 1:
        raise ValueError("Synthetic benchmark expects equal token lengths within each batch.")
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dense = torch.nn.functional.normalize(hidden[:, 0, :], p=2, dim=-1)
        sparse = torch.sum(hidden * sparse_weight, dim=-1)
        sparse = sparse + sparse_bias if sparse_bias_sign == "plus" else sparse - sparse_bias
        sparse = torch.relu(sparse)
    sync(device)
    return dense, sparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone Torch BGE-M3 vs MLX-style benchmark.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="HF model id or local Transformers checkpoint.")
    parser.add_argument("--sparse-head", type=Path, required=True, help="sparse_linear.safetensors or .pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flex_attention", "flash_attention_2"))
    parser.add_argument("--sparse-bias-sign", choices=("minus", "plus"), default="plus",
                        help="Plus matches the official torch.nn.Linear BGE-M3 sparse head semantics.")
    parser.add_argument("--words-per-chunk", type=int, default=WORDS_PER_CHUNK)
    parser.add_argument("--num-chunks", type=int, default=NUM_CHUNKS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print("Loading model...")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(args.model, **model_kwargs).eval().to(device)
    sparse_weight, sparse_bias = load_sparse_head(args.sparse_head, device=device, dtype=dtype)
    load_seconds = time.perf_counter() - started
    print(f"Model loaded in {load_seconds:.2f}s")
    print(f"Torch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Attention: {getattr(model.config, '_attn_implementation', None)}")

    print("Generating test data...")
    texts = make_chunks(args.words_per_chunk, args.num_chunks)
    print(f"Chunks: {len(texts)}")
    print(f"Approx words/chunk: {args.words_per_chunk}")
    print("Warmup...")
    warm_dense, warm_sparse = encode_batch(
        model, tokenizer, texts[:32], sparse_weight, sparse_bias,
        device=device, sparse_bias_sign=args.sparse_bias_sign,
    )
    print(f"Warmup dense shape: {tuple(warm_dense.shape)}")
    print(f"Warmup sparse shape: {tuple(warm_sparse.shape)}")
    del warm_dense, warm_sparse

    for batch_size in args.batch_sizes:
        print()
        print("=" * 60)
        print(f"Batch size: {batch_size}")
        sync(device)
        processed = 0
        started = time.perf_counter()
        for index in range(0, len(texts), batch_size):
            dense, sparse = encode_batch(
                model, tokenizer, texts[index:index + batch_size], sparse_weight, sparse_bias,
                device=device, sparse_bias_sign=args.sparse_bias_sign,
            )
            processed += len(texts[index:index + batch_size])
            del dense, sparse
        sync(device)
        elapsed = time.perf_counter() - started
        print(f"Processed: {processed}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Average: {elapsed / processed * 1000:.3f} ms/chunk")
        print(f"Throughput: {processed / elapsed:.1f} chunks/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
