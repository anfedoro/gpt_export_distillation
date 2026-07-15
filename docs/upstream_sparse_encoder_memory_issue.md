# SparseEncoder memory pressure with `convert_to_sparse_tensor=True`

## Summary

We observed strong CPU RSS growth when using `sentence_transformers.SparseEncoder.encode_document()` with:

- `convert_to_tensor=True`
- `convert_to_sparse_tensor=True`
- `device="mps"`
- `torch_dtype=float32`
- model: `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1`

In our application, changing only `convert_to_sparse_tensor` to `False` and then calling `decode()` on the returned tensor preserved the sparse term output count, while reducing peak RSS substantially.

This draft is intentionally diagnostic, not accusatory. The most likely issue is not that the sparse path briefly allocates memory, but that the memory pressure appears to persist after a batch and does not return to the lower steady-state level we observe with the dense-tensor path. The most likely issue is either:

- the sparse COO conversion path inside `encode()` / `encode_document()`
- or a memory-retention pattern triggered by that path on MPS / Apple Silicon

## Environment

- macOS: `26.5.2` (`ProductVersion`, build `25F84`)
- Hardware: Apple Silicon (`arm64`), user machine is an `M4 Max`
- Python: `3.13.13`
- `torch`: `2.12.1`
- `transformers`: `5.13.0`
- `sentence-transformers`: `5.6.0`
- device: `mps`
- dtype: `float32`
- model: `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1`

## Relevant upstream code

The public API currently routes `encode_document()` to `encode()` and forwards `convert_to_sparse_tensor` unchanged:

- `sentence_transformers/sparse_encoder/model.py:272`
- `sentence_transformers/sparse_encoder/model.py:348`

The dense/sparse conversion branch is here:

- `sentence_transformers/sparse_encoder/model.py:526-540`

Relevant lines:

```python
features = self.preprocess(inputs_batch, prompt=prompt, **kwargs)
features = batch_to_device(features, device)

with torch.inference_mode():
    embeddings = self.forward(features, **forward_kwargs)["sentence_embedding"]

    if max_active_dims is not None:
        embeddings = select_max_active_dims(embeddings, max_active_dims=max_active_dims)

if convert_to_sparse_tensor:
    embeddings = embeddings.to_sparse()
if save_to_cpu:
    embeddings = embeddings.cpu()
```

`decode()` explicitly supports dense tensors by converting them to sparse internally:

- `sentence_transformers/sparse_encoder/model.py:1213-1248`

Relevant lines:

```python
if not embeddings.is_sparse:
    embeddings = embeddings.to_sparse()

embeddings = embeddings.coalesce()
indices = embeddings.indices()
values = embeddings.values()
```

This means `decode()` can accept dense `sentence_embedding` tensors.

## Minimal reproduction

This script uses synthetic texts only and compares two modes:

1. `convert_to_sparse_tensor=True`
2. `convert_to_sparse_tensor=False`

Both modes then call `model.decode(embeddings, top_k=128)`.

```python
import gc
import json
import math
import os
import resource
import subprocess
import sys
import time

import torch
from sentence_transformers import SparseEncoder

MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
TEXTS = [
    f"Document {i}: synthetic text about memory pressure and sparse encoding."
    for i in range(3000)
]
BATCH_SIZE = 16


def rss_mb() -> float:
    out = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        text=True,
    ).strip()
    return float(out) / 1024


def cleanup() -> None:
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(mode: str, device: str) -> dict[str, object]:
    model = SparseEncoder(
        MODEL,
        device=device,
        model_kwargs={"torch_dtype": torch.float32},
    )
    peak = rss_mb()
    total_terms = 0
    started = time.time()

    for start in range(0, len(TEXTS), BATCH_SIZE):
        batch = TEXTS[start : start + BATCH_SIZE]
        with torch.inference_mode():
            if mode == "sparse_tensor":
                emb = model.encode_document(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=True,
                )
            else:
                emb = model.encode_document(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=False,
                )
            decoded = model.decode(emb, top_k=128)

        total_terms += sum(len(x) for x in decoded)
        before = rss_mb()
        del emb, decoded, batch
        cleanup()
        after = rss_mb()
        peak = max(peak, before, after)

    return {
        "mode": mode,
        "device": device,
        "peak_rss_mb": peak,
        "final_rss_mb": rss_mb(),
        "total_terms": total_terms,
        "seconds": round(time.time() - started, 2),
    }


results = []
for mode in ["sparse_tensor", "dense_tensor"]:
    results.append(run(mode, "mps"))
print(json.dumps(results, indent=2))
```

## Reproduction results

Synthetic texts, `mps`, `float32`, `300` docs, batch size `16`:

```json
[
  {
    "mode": "sparse_tensor",
    "device": "mps",
    "peak_rss_mb": 1370.3125,
    "final_rss_mb": 1370.3125,
    "total_terms": 38400,
    "seconds": 4.4
  },
  {
    "mode": "dense_tensor",
    "device": "mps",
    "peak_rss_mb": 2081.296875,
    "final_rss_mb": 1435.125,
    "total_terms": 38400,
    "seconds": 4.18
  }
]
```

Application experiment on real data, no SQLite writes, streaming JSONL only:

- real corpus: `3000` `knowledge_blocks`
- SQL writes: disabled
- output: streaming JSONL
- total sparse terms: `381774`
- output term count was identical between both modes

Observed peak RSS:

- `convert_to_sparse_tensor=True`: about `6818 MB`
- `convert_to_sparse_tensor=False`: about `1836 MB`

Same observed `sparse_terms`:

- `381774` in both modes

Short application log excerpt for the high-RSS path:

```text
batch=100/188 processed=1600/3000 rss_mb=4149.4 ...
batch=180/188 processed=2880/3000 rss_mb=6575.8 ...
batch=188/188 processed=3000/3000 rss_mb=6818.0 ...
```

Short application log excerpt for the lower-RSS path:

```text
batch=100/188 processed=1600/3000 rss_mb=1379.1 ...
batch=188/188 processed=3000/3000 rss_mb=1836.2 ...
```

## Observed vs expected

Observed:

- `convert_to_sparse_tensor=True` causes several-GB RSS growth on MPS / Apple Silicon during large batch document embedding.
- `gc.collect()` and `torch.mps.empty_cache()` do not reduce RSS after a batch.
- `mps_current_mb` remains stable in our measurements, so the growth appears to be CPU-side RSS rather than active MPS tensor memory.
- `decode()` works correctly on the dense tensor path.
- In our no-SQL application experiment, RSS did not fall back to a low baseline after cleanup; it climbed batch after batch.

Expected:

- If `decode()` supports dense tensor input, the documentation or API should clarify that `convert_to_sparse_tensor=False` is safe and may be preferable for decode-heavy workflows.
- Alternatively, the sparse conversion path should not leave the process at a much higher RSS steady state than the dense path after cleanup.

## Workaround used in our project

We changed our sparse embedding integration to call:

```python
encode_document(..., convert_to_tensor=True, convert_to_sparse_tensor=False)
```

and then call `decode()` on the dense tensor.

This kept the sparse term output unchanged in our test runs, while reducing peak RSS substantially.

## Questions for maintainers

1. Is `convert_to_sparse_tensor=True` expected to materialize a sparse COO tensor with significantly higher transient memory usage than the dense path?
2. Is `decode()` intentionally supported on dense tensors as a first-class workflow?
3. If dense decode is supported, should the docs recommend `convert_to_sparse_tensor=False` for large batch decode-heavy pipelines?
4. Is there a known interaction between `SparseEncoder.encode_document(..., convert_to_sparse_tensor=True)` and MPS / Apple Silicon memory pressure?

## Raw log excerpts

```text
SparseEncoder.encode_document(..., convert_to_sparse_tensor=True)
peak_rss_mb: 6818.0
final_rss_mb: 6818.0
sparse_terms: 381774
```

```text
SparseEncoder.encode_document(..., convert_to_sparse_tensor=False)
peak_rss_mb: 1836.2
final_rss_mb: 1836.2
sparse_terms: 381774
```
