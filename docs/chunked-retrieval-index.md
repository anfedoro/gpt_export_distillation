# Chunked Retrieval Index

The retrieval index is intentionally separated from the structural Markdown model.

```text
conversation
  -> message.raw_text
    -> structural block metadata and source ranges
      -> retrieval chunk text and source ranges
        -> dense/sparse representations
```

`messages.raw_text` remains the canonical source text. `blocks` store structural metadata, language, and source character ranges. `retrieval_chunks` store only chunk-local text, source ranges, token count, and a versioned chunk policy id. Conversation, project, role, block type, and source document metadata are inherited through joins.

Embeddings now use `owner_type = retrieval_chunk`. Existing `owner_type = knowledge_block` vectors are legacy block-level embeddings and must not be interpreted as valid chunk vectors.

## Chunk Policy

The current canonical policy is `canonical_token_chunks:v2`. The previous
`canonical_token_chunks:v1` policy remains a legacy identity and is not changed
retroactively.

The active dense and sparse providers expose their tokenizer and effective maximum sequence length. The indexer uses the strictest active provider limit, subtracts provider overhead and an explicit safety reserve, and builds tokenizer-aware chunks. In v1 the overlap was 15 percent of the content token budget.

In v2, overlap is not applied to chunks split on natural structural boundaries.
Prose prefers paragraph, sentence, punctuation, and whitespace boundaries. Code
prefers line boundaries. Lists and tables prefer item or row boundaries. Overlap
is applied only for forced tokenizer-window fallback splits where no safe
natural boundary is available. The fallback overlap is:

```text
floor(content_token_budget / 16)
```

For example, a content budget of 128 tokens uses 8 tokens of fallback overlap,
256 uses 16, and 512 uses 32. Audit output records
`chunks_with_overlap`, `overlap_token_count_total`,
`chunks_split_on_natural_boundary`, and `chunks_split_by_token_fallback`.

Before any embedding call, the final provider input is tokenized with truncation disabled. If it exceeds the provider limit, indexing fails with provider, chunk, block, source identity, actual token count, and allowed token count. Silent truncation is not accepted.

## Model Limits

Provider contracts expose model limits separately instead of flattening them
into one `max_length` value:

- tokenizer model limit;
- backbone `max_position_embeddings`;
- SentenceTransformer declared `max_seq_length`;
- configured effective max sequence length;
- special token overhead;
- safety reserve;
- computed retrieval chunk content budget.

The indexer does not automatically raise a SentenceTransformer declared limit to
the backbone architectural limit. A max sequence override must be explicit, is
stored in the embedding space identity, and should be treated as a distinct
index.

## Multilingual Dense Canary

Use the canary before a full rebuild when comparing dense multilingual models:

```bash
kb-model-canary \
  --work-dir benchmarks/model_canary \
  --models sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,BAAI/bge-m3 \
  --dense-device mps \
  --sparse-provider none \
  --batch-size 8 \
  --keep-databases
```

If `--input` is omitted, the command builds a small synthetic safe fixture. Use
`--input /path/to/distilled/export` only for a local private canary. The command
writes `report.json` and `report.md` under a timestamped run directory and
prints the resolved provider contract before indexing each model.

## Rebuild Path

Databases created with block-level embeddings should be rebuilt from the distilled Markdown archive:

```bash
kb-index import \
  --input /Users/anfedoro/Downloads/gpt-export-30.06.2026 \
  --db /Users/anfedoro/Downloads/gpt-export-30.06.2026/chat_memory.db \
  --dense-provider sentence-transformers \
  --sparse-provider sentence-transformers \
  --dense-device mps \
  --sparse-device mps \
  --dense-torch-dtype float16 \
  --sparse-torch-dtype float16 \
  --batch-size 32
```

Historical benchmark runs made on block-level embeddings remain useful as historical artifacts only. They are not valid for comparing embedding models after the chunked index change.
