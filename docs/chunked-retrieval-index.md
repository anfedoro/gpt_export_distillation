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

The first canonical policy is `canonical_token_chunks:v1`.

The active dense and sparse providers expose their tokenizer and effective maximum sequence length. The indexer uses the strictest active provider limit, subtracts provider overhead and an explicit safety reserve, and applies token-aware overlap. The default overlap is 15 percent of the content token budget.

Before any embedding call, the final provider input is tokenized with truncation disabled. If it exceeds the provider limit, indexing fails with provider, chunk, block, source identity, actual token count, and allowed token count. Silent truncation is not accepted.

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
