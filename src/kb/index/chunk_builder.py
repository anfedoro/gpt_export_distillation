from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any

from kb.model.ids import stable_id


CHUNK_POLICY_NAME = "canonical_token_chunks"
CHUNK_POLICY_VERSION = "v1"
DEFAULT_OVERLAP_RATIO = 0.15
DEFAULT_SAFETY_RESERVE = 4


@dataclass(frozen=True)
class ChunkPolicy:
    id: str
    max_input_tokens: int
    content_token_budget: int
    overlap_tokens: int
    safety_reserve: int


@dataclass(frozen=True)
class RetrievalChunk:
    id: str
    block_id: str
    ordinal: int
    source_char_start: int
    source_char_end: int
    token_count: int
    text: str
    chunk_policy_id: str


def build_chunk_policy(providers: list[Any]) -> ChunkPolicy:
    limits = [provider.effective_max_sequence_length for provider in providers if provider is not None]
    if not limits or any(limit is None for limit in limits):
        raise RuntimeError("All active embedding providers must expose effective_max_sequence_length.")
    max_input_tokens = min(int(limit) for limit in limits)
    max_overhead = 0
    for provider in providers:
        if provider is None:
            continue
        overhead = provider.token_count(provider.embedding_input("")) if hasattr(provider, "token_count") else None
        if overhead is None:
            raise RuntimeError(f"Provider {provider.model_name} does not expose a reliable tokenizer.")
        max_overhead = max(max_overhead, int(overhead))
    content_budget = max_input_tokens - max_overhead - DEFAULT_SAFETY_RESERVE
    if content_budget <= 0:
        raise ValueError(
            f"Embedding providers leave no chunk content budget: limit={max_input_tokens} overhead={max_overhead}"
        )
    overlap_tokens = max(1, int(math.floor(content_budget * DEFAULT_OVERLAP_RATIO)))
    policy_id = (
        f"{CHUNK_POLICY_NAME}:{CHUNK_POLICY_VERSION};limit={max_input_tokens};"
        f"content={content_budget};overlap={overlap_tokens};reserve={DEFAULT_SAFETY_RESERVE}"
    )
    return ChunkPolicy(
        id=policy_id,
        max_input_tokens=max_input_tokens,
        content_token_budget=content_budget,
        overlap_tokens=overlap_tokens,
        safety_reserve=DEFAULT_SAFETY_RESERVE,
    )


def build_retrieval_chunks(
    *,
    block_id: str,
    block_text: str,
    block_char_start: int,
    policy: ChunkPolicy,
    tokenizer_provider: Any,
) -> list[RetrievalChunk]:
    if not block_text:
        return []
    chunks: list[RetrievalChunk] = []
    local_start = 0
    text_len = len(block_text)
    ordinal = 1
    while local_start < text_len:
        local_end = _max_end_for_budget(block_text, local_start, text_len, policy.content_token_budget, tokenizer_provider)
        if local_end <= local_start:
            raise ValueError(f"Unable to create a fitting retrieval chunk for block {block_id} at char {local_start}.")
        chunk_text = block_text[local_start:local_end]
        token_count = tokenizer_provider.token_count(tokenizer_provider.embedding_input(chunk_text))
        chunks.append(
            RetrievalChunk(
                id=stable_id(block_id, policy.id, ordinal, local_start, local_end, prefix="chunk"),
                block_id=block_id,
                ordinal=ordinal,
                source_char_start=block_char_start + local_start,
                source_char_end=block_char_start + local_end,
                token_count=token_count,
                text=chunk_text,
                chunk_policy_id=policy.id,
            )
        )
        if local_end >= text_len:
            break
        next_start = _overlap_start(block_text, local_start, local_end, policy.overlap_tokens, tokenizer_provider)
        if next_start <= local_start:
            next_start = local_end
        local_start = next_start
        ordinal += 1
    return chunks


def audit_chunks(block_rows: list[dict[str, Any]], chunks: list[RetrievalChunk], policy: ChunkPolicy) -> dict[str, Any]:
    chunks_by_block: dict[str, list[RetrievalChunk]] = {}
    for chunk in chunks:
        chunks_by_block.setdefault(chunk.block_id, []).append(chunk)
    total_source_chars = 0
    covered_unique_chars = 0
    blocks_with_gaps = 0
    token_counts = [chunk.token_count for chunk in chunks]
    for block in block_rows:
        start = int(block["char_start"])
        end = int(block["char_end"])
        total_source_chars += max(0, end - start)
        ranges = sorted((chunk.source_char_start, chunk.source_char_end) for chunk in chunks_by_block.get(str(block["id"]), []))
        covered = _covered_length(ranges)
        covered_unique_chars += covered
        if covered != max(0, end - start):
            blocks_with_gaps += 1
    over_limit = sum(1 for count in token_counts if count > policy.max_input_tokens)
    return {
        "chunk_policy_id": policy.id,
        "total_source_characters": total_source_chars,
        "total_indexable_characters": total_source_chars,
        "covered_unique_characters": covered_unique_chars,
        "uncovered_characters": max(0, total_source_chars - covered_unique_chars),
        "total_retrieval_chunks": len(chunks),
        "maximum_chunk_token_count": max(token_counts) if token_counts else 0,
        "p50_chunk_token_count": _percentile(token_counts, 50),
        "p95_chunk_token_count": _percentile(token_counts, 95),
        "p99_chunk_token_count": _percentile(token_counts, 99),
        "chunks_over_limit": over_limit,
        "truncated_chunks": 0,
        "blocks_with_coverage_gaps": blocks_with_gaps,
    }


def _max_end_for_budget(text: str, start: int, max_end: int, budget: int, provider: Any) -> int:
    lo = start + 1
    hi = max_end
    best = start
    while lo <= hi:
        mid = (lo + hi) // 2
        count = provider.token_count(provider.embedding_input(text[start:mid]))
        if count <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    snapped = _snap_end(text, start, best, max_end)
    if snapped > start and provider.token_count(provider.embedding_input(text[start:snapped])) <= budget:
        return snapped
    return best


def _snap_end(text: str, start: int, proposed: int, max_end: int) -> int:
    if proposed >= max_end:
        return max_end
    window = text[start:proposed]
    for sep in ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "):
        idx = window.rfind(sep)
        if idx > 0 and proposed - (start + idx + len(sep)) < 256:
            return start + idx + len(sep)
    return proposed


def _overlap_start(text: str, chunk_start: int, chunk_end: int, overlap_tokens: int, provider: Any) -> int:
    lo = chunk_start
    hi = chunk_end
    best = chunk_end
    while lo <= hi:
        mid = (lo + hi) // 2
        count = provider.token_count(provider.embedding_input(text[mid:chunk_end]))
        if count >= overlap_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    while best < chunk_end and not text[best].isspace() and best > chunk_start:
        best -= 1
    return max(chunk_start, best)


def _covered_length(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
        return 0
    total = 0
    cur_start, cur_end = ranges[0]
    for start, end in ranges[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    if percentile == 50:
        return float(median(values))
    ordered = sorted(values)
    idx = math.ceil((percentile / 100) * len(ordered)) - 1
    return float(ordered[max(0, min(idx, len(ordered) - 1))])
