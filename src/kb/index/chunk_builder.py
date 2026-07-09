from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any

from kb.model.ids import stable_id


CHUNK_POLICY_NAME = "canonical_token_chunks"
CHUNK_POLICY_VERSION_V1 = "v1"
CHUNK_POLICY_VERSION_V2 = "v2"
DEFAULT_OVERLAP_RATIO = 0.15
V2_FALLBACK_OVERLAP_DIVISOR = 16
DEFAULT_SAFETY_RESERVE = 4


@dataclass(frozen=True)
class ChunkPolicy:
    id: str
    max_input_tokens: int
    content_token_budget: int
    overlap_tokens: int
    safety_reserve: int
    version: str = CHUNK_POLICY_VERSION_V2


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
    split_reason: str = "natural_boundary"
    overlap_token_count: int = 0


def build_chunk_policy(
    providers: list[Any],
    *,
    version: str = CHUNK_POLICY_VERSION_V2,
    content_budget_override: int | None = None,
) -> ChunkPolicy:
    if version not in {CHUNK_POLICY_VERSION_V1, CHUNK_POLICY_VERSION_V2}:
        raise ValueError(f"Unsupported chunk policy version: {version}")
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
    content_budget = (
        content_budget_override
        if content_budget_override is not None
        else max_input_tokens - max_overhead - DEFAULT_SAFETY_RESERVE
    )
    if content_budget <= 0:
        raise ValueError(
            f"Embedding providers leave no chunk content budget: limit={max_input_tokens} overhead={max_overhead}"
        )
    if content_budget > max_input_tokens:
        raise ValueError(f"Chunk content budget {content_budget} exceeds provider limit {max_input_tokens}.")
    overlap_tokens = (
        max(1, int(math.floor(content_budget * DEFAULT_OVERLAP_RATIO)))
        if version == CHUNK_POLICY_VERSION_V1
        else max(1, int(math.floor(content_budget / V2_FALLBACK_OVERLAP_DIVISOR)))
    )
    policy_id = (
        f"{CHUNK_POLICY_NAME}:{version};limit={max_input_tokens};"
        f"content={content_budget};fallback_overlap={overlap_tokens};reserve={DEFAULT_SAFETY_RESERVE}"
    )
    return ChunkPolicy(
        id=policy_id,
        max_input_tokens=max_input_tokens,
        content_token_budget=content_budget,
        overlap_tokens=overlap_tokens,
        safety_reserve=DEFAULT_SAFETY_RESERVE,
        version=version,
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
        local_end, split_reason = _max_end_for_budget(
            block_text,
            local_start,
            text_len,
            policy.content_token_budget,
            tokenizer_provider,
            prefer_natural=policy.version == CHUNK_POLICY_VERSION_V2,
        )
        if local_end <= local_start:
            raise ValueError(f"Unable to create a fitting retrieval chunk for block {block_id} at char {local_start}.")
        chunk_text = block_text[local_start:local_end]
        token_count = tokenizer_provider.token_count(tokenizer_provider.embedding_input(chunk_text))
        overlap_token_count = 0
        if chunks and local_start < chunks[-1].source_char_end - block_char_start:
            overlap_text = block_text[local_start:chunks[-1].source_char_end - block_char_start]
            overlap_token_count = tokenizer_provider.token_count(tokenizer_provider.embedding_input(overlap_text))
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
                split_reason=split_reason,
                overlap_token_count=overlap_token_count,
            )
        )
        if local_end >= text_len:
            break
        if policy.version == CHUNK_POLICY_VERSION_V1 or split_reason == "token_window_fallback":
            next_start = _overlap_start(block_text, local_start, local_end, policy.overlap_tokens, tokenizer_provider)
        else:
            next_start = local_end
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
    chunks_with_overlap = sum(1 for chunk in chunks if chunk.overlap_token_count > 0)
    split_by_fallback = sum(1 for chunk in chunks if chunk.split_reason == "token_window_fallback")
    split_on_natural = sum(1 for chunk in chunks if chunk.split_reason != "token_window_fallback")
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
        "chunk_policy_version": policy.version,
        "chunk_policy_max_input_tokens": policy.max_input_tokens,
        "chunk_policy_content_token_budget": policy.content_token_budget,
        "chunk_policy_overlap_tokens": policy.overlap_tokens,
        "chunk_policy_safety_reserve": policy.safety_reserve,
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
        "chunks_with_overlap": chunks_with_overlap,
        "overlap_token_count_total": sum(chunk.overlap_token_count for chunk in chunks),
        "chunks_split_on_natural_boundary": split_on_natural,
        "chunks_split_by_token_fallback": split_by_fallback,
    }


def _max_end_for_budget(
    text: str,
    start: int,
    max_end: int,
    budget: int,
    provider: Any,
    *,
    prefer_natural: bool,
) -> tuple[int, str]:
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
    snapped, natural = _snap_end(text, start, best, max_end) if prefer_natural else (best, False)
    if snapped > start and provider.token_count(provider.embedding_input(text[start:snapped])) <= budget:
        if snapped >= max_end:
            return snapped, "complete"
        return snapped, "natural_boundary" if natural else "token_window_fallback"
    return best, "token_window_fallback" if best < max_end else "complete"


def _snap_end(text: str, start: int, proposed: int, max_end: int) -> tuple[int, bool]:
    if proposed >= max_end:
        return max_end, True
    window = text[start:proposed]
    for sep in ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "):
        idx = window.rfind(sep)
        if idx > 0 and proposed - (start + idx + len(sep)) < 256:
            return start + idx + len(sep), True
    return proposed, False


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
