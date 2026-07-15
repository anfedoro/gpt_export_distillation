"""Deterministic compact assembly for focused archive retrieval results.

This module deliberately works only on the bounded fused candidate list.  It
does not query embeddings, alter retrieval scores, or inspect the database.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Iterable


MAX_GROUPS_PER_CONVERSATION = 2
MAX_POST_RETRIEVAL_CANDIDATES = 180
_INTRODUCTIONS = {"bgp", "yes", "no", "correct", "right", "верно", "да", "нет", "вот это и есть"}


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def trim(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)].rstrip() + "…"


def meaningful_terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[\w.-]{3,}", value.lower()) if not term.isdigit()}


def lexical_similarity(left: str, right: str) -> float:
    left_terms, right_terms = meaningful_terms(left), meaningful_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def information_score(text: str) -> float:
    """Return a small deterministic information signal in the 0..1 range."""
    stripped = text.strip()
    terms = meaningful_terms(stripped)
    if not stripped:
        return 0.0
    if stripped.lower().rstrip(":.!?…") in _INTRODUCTIONS:
        return 0.0
    if len(stripped) < 24 or len(terms) < 3:
        return 0.15
    punctuation_ratio = sum(char in ":;,.!?-—()[]{}" for char in stripped) / max(1, len(stripped))
    return min(1.0, 0.35 + min(len(terms), 14) / 20 - max(0.0, punctuation_ratio - 0.35))


def rerank_hits(rows: Iterable[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Apply a bounded, fused-score-primary intent-aware rerank.

    The bounded bonuses cannot reverse a material fused-score lead.  They only
    make close scores prefer complete query terms and informative evidence.
    """
    query_terms = meaningful_terms(query)
    ranked: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        text = str(item.get("text") or "")
        overlap = len(query_terms & meaningful_terms(text)) / max(1, len(query_terms))
        info = information_score(text)
        fused = float(item.get("scores", {}).get("fused", 0.0))
        short_penalty = 0.018 if info <= 0.15 else 0.0
        item["_rerank_score"] = fused + 0.035 * overlap + 0.012 * info - short_penalty
        item["_information_score"] = info
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-float(item["_rerank_score"]), -float(item.get("scores", {}).get("fused", 0.0)), str(item.get("chunk_id", ""))))


def _ranges_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("block_id")) != str(right.get("block_id")):
        return False
    left_start, left_end = int(left.get("source_char_start", 0)), int(left.get("source_char_end", 0))
    right_start, right_end = int(right.get("source_char_start", 0)), int(right.get("source_char_end", 0))
    return left_start <= right_end and right_start <= left_end


def _same_evidence_group(group: dict[str, Any], row: dict[str, Any]) -> bool:
    representative = group["representative"]
    if str(representative.get("conversation_id")) != str(row.get("conversation_id")):
        return False
    if str(representative.get("message_id")) == str(row.get("message_id")):
        return True
    if _ranges_overlap(representative, row):
        return True
    return lexical_similarity(str(representative.get("text") or ""), str(row.get("text") or "")) >= 0.86


def group_overlapping_hits(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse same-message, overlapping-range, and near-identical evidence."""
    groups: list[dict[str, Any]] = []
    dropped = 0
    seen_chunk_ids: set[str] = set()
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id and chunk_id in seen_chunk_ids:
            dropped += 1
            continue
        seen_chunk_ids.add(chunk_id)
        matching = next((group for group in groups if _same_evidence_group(group, row)), None)
        if matching is None:
            groups.append({
                "representative": row,
                "rows": [row],
                "contributing_chunk_ids": [chunk_id] if chunk_id else [],
                "block_ids": [str(row.get("block_id"))] if row.get("block_id") is not None else [],
                "message_ordinal_from": int(row.get("message_ordinal", 0)),
                "message_ordinal_to": int(row.get("message_ordinal", 0)),
            })
            continue
        matching["rows"].append(row)
        if chunk_id:
            matching["contributing_chunk_ids"].append(chunk_id)
        block_id = str(row.get("block_id"))
        if row.get("block_id") is not None and block_id not in matching["block_ids"]:
            matching["block_ids"].append(block_id)
        ordinal = int(row.get("message_ordinal", 0))
        matching["message_ordinal_from"] = min(matching["message_ordinal_from"], ordinal)
        matching["message_ordinal_to"] = max(matching["message_ordinal_to"], ordinal)
        if float(row["_rerank_score"]) > float(matching["representative"]["_rerank_score"]):
            matching["representative"] = row
        dropped += 1
    return groups, dropped


def filter_low_information_groups(
    groups: Iterable[dict[str, Any]],
    context_text: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Drop weak anchors unless immediate context supplies distinct evidence."""
    retained: list[dict[str, Any]] = []
    dropped = 0
    for group in groups:
        row = group["representative"]
        info = float(row.get("_information_score", information_score(str(row.get("text") or ""))))
        nearby = context_text(group) if context_text else str(row.get("context_text") or "")
        useful_context = information_score(nearby) >= 0.45 and lexical_similarity(nearby, str(row.get("text") or "")) < 0.92
        strong_score = float(row.get("scores", {}).get("fused", 0.0)) >= 0.90
        if info <= 0.15 and not useful_context and not strong_score:
            dropped += 1
            continue
        retained.append(group)
    return retained, dropped


def diversify_groups(groups: Iterable[dict[str, Any]], *, limit: int, max_per_conversation: int = MAX_GROUPS_PER_CONVERSATION) -> list[dict[str, Any]]:
    """Greedy MMR-like selection over the small final candidate set."""
    remaining = list(groups)
    selected: list[dict[str, Any]] = []
    conversation_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        scored: list[tuple[float, dict[str, Any]]] = []
        for group in remaining:
            row = group["representative"]
            conversation = str(row.get("conversation_id"))
            if conversation_counts[conversation] >= max_per_conversation:
                continue
            duplicate_penalty = max((lexical_similarity(str(row.get("text") or ""), str(chosen["representative"].get("text") or "")) for chosen in selected), default=0.0)
            same_conversation_penalty = 0.06 * conversation_counts[conversation]
            project_bonus = 0.012 if selected and all(str(chosen["representative"].get("project_id")) != str(row.get("project_id")) for chosen in selected) else 0.0
            adjusted = float(row["_rerank_score"]) - 0.10 * duplicate_penalty - same_conversation_penalty + project_bonus
            scored.append((adjusted, group))
        if not scored:
            break
        _, winner = max(scored, key=lambda pair: (pair[0], float(pair[1]["representative"].get("scores", {}).get("fused", 0.0)), str(pair[1]["representative"].get("chunk_id", ""))))
        selected.append(winner)
        conversation_counts[str(winner["representative"].get("conversation_id"))] += 1
        remaining.remove(winner)
    return selected


def build_compact_items(
    groups: Iterable[dict[str, Any]],
    *,
    neighbors: int,
    budget_tokens: int,
    messages_by_conversation: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], bool]:
    """Assemble excerpts and unique, bounded neighbour messages for selected groups."""
    groups = list(groups)
    total_chars = budget_tokens * 4
    per_item_chars = max(160, total_chars // max(1, len(groups)))
    used_chars = 0
    used_neighbors: set[str] = set()
    items: list[dict[str, Any]] = []
    limited = False
    for group in groups:
        row = group["representative"]
        remaining = total_chars - used_chars
        if items and remaining < 120:
            limited = True
            break
        item_char_cap = min(per_item_chars, remaining)
        excerpt_budget = min(item_char_cap, max(100, int(item_char_cap * 0.65)))
        excerpt = trim(str(row.get("text") or ""), excerpt_budget)
        ordinal = int(row.get("message_ordinal", 0))
        conversation = str(row.get("conversation_id"))
        context_limit = max(0, item_char_cap - len(excerpt))
        context: list[dict[str, Any]] = []
        if neighbors:
            candidates = sorted(messages_by_conversation.get(conversation, []), key=lambda item: (abs(int(item["ordinal"]) - ordinal), int(item["ordinal"])))
            for message in candidates:
                message_id = str(message["id"])
                if message_id == str(row.get("message_id")) or message_id in used_neighbors:
                    continue
                if abs(int(message["ordinal"]) - ordinal) > neighbors:
                    continue
                remaining_context = context_limit - sum(len(item["text"]) for item in context)
                if remaining_context < 80:
                    break
                context.append({
                    "message_id": message["id"], "role": message["role"], "ordinal": message["ordinal"],
                    "timestamp": message["time_utc"], "text": trim(str(message["raw_text"]), min(320, remaining_context)),
                })
                used_neighbors.add(message_id)
        before = [item for item in context if int(item["ordinal"]) < ordinal]
        after = [item for item in context if int(item["ordinal"]) > ordinal]
        used_chars += len(excerpt) + sum(len(item["text"]) for item in context)
        items.append({
            "conversation_id": row.get("conversation_id"), "message_id": row.get("message_id"), "source_message_id": row.get("source_message_id"),
            "title": row.get("conversation_title"), "timestamp": row.get("time_utc") or row.get("update_time_utc"),
            "project": row.get("project_id"), "role": row.get("role"), "excerpt": excerpt, "text": excerpt,
            "supporting_context": context, "context_before": before, "context_after": after,
            "scores": row.get("scores", {}), "reason": row.get("reason"),
            "provenance": {
                "chunk_id": row.get("chunk_id"), "representative_chunk_id": row.get("chunk_id"),
                "contributing_chunk_ids": group["contributing_chunk_ids"], "block_id": row.get("block_id"),
                "block_ids": group["block_ids"], "source_path": row.get("source_path"),
                "message_ordinal": ordinal, "message_ordinal_from": group["message_ordinal_from"],
                "message_ordinal_to": group["message_ordinal_to"], "source_char_start": row.get("source_char_start"),
                "source_char_end": row.get("source_char_end"),
            },
        })
    return items, limited
