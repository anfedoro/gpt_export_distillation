"""Read-only MCP-facing assembly over the clean native retrieval runtime."""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.embeddings.bge_m3_provider import embed_joint_documents
from kb.index.chunk_builder import build_chunk_policy
from kb.storage.native_pre_mvp import NativePreMvpError, NativePreMvpRetriever, _chunked_space


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _trim(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)].rstrip() + "…"


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[\w.-]{3,}", value.lower()) if not term.isdigit()][:24]


@dataclass(frozen=True)
class ArchiveConfig:
    db_path: Path
    candidate_pool: int = 500
    default_output_tokens: int = 1800
    max_output_tokens: int = 6000
    max_conversations: int = 6
    max_messages_per_conversation: int = 3


class ArchiveSession:
    """One process-lifetime native retriever and model pair.

    Calls are serialized because sqlite-vec and the compact sparse arrays are
    process resources with no useful per-request mutation. This makes a shared
    session safe without silently opening a legacy fallback connection.
    """

    def __init__(self, config: ArchiveConfig, dense: Any, sparse: Any) -> None:
        self.config = config
        self.dense = dense
        self.sparse = sparse
        self.lock = threading.RLock()
        self.retriever = NativePreMvpRetriever(config.db_path)
        policy = build_chunk_policy([dense, sparse])
        if self.retriever.dense_model != dense.model_name or self.retriever.dense_space != _chunked_space(dense.embedding_space_id, policy.id):
            self.close()
            raise NativePreMvpError("Dense provider is incompatible with the clean native DB embedding space.")
        if self.retriever.sparse.model_name != sparse.model_name or self.retriever.sparse.embedding_space_id != _chunked_space(sparse.embedding_space_id, policy.id):
            self.close()
            raise NativePreMvpError("Sparse provider is incompatible with the clean native DB embedding space.")
        self.calls = 0

    def close(self) -> None:
        self.retriever.close()

    def search(self, query: str, *, limit: int, mode: str = "hybrid", timeout_ms: int | None = None) -> list[dict[str, Any]]:
        if mode != "hybrid":
            raise ValueError("Only retrieval_mode='hybrid' is available in the clean native pre-MVP.")
        started = time.monotonic()
        with self.lock:
            dense_rows, sparse_rows = embed_joint_documents(self.dense, self.sparse, [query])
            dense, sparse = dense_rows[0], sparse_rows[0]
            if timeout_ms and (time.monotonic() - started) * 1000 > timeout_ms:
                raise TimeoutError("Query encoding exceeded timeout_ms.")
            hits = self.retriever.search(
                query_dense=dense, query_sparse=sparse, limit=max(limit, self.config.candidate_pool),
                dense_candidate_k=self.config.candidate_pool, sparse_candidate_k=self.config.candidate_pool,
            )
            self.calls += 1
        return [
            {
                **hit.provenance,
                "chunk_id": hit.chunk_id,
                "scores": {"dense": round(hit.dense_score, 6), "sparse": round(hit.sparse_score, 6), "fused": round(hit.final_score, 6)},
                "reason": _reason(hit.dense_score, hit.sparse_score),
            }
            for hit in hits
        ]

    def search_archive(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _required_text(arguments, "query")
        limit = _bounded_int(arguments.get("limit", 8), "limit", 1, 30)
        neighbors = _bounded_int(arguments.get("include_neighbors", 1), "include_neighbors", 0, 4)
        budget = self._budget(arguments)
        rows = self.search(query, limit=max(limit * 4, 20), mode=str(arguments.get("retrieval_mode", "hybrid")), timeout_ms=arguments.get("timeout_ms"))
        rows = _filter(rows, arguments)
        selected = _dedupe_messages(rows, limit=limit, max_per_conversation=self.config.max_messages_per_conversation)
        items = self._assemble(selected, neighbors=neighbors, budget=budget)
        return _payload("focused", items, budget, self, warnings=[] if items else ["No relevant archive memory found."])

    def construct_archive_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        context = _required_text(arguments, "current_context")
        budget = self._budget(arguments)
        queries = [context]
        terms = " ".join(_terms(context)[:8])
        if terms and terms != context:
            queries.append(terms)
        if arguments.get("include_decisions", True):
            queries.append(f"{terms} decision decision accepted final")
        if arguments.get("include_preferences", True):
            queries.append(f"{terms} preference prefer avoid")
        all_rows: list[dict[str, Any]] = []
        for query in dict.fromkeys(queries):
            all_rows.extend(self.search(query, limit=30, timeout_ms=arguments.get("timeout_ms")))
        all_rows = _filter(all_rows, {"project": arguments.get("project_hint"), "date_from": _range(arguments, 0), "date_to": _range(arguments, 1)})
        selected = _diverse_messages(all_rows, limit=self.config.max_conversations * self.config.max_messages_per_conversation,
                                     max_conversations=self.config.max_conversations, max_per_conversation=self.config.max_messages_per_conversation)
        items = self._assemble(selected, neighbors=1 if arguments.get("include_recent_related", True) else 0, budget=budget)
        warnings: list[str] = []
        if not items:
            warnings.append("No relevant archive memory found; do not infer personal history from this result.")
        return _payload("broad", items, budget, self, warnings=warnings)

    def _budget(self, arguments: dict[str, Any]) -> int:
        value = arguments.get("max_tokens", arguments.get("max_chars", self.config.default_output_tokens))
        if "max_chars" in arguments and "max_tokens" not in arguments:
            value = max(1, int(value) // 4)
        return _bounded_int(value, "output budget", 100, self.config.max_output_tokens)

    def _assemble(self, rows: list[dict[str, Any]], *, neighbors: int, budget: int) -> list[dict[str, Any]]:
        windows: dict[str, tuple[int, int]] = {}
        for row in rows:
            ordinal = int(row["message_ordinal"])
            conversation = str(row["conversation_id"])
            prior = windows.get(conversation, (ordinal, ordinal))
            windows[conversation] = (min(prior[0], ordinal - neighbors), max(prior[1], ordinal + neighbors))
        raw_windows = self.retriever.messages_for_windows(windows)
        items: list[dict[str, Any]] = []
        used = 0
        for row in rows:
            text = str(row.get("text") or "")
            available_chars = max(160, (budget - used) * 4)
            text = _trim(text, available_chars)
            item_tokens = _estimate_tokens(text)
            if items and used + item_tokens > budget:
                continue
            conversation_messages = raw_windows.get(str(row["conversation_id"]), [])
            ordinal = int(row["message_ordinal"])
            before = [_message_view(value) for value in conversation_messages if int(value["ordinal"]) < ordinal]
            after = [_message_view(value) for value in conversation_messages if int(value["ordinal"]) > ordinal]
            items.append({
                "conversation_id": row["conversation_id"], "message_id": row["message_id"], "source_message_id": row.get("source_message_id"),
                "title": row.get("conversation_title"), "timestamp": row.get("time_utc") or row.get("update_time_utc"), "project": row.get("project_id"),
                "role": row["role"], "text": text, "context_before": before, "context_after": after,
                "scores": row["scores"], "reason": row["reason"],
                "provenance": {"chunk_id": row["chunk_id"], "block_id": row["block_id"], "source_path": row["source_path"],
                               "message_ordinal": ordinal, "source_char_start": row["source_char_start"], "source_char_end": row["source_char_end"]},
            })
            used += item_tokens
        return items


def _payload(mode: str, items: list[dict[str, Any]], budget: int, session: ArchiveSession, *, warnings: list[str]) -> dict[str, Any]:
    return {"schema_version": "kb.mcp.memory.v1", "mode": mode, "summary": _summary(items), "items": items,
            "coverage": {"conversation_count": len({item["conversation_id"] for item in items}), "item_count": len(items),
                         "estimated_tokens": sum(_estimate_tokens(item["text"]) for item in items), "budget_tokens": budget},
            "warnings": warnings, "runtime": {"candidate_pool": session.config.candidate_pool, "session_calls": session.calls,
                                                   "sparse_materialized_once": True}}


def _summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No supported archive context was found."
    refs = ", ".join(filter(None, [str(item.get("title") or item["conversation_id"]) for item in items[:3]]))
    return f"Retrieved {len(items)} traceable message excerpts from {len({item['conversation_id'] for item in items})} conversation(s): {refs}."


def _message_view(row: dict[str, Any]) -> dict[str, Any]:
    return {"message_id": row["id"], "role": row["role"], "ordinal": row["ordinal"], "timestamp": row["time_utc"], "text": _trim(str(row["raw_text"]), 800)}


def _reason(dense: float, sparse: float) -> str:
    if dense > 0 and sparse > 0:
        return "Matched by both semantic and lexical retrieval."
    return "Matched by semantic retrieval." if dense > 0 else "Matched by lexical retrieval."


def _required_text(arguments: dict[str, Any], name: str) -> str:
    value = str(arguments.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def _bounded_int(value: Any, name: str, low: int, high: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return result


def _range(arguments: dict[str, Any], index: int) -> Any:
    value = arguments.get("time_range")
    return value[index] if isinstance(value, list) and len(value) == 2 else None


def _filter(rows: list[dict[str, Any]], arguments: dict[str, Any]) -> list[dict[str, Any]]:
    project = arguments.get("project") or arguments.get("project_hint")
    conversation_id = arguments.get("conversation_id")
    roles = set(arguments.get("roles") or [])
    date_from, date_to = arguments.get("date_from"), arguments.get("date_to")
    if roles - {"user", "assistant", "system", "tool"}:
        raise ValueError("roles contains an unsupported role.")
    result = []
    for row in rows:
        stamp = str(row.get("time_utc") or row.get("update_time_utc") or "")
        if project and str(project).lower() not in str(row.get("project_id") or "").lower():
            continue
        if conversation_id and conversation_id not in {row.get("conversation_id"), row.get("dialogue_id")}:
            continue
        if roles and row.get("role") not in roles:
            continue
        if date_from and stamp and stamp < str(date_from):
            continue
        if date_to and stamp and stamp > str(date_to):
            continue
        result.append(row)
    return result


def _dedupe_messages(rows: list[dict[str, Any]], *, limit: int, max_per_conversation: int) -> list[dict[str, Any]]:
    return _diverse_messages(rows, limit=limit, max_conversations=limit, max_per_conversation=max_per_conversation)


def _diverse_messages(rows: list[dict[str, Any]], *, limit: int, max_conversations: int, max_per_conversation: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row["message_id"])
        if identifier not in best or row["scores"]["fused"] > best[identifier]["scores"]["fused"]:
            best[identifier] = row
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(best.values(), key=lambda item: (-item["scores"]["fused"], str(item["message_id"]))):
        buckets[str(row["conversation_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for conversation in sorted(buckets, key=lambda key: -buckets[key][0]["scores"]["fused"])[:max_conversations]:
        selected.extend(buckets[conversation][:max_per_conversation])
    return sorted(selected, key=lambda item: (-item["scores"]["fused"], str(item["message_id"])))[:limit]
