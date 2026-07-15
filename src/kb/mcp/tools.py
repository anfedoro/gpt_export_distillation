"""Canonical MCP archive tool names and schemas shared by all transports."""

from __future__ import annotations

from typing import Any

TOOL_NAMES = ("construct_archive_context", "search_archive")


def archive_tools() -> list[dict[str, Any]]:
    return [construct_archive_context_tool(), search_archive_tool()]


def construct_archive_context_tool() -> dict[str, Any]:
    return {
        "name": "construct_archive_context",
        "description": "Use only when the user request may depend on their prior ChatGPT archive: a previous decision, project continuation, preference, old discussion, historical comparison, or a named entity likely discussed before. Do not use for general questions, when this conversation is sufficient, or when the user asks not to access the archive. Never claim that the archive contains something before calling this tool. Returns a compact, provenance-linked broad context package, not a raw archive dump.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "current_context": {"type": "string"},
                "max_tokens": {"type": "integer", "minimum": 100, "maximum": 6000},
                "max_chars": {"type": "integer", "minimum": 400},
                "project_hint": {"type": "string"},
                "time_range": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                "include_preferences": {"type": "boolean", "default": True},
                "include_decisions": {"type": "boolean", "default": True},
                "include_recent_related": {"type": "boolean", "default": True},
                "timeout_ms": {"type": "integer", "minimum": 1},
            },
            "required": ["current_context"],
            "additionalProperties": False,
        },
    }


def search_archive_tool() -> dict[str, Any]:
    return {
        "name": "search_archive",
        "description": "Use for a focused factual lookup in the user's prior ChatGPT archive: find a specific decision, project, person, issue, conversation, or exact historical evidence. Do not use for broad background synthesis; use construct_archive_context instead. Do not claim archive contents before this tool returns. Results contain bounded excerpts, optional chronological neighbours, scores, and source IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
                "project": {"type": "string"},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "roles": {"type": "array", "items": {"type": "string", "enum": ["user", "assistant", "system", "tool"]}},
                "conversation_id": {"type": "string"},
                "retrieval_mode": {"type": "string", "enum": ["hybrid"], "default": "hybrid"},
                "include_neighbors": {"type": "integer", "minimum": 0, "maximum": 4, "default": 1},
                "max_tokens": {"type": "integer", "minimum": 100, "maximum": 6000},
                "max_chars": {"type": "integer", "minimum": 400},
                "timeout_ms": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
