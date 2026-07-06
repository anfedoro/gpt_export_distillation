from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.cli import _build_dense_provider, _build_sparse_provider
from kb.retrieval.context_pack import ContextPackOptions, build_context_pack


TOOL_NAME = "build_context_pack"
PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class ServerConfig:
    db_path: Path
    dense_provider: str = "sentence-transformers"
    sparse_provider: str = "sentence-transformers"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "naver/splade-cocondenser-ensembledistil"
    sparse_top_k: int = 128
    max_block_chars: int = 4000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local knowledge base MCP server over stdio.")
    parser.add_argument("--db", required=True, help="Path to a built chat_memory.db SQLite database.")
    parser.add_argument("--dense-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    parser.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sparse-model", default="naver/splade-cocondenser-ensembledistil")
    parser.add_argument("--sparse-top-k", type=int, default=128)
    parser.add_argument("--max-block-chars", type=int, default=4000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ServerConfig(
        db_path=Path(args.db).expanduser().resolve(),
        dense_provider=args.dense_provider,
        sparse_provider=args.sparse_provider,
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        sparse_top_k=args.sparse_top_k,
        max_block_chars=args.max_block_chars,
    )
    serve_stdio(config)


def serve_stdio(config: ServerConfig) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle_request(config, request)
        except Exception as exc:  # noqa: BLE001
            response = _error_response(None, -32603, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def handle_request(config: ServerConfig, request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        return _result_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": "gpt-export-distillation-kb", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        )
    if method == "ping":
        return _result_response(request_id, {})
    if method == "tools/list":
        return _result_response(request_id, {"tools": [_tool_description()]})
    if method == "tools/call":
        params = request.get("params") or {}
        if params.get("name") != TOOL_NAME:
            return _error_response(request_id, -32602, f"Unknown tool: {params.get('name')}")
        try:
            payload = call_build_context_pack(config, params.get("arguments") or {})
            return _result_response(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                        }
                    ],
                    "isError": False,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _result_response(
                request_id,
                {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True,
                },
            )
    return _error_response(request_id, -32601, f"Method not found: {method}")


def call_build_context_pack(config: ServerConfig, arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required.")
    token_budget = int(arguments.get("token_budget") or arguments.get("budget_tokens") or 4000)
    project_filter = arguments.get("project_filter")
    if project_filter is not None:
        project_filter = str(project_filter)
    include_low_interest = bool(arguments.get("include_low_interest", False))
    dense = _build_dense_provider(config.dense_provider, config.dense_model)
    sparse = _build_sparse_provider(config.sparse_provider, config.sparse_model, config.sparse_top_k)
    payload = build_context_pack(
        db_path=config.db_path,
        query=query,
        dense=dense,
        sparse=sparse,
        dense_provider=config.dense_provider,
        sparse_provider=config.sparse_provider,
        dense_model=config.dense_model,
        sparse_model=config.sparse_model,
        sparse_top_k=config.sparse_top_k,
        include_low_interest=include_low_interest,
        project=project_filter,
        options=ContextPackOptions(
            budget_tokens=token_budget,
            direct_limit=int(arguments.get("direct_limit") or 10),
            node_limit=int(arguments.get("node_limit") or 5),
            node_member_limit=int(arguments.get("node_member_limit") or 5),
            neighbor_limit=int(arguments.get("neighbor_limit") or 5),
        ),
        ensure_schema=False,
        read_only=True,
    )
    payload["context_text"] = format_context_text(payload, max_block_chars=config.max_block_chars)
    payload["source_references"] = source_references(payload)
    return payload


def format_context_text(payload: dict[str, Any], *, max_block_chars: int) -> str:
    lines: list[str] = []
    for idx, block in enumerate(payload.get("selected_blocks", []), start=1):
        text = str(block.get("text") or "")
        if len(text) > max_block_chars:
            text = text[: max_block_chars - 1] + "…"
        lines.extend(
            [
                f"[{idx}] Source: {block.get('source_path')}",
                f"Role: {block.get('role')} | Block: {block.get('block_type')} | Reason: {block.get('reason')}",
                text,
                "",
            ]
        )
    return "\n".join(lines).strip()


def source_references(payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    seen: set[tuple[Any, Any]] = set()
    for block in payload.get("selected_blocks", []):
        key = (block.get("source_path"), block.get("block_id"))
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            {
                "block_id": block.get("block_id"),
                "source_path": block.get("source_path"),
                "conversation_id": block.get("conversation_id"),
                "message_id": block.get("message_id"),
                "interest_tier": block.get("interest_tier"),
                "reason": block.get("reason"),
                "score": block.get("score"),
            }
        )
    return refs


def _tool_description() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Build a traceable augmentation context pack from the local ChatGPT export knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user query to retrieve memory for."},
                "token_budget": {"type": "integer", "default": 4000, "minimum": 1},
                "project_filter": {"type": ["string", "null"], "default": None},
                "include_low_interest": {"type": "boolean", "default": False},
                "direct_limit": {"type": "integer", "default": 10, "minimum": 1},
                "node_limit": {"type": "integer", "default": 5, "minimum": 0},
                "node_member_limit": {"type": "integer", "default": 5, "minimum": 0},
                "neighbor_limit": {"type": "integer", "default": 5, "minimum": 0},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    main()
