from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.cli import _build_dense_provider, _build_sparse_provider
from kb.mcp.archive import ArchiveConfig, ArchiveSession
from kb.mcp.tools import TOOL_NAMES, archive_tools, construct_archive_context_tool, search_archive_tool


PROTOCOL_VERSION = "2024-11-05"
LOG = logging.getLogger("kb.mcp")


@dataclass(frozen=True)
class ServerConfig:
    db_path: Path
    dense_provider: str = "sentence-transformers"
    sparse_provider: str = "sentence-transformers"
    dense_model: str = "BAAI/bge-m3"
    sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    sparse_top_k: int = 128
    candidate_pool: int = 500
    default_output_tokens: int = 1800
    max_output_tokens: int = 6000


class MCPServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        # Providers and sparse arrays are deliberately constructed once, before readiness.
        dense = _build_dense_provider(config.dense_provider, config.dense_model)
        sparse = _build_sparse_provider(config.sparse_provider, config.sparse_model, config.sparse_top_k)
        self.session = ArchiveSession(ArchiveConfig(config.db_path, config.candidate_pool, config.default_output_tokens, config.max_output_tokens), dense, sparse)

    def close(self) -> None:
        self.session.close()

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method, request_id = request.get("method"), request.get("id")
        if request_id is None:
            return None
        if method == "initialize":
            return _result(request_id, {"protocolVersion": PROTOCOL_VERSION, "serverInfo": {"name": "gpt-export-distillation-kb", "version": "0.4.1"}, "capabilities": {"tools": {}}})
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": archive_tools()})
        if method == "tools/call":
            params = request.get("params") or {}
            name, arguments = params.get("name"), params.get("arguments") or {}
            if name not in TOOL_NAMES:
                return _error(request_id, -32602, f"Unknown tool: {name}")
            try:
                payload = self.session.construct_archive_context(arguments) if name == "construct_archive_context" else self.session.search_archive(arguments)
                return _result(request_id, {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}], "isError": False})
            except Exception as exc:  # noqa: BLE001
                LOG.info("tool_call_failed tool=%s error=%s", name, type(exc).__name__)
                return _result(request_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})
        return _error(request_id, -32601, f"Method not found: {method}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the read-only native archive MCP server.")
    parser.add_argument("command", choices=["serve", "test-call"], nargs="?", default="serve")
    parser.add_argument("--db", required=True, help="Path to the clean native SQLite database.")
    parser.add_argument("--transport", choices=["stdio"], default="stdio")
    parser.add_argument("--tool", choices=TOOL_NAMES)
    parser.add_argument("--query", help="Query for test-call (maps to current_context for broad context).")
    parser.add_argument("--dense-provider", choices=["sentence-transformers"], default="sentence-transformers")
    parser.add_argument("--sparse-provider", choices=["sentence-transformers"], default="sentence-transformers")
    parser.add_argument("--dense-model", default="BAAI/bge-m3")
    parser.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
    parser.add_argument("--sparse-top-k", type=int, default=128)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--default-output-tokens", type=int, default=1800)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ServerConfig(Path(args.db).expanduser().resolve(), args.dense_provider, args.sparse_provider, args.dense_model, args.sparse_model, args.sparse_top_k, args.candidate_pool, args.default_output_tokens, args.max_output_tokens)
    server = MCPServer(config)
    try:
        if args.command == "test-call":
            if not args.tool or not args.query:
                raise SystemExit("test-call requires --tool and --query.")
            arguments = {"query": args.query} if args.tool == "search_archive" else {"current_context": args.query}
            response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": args.tool, "arguments": arguments}})
            print(json.dumps(response, ensure_ascii=False, indent=2))
        else:
            serve_stdio(server)
    finally:
        server.close()


def serve_stdio(server: MCPServer) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = server.handle(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            response = _error(None, -32603, type(exc).__name__)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def handle_request(config: ServerConfig, request: dict[str, Any]) -> dict[str, Any] | None:
    """Compatibility helper for one-shot tests; production uses one MCPServer."""
    server = MCPServer(config)
    try:
        return server.handle(request)
    finally:
        server.close()


def _context_tool() -> dict[str, Any]:
    return construct_archive_context_tool()


def _search_tool() -> dict[str, Any]:
    return search_archive_tool()


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    main()
