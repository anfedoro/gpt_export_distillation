"""Lightweight stdio MCP adapter for the persistent PTHA service."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from kb.mcp.tools import TOOL_NAMES, archive_tools
from ptha.config import PthaConfig
from ptha.ipc import IPCError, PROTOCOL_VERSION, RemoteError, ServiceUnavailable, request

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPAdapter:
    def __init__(self, config: PthaConfig, *, stderr: TextIO | None = None) -> None:
        self.config = config
        self.stderr = stderr or sys.stderr

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return None
        if method == "initialize":
            try:
                self._request("ping", timeout_ms=self.config.request_timeout_seconds * 1000)
            except IPCError:
                self._service_unavailable()
                return _error(request_id, -32001, "PTHA service is not running. Start it with: ptha service run")
            return _result(request_id, {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {"name": "ptha", "version": "1"},
                "capabilities": {"tools": {}},
            })
        if method == "ping":
            try:
                self._request("ping", timeout_ms=self.config.request_timeout_seconds * 1000)
                return _result(request_id, {})
            except IPCError:
                self._service_unavailable()
                return _error(request_id, -32001, "PTHA service is not running.")
        if method == "tools/list":
            return _result(request_id, {"tools": archive_tools()})
        if method == "tools/call":
            params = message.get("params")
            if not isinstance(params, dict):
                return _error(request_id, -32602, "Invalid tool parameters.")
            name = params.get("name")
            arguments = params.get("arguments", {})
            if name not in TOOL_NAMES or not isinstance(arguments, dict):
                return _error(request_id, -32602, "Unknown tool or invalid arguments.")
            timeout_ms = arguments.get("timeout_ms", self.config.request_timeout_seconds * 1000)
            if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
                return _tool_error(request_id, "invalid_arguments", "Archive tool arguments are invalid.")
            try:
                payload = self._request(str(name), arguments, timeout_ms=timeout_ms)
                return _result(request_id, {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False,
                                                                                                 separators=(",", ":"))}],
                                            "isError": False})
            except ServiceUnavailable:
                self._service_unavailable()
                return _tool_error(request_id, "service_not_running", "PTHA service is not running.")
            except RemoteError as exc:
                return _tool_error(request_id, exc.code, _safe_remote_message(exc.code))
            except IPCError:
                return _tool_error(request_id, "ipc_error", "PTHA service request failed.")
        return _error(request_id, -32601, "Method not found.")

    def _service_unavailable(self) -> None:
        print("PTHA service is not running.\n\nStart it with:\n  ptha service run", file=self.stderr)

    def _request(self, operation: str, arguments: dict[str, Any] | None = None, *, timeout_ms: int) -> Any:
        return request(self.config.paths.socket, operation, arguments, timeout_ms=timeout_ms,
                       max_request_size=self.config.max_request_bytes,
                       max_response_size=self.config.max_response_bytes)


def serve_stdio(config: PthaConfig, *, stdin: TextIO | None = None, stdout: TextIO | None = None,
                stderr: TextIO | None = None) -> None:
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    adapter = MCPAdapter(config, stderr=stderr)
    for line in input_stream:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError
            response = adapter.handle(message)
        except (json.JSONDecodeError, ValueError):
            response = _error(None, -32700, "Invalid JSON-RPC request.")
        except Exception as exc:  # noqa: BLE001
            print(f"PTHA MCP adapter error: {type(exc).__name__}", file=adapter.stderr)
            response = _error(None, -32603, "Internal adapter error.")
        if response is not None:
            output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_stream.flush()


def _safe_remote_message(code: str) -> str:
    return {
        "invalid_arguments": "Archive tool arguments are invalid.",
        "retrieval_timeout": "PTHA retrieval exceeded its cooperative deadline.",
        "response_too_large": "PTHA response exceeds the configured limit.",
        "service_shutting_down": "PTHA service is shutting down.",
    }.get(code, "PTHA service request failed.")


def _tool_error(request_id: Any, code: str, message: str) -> dict[str, Any]:
    payload = json.dumps({"code": code, "message": message}, separators=(",", ":"))
    return _result(request_id, {"content": [{"type": "text", "text": payload}], "isError": True})


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
