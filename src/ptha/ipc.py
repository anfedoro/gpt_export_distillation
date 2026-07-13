"""Length-prefixed JSON protocol used by the local PTHA service."""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import uuid
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_VERSION = 1
DEFAULT_MAX_REQUEST_SIZE = 1_048_576
DEFAULT_MAX_RESPONSE_SIZE = 16_777_216
HEADER_SIZE = 4


class IPCError(Exception):
    code = "ipc_error"


class ConnectionClosed(IPCError):
    code = "invalid_request"


class FrameError(IPCError):
    code = "invalid_request"


class FrameTooLarge(IPCError):
    def __init__(self, code: str, maximum: int) -> None:
        super().__init__(f"Frame exceeds the configured {maximum}-byte limit.")
        self.code = code


class ServiceUnavailable(IPCError):
    code = "service_not_running"


class RemoteError(IPCError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def encode_frame(payload: Mapping[str, Any], *, maximum: int, oversized_code: str = "response_too_large") -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not body:
        raise FrameError("Empty JSON frames are not valid.")
    if len(body) > maximum:
        raise FrameTooLarge(oversized_code, maximum)
    return struct.pack(">I", len(body)) + body


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionClosed("Connection closed before the frame was complete.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(connection: socket.socket, *, maximum: int, oversized_code: str = "request_too_large") -> dict[str, Any]:
    length = struct.unpack(">I", recv_exact(connection, HEADER_SIZE))[0]
    if length == 0:
        raise FrameError("Zero-length frames are not valid.")
    if length > maximum:
        raise FrameTooLarge(oversized_code, maximum)
    body = recv_exact(connection, length)
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrameError("Frame is not valid UTF-8.") from exc
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise FrameError("Frame is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise FrameError("Frame JSON must be an object.")
    return value


def send_frame(connection: socket.socket, payload: Mapping[str, Any], *, maximum: int) -> None:
    connection.sendall(encode_frame(payload, maximum=maximum))


def make_request(operation: str, arguments: Mapping[str, Any] | None = None, *, timeout_ms: int = 30_000,
                 request_id: str | None = None) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id or str(uuid.uuid4()),
        "operation": operation,
        "arguments": dict(arguments or {}),
        "timeout_ms": timeout_ms,
    }


def request(socket_path: Path, operation: str, arguments: Mapping[str, Any] | None = None, *, timeout_ms: int = 30_000,
            max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
            max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE) -> Any:
    payload = make_request(operation, arguments, timeout_ms=timeout_ms)
    timeout = max(0.001, timeout_ms / 1000)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(socket_path))
            connection.sendall(encode_frame(payload, maximum=max_request_size, oversized_code="request_too_large"))
            response = recv_frame(connection, maximum=max_response_size, oversized_code="response_too_large")
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as exc:
        raise ServiceUnavailable("PTHA service is not available.") from exc
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise RemoteError("unsupported_protocol", "PTHA service protocol is incompatible.")
    if response.get("request_id") != payload["request_id"]:
        raise RemoteError("invalid_request", "PTHA service returned a mismatched request ID.")
    if not response.get("ok"):
        error = response.get("error") if isinstance(response.get("error"), dict) else {}
        raise RemoteError(str(error.get("code") or "internal_error"), str(error.get("message") or "PTHA service request failed."))
    return response.get("result")


def socket_state(path: Path, *, timeout_ms: int = 250) -> str:
    if not path.exists():
        return "missing"
    try:
        mode = path.lstat().st_mode
    except OSError:
        return "stale"
    if not stat.S_ISSOCK(mode):
        return "unsafe"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(max(0.001, timeout_ms / 1000))
            connection.connect(str(path))
    except OSError:
        return "stale"
    try:
        request(path, "ping", timeout_ms=timeout_ms)
        return "healthy"
    except IPCError:
        return "active"


def remove_stale_socket(path: Path) -> None:
    state = socket_state(path)
    if state in {"healthy", "active"}:
        raise IPCError("A healthy PTHA service is already running.")
    if state == "unsafe":
        raise IPCError("The configured socket path is not a Unix socket.")
    if path.exists():
        owner = path.lstat().st_uid
        if owner != os.geteuid():
            raise IPCError("The stale socket is not owned by the current user.")
        path.unlink()


def peer_is_current_user(connection: socket.socket) -> bool:
    if hasattr(connection, "getpeereid"):
        uid, _ = connection.getpeereid()  # type: ignore[attr-defined]
        return uid == os.geteuid()
    if hasattr(socket, "SO_PEERCRED"):
        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _, uid, _ = struct.unpack("3i", credentials)
        return uid == os.geteuid()
    return True
