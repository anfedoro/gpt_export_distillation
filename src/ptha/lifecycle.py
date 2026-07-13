"""Background process orchestration for the foreground PTHA service."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from ptha.config import PthaConfig
from ptha.database import inspect_database
from ptha.errors import PthaError
from ptha.ipc import IPCError, request, socket_state
from ptha.process import ProcessIdentity, identity_matches, inspect_process, instance_matches_process, send_termination, wait_for_exit

STATE_SCHEMA_VERSION = 1
_DETACHED_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


class LifecycleError(PthaError):
    code = "service_start_failed"
    exit_code = 7

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


@dataclass(frozen=True)
class ServiceState:
    identity: ProcessIdentity
    database_path: str
    socket_path: str
    started_at: str
    service_protocol_version: int = 1
    phase: str = "starting"
    instance_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": STATE_SCHEMA_VERSION, **self.identity.to_dict(),
                "database_path": self.database_path, "socket_path": self.socket_path,
                "started_at": self.started_at, "service_protocol_version": self.service_protocol_version,
                "phase": self.phase, "instance_id": self.instance_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServiceState":
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported service state schema.")
        command = value.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("Invalid service command metadata.")
        identity = ProcessIdentity(int(value["pid"]), float(value["process_start_time"]),
                                   str(value["executable"]), tuple(command))
        started_at = str(value["started_at"])
        datetime.fromisoformat(started_at)
        instance_id = str(value["instance_id"])
        if len(instance_id) < 20:
            raise ValueError("Invalid service instance identity.")
        return cls(identity, str(value["database_path"]), str(value["socket_path"]),
                   started_at, int(value.get("service_protocol_version", 1)),
                   str(value.get("phase", "starting")), instance_id)


def state_path(config: PthaConfig) -> Path:
    return config.paths.state_dir / "service-state.json"


def read_state(config: PthaConfig) -> ServiceState | None:
    path = state_path(config)
    if not path.exists():
        return None
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise LifecycleError("PTHA service state is not a safe regular file.", code="service_state_stale")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return ServiceState.from_dict(value)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LifecycleError("PTHA service state is invalid.", code="service_state_stale") from exc


def write_state(config: PthaConfig, state: ServiceState) -> None:
    path = state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


@contextmanager
def lifecycle_lock(config: PthaConfig, *, timeout: float = 5.0) -> Iterator[None]:
    path = config.paths.state_dir / "service.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        os.chmod(path, 0o600)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LifecycleError("Another PTHA lifecycle command is running.", code="service_busy")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def service_status(config: PthaConfig) -> dict[str, Any]:
    metadata_error = False
    try:
        state = read_state(config)
    except LifecycleError:
        state = None
        metadata_error = True
    current = inspect_process(state.identity.pid) if state else None
    identity_valid = bool(state and identity_matches(state.identity, current))
    ipc: dict[str, Any] | None = None
    try:
        value = request(config.paths.socket, "status", timeout_ms=500,
                        max_request_size=config.max_request_bytes, max_response_size=config.max_response_bytes)
        ipc = value if isinstance(value, dict) else None
    except IPCError:
        pass
    instance_valid = bool(state and identity_valid and (
        (ipc and ipc.get("instance_id") == state.instance_id) or
        (not ipc and instance_matches_process(state.identity.pid, state.instance_id))
    ))
    if state and current and not identity_valid:
        classification = "unknown-process"
    elif ipc and identity_valid and not instance_valid:
        classification = "unknown-process"
    elif ipc and identity_valid:
        if ipc.get("state") == "stopping":
            classification = "stopping"
        else:
            classification = "ready" if ipc.get("state") == "ready" and ipc.get("models_loaded") else "degraded"
    elif ipc and not state:
        classification = "ready" if ipc.get("state") == "ready" and ipc.get("models_loaded") else "degraded"
    elif identity_valid and instance_valid:
        elapsed = time.time() - datetime.fromisoformat(state.started_at).timestamp()
        classification = "starting" if state.phase == "starting" and elapsed <= config.startup_timeout_seconds else "degraded"
    elif state or metadata_error or config.paths.socket.exists():
        classification = "stale-state"
    else:
        classification = "stopped"
    return {
        "schema_version": 1, "state": classification, "pid": state.identity.pid if state else None,
        "process_identity_valid": identity_valid, "ipc_ready": bool(ipc),
        "instance_identity_valid": instance_valid,
        "models_loaded": bool(ipc and ipc.get("models_loaded")),
        "uptime_seconds": ipc.get("uptime_seconds") if ipc else None,
        "active_requests": ipc.get("active_requests") if ipc else None,
        "socket_path": str(config.paths.socket), "log_path": str(config.paths.service_log),
        "message": None,
    }


def start_service(config: PthaConfig, *, timeout: float | None = None) -> dict[str, Any]:
    from ptha.operations import maintenance_lock
    with lifecycle_lock(config):
        with maintenance_lock(config):
            return _start_locked(config, timeout=timeout)


def _start_locked(config: PthaConfig, *, timeout: float | None) -> dict[str, Any]:
    current_status = service_status(config)
    if current_status["state"] == "ready":
        return {**current_status, "already_running": True}
    if current_status["state"] == "unknown-process":
        raise LifecycleError("Saved PID belongs to another process; no signal or cleanup was performed.",
                             code="service_identity_mismatch")
    if current_status["state"] in {"starting", "degraded"} and current_status["process_identity_valid"]:
        raise LifecycleError("A PTHA service process is already starting or degraded.", code="service_already_running")
    database = inspect_database(config.database)
    if database.get("state") != "ready":
        raise LifecycleError("PTHA database is not ready. Import an archive before starting the service.",
                             code="database_not_ready")
    _cleanup_stale(config)
    config.paths.log_dir.mkdir(parents=True, exist_ok=True)
    command = _service_command(config)
    instance_id = secrets.token_urlsafe(32)
    environment = {**os.environ, "PTHA_INSTANCE_ID": instance_id}
    with config.paths.service_log.open("ab", buffering=0) as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, close_fds=True, env=environment)
    _DETACHED_CHILDREN[child.pid] = child
    identity = _wait_identity(child.pid)
    if identity is None:
        child.wait(timeout=1)
        _DETACHED_CHILDREN.pop(child.pid, None)
        raise LifecycleError("PTHA service process exited before identity could be recorded.", code="service_start_failed")
    state = ServiceState(identity, str(config.database), str(config.paths.socket), datetime.now(UTC).isoformat(),
                         instance_id=instance_id)
    write_state(config, state)
    deadline = time.monotonic() + (timeout if timeout is not None else config.startup_timeout_seconds)
    while time.monotonic() < deadline:
        if not identity_matches(identity, inspect_process(identity.pid)):
            child.wait(timeout=1)
            _DETACHED_CHILDREN.pop(child.pid, None)
            state_path(config).unlink(missing_ok=True)
            raise LifecycleError(_failure_message(config, "PTHA service exited before readiness."), code="service_start_failed")
        status = service_status(config)
        if status["state"] == "ready":
            write_state(config, ServiceState(identity, state.database_path, state.socket_path, state.started_at,
                                             state.service_protocol_version, "ready", state.instance_id))
            return {**status, "already_running": False}
        time.sleep(0.1)
    if identity_matches(identity, inspect_process(identity.pid)):
        try:
            request(config.paths.socket, "shutdown", timeout_ms=500)
        except IPCError:
            send_termination(identity)
        exited = wait_for_exit(identity, 5)
        child_handle = _DETACHED_CHILDREN.pop(identity.pid, None) if exited else None
        if child_handle is not None:
            child_handle.wait(timeout=1)
        if exited:
            _cleanup_stale(config)
    raise LifecycleError(_failure_message(config, "PTHA service did not become ready before timeout."),
                         code="service_start_timeout")


def stop_service(config: PthaConfig, *, timeout: float | None = None, force: bool = False) -> dict[str, Any]:
    with lifecycle_lock(config):
        return _stop_locked(config, timeout=timeout, force=force)


def _stop_locked(config: PthaConfig, *, timeout: float | None, force: bool) -> dict[str, Any]:
    state = read_state(config)
    status = service_status(config)
    if status["state"] == "unknown-process":
        raise LifecycleError("Saved PID belongs to another process; no signal was sent.", code="service_identity_mismatch")
    if state is None:
        if status["ipc_ready"]:
            try:
                request(config.paths.socket, "shutdown", timeout_ms=1000)
            except IPCError:
                pass
            deadline = time.monotonic() + (timeout if timeout is not None else config.shutdown_timeout_seconds)
            while time.monotonic() < deadline and socket_state(config.paths.socket) in {"healthy", "active"}:
                time.sleep(0.05)
        _cleanup_stale(config)
        return {**service_status(config), "already_stopped": not status["ipc_ready"]}
    identity = state.identity
    identity_valid = identity_matches(identity, inspect_process(identity.pid))
    if not identity_valid:
        _cleanup_stale(config)
        return {**service_status(config), "already_stopped": True}
    try:
        request(config.paths.socket, "shutdown", timeout_ms=1000)
    except IPCError:
        if not instance_matches_process(identity.pid, state.instance_id):
            raise LifecycleError("PTHA instance identity could not be verified; no signal was sent.",
                                 code="service_identity_mismatch")
        if not send_termination(identity):
            raise LifecycleError("PTHA process identity changed before SIGTERM.", code="service_identity_mismatch")
    wait = timeout if timeout is not None else config.shutdown_timeout_seconds
    if not wait_for_exit(identity, wait):
        if not force:
            raise LifecycleError("PTHA service did not stop before timeout.", code="service_stop_timeout")
        if not instance_matches_process(identity.pid, state.instance_id):
            raise LifecycleError("PTHA instance identity changed before forced termination.",
                                 code="service_identity_mismatch")
        if not send_termination(identity, force=True):
            raise LifecycleError("PTHA process identity changed before forced termination.", code="service_identity_mismatch")
        if not wait_for_exit(identity, 5):
            raise LifecycleError("PTHA service could not be stopped.", code="service_stop_failed")
    child = _DETACHED_CHILDREN.pop(identity.pid, None)
    if child is not None:
        child.wait(timeout=1)
    _cleanup_stale(config)
    return {**service_status(config), "already_stopped": False, "stopped_pid": identity.pid}


def restart_service(config: PthaConfig, *, start_timeout: float | None = None,
                    stop_timeout: float | None = None, force: bool = False) -> dict[str, Any]:
    from ptha.operations import maintenance_lock
    with lifecycle_lock(config):
        with maintenance_lock(config):
            old = read_state(config)
            stopped = _stop_locked(config, timeout=stop_timeout, force=force)
            started = _start_locked(config, timeout=start_timeout)
            return {"schema_version": 1, "old_pid": old.identity.pid if old else None,
                    "new_pid": started.get("pid"), "stop": stopped, "start": started}


def cleanup_service_state(config: PthaConfig, *, force_state: bool = False) -> dict[str, Any]:
    with lifecycle_lock(config):
        status = service_status(config)
        if status["state"] in {"ready", "starting", "degraded", "stopping"}:
            raise LifecycleError("Active PTHA service state cannot be cleaned.", code="service_already_running")
        if status["state"] == "unknown-process" and not force_state:
            raise LifecycleError(
                "PTHA cannot safely remove lifecycle state because the stored PID belongs to another running process. "
                "No process was signalled and no state was removed.", code="service_identity_mismatch"
            )
        socket_status = socket_state(config.paths.socket)
        if socket_status in {"healthy", "active"}:
            raise LifecycleError("An active PTHA socket cannot be removed by cleanup.", code="service_already_running")
        _safe_unlink_owned(state_path(config), allow_regular=True, allow_socket=False)
        _safe_unlink_owned(config.paths.socket, allow_regular=False, allow_socket=True)
        from ptha.operations import cleanup_stale_operations
        removed_operations = cleanup_stale_operations(config, force_state=force_state)
        return {"schema_version": 1, "state": "stopped", "cleaned": True,
                "forced_state_cleanup": bool(force_state), "removed_operation_paths": removed_operations}


def _service_command(config: PthaConfig) -> list[str]:
    executable = shutil.which("ptha")
    prefix = [executable] if executable else [sys.executable, "-m", "ptha.cli"]
    return [*prefix, "--config", str(config.config_file), "service", "run"]


def _wait_identity(pid: int) -> ProcessIdentity | None:
    for _ in range(50):
        identity = inspect_process(pid)
        if identity:
            return identity
        time.sleep(0.01)
    return None


def _cleanup_stale(config: PthaConfig) -> None:
    _safe_unlink_owned(state_path(config), allow_regular=True, allow_socket=False)
    if socket_state(config.paths.socket) != "healthy":
        _safe_unlink_owned(config.paths.socket, allow_regular=False, allow_socket=True)


def _safe_unlink_owned(path: Path, *, allow_regular: bool, allow_socket: bool) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid():
        raise LifecycleError("PTHA state path is not safe to remove.", code="service_state_stale")
    if not ((allow_regular and stat.S_ISREG(info.st_mode)) or (allow_socket and stat.S_ISSOCK(info.st_mode))):
        raise LifecycleError("PTHA state path has an unexpected type.", code="service_state_stale")
    path.unlink()


def _failure_message(config: PthaConfig, message: str) -> str:
    tail = _log_tail(config.paths.service_log)
    suffix = f"\nLog: {config.paths.service_log}"
    if tail:
        suffix += "\nRecent log:\n" + "\n".join(tail)
    return message + suffix


def _log_tail(path: Path, lines: int = 8) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []
