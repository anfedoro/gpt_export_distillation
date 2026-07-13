"""Process identity primitives for safe PTHA lifecycle management."""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    create_time: float
    executable: str
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "process_start_time": self.create_time,
                "executable": self.executable, "command": list(self.command)}


def inspect_process(pid: int) -> ProcessIdentity | None:
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            return ProcessIdentity(pid, process.create_time(), process.exe(), tuple(process.cmdline()))
    except (psutil.Error, OSError):
        return None


def identity_matches(expected: ProcessIdentity, current: ProcessIdentity | None, *, tolerance: float = 0.01) -> bool:
    if current is None or expected.pid != current.pid:
        return False
    if abs(expected.create_time - current.create_time) > tolerance:
        return False
    try:
        return Path(expected.executable).resolve() == Path(current.executable).resolve()
    except OSError:
        return expected.executable == current.executable


def wait_for_exit(identity: ProcessIdentity, timeout: float) -> bool:
    current = inspect_process(identity.pid)
    if not identity_matches(identity, current):
        return True
    try:
        psutil.Process(identity.pid).wait(timeout=max(0.0, timeout))
        return True
    except psutil.TimeoutExpired:
        return not identity_matches(identity, inspect_process(identity.pid))
    except psutil.Error:
        return True


def send_termination(identity: ProcessIdentity, *, force: bool = False) -> bool:
    if not identity_matches(identity, inspect_process(identity.pid)):
        return False
    os.kill(identity.pid, signal.SIGKILL if force else signal.SIGTERM)
    return True


def instance_matches_process(pid: int, instance_id: str) -> bool:
    try:
        return psutil.Process(pid).environ().get("PTHA_INSTANCE_ID") == instance_id
    except (psutil.Error, OSError):
        return False
