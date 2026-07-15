"""Foreground PTHA retrieval service over local Unix IPC."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import socket
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from kb.embeddings.bge_m3_provider import build_bge_m3_providers
from kb.mcp.archive import ArchiveConfig, ArchiveSession
from ptha.config import PthaConfig
from ptha import application_version
from ptha.database import inspect_database
from ptha.errors import PthaError
from ptha.ipc import (
    PROTOCOL_VERSION,
    FrameError,
    FrameTooLarge,
    IPCError,
    peer_is_current_user,
    recv_frame,
    remove_stale_socket,
    send_frame,
)

LOG = logging.getLogger("ptha.service")
OPERATIONS = {"ping", "status", "search_archive", "construct_archive_context", "shutdown"}


class ServiceError(PthaError):
    code = "service_start_failed"
    exit_code = 7


class DatabaseNotReadyError(ServiceError):
    code = "database_not_ready"


class ModelLoadError(ServiceError):
    code = "model_load_failed"


class RetrievalService:
    def __init__(
        self,
        config: PthaConfig,
        *,
        session_factory: Callable[[PthaConfig], Any] | None = None,
        max_request_size: int | None = None,
        max_response_size: int | None = None,
    ) -> None:
        self.config = config
        self.instance_id = os.environ.get("PTHA_INSTANCE_ID") or secrets.token_urlsafe(32)
        self.socket_path = config.paths.socket
        self.session_factory = session_factory or build_archive_session
        self.max_request_size = max_request_size or config.max_request_bytes
        self.max_response_size = max_response_size or config.max_response_bytes
        self.shutdown_event = threading.Event()
        self.ready_event = threading.Event()
        self.started = time.monotonic()
        self.active_requests = 0
        self.request_count = 0
        self._active_lock = threading.Lock()
        self._listener: socket.socket | None = None
        self._session: Any = None
        self.database_info: dict[str, Any] = {}

    def run(self) -> None:
        startup_started = time.monotonic()
        self._prepare_database()
        self._prepare_socket()
        try:
            LOG.info("loading_retrieval_runtime")
            try:
                self._session = self.session_factory(self.config)
            except Exception as exc:
                raise ModelLoadError("PTHA retrieval models could not be loaded.") from exc
            self.ready_event.set()
            LOG.info("service_ready protocol=%s schema=%s dense_model=%s sparse_model=%s startup_seconds=%.3f",
                     PROTOCOL_VERSION, self.database_info.get("schema_version"), self.config.dense_model,
                     self.config.sparse_model, time.monotonic() - startup_started)
            self._accept_loop()
        finally:
            self.ready_event.clear()
            if self._session is not None:
                self._session.close()
            if self._listener is not None:
                self._listener.close()
            self.socket_path.unlink(missing_ok=True)
            LOG.info("service_stopped request_count=%s", self.request_count)

    def request_shutdown(self) -> None:
        self.shutdown_event.set()

    def status(self) -> dict[str, Any]:
        return {
            "state": "ready" if self.ready_event.is_set() and not self.shutdown_event.is_set() else "stopping",
            "protocol_version": PROTOCOL_VERSION,
            "database_schema_version": self.database_info.get("schema_version"),
            "models_loaded": self._session is not None,
            "dense_model": self.config.dense_model,
            "sparse_model": self.config.sparse_model,
            "active_requests": self.active_requests,
            "uptime_seconds": round(time.monotonic() - self.started, 3),
            "serialized_retrieval": True,
            "instance_id": self.instance_id,
            "database_path": str(self.config.database),
        }

    def _prepare_database(self) -> None:
        self.database_info = inspect_database(self.config.database)
        if self.database_info.get("state") != "ready":
            raise DatabaseNotReadyError("PTHA database is not ready. Import an archive before starting the service.")

    def _prepare_socket(self) -> None:
        runtime = self.socket_path.parent
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(runtime, 0o700)
        try:
            remove_stale_socket(self.socket_path)
        except IPCError as exc:
            raise ServiceError("A PTHA service is already running or the socket path is unsafe.") from exc
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(16)
            listener.settimeout(0.25)
        except Exception:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self.shutdown_event.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.shutdown_event.is_set():
                    break
                raise
            self._serve_connection(connection)

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection:
            if not peer_is_current_user(connection):
                return
            request_id: Any = None
            try:
                request = recv_frame(connection, maximum=self.max_request_size)
                request_id = request.get("request_id")
                response = self._dispatch(request)
                try:
                    send_frame(connection, response, maximum=self.max_response_size)
                except FrameTooLarge:
                    send_frame(connection, _error(request_id, "response_too_large", "PTHA response exceeds the configured limit."),
                               maximum=self.max_response_size)
            except FrameTooLarge as exc:
                self._try_error(connection, request_id, exc.code, "PTHA request exceeds the configured limit.")
            except (FrameError, IPCError):
                self._try_error(connection, request_id, "invalid_request", "PTHA request is malformed.")
            except Exception as exc:  # noqa: BLE001
                LOG.warning("connection_failed error_class=%s", type(exc).__name__)
                self._try_error(connection, request_id, "internal_error", "PTHA could not process the request.")

    def _dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _error(None, "invalid_request", "request_id must be a non-empty string.")
        if request.get("protocol_version") != PROTOCOL_VERSION:
            return _error(request_id, "unsupported_protocol", "PTHA IPC protocol version is unsupported.")
        operation = request.get("operation")
        if operation not in OPERATIONS:
            return _error(request_id, "unsupported_operation", "PTHA IPC operation is unsupported.")
        arguments = request.get("arguments", {})
        if not isinstance(arguments, dict):
            return _error(request_id, "invalid_arguments", "arguments must be a JSON object.")
        timeout_ms = request.get("timeout_ms", self.config.request_timeout_seconds * 1000)
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
            return _error(request_id, "invalid_arguments", "timeout_ms must be a positive integer.")
        if self.shutdown_event.is_set() and operation != "status":
            return _error(request_id, "service_shutting_down", "PTHA service is shutting down.")
        if operation == "ping":
            return _success(request_id, {"state": "ready", "protocol_version": PROTOCOL_VERSION})
        if operation == "status":
            return _success(request_id, self.status())
        if operation == "shutdown":
            self.request_shutdown()
            return _success(request_id, {"state": "stopping"})
        with self._active_lock:
            self.active_requests += 1
            self.request_count += 1
        request_started = time.monotonic()
        try:
            forwarded = {**arguments, "timeout_ms": timeout_ms}
            if operation == "search_archive":
                result = self._session.search_archive(forwarded)
            else:
                result = self._session.construct_archive_context(forwarded)
            return _success(request_id, result)
        except TimeoutError:
            return _error(request_id, "retrieval_timeout", "PTHA retrieval exceeded its cooperative deadline.")
        except (TypeError, ValueError):
            return _error(request_id, "invalid_arguments", "Archive tool arguments are invalid.")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("retrieval_failed request_id=%s operation=%s error_class=%s", request_id, operation, type(exc).__name__)
            return _error(request_id, "internal_error", "PTHA retrieval failed.")
        finally:
            LOG.info("request_completed request_id=%s operation=%s duration_ms=%.3f",
                     request_id, operation, (time.monotonic() - request_started) * 1000)
            with self._active_lock:
                self.active_requests -= 1

    def _try_error(self, connection: socket.socket, request_id: Any, code: str, message: str) -> None:
        try:
            send_frame(connection, _error(request_id, code, message), maximum=self.max_response_size)
        except (OSError, IPCError):
            pass


def build_archive_session(config: PthaConfig) -> ArchiveSession:
    dense, sparse = build_providers(config)
    return ArchiveSession(
        ArchiveConfig(config.database, config.candidate_pool, config.default_output_tokens, config.max_output_tokens),
        dense,
        sparse,
    )


def build_providers(config: PthaConfig) -> tuple[Any, Any]:
    if config.embedding_backend != "mlx":
        raise RuntimeError("PTHA v1 production embeddings require embedding_backend=mlx.")
    if config.sparse_model != config.dense_model:
        raise RuntimeError("PTHA requires one shared BGE-M3 model for dense and sparse embeddings.")
    requested = {value for value in (config.dense_device, config.sparse_device) if value != "auto"}
    if len(requested) > 1:
        raise RuntimeError("Dense and sparse embeddings must use the same device.")
    return build_bge_m3_providers(
        config.embedding_model,
        model_revision=config.embedding_model_revision,
        device=config.embedding_device,
        dtype=config.embedding_dtype,
        max_seq_length=512,
        sparse_top_k=config.sparse_top_k,
        batch_size=config.batch_size,
        max_padded_tokens=config.embedding_max_padded_tokens,
        sparse_head=config.embedding_sparse_head,
        colbert_head=config.embedding_colbert_head,
        model_cache=config.model_cache,
    )


def install_signal_handlers(service: RetrievalService) -> None:
    def handle_signal(_signum: int, _frame: Any) -> None:
        service.request_shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def configure_service_logging(config: PthaConfig) -> None:
    config.paths.log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(config.paths.service_log, maxBytes=config.service_max_bytes,
                                  backupCount=config.service_backup_count, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    LOG.info("service_starting version=%s", application_version())


def _success(request_id: Any, result: Any) -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id, "ok": True, "result": result}


def _error(request_id: Any, code: str, message: str) -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id, "ok": False,
            "error": {"code": code, "message": message}}
