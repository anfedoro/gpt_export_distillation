from __future__ import annotations

import io
import ast
from contextlib import closing
import json
import os
import signal
import socket
import sqlite3
import stat
import struct
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kb.mcp.server import _context_tool, _search_tool
from kb.mcp.archive import ArchiveConfig, ArchiveSession
from kb.mcp.tools import archive_tools
from kb.index.chunk_builder import build_chunk_policy
from kb.storage.native_pre_mvp import NativeBuildStore, _chunked_space
from ptha.config import load_config
from ptha.cli import main
from ptha.ipc import (
    FrameError,
    FrameTooLarge,
    PROTOCOL_VERSION,
    encode_frame,
    make_request,
    recv_frame,
    request,
    socket_state,
)
from ptha.mcp import MCPAdapter, serve_stdio
from ptha.paths import PthaPaths
from ptha.service import DatabaseNotReadyError, RetrievalService, ServiceError, install_signal_handlers


class FakeSession:
    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1
        self.closed = False

    def search_archive(self, arguments: dict[str, object]) -> dict[str, object]:
        if not arguments.get("query"):
            raise ValueError("private exception detail")
        return {"schema_version": "kb.mcp.memory.v1", "mode": "focused", "items": [{"text": "synthetic result"}]}

    def construct_archive_context(self, arguments: dict[str, object]) -> dict[str, object]:
        if not arguments.get("current_context"):
            raise ValueError("private exception detail")
        return {"schema_version": "kb.mcp.memory.v1", "mode": "broad", "items": [{"text": "synthetic context"}]}

    def close(self) -> None:
        self.closed = True


class SyntheticDenseProvider:
    model_name = "synthetic-dense"
    embedding_space_id = "synthetic-dense;dim=1024;normalize=true;max_seq=64"
    effective_max_sequence_length = 64
    document_prefix = ""

    def embed_query(self, _query: str) -> list[float]:
        return [1.0] + [0.0] * 1023

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def token_count(self, text: str) -> int:
        return max(1, len(text.split()))

    def embedding_input(self, text: str) -> str:
        return text


class SyntheticSparseProvider:
    model_name = "synthetic-sparse"
    embedding_space_id = "synthetic-sparse;document_encoder=documents;query_encoder=query;top_k=128;max_seq=64"
    effective_max_sequence_length = 64
    document_prefix = ""

    def embed_query(self, _query: str) -> dict[str, float]:
        return {"synthetic": 1.0}

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        return [{"synthetic": 1.0} for _ in texts]

    def token_count(self, text: str) -> int:
        return max(1, len(text.split()))

    def embedding_input(self, text: str) -> str:
        return text


class IPCFrameTests(unittest.TestCase):
    def test_frame_round_trip_and_request_id(self) -> None:
        left, right = socket.socketpair()
        with left, right:
            payload = make_request("ping", request_id="fixed-id")
            left.sendall(encode_frame(payload, maximum=1000))
            self.assertEqual(recv_frame(right, maximum=1000), payload)
            self.assertEqual(payload["request_id"], "fixed-id")

    def test_fragmented_header_and_payload(self) -> None:
        left, right = socket.socketpair()
        payload = make_request("ping", request_id="fragmented")
        frame = encode_frame(payload, maximum=1000)
        with left, right:
            def writer() -> None:
                for piece in (frame[:1], frame[1:4], frame[4:8], frame[8:]):
                    left.sendall(piece)
                    time.sleep(0.005)
            thread = threading.Thread(target=writer)
            thread.start()
            self.assertEqual(recv_frame(right, maximum=1000), payload)
            thread.join()

    def test_zero_length_invalid_utf8_malformed_json_and_oversize(self) -> None:
        for raw, exception in (
            (struct.pack(">I", 0), FrameError),
            (struct.pack(">I", 1) + b"\xff", FrameError),
            (struct.pack(">I", 1) + b"{", FrameError),
            (struct.pack(">I", 100), FrameTooLarge),
        ):
            with self.subTest(raw=raw), self.assertRaises(exception):
                left, right = socket.socketpair()
                with left, right:
                    left.sendall(raw)
                    recv_frame(right, maximum=10)

    def test_response_encoding_limit(self) -> None:
        with self.assertRaises(FrameTooLarge) as raised:
            encode_frame({"value": "x" * 100}, maximum=10)
        self.assertEqual(raised.exception.code, "response_too_large")
        with self.assertRaises(FrameTooLarge) as request_raised:
            encode_frame({"value": "x" * 100}, maximum=10, oversized_code="request_too_large")
        self.assertEqual(request_raised.exception.code, "request_too_large")


class RuntimeVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        paths = PthaPaths(root / "config", root / "data", root / "cache", root / "state", root / "logs", root / "run")
        paths.config_dir.mkdir()
        config_file = paths.config_file
        config_file.write_text("config_version = 1\n", encoding="utf-8")
        base = load_config(config_file)
        self.database = paths.database
        _create_ready_database(self.database)
        self.config = replace(base, paths=paths, database=self.database, config_file=config_file,
                              dense_model="synthetic-dense", sparse_model="synthetic-sparse")
        FakeSession.instances = 0
        self.service = RetrievalService(self.config, session_factory=lambda _config: FakeSession())
        self.thread = threading.Thread(target=self.service.run)
        self.thread.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                if request(paths.socket, "ping", timeout_ms=100).get("state") == "ready":
                    break
            except Exception:
                time.sleep(0.01)
        else:
            self.fail("service did not become ready")

    def tearDown(self) -> None:
        if self.thread.is_alive():
            try:
                request(self.config.paths.socket, "shutdown", timeout_ms=500)
            except Exception:
                self.service.request_shutdown()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def test_real_ipc_status_socket_permissions_and_single_session(self) -> None:
        status = request(self.config.paths.socket, "status")
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["models_loaded"])
        self.assertTrue(status["serialized_retrieval"])
        self.assertEqual(stat.S_IMODE(self.config.paths.socket.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.config.paths.runtime_dir.stat().st_mode), 0o700)
        self.assertEqual(FakeSession.instances, 1)

    def test_cli_service_status_uses_real_ipc(self) -> None:
        stdout = io.StringIO()
        with patch("ptha.cli.load_config", return_value=self.config), patch("sys.stdout", stdout):
            self.assertEqual(main(["--config", str(self.config.config_file), "service", "status", "--json"]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["state"], "ready")
        self.assertTrue(payload["models_loaded"])

    def test_second_service_refuses_active_socket(self) -> None:
        second = RetrievalService(self.config, session_factory=lambda _config: FakeSession())
        second._prepare_database()
        with self.assertRaises(ServiceError):
            second._prepare_socket()
        self.assertEqual(request(self.config.paths.socket, "ping")["state"], "ready")
        self.assertEqual(request(self.config.paths.socket, "search_archive", {"query": "synthetic"})["mode"], "focused")
        self.assertEqual(request(self.config.paths.socket, "construct_archive_context", {"current_context": "synthetic"})["mode"], "broad")
        self.assertEqual(FakeSession.instances, 1)

    def test_protocol_operation_and_arguments_errors_are_sanitized(self) -> None:
        self.assertEqual(self.service._dispatch({**make_request("ping"), "protocol_version": 99})["error"]["code"], "unsupported_protocol")
        self.assertEqual(self.service._dispatch(make_request("unknown"))["error"]["code"], "unsupported_operation")
        response = self.service._dispatch(make_request("search_archive", {}))
        self.assertEqual(response["error"]["code"], "invalid_arguments")
        self.assertNotIn("private exception detail", json.dumps(response))

    def test_malformed_connection_does_not_stop_service_and_stale_socket_detection(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.config.paths.socket))
            connection.sendall(struct.pack(">I", 1) + b"{")
            response = recv_frame(connection, maximum=1000)
            self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertEqual(request(self.config.paths.socket, "ping")["state"], "ready")
        self.assertEqual(socket_state(self.config.paths.socket), "healthy")

    def test_two_mcp_adapters_share_service_and_stdout_is_protocol_only(self) -> None:
        initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        tool_list = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        search = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                  "params": {"name": "search_archive", "arguments": {"query": "synthetic"}}}
        context = {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                   "params": {"name": "construct_archive_context", "arguments": {"current_context": "synthetic"}}}
        for messages in ((initialize, tool_list, search), (initialize, context)):
            stdout, stderr = io.StringIO(), io.StringIO()
            serve_stdio(self.config, stdin=io.StringIO("".join(json.dumps(item) + "\n" for item in messages)),
                        stdout=stdout, stderr=stderr)
            lines = stdout.getvalue().splitlines()
            self.assertEqual(len(lines), len(messages))
            self.assertTrue(all(json.loads(line)["jsonrpc"] == "2.0" for line in lines))
            self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(FakeSession.instances, 1)

    def test_mcp_cannot_dispatch_internal_shutdown(self) -> None:
        adapter = MCPAdapter(self.config, stderr=io.StringIO())
        response = adapter.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                   "params": {"name": "shutdown", "arguments": {}}})
        self.assertEqual(response["error"]["code"], -32602)
        self.assertEqual(request(self.config.paths.socket, "ping")["state"], "ready")

    def test_shutdown_cleans_socket(self) -> None:
        request(self.config.paths.socket, "shutdown")
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.config.paths.socket.exists())

    def test_tool_schemas_match_legacy_server(self) -> None:
        self.assertEqual(archive_tools(), [_context_tool(), _search_tool()])


class StaleSocketTests(unittest.TestCase):
    def test_closed_listener_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "stale.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            listener.close()
            self.assertEqual(socket_state(path), "stale")

    def test_service_without_database_fails_before_socket_creation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            paths = PthaPaths(root_path / "config", root_path / "data", root_path / "cache", root_path / "state",
                              root_path / "logs", root_path / "run")
            config = replace(load_config(root_path / "missing.toml"), paths=paths, database=paths.database)
            with self.assertRaises(DatabaseNotReadyError):
                RetrievalService(config, session_factory=lambda _config: FakeSession()).run()
            self.assertFalse(paths.socket.exists())

    def test_adapter_without_service_returns_sanitized_error(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            paths = PthaPaths(root_path / "config", root_path / "data", root_path / "cache", root_path / "state",
                              root_path / "logs", root_path / "run")
            base = load_config(root_path / "missing.toml")
            config = replace(base, paths=paths)
            stderr = io.StringIO()
            response = MCPAdapter(config, stderr=stderr).handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            self.assertEqual(response["error"]["code"], -32001)
            self.assertIn("ptha service run", stderr.getvalue())
            self.assertNotIn(str(config.database), json.dumps(response))

    def test_adapter_module_has_no_provider_database_or_session_imports(self) -> None:
        import ptha.mcp as adapter_module
        tree = ast.parse(Path(adapter_module.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any("embeddings" in name or "database" in name or "ArchiveSession" in name for name in imported))

    def test_signal_handler_requests_shutdown(self) -> None:
        class Target:
            called = False

            def request_shutdown(self) -> None:
                self.called = True

        target = Target()
        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)
        try:
            install_signal_handlers(target)  # type: ignore[arg-type]
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(target.called)
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)


class RealArchiveSessionIntegrationTests(unittest.TestCase):
    def test_synthetic_native_db_retrieves_through_service_and_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            paths = PthaPaths(root_path / "config", root_path / "data", root_path / "cache", root_path / "state",
                              root_path / "logs", root_path / "run")
            paths.config_dir.mkdir()
            paths.config_file.write_text("config_version = 1\n", encoding="utf-8")
            _create_retrievable_database(paths.database)
            base = load_config(paths.config_file)
            config = replace(base, paths=paths, database=paths.database, dense_model="synthetic-dense",
                             sparse_model="synthetic-sparse", candidate_pool=10)
            sessions: list[ArchiveSession] = []

            def session_factory(_config: object) -> ArchiveSession:
                session = ArchiveSession(ArchiveConfig(paths.database, candidate_pool=10),
                                         SyntheticDenseProvider(), SyntheticSparseProvider())
                sessions.append(session)
                return session

            service = RetrievalService(config, session_factory=session_factory)
            thread = threading.Thread(target=service.run)
            thread.start()
            _wait_ready(paths.socket)
            try:
                adapter = MCPAdapter(config, stderr=io.StringIO())
                search = adapter.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": "search_archive", "arguments": {"query": "synthetic memory"}}})
                context = adapter.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                          "params": {"name": "construct_archive_context",
                                                     "arguments": {"current_context": "synthetic memory"}}})
                self.assertFalse(search["result"]["isError"])
                self.assertFalse(context["result"]["isError"])
                search_payload = json.loads(search["result"]["content"][0]["text"])
                self.assertEqual(search_payload["items"][0]["text"], "synthetic memory")
                self.assertEqual(len(sessions), 1)
            finally:
                request(paths.socket, "shutdown")
                thread.join(timeout=3)
            self.assertFalse(paths.socket.exists())


def _create_ready_database(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
            CREATE TABLE source_documents(id TEXT);
            CREATE TABLE conversations(id TEXT);
            CREATE TABLE messages(id TEXT);
            CREATE TABLE blocks(id TEXT);
            CREATE TABLE retrieval_chunks(id TEXT);
            CREATE TABLE dense_native_metadata(id TEXT);
            CREATE TABLE sparse_vector_metadata(id TEXT);
            CREATE TABLE sparse_vectors_compact(id TEXT);
            CREATE TABLE native_build_audit(id INTEGER, schema_version TEXT, status TEXT, started_at TEXT,
                                             finished_at TEXT, contracts_json TEXT);
            INSERT INTO native_build_audit VALUES(1, 'native-pre-mvp-v1', 'completed', 'start', 'finish', '{}');
        """)


def _create_retrievable_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense, sparse = SyntheticDenseProvider(), SyntheticSparseProvider()
    policy = build_chunk_policy([dense, sparse])
    with NativeBuildStore(path) as store:
        store.conn.execute("INSERT INTO source_documents VALUES('src','synthetic.md','synthetic.md','chat_md',NULL,'normal',NULL,NULL,'synthetic.md','md','hash',NULL,NULL,'{}')")
        store.conn.execute("INSERT INTO conversations VALUES('conv','src','conversation',NULL,'Synthetic',NULL,NULL,1,0,1,16,0,NULL,NULL,'{}')")
        store.conn.execute("INSERT INTO messages VALUES('msg','conv',1,'user','source-msg',NULL,'synthetic memory','{}')")
        store.conn.execute(
            "INSERT INTO blocks(id,message_id,parent_block_id,ordinal,block_type,language,"
            "source_char_start,source_char_end,raw_content,canonical_content,canonical_content_hash,"
            "parser_version,canonicalizer_version,semantic_status,dense_index_policy,sparse_index_policy,"
            "graph_eligibility,artifact_policy,context_policy,exclusion_reasons_json,metadata_json) "
            "VALUES('block','msg',NULL,1,'prose',NULL,0,16,'synthetic','synthetic','hash',"
            "'test-parser','test-canonicalizer','graph_eligible','include','include',1,'no','include','[]','{}')"
        )
        store.conn.execute("INSERT INTO retrieval_chunks VALUES('chunk','block',1,0,16,2,'synthetic memory',?,'{}')", (policy.id,))
        rows = store.conn.execute("SELECT id,block_id,text FROM retrieval_chunks").fetchall()
        store.write_embedding_batch(rows=rows, dense_vectors=[[1.0] + [0.0] * 1023],
                                    sparse_vectors=[{"synthetic": 1.0}], dense_model=dense.model_name,
                                    dense_space=_chunked_space(dense.embedding_space_id, policy.id),
                                    sparse_model=sparse.model_name,
                                    sparse_space=_chunked_space(sparse.embedding_space_id, policy.id))
        store.conn.execute("INSERT INTO native_build_audit VALUES(1,'kb.native_pre_mvp.v1','synthetic','completed','start','finish','{}','{}')")
        store.commit()


def _wait_ready(path: Path) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            if request(path, "ping", timeout_ms=100).get("state") == "ready":
                return
        except Exception:
            time.sleep(0.01)
    raise AssertionError("service did not become ready")


if __name__ == "__main__":
    unittest.main()
