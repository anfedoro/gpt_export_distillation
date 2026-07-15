"""Unified PTHA command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

from ptha import application_version
from ptha.config import DEFAULT_CONFIG_TEXT, config_path, load_config
from ptha.database import inspect_database
from ptha.errors import PthaError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptha", description="Personal Thought Archive")
    parser.add_argument("--config", help="Path to config.toml.")
    parser.add_argument("--debug", action="store_true", help="Show local tracebacks.")
    parser.add_argument("--version", action="version", version=f"PTHA {application_version()}")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create platform directories and default configuration.")
    init.add_argument("--force", action="store_true", help="Replace the existing configuration.")
    status = commands.add_parser("status", help="Show database, model, and service state.")
    status.add_argument("--json", action="store_true", help="Print stable JSON output.")
    doctor = commands.add_parser("doctor", help="Validate the local PTHA installation.")
    doctor.add_argument("--full", action="store_true", help="Also load models and run retrieval.")
    doctor.add_argument("--query", help="Optional user-supplied full-doctor smoke query.")
    doctor.add_argument("--json", action="store_true", help="Print stable JSON output.")
    import_command = commands.add_parser("import", help="Import a ChatGPT export into the native database.")
    import_command.add_argument("source", type=Path)
    import_command.add_argument("--db", type=Path)
    import_command.add_argument("--working-dir", type=Path)
    import_command.add_argument("--keep-distilled", action="store_true")
    import_command.add_argument("--replace", action="store_true")
    import_command.add_argument("--discard-failed", action="store_true", help="Discard a failed resumable import and start fresh.")
    import_command.add_argument("--include-low-interest", action="store_true")
    import_command.add_argument("--dense-device")
    import_command.add_argument("--sparse-device")
    import_command.add_argument("--dense-dtype")
    import_command.add_argument("--sparse-dtype")
    import_command.add_argument("--batch-size", type=int)
    import_command.add_argument("--quiet", action="store_true")
    import_command.add_argument("--json", action="store_true")
    reindex = commands.add_parser("reindex", help="Atomically rebuild all derived retrieval structures.")
    reindex.add_argument("--force", action="store_true", help="Clean a proven interrupted reindex and restart it.")
    reindex.add_argument("--batch-size", type=int)
    reindex.add_argument("--dense-device")
    reindex.add_argument("--sparse-device")
    reindex.add_argument("--quiet", action="store_true")
    reindex.add_argument("--json", action="store_true")
    service = commands.add_parser("service", help="Run or inspect the local retrieval service.")
    service_commands = service.add_subparsers(dest="service_command", required=True)
    service_commands.add_parser("run", help="Run the retrieval service in the foreground.")
    service_start = service_commands.add_parser("start", help="Start the retrieval service in the background.")
    service_start.add_argument("--timeout", type=float)
    service_start.add_argument("--json", action="store_true")
    service_stop = service_commands.add_parser("stop", help="Stop the background retrieval service.")
    service_stop.add_argument("--timeout", type=float)
    service_stop.add_argument("--force", action="store_true")
    service_stop.add_argument("--json", action="store_true")
    service_restart = service_commands.add_parser("restart", help="Stop and start the background retrieval service.")
    service_restart.add_argument("--start-timeout", type=float)
    service_restart.add_argument("--stop-timeout", type=float)
    service_restart.add_argument("--force", action="store_true")
    service_restart.add_argument("--json", action="store_true")
    service_cleanup = service_commands.add_parser("cleanup", help="Remove proven stale lifecycle state without signalling processes.")
    service_cleanup.add_argument("--force-state", action="store_true")
    service_cleanup.add_argument("--json", action="store_true")
    service_status = service_commands.add_parser("status", help="Query service status through local IPC.")
    service_status.add_argument("--json", action="store_true")
    mcp = commands.add_parser("mcp", help="Run or configure the stdio MCP adapter.")
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_commands.add_parser("serve", help="Run the lightweight stdio MCP adapter.")
    mcp_config = mcp_commands.add_parser("config", help="Print MCP configuration.")
    mcp_config.add_argument("--absolute", action="store_true",
                            help="Print a complete configuration with this executable and config path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            return _init(args)
        selected_config = config_path(args.config)
        if args.command == "import":
            created_paths = _ensure_initialized(selected_config)
            if created_paths and not args.quiet and not args.json:
                print("Created PTHA paths:\n  " + "\n  ".join(str(path) for path in created_paths), file=sys.stderr)
        config = load_config(selected_config)
        if args.command == "status":
            return _status(config, json_output=args.json)
        if args.command == "doctor":
            return _doctor(config, full=args.full, query=args.query, json_output=args.json)
        if args.command == "import":
            from ptha.importer import import_archive
            config = load_config(config.config_file, overrides={
                "database": args.db, "working_dir": args.working_dir, "dense_device": args.dense_device,
                "sparse_device": args.sparse_device, "dense_dtype": args.dense_dtype,
                "sparse_dtype": args.sparse_dtype, "batch_size": args.batch_size,
            })
            report = import_archive(args.source, config, replace=args.replace,
                                    keep_distilled=args.keep_distilled,
                                    include_low_interest=args.include_low_interest,
                                    discard_failed=args.discard_failed,
                                    progress=None if args.quiet or args.json else lambda value: print(value, file=sys.stderr))
            if args.json:
                print(json.dumps({"schema_version": 1, **report}, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                metadata = report["metadata"]
                print(f"Import completed.\n\nDatabase:\n  {report['database']}\n\nContent:\n  Conversations: {metadata['conversation_count']}\n  Messages: {metadata['message_count']}\n  Retrieval chunks: {metadata['retrieval_chunk_count']}\n\nNext:\n  ptha doctor\n  ptha service start")
            return 0
        if args.command == "reindex":
            from ptha.reindex import reindex_database
            report = reindex_database(config, force=args.force, batch_size=args.batch_size,
                                      dense_device=args.dense_device, sparse_device=args.sparse_device,
                                      progress=None if args.quiet or args.json else lambda value: print(value, file=sys.stderr))
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(f"Reindex completed.\n\nDatabase:\n  {report['database']}\n\nCanonical SHA-256:\n  {report['canonical_sha256']}\n\nRetrieval chunks:\n  {report['retrieval_chunk_count']}\n\nDuration:\n  {_duration(report['duration_seconds'])}")
            return 0
        if args.command == "service" and args.service_command == "run":
            from ptha.service import RetrievalService, configure_service_logging, install_signal_handlers
            configure_service_logging(config)
            service = RetrievalService(config)
            install_signal_handlers(service)
            print("Starting PTHA retrieval service...", file=sys.stderr)
            service.run()
            return 0
        if args.command == "service" and args.service_command == "status":
            return _service_status(config, json_output=args.json)
        if args.command == "service" and args.service_command == "start":
            from ptha.lifecycle import start_service
            result = start_service(config, timeout=args.timeout)
            _print_lifecycle("start", result, json_output=args.json, config=config)
            return 0
        if args.command == "service" and args.service_command == "stop":
            from ptha.lifecycle import stop_service
            result = stop_service(config, timeout=args.timeout, force=args.force)
            _print_lifecycle("stop", result, json_output=args.json, config=config)
            return 0
        if args.command == "service" and args.service_command == "restart":
            from ptha.lifecycle import restart_service
            result = restart_service(config, start_timeout=args.start_timeout,
                                     stop_timeout=args.stop_timeout, force=args.force)
            _print_lifecycle("restart", result, json_output=args.json, config=config)
            return 0
        if args.command == "service" and args.service_command == "cleanup":
            from ptha.lifecycle import cleanup_service_state
            result = cleanup_service_state(config, force_state=args.force_state)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print("PTHA stale lifecycle state cleaned.\n\nNo process was signalled.")
            return 0
        if args.command == "mcp" and args.mcp_command == "serve":
            from ptha.mcp import serve_stdio
            serve_stdio(config)
            return 0
        if args.command == "mcp" and args.mcp_command == "config":
            if args.absolute:
                executable = shutil.which("ptha") or sys.argv[0]
                config_file = str(config.config_file.expanduser().resolve())
                print(json.dumps({"mcpServers": {"ptha": {
                    "command": str(Path(executable).expanduser().resolve()),
                    "args": ["--config", config_file, "mcp", "serve"],
                }}}, indent=2))
            else:
                print(json.dumps({"command": "ptha", "args": ["mcp", "serve"]}, indent=2))
            return 0
        raise PthaError("Command is not implemented.")
    except KeyboardInterrupt:
        if args.command == "import":
            message = ("PTHA import interrupted.\n\n"
                       "Checkpoint data was preserved; the active database was not changed.\n\n"
                       "Next:\n"
                       "  ptha import /path/to/chatgpt-export.zip\n"
                       "\nTo discard the checkpoint and start over:\n"
                       "  ptha import /path/to/chatgpt-export.zip --discard-failed")
        elif args.command == "reindex":
            message = ("PTHA reindex interrupted.\n\n"
                       "The active database was not changed. Recovery state was preserved for inspection.\n\n"
                       "Next:\n"
                       "  ptha doctor\n"
                       "  ptha reindex --force")
        else:
            message = "PTHA operation interrupted."
        print(message, file=sys.stderr)
        return 130
    except PthaError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"schema_version": 1, "ok": False,
                              "error": {"code": exc.code, "message": str(exc)}},
                             ensure_ascii=False, indent=2, sort_keys=True))
        else:
            _print_error(exc.code, str(exc), debug=args.debug)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001
        _print_error("internal_error", "PTHA could not complete the command.", debug=args.debug)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1


def _init(args: argparse.Namespace) -> int:
    selected = config_path(args.config)
    existed = selected.exists()
    if args.force and existed:
        selected.unlink()
    _ensure_initialized(selected)
    config = load_config(selected)
    locations = config.paths
    if args.force or not existed:
        selected.parent.mkdir(parents=True, exist_ok=True)
        created = True
    else:
        created = False
    print("PTHA initialized." if created else "PTHA is already initialized.")
    print(f"\nConfiguration:\n  {selected}\n\nDatabase:\n  {config.database}\n\nLogs:\n  {locations.log_dir}/")
    print("\nNext:\n  ptha import /path/to/chatgpt-export.zip")
    return 0


def _status(config: Any, *, json_output: bool) -> int:
    database = inspect_database(config.database)
    from ptha.ipc import IPCError, request
    try:
        service = request(config.paths.socket, "status", timeout_ms=500)
        service["socket"] = str(config.paths.socket)
    except IPCError:
        service = {"state": "stopped", "socket": str(config.paths.socket), "models_loaded": False}
    payload = {
        "schema_version": 1,
        "configuration": {"path": str(config.config_file), "exists": config.config_file.is_file()},
        "database": database,
        "models": {"dense": config.dense_model, "sparse": config.sparse_model, "cache": str(config.model_cache) if config.model_cache else "huggingface-default"},
        "service": service,
        "logs": {"path": str(config.paths.log_dir)},
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        counts = database.get("counts", {})
        print("PTHA status")
        print(f"\nConfiguration:\n  {config.config_file}")
        print(f"\nDatabase:\n  {config.database}\n  State: {database['state']}\n  Conversations: {counts.get('conversations', 0)}\n  Messages: {counts.get('messages', 0)}\n  Retrieval chunks: {counts.get('retrieval_chunks', 0)}")
        incremental = database.get("incremental_metadata", {})
        generation = incremental.get("generation", {}) if isinstance(incremental, dict) else {}
        print(f"\nIncremental metadata:\n  {'available' if incremental.get('available') else 'unavailable for this legacy generation'}")
        if generation.get("id"):
            print(f"  Generation: {generation['id']}")
        print(f"\nModels:\n  Dense: {config.dense_model}\n  Sparse: {config.sparse_model}\n  Cache: {payload['models']['cache']}")
        print(f"\nService:\n  State: {service['state']}\n  Socket: {config.paths.socket}")
    return 0


def _doctor(config: Any, *, full: bool, query: str | None, json_output: bool) -> int:
    from ptha.doctor import run_doctor
    payload = run_doctor(config, full=full, query=query)
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PTHA doctor\n")
        for item in payload["checks"]:
            print(f"{item['status'].upper():5} {item['id']}\n  {item['message']}")
            if item["remediation"]:
                print(f"  Action: {item['remediation']}")
        summary = payload["summary"]
        print(f"\nResult:\n  {summary['passed']} passed\n  {summary['warnings']} warning(s)\n  {summary['failed']} failed")
    return 0 if payload["result"] == "pass" else 1


def _print_error(code: str, message: str, *, debug: bool) -> None:
    print(f"PTHA error [{code}]: {message}", file=sys.stderr)


def _ensure_initialized(selected: Path) -> list[Path]:
    config = load_config(selected)
    locations = config.paths
    created: list[Path] = []
    for directory in (locations.config_dir, locations.data_dir, locations.cache_dir, locations.state_dir,
                      locations.log_dir, locations.runtime_dir, selected.parent):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    if not selected.exists():
        selected.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
        os.chmod(selected, 0o600)
        created.append(selected)
    return created


def _service_status(config: Any, *, json_output: bool) -> int:
    from ptha.lifecycle import service_status
    result = service_status(config)
    if json_output:
        print(json.dumps({"schema_version": 1, **result}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"PTHA service\n\nState:\n  {result['state']}")
        if result.get("pid"):
            print(f"\nPID:\n  {result['pid']}")
        print(f"\nModels:\n  {'loaded' if result.get('models_loaded') else 'not loaded'}")
        if result.get("uptime_seconds") is not None:
            print(f"\nUptime:\n  {_duration(float(result['uptime_seconds']))}")
        print(f"\nActive requests:\n  {result.get('active_requests') or 0}\n\nSocket:\n  {result['socket_path']}\n\nLog:\n  {result['log_path']}")
    return 0 if result["state"] == "ready" else 1


def _print_lifecycle(action: str, result: dict[str, Any], *, json_output: bool, config: Any) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if action == "start":
        heading = "PTHA service is already running." if result.get("already_running") else "PTHA service started."
        print(f"{heading}\n\nState:\n  {result['state']}\n\nPID:\n  {result.get('pid')}\n\nSocket:\n  {config.paths.socket}\n\nDatabase:\n  {config.database}\n\nModels:\n  {'loaded' if result.get('models_loaded') else 'not loaded'}\n\nLog:\n  {config.paths.service_log}")
    elif action == "stop":
        heading = "PTHA service is already stopped." if result.get("already_stopped") else "PTHA service stopped."
        print(f"{heading}\n\nState:\n  {result['state']}\n\nLog:\n  {config.paths.service_log}")
    else:
        print(f"PTHA service restarted.\n\nOld PID:\n  {result.get('old_pid')}\n\nNew PID:\n  {result.get('new_pid')}\n\nState:\n  {result['start']['state']}\n\nLog:\n  {config.paths.service_log}")


def _duration(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
