from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypeVar

from kb.embeddings.mock_provider import MockDenseProvider, MockSparseProvider
from kb.embeddings.sentence_transformer_provider import (
    SentenceTransformerDenseProvider,
    SentenceTransformerSparseProvider,
)
from kb.block_chunk_audit import audit_block_chunks
from kb.storage_audit import audit_storage
from kb.storage.dense_native import audit_dense_native, migrate_dense_native, write_dense_native_report
from kb.index.chunk_builder import ChunkPolicy, build_chunk_policy
from kb.index.edge_builder import build_similarity_edges
from kb.index.semantic_node_builder import build_deterministic_nodes
from kb.ingest.attachment_parser import parse_attachment
from kb.ingest.chat_md_parser import parse_chat_file, write_parsed_chat_json
from kb.ingest.tree_walker import scan_tree, write_inventory_jsonl
from kb.storage.sqlite_store import SQLiteStore, init_db


T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect a local ChatGPT export knowledge DB.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan a distilled Markdown export tree.")
    scan.add_argument("--input", required=True)
    scan.add_argument("--output", required=True)

    parse_chat = sub.add_parser("parse-chat", help="Parse one chat markdown file.")
    parse_chat.add_argument("path")
    parse_chat.add_argument("--json", required=True, dest="json_output")

    init = sub.add_parser("init-db", help="Create or migrate the SQLite DB.")
    init.add_argument("--db", required=True)

    ingest = sub.add_parser("ingest-chats", help="Scan and ingest chat markdown files into SQLite.")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--db", required=True)
    ingest.add_argument("--limit", type=int)
    ingest.add_argument("--project")

    import_cmd = sub.add_parser("import", help="Build a query-ready knowledge DB from a distilled export tree.")
    import_cmd.add_argument("--input", required=True)
    import_cmd.add_argument("--db", required=True)
    import_cmd.add_argument("--limit", type=int, help="Limit chat markdown ingestion; attachments are still scanned.")
    import_cmd.add_argument("--project")
    import_cmd.add_argument("--provider", choices=["sentence-transformers", "mock"], default="sentence-transformers")
    import_cmd.add_argument("--dense-provider", choices=["sentence-transformers", "mock", "none"])
    import_cmd.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    import_cmd.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    import_cmd.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
    import_cmd.add_argument("--dense-device")
    import_cmd.add_argument("--sparse-device")
    import_cmd.add_argument("--dense-backend", choices=["torch", "onnx", "openvino"], default="torch")
    import_cmd.add_argument("--sparse-backend", choices=["torch", "onnx", "openvino"], default="torch")
    import_cmd.add_argument("--dense-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    import_cmd.add_argument("--sparse-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    import_cmd.add_argument("--dense-torch-compile", action="store_true")
    import_cmd.add_argument("--sparse-torch-compile", action="store_true")
    import_cmd.add_argument("--dense-effective-max-seq-length", type=int)
    import_cmd.add_argument("--chunk-policy", choices=["canonical_token_chunks:v2"], default="canonical_token_chunks:v2")
    import_cmd.add_argument("--chunk-content-budget", type=int)
    import_cmd.add_argument("--sparse-top-k", type=int, default=128)
    import_cmd.add_argument("--batch-size", type=int, default=32)
    import_cmd.add_argument("--embedding-pass-mode", choices=["joint", "separate"], default="separate")
    import_cmd.add_argument("--memory-report-every", type=int, default=0, help="Print process memory every N embedding batches.")
    import_cmd.add_argument(
        "--dependency-log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="warning",
        help="Enable Python logging for model dependencies such as sentence-transformers, transformers, torch, and huggingface_hub.",
    )
    import_cmd.add_argument("--force-embeddings", action="store_true")
    import_cmd.add_argument("--skip-low-interest-content", action=argparse.BooleanOptionalAction, default=True)
    import_cmd.add_argument("--edge-scope", choices=["conversation", "project", "attachment"], default="project")
    import_cmd.add_argument("--edge-top-k", type=int, default=10)
    import_cmd.add_argument("--edge-max-group-size", type=int, default=1000)
    import_cmd.add_argument("--no-attachments", action="store_true")
    import_cmd.add_argument("--no-embeddings", action="store_true")
    import_cmd.add_argument("--no-nodes", action="store_true")
    import_cmd.add_argument("--no-edges", action="store_true")
    import_cmd.add_argument("--quiet", action="store_true", help="Disable progress output on stderr.")

    ingest_attachments_parser = sub.add_parser("ingest-attachments", help="Extract supported attachments into knowledge blocks.")
    ingest_attachments_parser.add_argument("--input", required=True)
    ingest_attachments_parser.add_argument("--db", required=True)
    ingest_attachments_parser.add_argument("--limit", type=int)
    ingest_attachments_parser.add_argument("--project")

    embed = sub.add_parser("embed", help="Embed knowledge blocks with pluggable providers.")
    embed.add_argument("--db", required=True)
    embed.add_argument("--provider", choices=["sentence-transformers", "mock"], default="sentence-transformers")
    embed.add_argument("--dense-provider", choices=["sentence-transformers", "mock", "none"])
    embed.add_argument("--sparse-provider", choices=["sentence-transformers", "mock", "none"], default="sentence-transformers")
    embed.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    embed.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
    embed.add_argument("--dense-device")
    embed.add_argument("--sparse-device")
    embed.add_argument("--dense-backend", choices=["torch", "onnx", "openvino"], default="torch")
    embed.add_argument("--sparse-backend", choices=["torch", "onnx", "openvino"], default="torch")
    embed.add_argument("--dense-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    embed.add_argument("--sparse-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    embed.add_argument("--dense-torch-compile", action="store_true")
    embed.add_argument("--sparse-torch-compile", action="store_true")
    embed.add_argument("--dense-effective-max-seq-length", type=int)
    embed.add_argument("--chunk-policy", choices=["canonical_token_chunks:v2"], default="canonical_token_chunks:v2")
    embed.add_argument("--chunk-content-budget", type=int)
    embed.add_argument("--sparse-top-k", type=int, default=128)
    embed.add_argument("--limit", type=int)
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--embedding-pass-mode", choices=["joint", "separate"], default="separate")
    embed.add_argument("--memory-report-every", type=int, default=0, help="Print process memory every N batches.")
    embed.add_argument(
        "--dependency-log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="warning",
        help="Enable Python logging for model dependencies such as sentence-transformers, transformers, torch, and huggingface_hub.",
    )
    embed.add_argument("--force", action="store_true")
    embed.add_argument("--skip-low-interest-content", action=argparse.BooleanOptionalAction, default=True)
    embed.add_argument("--quiet", action="store_true", help="Disable progress output on stderr.")

    audit_blocks = sub.add_parser("audit-block-chunks", help="Audit structural block to retrieval chunk distribution.")
    audit_blocks.add_argument("--db", required=True)
    audit_blocks.add_argument("--output-dir", help="Report directory; defaults to benchmarks/block_chunk_audit/<timestamp>.")

    audit_storage_parser = sub.add_parser("audit-storage", help="Audit SQLite storage without mutating the DB.")
    audit_storage_parser.add_argument("--db", required=True)
    audit_storage_parser.add_argument("--output-dir", help="Report directory; defaults to benchmarks/storage_audit/<timestamp>.")

    migrate_dense_native_parser = sub.add_parser(
        "migrate-dense-native",
        help="Copy a DB and migrate dense JSON vectors into sqlite-vec float32 storage.",
    )
    migrate_dense_native_parser.add_argument("--source-db", required=True)
    migrate_dense_native_parser.add_argument("--target-db", required=True)
    migrate_dense_native_parser.add_argument("--batch-size", type=int, default=256)
    migrate_dense_native_parser.add_argument("--report-dir")

    audit_dense_native_parser = sub.add_parser(
        "audit-dense-native",
        help="Audit an existing sqlite-vec dense migration without modifying it.",
    )
    audit_dense_native_parser.add_argument("--db", required=True)
    audit_dense_native_parser.add_argument("--source-db")
    audit_dense_native_parser.add_argument("--sample-size", type=int, default=1000)
    audit_dense_native_parser.add_argument("--report-dir")

    build_nodes = sub.add_parser("build-nodes", help="Build deterministic semantic nodes.")
    build_nodes.add_argument("--db", required=True)
    build_nodes.add_argument("--mode", choices=["deterministic"], default="deterministic")
    build_nodes.add_argument("--sparse-top-k", type=int, default=50)

    build_edges = sub.add_parser("build-edges", help="Build computed semantic edges.")
    build_edges.add_argument("--db", required=True)
    build_edges.add_argument("--scope", choices=["conversation", "project", "attachment"], default="project")
    build_edges.add_argument("--top-k", type=int, default=10)
    build_edges.add_argument("--max-group-size", type=int, default=1000)
    build_edges.add_argument("--no-dense", action="store_true")
    build_edges.add_argument("--no-sparse", action="store_true")

    stats = sub.add_parser("stats", help="Print DB table counts.")
    stats.add_argument("--db", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "scan":
        input_dir = Path(args.input).expanduser()
        output_path = Path(args.output).expanduser()
        count = write_inventory_jsonl(scan_tree(input_dir), output_path)
        print(f"inventory_items: {count}")
        return

    if args.command == "parse-chat":
        path = Path(args.path).expanduser()
        parsed = parse_chat_file(path, source_document_id=f"parse-only:{path}")
        write_parsed_chat_json(parsed, Path(args.json_output).expanduser())
        print(
            f"messages: {len(parsed.messages)} blocks: {len(parsed.blocks)} "
            f"conversation_id: {parsed.conversation.conversation_id or parsed.conversation.id}"
        )
        return

    if args.command == "init-db":
        init_db(Path(args.db).expanduser())
        print(f"initialized: {args.db}")
        return

    if args.command == "ingest-chats":
        stats = ingest_chats(
            input_dir=Path(args.input).expanduser(),
            db_path=Path(args.db).expanduser(),
            limit=args.limit,
            project=args.project,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "import":
        _configure_dependency_logging(args.dependency_log_level)
        stats = import_knowledge_base(
            input_dir=Path(args.input).expanduser(),
            db_path=Path(args.db).expanduser(),
            limit=args.limit,
            project=args.project,
            provider=args.provider,
            dense_provider=args.dense_provider,
            sparse_provider=args.sparse_provider,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            dense_device=args.dense_device,
            sparse_device=args.sparse_device,
            dense_backend=args.dense_backend,
            sparse_backend=args.sparse_backend,
            dense_torch_dtype=args.dense_torch_dtype,
            sparse_torch_dtype=args.sparse_torch_dtype,
            dense_torch_compile=args.dense_torch_compile,
            sparse_torch_compile=args.sparse_torch_compile,
            dense_effective_max_seq_length=args.dense_effective_max_seq_length,
            chunk_policy_version=_chunk_policy_version(args.chunk_policy),
            chunk_content_budget=args.chunk_content_budget,
            sparse_top_k=args.sparse_top_k,
            batch_size=args.batch_size,
            embedding_pass_mode=args.embedding_pass_mode,
            memory_report_every=args.memory_report_every,
            force_embeddings=args.force_embeddings,
            skip_low_interest_content=args.skip_low_interest_content,
            edge_scope=args.edge_scope,
            edge_top_k=args.edge_top_k,
            edge_max_group_size=args.edge_max_group_size,
            include_attachments=not args.no_attachments,
            include_embeddings=not args.no_embeddings,
            include_nodes=not args.no_nodes,
            include_edges=not args.no_edges,
            progress=not args.quiet,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "ingest-attachments":
        stats = ingest_attachments(
            input_dir=Path(args.input).expanduser(),
            db_path=Path(args.db).expanduser(),
            limit=args.limit,
            project=args.project,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "stats":
        with SQLiteStore(Path(args.db).expanduser()) as store:
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "audit-block-chunks":
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else Path("benchmarks/block_chunk_audit") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        report = audit_block_chunks(Path(args.db).expanduser(), output_dir)
        print(json.dumps({"output_dir": str(output_dir), **report["summary"], "consistency": report["consistency"]}, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "audit-storage":
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else Path("benchmarks/storage_audit") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        report = audit_storage(Path(args.db).expanduser(), output_dir)
        print(json.dumps({"output_dir": str(output_dir), **report["object_totals"], "dense": report["dense"], "sparse": report["sparse"], "recommendation": report["recommendation"]}, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "migrate-dense-native":
        report = migrate_dense_native(
            source_db=Path(args.source_db).expanduser(),
            target_db=Path(args.target_db).expanduser(),
            batch_size=args.batch_size,
            report_dir=Path(args.report_dir).expanduser() if args.report_dir else None,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "audit-dense-native":
        report = audit_dense_native(
            Path(args.db).expanduser(),
            source_db=Path(args.source_db).expanduser() if args.source_db else None,
            sample_size=args.sample_size,
        )
        if args.report_dir:
            write_dense_native_report(report, Path(args.report_dir).expanduser())
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "embed":
        _configure_dependency_logging(args.dependency_log_level)
        stats = embed_knowledge_blocks(
            db_path=Path(args.db).expanduser(),
            provider=args.provider,
            dense_provider=args.dense_provider,
            sparse_provider=args.sparse_provider,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            dense_device=args.dense_device,
            sparse_device=args.sparse_device,
            dense_backend=args.dense_backend,
            sparse_backend=args.sparse_backend,
            dense_torch_dtype=args.dense_torch_dtype,
            sparse_torch_dtype=args.sparse_torch_dtype,
            dense_torch_compile=args.dense_torch_compile,
            sparse_torch_compile=args.sparse_torch_compile,
            chunk_policy_version=_chunk_policy_version(args.chunk_policy),
            sparse_top_k=args.sparse_top_k,
            limit=args.limit,
            batch_size=args.batch_size,
            embedding_pass_mode=args.embedding_pass_mode,
            memory_report_every=args.memory_report_every,
            force=args.force,
            skip_low_interest_content=args.skip_low_interest_content,
            progress=not args.quiet,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "build-nodes":
        stats = build_nodes_command(
            db_path=Path(args.db).expanduser(),
            mode=args.mode,
            sparse_top_k=args.sparse_top_k,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "build-edges":
        stats = build_edges_command(
            db_path=Path(args.db).expanduser(),
            scope=args.scope,
            top_k=args.top_k,
            include_dense=not args.no_dense,
            include_sparse=not args.no_sparse,
            max_group_size=args.max_group_size,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return


def _configure_dependency_logging(level_name: str) -> None:
    level_name = level_name.upper()
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for logger_name in (
        "kb",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "torch",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(level)
    os.environ["TRANSFORMERS_VERBOSITY"] = level_name.lower()
    os.environ["HF_HUB_VERBOSITY"] = level_name.lower()
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity(level)
        transformers_logging.enable_default_handler()
        transformers_logging.enable_explicit_format()
    except Exception:  # noqa: BLE001
        logging.getLogger("kb").debug("transformers logging setup skipped", exc_info=True)
    try:
        import torch

        if level <= logging.DEBUG and hasattr(torch, "_logging"):
            # Torch exposes only selected component logs. If the API changes, keep CLI logging usable.
            torch._logging.set_logs(all=logging.DEBUG)
    except Exception:  # noqa: BLE001
        logging.getLogger("kb").debug("torch logging setup skipped", exc_info=True)


def ingest_chats(input_dir: Path, db_path: Path, limit: int | None = None, project: str | None = None) -> dict[str, int]:
    input_root = input_dir.expanduser().resolve()
    init_db(db_path)
    scanned = 0
    parsed_count = 0
    failed = 0
    skipped = 0
    attachments_seen = 0
    with SQLiteStore(db_path) as store:
        for item in scan_tree(input_root):
            scanned += 1
            if item.is_attachment:
                attachments_seen += 1
            if project and item.project_path != project:
                skipped += 1
                continue
            source_id = store.upsert_source_document(input_root, item)
            if item.detected_kind != "chat_md":
                continue
            if limit is not None and parsed_count >= limit:
                skipped += 1
                continue
            try:
                parsed = parse_chat_file(
                    input_root / item.relative_path,
                    source_document_id=source_id,
                    project_id=item.project_path,
                    folder_kind=item.folder_kind,
                )
                store.insert_parsed_chat(parsed)
                parsed_count += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"failed: {item.relative_path}: {exc}")
        store.commit()
        table_stats = store.stats()
    return {
        **table_stats,
        "scanned": scanned,
        "parsed_chats": parsed_count,
        "failed_chats": failed,
        "skipped": skipped,
        "attachments_seen": attachments_seen,
    }


def ingest_attachments(input_dir: Path, db_path: Path, limit: int | None = None, project: str | None = None) -> dict[str, int]:
    input_root = input_dir.expanduser().resolve()
    init_db(db_path)
    scanned = 0
    attachments_seen = 0
    attempted = 0
    extracted = 0
    unsupported = 0
    failed = 0
    blocks_created = 0
    skipped = 0
    with SQLiteStore(db_path) as store:
        for item in scan_tree(input_root):
            scanned += 1
            if not item.is_attachment:
                continue
            attachments_seen += 1
            if project and item.project_path != project:
                skipped += 1
                continue
            if limit is not None and attempted >= limit:
                skipped += 1
                continue
            source_id = store.upsert_source_document(input_root, item)
            parsed = parse_attachment(input_root / item.relative_path)
            store.insert_parsed_attachment(input_root, item, source_id, parsed)
            attempted += 1
            blocks_created += len(parsed.blocks)
            if parsed.extraction_status == "extracted":
                extracted += 1
            elif parsed.extraction_status == "unsupported":
                unsupported += 1
            else:
                failed += 1
        store.commit()
        table_stats = store.stats()
    return {
        **table_stats,
        "scanned": scanned,
        "attachments_seen": attachments_seen,
        "attempted": attempted,
        "extracted": extracted,
        "unsupported": unsupported,
        "failed": failed,
        "blocks_created": blocks_created,
        "skipped": skipped,
    }


def import_knowledge_base(
    *,
    input_dir: Path,
    db_path: Path,
    limit: int | None = None,
    project: str | None = None,
    provider: str = "sentence-transformers",
    dense_provider: str | None = None,
    sparse_provider: str = "sentence-transformers",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    dense_device: str | None = None,
    sparse_device: str | None = None,
    dense_backend: str = "torch",
    sparse_backend: str = "torch",
    dense_torch_dtype: str = "auto",
    sparse_torch_dtype: str = "auto",
    dense_torch_compile: bool = False,
    sparse_torch_compile: bool = False,
    dense_effective_max_seq_length: int | None = None,
    chunk_policy_version: str = "v2",
    chunk_content_budget: int | None = None,
    sparse_top_k: int = 128,
    batch_size: int = 32,
    embedding_pass_mode: str = "separate",
    memory_report_every: int = 0,
    force_embeddings: bool = False,
    skip_low_interest_content: bool = True,
    edge_scope: str = "project",
    edge_top_k: int = 10,
    edge_max_group_size: int = 1000,
    include_attachments: bool = True,
    include_embeddings: bool = True,
    include_nodes: bool = True,
    include_edges: bool = True,
    progress: bool = False,
) -> dict[str, object]:
    stages: dict[str, object] = {}
    _progress_message("stage 1/5 ingest chats", enabled=progress)
    stages["ingest_chats"] = ingest_chats(input_dir=input_dir, db_path=db_path, limit=limit, project=project)
    if include_attachments:
        _progress_message("stage 2/5 ingest attachments", enabled=progress)
        stages["ingest_attachments"] = ingest_attachments(input_dir=input_dir, db_path=db_path, project=project)
    else:
        stages["ingest_attachments"] = {"skipped": True}
    if include_embeddings:
        _progress_message("stage 3/5 embed knowledge blocks", enabled=progress)
        stages["embed"] = embed_knowledge_blocks(
            db_path=db_path,
            provider=provider,
            dense_provider=dense_provider,
            sparse_provider=sparse_provider,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_device=dense_device,
            sparse_device=sparse_device,
            dense_backend=dense_backend,
            sparse_backend=sparse_backend,
            dense_torch_dtype=dense_torch_dtype,
            sparse_torch_dtype=sparse_torch_dtype,
            dense_torch_compile=dense_torch_compile,
            sparse_torch_compile=sparse_torch_compile,
            dense_effective_max_seq_length=dense_effective_max_seq_length,
            chunk_policy_version=chunk_policy_version,
            chunk_content_budget=chunk_content_budget,
            sparse_top_k=sparse_top_k,
            batch_size=batch_size,
            embedding_pass_mode=embedding_pass_mode,
            memory_report_every=memory_report_every,
            force=force_embeddings,
            skip_low_interest_content=skip_low_interest_content,
            progress=progress,
        )
    else:
        stages["embed"] = {"skipped": True}
    if include_nodes:
        _progress_message("stage 4/5 build semantic nodes", enabled=progress)
        stages["build_nodes"] = build_nodes_command(db_path=db_path, mode="deterministic", sparse_top_k=min(sparse_top_k, 50))
    else:
        stages["build_nodes"] = {"skipped": True}
    if include_edges:
        _progress_message("stage 5/5 build semantic edges", enabled=progress)
        stages["build_edges"] = build_edges_command(
            db_path=db_path,
            scope=edge_scope,
            top_k=edge_top_k,
            max_group_size=edge_max_group_size,
        )
    else:
        stages["build_edges"] = {"skipped": True}
    with SQLiteStore(db_path) as store:
        final_stats = store.stats()
    _progress_message("done", enabled=progress)
    return {"stages": stages, "final": final_stats}


def embed_knowledge_blocks(
    *,
    db_path: Path,
    provider: str = "sentence-transformers",
    dense_provider: str | None = None,
    sparse_provider: str = "sentence-transformers",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    dense_device: str | None = None,
    sparse_device: str | None = None,
    dense_backend: str = "torch",
    sparse_backend: str = "torch",
    dense_torch_dtype: str = "auto",
    sparse_torch_dtype: str = "auto",
    dense_torch_compile: bool = False,
    sparse_torch_compile: bool = False,
    dense_effective_max_seq_length: int | None = None,
    chunk_policy_version: str = "v2",
    chunk_content_budget: int | None = None,
    sparse_top_k: int = 128,
    limit: int | None = None,
    batch_size: int = 32,
    embedding_pass_mode: str = "separate",
    memory_report_every: int = 0,
    force: bool = False,
    skip_low_interest_content: bool = True,
    progress: bool = False,
) -> dict[str, int | float | str | None]:
    if embedding_pass_mode == "separate":
        return _embed_knowledge_blocks_separate(
            db_path=db_path,
            provider=provider,
            dense_provider=dense_provider,
            sparse_provider=sparse_provider,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_device=dense_device,
            sparse_device=sparse_device,
            dense_backend=dense_backend,
            sparse_backend=sparse_backend,
            dense_torch_dtype=dense_torch_dtype,
            sparse_torch_dtype=sparse_torch_dtype,
            dense_torch_compile=dense_torch_compile,
            sparse_torch_compile=sparse_torch_compile,
            dense_effective_max_seq_length=dense_effective_max_seq_length,
            chunk_content_budget=chunk_content_budget,
            sparse_top_k=sparse_top_k,
            limit=limit,
            batch_size=batch_size,
            memory_report_every=memory_report_every,
            force=force,
            skip_low_interest_content=skip_low_interest_content,
            progress=progress,
        )
    if embedding_pass_mode != "joint":
        raise ValueError(f"Unsupported embedding_pass_mode: {embedding_pass_mode}")
    return _embed_knowledge_blocks_joint(
        db_path=db_path,
        provider=provider,
        dense_provider=dense_provider,
        sparse_provider=sparse_provider,
        dense_model=dense_model,
        sparse_model=sparse_model,
        dense_device=dense_device,
        sparse_device=sparse_device,
        dense_backend=dense_backend,
        sparse_backend=sparse_backend,
        dense_torch_dtype=dense_torch_dtype,
        sparse_torch_dtype=sparse_torch_dtype,
        dense_torch_compile=dense_torch_compile,
        sparse_torch_compile=sparse_torch_compile,
        dense_effective_max_seq_length=dense_effective_max_seq_length,
        chunk_policy_version=chunk_policy_version,
        chunk_content_budget=chunk_content_budget,
        sparse_top_k=sparse_top_k,
        limit=limit,
        batch_size=batch_size,
        memory_report_every=memory_report_every,
        force=force,
        skip_low_interest_content=skip_low_interest_content,
        progress=progress,
    )


def _embed_knowledge_blocks_joint(
    *,
    db_path: Path,
    provider: str = "sentence-transformers",
    dense_provider: str | None = None,
    sparse_provider: str = "sentence-transformers",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    dense_device: str | None = None,
    sparse_device: str | None = None,
    dense_backend: str = "torch",
    sparse_backend: str = "torch",
    dense_torch_dtype: str = "auto",
    sparse_torch_dtype: str = "auto",
    dense_torch_compile: bool = False,
    sparse_torch_compile: bool = False,
    dense_effective_max_seq_length: int | None = None,
    chunk_policy_version: str = "v2",
    chunk_content_budget: int | None = None,
    sparse_top_k: int = 128,
    limit: int | None = None,
    batch_size: int = 32,
    memory_report_every: int = 0,
    force: bool = False,
    skip_low_interest_content: bool = True,
    progress: bool = False,
    chunk_policy: ChunkPolicy | None = None,
) -> dict[str, int | float | str | None]:
    dense_name = dense_provider or provider
    dense = _build_dense_provider(
        dense_name,
        dense_model,
        device=dense_device,
        backend=dense_backend,
        torch_dtype=dense_torch_dtype,
        torch_compile=dense_torch_compile,
        effective_max_seq_length=dense_effective_max_seq_length,
    )
    sparse = _build_sparse_provider(
        sparse_provider,
        sparse_model,
        sparse_top_k,
        device=sparse_device,
        backend=sparse_backend,
        torch_dtype=sparse_torch_dtype,
        torch_compile=sparse_torch_compile,
    )
    if dense is None and sparse is None:
        raise ValueError("At least one embedding provider must be enabled.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if memory_report_every < 0:
        raise ValueError("--memory-report-every must be non-negative.")

    init_db(db_path)
    dense_vectors = 0
    sparse_vectors = 0
    sparse_terms = 0
    errors = 0
    dense_dim_total = 0
    sparse_nnz_total = 0
    with SQLiteStore(db_path) as store:
        active_providers = [item for item in (dense, sparse) if item is not None]
        policy = chunk_policy or build_chunk_policy(
            active_providers,
            version=chunk_policy_version,
            content_budget_override=chunk_content_budget,
        )
        tokenizer_provider = min(active_providers, key=lambda item: int(item.effective_max_sequence_length or 0))
        audit = store.rebuild_retrieval_chunks(
            policy=policy,
            tokenizer_provider=tokenizer_provider,
            skip_low_interest_content=skip_low_interest_content,
        )
        _raise_on_failed_chunk_audit(audit)
        dense_space_id = _chunked_embedding_space_id(dense.embedding_space_id, policy.id) if dense else None
        sparse_space_id = _chunked_embedding_space_id(sparse.embedding_space_id, policy.id) if sparse else None
        candidate_count = store.count_retrieval_chunks_for_embedding(
            chunk_policy_id=policy.id,
            limit=limit,
            dense_model_name=dense.model_name if dense else None,
            dense_model_version=dense_space_id if dense else None,
            sparse_model_name=sparse.model_name if sparse else None,
            sparse_embedding_space_id=sparse_space_id if sparse else None,
            force=force,
        )
        total_batches = math.ceil(candidate_count / batch_size) if candidate_count else 0
        batch_iter = _progress_iter(
            range(total_batches),
            enabled=progress,
            total=total_batches,
            description="embedding batches",
            unit="batch",
        )
        after_id = None
        processed_candidates = 0
        peak_rss_mb = _process_memory_mb().get("rss_mb") or _process_memory_mb().get("max_rss_mb") or 0.0
        for batch_index in batch_iter:
            remaining = candidate_count - processed_candidates
            if remaining <= 0:
                break
            batch = store.retrieval_chunks_for_embedding_batch(
                chunk_policy_id=policy.id,
                after_id=after_id,
                batch_size=min(batch_size, remaining),
                dense_model_name=dense.model_name if dense else None,
                dense_model_version=dense_space_id if dense else None,
                sparse_model_name=sparse.model_name if sparse else None,
                sparse_embedding_space_id=sparse_space_id if sparse else None,
                force=force,
            )
            if not batch:
                break
            processed_candidates += len(batch)
            after_id = str(batch[-1]["id"])
            texts = [str(row["text"]) for row in batch]
            for row, text in zip(batch, texts, strict=True):
                if dense is not None:
                    dense.assert_fits(
                        text,
                        chunk_id=str(row["id"]),
                        block_id=str(row["block_id"]),
                        source_identity=str(row["block_id"]),
                    )
                if sparse is not None:
                    sparse.assert_fits(
                        text,
                        chunk_id=str(row["id"]),
                        block_id=str(row["block_id"]),
                        source_identity=str(row["block_id"]),
                    )
            dense_results = dense.embed_documents(texts) if dense else [None] * len(batch)
            sparse_results = sparse.embed_documents(texts) if sparse else [None] * len(batch)
            batch_sparse_terms_count = sum(len(sparse_vector) for sparse_vector in sparse_results if sparse_vector)
            pending_insert_rows_count = 0
            for row, dense_vector, sparse_vector in zip(batch, dense_results, sparse_results, strict=True):
                try:
                    owner_id = str(row["id"])
                    dense_vector_id = None
                    sparse_vector_id = None
                    if dense_vector is not None and dense is not None:
                        dense_vector_id = store.upsert_dense_vector(
                            owner_type="retrieval_chunk",
                            owner_id=owner_id,
                            model_name=dense.model_name,
                            model_version=dense_space_id,
                            runtime_metadata_json=json.dumps(dense.runtime_metadata, sort_keys=True, separators=(",", ":")),
                            vector=dense_vector,
                        )
                        dense_vectors += 1
                        dense_dim_total += len(dense_vector)
                    if sparse_vector is not None and sparse is not None:
                        pending_insert_rows_count += len(sparse_vector)
                        sparse_vector_id = store.replace_sparse_terms(
                            owner_type="retrieval_chunk",
                            owner_id=owner_id,
                            model_name=sparse.model_name,
                            embedding_space_id=sparse_space_id,
                            terms=sparse_vector,
                        )
                        sparse_vectors += 1
                        sparse_terms += len(sparse_vector)
                        sparse_nnz_total += len(sparse_vector)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"failed embedding retrieval_chunk {row['id']}: {exc}")
            store.commit()
            store.shrink_memory()
            memory_before_cleanup = _process_memory_mb()
            del texts, dense_results, sparse_results, batch
            _release_batch_memory()
            memory = _process_memory_mb()
            peak_rss_mb = max(peak_rss_mb, memory.get("rss_mb") or memory.get("max_rss_mb") or 0.0)
            if memory_report_every and ((batch_index + 1) % memory_report_every == 0 or processed_candidates >= candidate_count):
                _progress_message(
                    f"memory batch={batch_index + 1}/{total_batches} processed={processed_candidates}/{candidate_count} "
                    f"rss_before_cleanup_mb={memory_before_cleanup.get('rss_mb', 0.0):.1f} "
                    f"rss_mb={memory.get('rss_mb', 0.0):.1f} max_rss_mb={memory.get('max_rss_mb', 0.0):.1f} "
                    f"rss_delta_after_gc_mb={memory.get('rss_mb', 0.0) - memory_before_cleanup.get('rss_mb', 0.0):.1f} "
                    f"mps_current_mb={memory.get('mps_current_mb', 0.0):.1f} "
                    f"mps_driver_mb={memory.get('mps_driver_mb', 0.0):.1f} "
                    f"cuda_allocated_mb={memory.get('cuda_allocated_mb', 0.0):.1f} "
                    f"cuda_reserved_mb={memory.get('cuda_reserved_mb', 0.0):.1f} "
                    f"batch_sparse_terms_count={batch_sparse_terms_count} "
                    f"pending_insert_rows_count={pending_insert_rows_count} retained_terms_buffer_count=0",
                    enabled=True,
                )
        store.commit()
    return {
        "dense_model": dense.model_name if dense else None,
        "dense_model_version": dense_space_id if dense else None,
        "dense_embedding_space_id": dense_space_id if dense else None,
        "sparse_model": sparse.model_name if sparse else None,
        "sparse_model_version": sparse_space_id if sparse else None,
        "sparse_embedding_space_id": sparse_space_id if sparse else None,
        "chunk_policy_id": policy.id,
        "indexing_audit": audit,
        "candidate_blocks": candidate_count,
        "candidate_chunks": candidate_count,
        "blocks_embedded": max(dense_vectors, sparse_vectors),
        "chunks_embedded": max(dense_vectors, sparse_vectors),
        "dense_vectors": dense_vectors,
        "sparse_vectors": sparse_vectors,
        "sparse_terms": sparse_terms,
        "avg_dense_dim": dense_dim_total / dense_vectors if dense_vectors else 0,
        "avg_sparse_non_zero_count": sparse_nnz_total / sparse_vectors if sparse_vectors else 0,
        "peak_rss_mb": peak_rss_mb,
        "errors": errors,
    }


def _embed_knowledge_blocks_separate(
    *,
    db_path: Path,
    provider: str,
    dense_provider: str | None,
    sparse_provider: str,
    dense_model: str,
    sparse_model: str,
    dense_device: str | None,
    sparse_device: str | None,
    dense_backend: str,
    sparse_backend: str,
    dense_torch_dtype: str,
    sparse_torch_dtype: str,
    dense_torch_compile: bool,
    sparse_torch_compile: bool,
    dense_effective_max_seq_length: int | None = None,
    chunk_policy_version: str = "v2",
    chunk_content_budget: int | None = None,
    sparse_top_k: int,
    limit: int | None,
    batch_size: int,
    memory_report_every: int,
    force: bool,
    skip_low_interest_content: bool,
    progress: bool,
) -> dict[str, int | float | str | None | dict[str, int | float | str | None]]:
    dense_name = dense_provider or provider
    if dense_name == "none" or sparse_provider == "none":
        return _embed_knowledge_blocks_joint(
            db_path=db_path,
            provider=provider,
            dense_provider=dense_provider,
            sparse_provider=sparse_provider,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_device=dense_device,
            sparse_device=sparse_device,
            dense_backend=dense_backend,
            sparse_backend=sparse_backend,
            dense_torch_dtype=dense_torch_dtype,
            sparse_torch_dtype=sparse_torch_dtype,
            dense_torch_compile=dense_torch_compile,
            sparse_torch_compile=sparse_torch_compile,
            dense_effective_max_seq_length=dense_effective_max_seq_length,
            chunk_policy_version=chunk_policy_version,
            chunk_content_budget=chunk_content_budget,
            sparse_top_k=sparse_top_k,
            limit=limit,
            batch_size=batch_size,
            memory_report_every=memory_report_every,
            force=force,
            skip_low_interest_content=skip_low_interest_content,
            progress=progress,
        )
    _progress_message("embedding pass 1/2 dense", enabled=progress)
    dense_stats = _embed_knowledge_blocks_joint(
        db_path=db_path,
        provider=provider,
        dense_provider=dense_provider,
        sparse_provider="none",
        dense_model=dense_model,
        sparse_model=sparse_model,
        dense_device=dense_device,
        sparse_device=sparse_device,
        dense_backend=dense_backend,
        sparse_backend=sparse_backend,
        dense_torch_dtype=dense_torch_dtype,
        sparse_torch_dtype=sparse_torch_dtype,
        dense_torch_compile=dense_torch_compile,
        sparse_torch_compile=sparse_torch_compile,
        dense_effective_max_seq_length=dense_effective_max_seq_length,
        chunk_policy_version=chunk_policy_version,
        chunk_content_budget=chunk_content_budget,
        sparse_top_k=sparse_top_k,
        limit=limit,
        batch_size=batch_size,
        memory_report_every=memory_report_every,
        force=force,
        skip_low_interest_content=skip_low_interest_content,
        progress=progress,
    )
    shared_policy = _chunk_policy_from_audit(dense_stats.get("indexing_audit"))
    _release_batch_memory()
    _progress_message("embedding pass 2/2 sparse", enabled=progress)
    sparse_stats = _embed_knowledge_blocks_joint(
        db_path=db_path,
        provider=provider,
        dense_provider="none",
        sparse_provider=sparse_provider,
        dense_model=dense_model,
        sparse_model=sparse_model,
        dense_device=dense_device,
        sparse_device=sparse_device,
        dense_backend=dense_backend,
        sparse_backend=sparse_backend,
        dense_torch_dtype=dense_torch_dtype,
        sparse_torch_dtype=sparse_torch_dtype,
        dense_torch_compile=dense_torch_compile,
        sparse_torch_compile=sparse_torch_compile,
        dense_effective_max_seq_length=dense_effective_max_seq_length,
        chunk_policy_version=chunk_policy_version,
        chunk_content_budget=chunk_content_budget,
        sparse_top_k=sparse_top_k,
        limit=limit,
        batch_size=batch_size,
        memory_report_every=memory_report_every,
        force=force,
        skip_low_interest_content=skip_low_interest_content,
        progress=progress,
        chunk_policy=shared_policy,
    )
    return {
        "embedding_pass_mode": "separate",
        "dense_pass": dense_stats,
        "sparse_pass": sparse_stats,
        "dense_model": dense_stats["dense_model"],
        "dense_model_version": dense_stats["dense_model_version"],
        "dense_embedding_space_id": dense_stats["dense_embedding_space_id"],
        "sparse_model": sparse_stats["sparse_model"],
        "sparse_model_version": sparse_stats["sparse_model_version"],
        "sparse_embedding_space_id": sparse_stats["sparse_embedding_space_id"],
        "candidate_blocks": max(int(dense_stats["candidate_blocks"]), int(sparse_stats["candidate_blocks"])),
        "blocks_embedded": max(int(dense_stats["blocks_embedded"]), int(sparse_stats["blocks_embedded"])),
        "dense_vectors": int(dense_stats["dense_vectors"]),
        "sparse_vectors": int(sparse_stats["sparse_vectors"]),
        "sparse_terms": int(sparse_stats["sparse_terms"]),
        "chunk_policy_id": dense_stats["chunk_policy_id"],
        "indexing_audit": dense_stats["indexing_audit"],
        "avg_dense_dim": float(dense_stats["avg_dense_dim"]),
        "avg_sparse_non_zero_count": float(sparse_stats["avg_sparse_non_zero_count"]),
        "peak_rss_mb": max(float(dense_stats["peak_rss_mb"]), float(sparse_stats["peak_rss_mb"])),
        "errors": int(dense_stats["errors"]) + int(sparse_stats["errors"]),
    }


def _chunk_policy_from_audit(audit: object) -> ChunkPolicy | None:
    if not isinstance(audit, dict):
        return None
    policy_id = audit.get("chunk_policy_id")
    max_input_tokens = audit.get("chunk_policy_max_input_tokens")
    content_budget = audit.get("chunk_policy_content_token_budget")
    overlap_tokens = audit.get("chunk_policy_overlap_tokens")
    safety_reserve = audit.get("chunk_policy_safety_reserve")
    version = audit.get("chunk_policy_version")
    if not all(
        value is not None
        for value in (policy_id, max_input_tokens, content_budget, overlap_tokens, safety_reserve, version)
    ):
        return None
    return ChunkPolicy(
        id=str(policy_id),
        max_input_tokens=int(max_input_tokens),
        content_token_budget=int(content_budget),
        overlap_tokens=int(overlap_tokens),
        safety_reserve=int(safety_reserve),
        version=str(version),
    )


def _chunk_policy_version(policy: str) -> str:
    """Convert the public policy identifier into the internal version token."""
    if policy != "canonical_token_chunks:v2":
        raise ValueError(f"Unsupported chunk policy: {policy}")
    return "v2"


def _build_dense_provider(
    provider_name: str,
    dense_model: str,
    *,
    device: str | None = None,
    backend: str = "torch",
    torch_dtype: str = "auto",
    torch_compile: bool = False,
    effective_max_seq_length: int | None = None,
):
    if provider_name == "none":
        return None
    if provider_name == "mock":
        return MockDenseProvider()
    if provider_name == "sentence-transformers":
        return SentenceTransformerDenseProvider(
            dense_model,
            device=device,
            backend=backend,
            torch_dtype=torch_dtype,
            torch_compile=torch_compile,
            effective_max_seq_length=effective_max_seq_length,
        )
    raise ValueError(f"Unsupported dense provider: {provider_name}")


def _build_sparse_provider(
    provider_name: str,
    sparse_model: str,
    sparse_top_k: int,
    *,
    device: str | None = None,
    backend: str = "torch",
    torch_dtype: str = "auto",
    torch_compile: bool = False,
):
    if provider_name == "none":
        return None
    if provider_name == "mock":
        return MockSparseProvider()
    if provider_name == "sentence-transformers":
        if sparse_top_k <= 0:
            raise ValueError("--sparse-top-k must be positive.")
        return SentenceTransformerSparseProvider(
            sparse_model,
            top_k=sparse_top_k,
            device=device,
            backend=backend,
            torch_dtype=torch_dtype,
            torch_compile=torch_compile,
        )
    raise ValueError(f"Unsupported sparse provider: {provider_name}")


def _chunked_embedding_space_id(embedding_space_id: str, chunk_policy_id: str) -> str:
    return f"{embedding_space_id};chunk_policy={chunk_policy_id}"


def _raise_on_failed_chunk_audit(audit: dict[str, object]) -> None:
    failures = {
        key: audit.get(key)
        for key in (
            "uncovered_characters",
            "chunks_over_limit",
            "truncated_chunks",
            "blocks_with_coverage_gaps",
        )
        if audit.get(key)
    }
    if failures:
        raise RuntimeError(f"Retrieval chunk audit failed: {json.dumps(failures, sort_keys=True)}")


def _release_batch_memory() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        return


def _process_memory_mb() -> dict[str, float]:
    rss_mb = 0.0
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        rss_mb = _process_rss_mb_from_ps()
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    max_rss_mb = max_rss / (1024 * 1024) if sys.platform == "darwin" else max_rss / 1024
    memory = {"rss_mb": rss_mb, "max_rss_mb": max_rss_mb}
    memory.update(_torch_memory_mb())
    return memory


def _torch_memory_mb() -> dict[str, float]:
    memory = {
        "mps_current_mb": 0.0,
        "mps_driver_mb": 0.0,
        "cuda_allocated_mb": 0.0,
        "cuda_reserved_mb": 0.0,
    }
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            memory["mps_current_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            if hasattr(torch.mps, "driver_allocated_memory"):
                memory["mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024)
        if torch.cuda.is_available():
            memory["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return memory
    return memory


def _process_rss_mb_from_ps() -> float:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(output) / 1024 if output else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _progress_message(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[kb-index] {message}", file=sys.stderr, flush=True)


def _progress_iter(
    items: Iterable[T],
    *,
    enabled: bool,
    total: int,
    description: str,
    unit: str,
) -> Iterable[T]:
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm

        return tqdm(items, total=total, desc=description, unit=unit)
    except Exception:  # noqa: BLE001
        return _plain_progress_iter(items, total=total, description=description, unit=unit)


def _plain_progress_iter(items: Iterable[T], *, total: int, description: str, unit: str) -> Iterable[T]:
    step = max(1, total // 20)
    for index, item in enumerate(items, start=1):
        if index == 1 or index == total or index % step == 0:
            print(f"[kb-index] {description}: {index}/{total} {unit}s", file=sys.stderr, flush=True)
        yield item


def build_nodes_command(*, db_path: Path, mode: str = "deterministic", sparse_top_k: int = 50) -> dict[str, int]:
    if mode != "deterministic":
        raise ValueError(f"Unsupported node build mode: {mode}")
    if sparse_top_k <= 0:
        raise ValueError("--sparse-top-k must be positive.")
    init_db(db_path)
    with SQLiteStore(db_path) as store:
        stats = build_deterministic_nodes(store, sparse_top_k=sparse_top_k)
        store.commit()
        table_stats = store.stats()
    return {
        **table_stats,
        "nodes_created": stats.nodes_created,
        "memberships_created": stats.memberships_created,
        "nodes_with_dense_vectors": stats.nodes_with_dense_vectors,
        "nodes_with_sparse_terms": stats.nodes_with_sparse_terms,
    }


def build_edges_command(
    *,
    db_path: Path,
    scope: str = "project",
    top_k: int = 10,
    include_dense: bool = True,
    include_sparse: bool = True,
    max_group_size: int = 1000,
) -> dict[str, int]:
    if not include_dense and not include_sparse:
        raise ValueError("At least one of dense or sparse edges must be enabled.")
    init_db(db_path)
    with SQLiteStore(db_path) as store:
        stats = build_similarity_edges(
            store,
            scope=scope,
            top_k=top_k,
            include_dense=include_dense,
            include_sparse=include_sparse,
            max_group_size=max_group_size,
        )
        store.commit()
        table_stats = store.stats()
    return {
        **table_stats,
        "edges_created": stats.edges_created,
        "groups_processed": stats.groups_processed,
        "candidate_pairs": stats.candidate_pairs,
        "groups_skipped": stats.groups_skipped,
    }


if __name__ == "__main__":
    main()
