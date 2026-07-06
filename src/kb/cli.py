from __future__ import annotations

import argparse
import json
from pathlib import Path

from kb.embeddings.mock_provider import MockDenseProvider, MockSparseProvider
from kb.embeddings.sentence_transformer_provider import (
    SentenceTransformerDenseProvider,
    SentenceTransformerSparseProvider,
)
from kb.index.edge_builder import build_similarity_edges
from kb.index.semantic_node_builder import build_deterministic_nodes
from kb.ingest.attachment_parser import parse_attachment
from kb.ingest.chat_md_parser import parse_chat_file, write_parsed_chat_json
from kb.ingest.tree_walker import scan_tree, write_inventory_jsonl
from kb.storage.sqlite_store import SQLiteStore, init_db


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
    embed.add_argument("--sparse-model", default="naver/splade-cocondenser-ensembledistil")
    embed.add_argument("--sparse-top-k", type=int, default=128)
    embed.add_argument("--limit", type=int)
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--force", action="store_true")

    build_nodes = sub.add_parser("build-nodes", help="Build deterministic semantic nodes.")
    build_nodes.add_argument("--db", required=True)
    build_nodes.add_argument("--mode", choices=["deterministic"], default="deterministic")
    build_nodes.add_argument("--sparse-top-k", type=int, default=50)

    build_edges = sub.add_parser("build-edges", help="Build computed semantic edges.")
    build_edges.add_argument("--db", required=True)
    build_edges.add_argument("--scope", choices=["conversation", "project", "attachment"], default="project")
    build_edges.add_argument("--top-k", type=int, default=10)
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

    if args.command == "embed":
        stats = embed_knowledge_blocks(
            db_path=Path(args.db).expanduser(),
            provider=args.provider,
            dense_provider=args.dense_provider,
            sparse_provider=args.sparse_provider,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            sparse_top_k=args.sparse_top_k,
            limit=args.limit,
            batch_size=args.batch_size,
            force=args.force,
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
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return


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


def embed_knowledge_blocks(
    *,
    db_path: Path,
    provider: str = "sentence-transformers",
    dense_provider: str | None = None,
    sparse_provider: str = "sentence-transformers",
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sparse_model: str = "naver/splade-cocondenser-ensembledistil",
    sparse_top_k: int = 128,
    limit: int | None = None,
    batch_size: int = 32,
    force: bool = False,
) -> dict[str, int | float | str | None]:
    dense_name = dense_provider or provider
    dense = _build_dense_provider(dense_name, dense_model)
    sparse = _build_sparse_provider(sparse_provider, sparse_model, sparse_top_k)
    if dense is None and sparse is None:
        raise ValueError("At least one embedding provider must be enabled.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    init_db(db_path)
    dense_vectors = 0
    sparse_vectors = 0
    sparse_terms = 0
    errors = 0
    dense_dim_total = 0
    sparse_nnz_total = 0
    with SQLiteStore(db_path) as store:
        rows = store.knowledge_blocks_for_embedding(
            limit=limit,
            dense_model_name=dense.model_name if dense else None,
            dense_model_version=dense.model_version if dense else None,
            sparse_model_name=sparse.model_name if sparse else None,
            force=force,
        )
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [str(row["text_for_embedding"]) for row in batch]
            dense_results = dense.embed_texts(texts) if dense else [None] * len(batch)
            sparse_results = sparse.embed_texts(texts) if sparse else [None] * len(batch)
            for row, dense_vector, sparse_vector in zip(batch, dense_results, sparse_results, strict=True):
                try:
                    owner_id = str(row["id"])
                    dense_vector_id = None
                    sparse_vector_id = None
                    if dense_vector is not None and dense is not None:
                        dense_vector_id = store.upsert_dense_vector(
                            owner_type="knowledge_block",
                            owner_id=owner_id,
                            model_name=dense.model_name,
                            model_version=dense.model_version,
                            vector=dense_vector,
                        )
                        dense_vectors += 1
                        dense_dim_total += len(dense_vector)
                    if sparse_vector is not None and sparse is not None:
                        sparse_vector_id = store.replace_sparse_terms(
                            owner_type="knowledge_block",
                            owner_id=owner_id,
                            model_name=sparse.model_name,
                            terms=sparse_vector,
                        )
                        sparse_vectors += 1
                        sparse_terms += len(sparse_vector)
                        sparse_nnz_total += len(sparse_vector)
                    store.set_knowledge_block_vector_ids(
                        knowledge_block_id=owner_id,
                        dense_vector_id=dense_vector_id,
                        sparse_vector_id=sparse_vector_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"failed embedding knowledge_block {row['id']}: {exc}")
        store.commit()
    return {
        "dense_model": dense.model_name if dense else None,
        "dense_model_version": dense.model_version if dense else None,
        "sparse_model": sparse.model_name if sparse else None,
        "sparse_model_version": sparse.model_version if sparse else None,
        "candidate_blocks": len(rows),
        "blocks_embedded": max(dense_vectors, sparse_vectors),
        "dense_vectors": dense_vectors,
        "sparse_vectors": sparse_vectors,
        "sparse_terms": sparse_terms,
        "avg_dense_dim": dense_dim_total / dense_vectors if dense_vectors else 0,
        "avg_sparse_non_zero_count": sparse_nnz_total / sparse_vectors if sparse_vectors else 0,
        "errors": errors,
    }


def _build_dense_provider(provider_name: str, dense_model: str):
    if provider_name == "none":
        return None
    if provider_name == "mock":
        return MockDenseProvider()
    if provider_name == "sentence-transformers":
        return SentenceTransformerDenseProvider(dense_model)
    raise ValueError(f"Unsupported dense provider: {provider_name}")


def _build_sparse_provider(provider_name: str, sparse_model: str, sparse_top_k: int):
    if provider_name == "none":
        return None
    if provider_name == "mock":
        return MockSparseProvider()
    if provider_name == "sentence-transformers":
        if sparse_top_k <= 0:
            raise ValueError("--sparse-top-k must be positive.")
        return SentenceTransformerSparseProvider(sparse_model, top_k=sparse_top_k)
    raise ValueError(f"Unsupported sparse provider: {provider_name}")


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
        )
        store.commit()
        table_stats = store.stats()
    return {
        **table_stats,
        "edges_created": stats.edges_created,
        "groups_processed": stats.groups_processed,
        "candidate_pairs": stats.candidate_pairs,
    }


if __name__ == "__main__":
    main()
