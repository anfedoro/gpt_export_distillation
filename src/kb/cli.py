from __future__ import annotations

import argparse
import json
from pathlib import Path

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


if __name__ == "__main__":
    main()
