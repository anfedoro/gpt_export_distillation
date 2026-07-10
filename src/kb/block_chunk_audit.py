from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any


def audit_block_chunks(db_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run a read-only structural-block to retrieval-chunk audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        blocks = conn.execute(
            """
            SELECT
                b.id AS block_id,
                b.message_id,
                b.conversation_id,
                b.block_type,
                b.raw_text,
                b.char_start,
                b.char_end,
                m.role,
                sd.interest_tier
            FROM blocks b
            JOIN messages m ON m.id = b.message_id
            JOIN conversations c ON c.id = m.conversation_id
            JOIN source_documents sd ON sd.id = c.source_document_id
            ORDER BY b.id
            """
        ).fetchall()
        chunks = conn.execute(
            """
            SELECT id, block_id, ordinal, source_char_start, source_char_end,
                   token_count
            FROM retrieval_chunks
            ORDER BY block_id, ordinal
            """
        ).fetchall()
        orphan_chunks = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM retrieval_chunks rc
                LEFT JOIN blocks b ON b.id = rc.block_id
                WHERE b.id IS NULL
                """
            ).fetchone()[0]
        )

    chunks_by_block: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_block[str(chunk["block_id"])].append(chunk)

    counts = [len(chunks_by_block.get(str(block["block_id"]), [])) for block in blocks]
    blocks_with_chunks = sum(count > 0 for count in counts)
    total_chunks = len(chunks)
    block_count = len(blocks)
    token_counts_by_block = {
        str(block["block_id"]): [int(row["token_count"]) for row in chunks_by_block.get(str(block["block_id"]), [])]
        for block in blocks
    }

    distribution = []
    for value in sorted(set(counts)):
        block_count_for_value = counts.count(value)
        distribution.append(
            {
                "chunks_per_block": value,
                "block_count": block_count_for_value,
                "percent_of_blocks": _percent(block_count_for_value, block_count),
                "total_chunks": value * block_count_for_value,
            }
        )

    buckets = {}
    for label, predicate in (
        ("0", lambda value: value == 0),
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3", lambda value: value == 3),
        ("4", lambda value: value == 4),
        ("5", lambda value: value == 5),
        ("6-10", lambda value: 6 <= value <= 10),
        ("11-20", lambda value: 11 <= value <= 20),
        ("21-50", lambda value: 21 <= value <= 50),
        (">50", lambda value: value > 50),
    ):
        matching = [value for value in counts if predicate(value)]
        buckets[label] = {
            "block_count": len(matching),
            "percent_of_blocks": _percent(len(matching), block_count),
            "total_chunks": sum(matching),
        }

    by_type: dict[str, list[int]] = defaultdict(list)
    for block, count in zip(blocks, counts, strict=True):
        by_type[str(block["block_type"])].append(count)
    breakdown_by_type = {
        block_type: _count_summary(values)
        for block_type, values in sorted(by_type.items())
    }
    for block_type, values in breakdown_by_type.items():
        values_for_type = by_type[block_type]
        values_for_type_chunks = sum(values_for_type)
        values["block_count"] = len(values_for_type)
        values["chunk_count"] = values_for_type_chunks
        values["blocks_with_more_than_1_chunk"] = sum(value > 1 for value in values_for_type)
        values["blocks_with_at_least_5_chunks"] = sum(value >= 5 for value in values_for_type)
        values["blocks_with_at_least_10_chunks"] = sum(value >= 10 for value in values_for_type)

    top_blocks = []
    for block, count in sorted(
        zip(blocks, counts, strict=True),
        key=lambda item: (-item[1], str(item[0]["block_id"])),
    )[:30]:
        block_chunks = chunks_by_block.get(str(block["block_id"]), [])
        token_counts = token_counts_by_block[str(block["block_id"])]
        top_blocks.append(
            {
                "block_id": block["block_id"],
                "conversation_id": block["conversation_id"],
                "message_id": block["message_id"],
                "role": block["role"],
                "block_type": block["block_type"],
                "source_character_length": int(block["char_end"]) - int(block["char_start"]),
                "chunk_count": count,
                "chunk_token_count": {
                    "min": min(token_counts) if token_counts else 0,
                    "avg": fmean(token_counts) if token_counts else 0.0,
                    "max": max(token_counts) if token_counts else 0,
                },
                "first_source_offset": min(
                    int(chunk["source_char_start"]) for chunk in block_chunks
                ) if block_chunks else None,
                "last_source_offset": max(
                    int(chunk["source_char_end"]) for chunk in block_chunks
                ) if block_chunks else None,
            }
        )

    no_chunk_by_type = Counter(str(block["block_type"]) for block, count in zip(blocks, counts, strict=True) if count == 0)
    no_chunk_reasons = Counter()
    for block, count in zip(blocks, counts, strict=True):
        if count != 0:
            continue
        raw_text = str(block["raw_text"] or "")
        if not raw_text.strip():
            reason = "empty_or_whitespace"
        elif str(block["interest_tier"]) in {"low", "quarantine"}:
            reason = "low_or_quarantine_interest"
        else:
            reason = "nonempty_other"
        no_chunk_reasons[reason] += 1

    ordinal_gaps = []
    duplicate_ordinals = []
    for block_id, block_chunks in chunks_by_block.items():
        ordinals = [int(chunk["ordinal"]) for chunk in block_chunks]
        if len(ordinals) != len(set(ordinals)):
            duplicate_ordinals.append(block_id)
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            ordinal_gaps.append(block_id)

    result = {
        "schema_version": "kb.block_chunk_audit.v1",
        "database": str(db_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_structural_blocks": block_count,
            "blocks_with_at_least_one_retrieval_chunk": blocks_with_chunks,
            "blocks_without_retrieval_chunks": block_count - blocks_with_chunks,
            "total_retrieval_chunks": total_chunks,
            "average_chunks_per_block_all": _average(counts),
            "median_chunks_per_block_all": median(counts) if counts else 0,
            "average_chunks_per_block_with_chunks": _average([value for value in counts if value > 0]),
            "median_chunks_per_block_with_chunks": median([value for value in counts if value > 0]) if blocks_with_chunks else 0,
            "p90_chunks_per_block": _percentile(counts, 90),
            "p95_chunks_per_block": _percentile(counts, 95),
            "p99_chunks_per_block": _percentile(counts, 99),
            "maximum_chunks_per_block": max(counts) if counts else 0,
        },
        "distribution": distribution,
        "buckets": buckets,
        "by_block_type": breakdown_by_type,
        "top_30_blocks": top_blocks,
        "blocks_without_chunks": {
            "by_block_type": dict(sorted(no_chunk_by_type.items())),
            "by_reason": dict(sorted(no_chunk_reasons.items())),
        },
        "consistency": {
            "orphan_retrieval_chunks": orphan_chunks,
            "chunk_count_sum_matches_total": sum(counts) == total_chunks,
            "chunk_count_sum": sum(counts),
            "ordinal_gaps": len(ordinal_gaps),
            "ordinal_gap_block_ids": ordinal_gaps[:100],
            "duplicate_ordinals": len(duplicate_ordinals),
            "duplicate_ordinal_block_ids": duplicate_ordinals[:100],
            "all_checks_passed": (
                orphan_chunks == 0
                and sum(counts) == total_chunks
                and not ordinal_gaps
                and not duplicate_ordinals
            ),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "report.md").write_text(_render_markdown(result))
    return result


def _average(values: list[int]) -> float:
    return fmean(values) if values else 0.0


def _percent(value: int, total: int) -> float:
    return (value * 100.0 / total) if total else 0.0


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _count_summary(values: list[int]) -> dict[str, Any]:
    return {
        "block_count": len(values),
        "chunk_count": sum(values),
        "average_chunks_per_block": _average(values),
        "median_chunks_per_block": median(values) if values else 0,
        "p95_chunks_per_block": _percentile(values, 95),
        "maximum_chunks_per_block": max(values) if values else 0,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Structural Blocks and Retrieval Chunks Audit",
        "",
        f"Database: `{report['database']}`",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Full Distribution", "", "| chunks_per_block | block_count | percent_of_blocks | total_chunks |", "|---:|---:|---:|---:|"]
    for row in report["distribution"]:
        lines.append(f"| {row['chunks_per_block']} | {row['block_count']} | {row['percent_of_blocks']:.4f} | {row['total_chunks']} |")
    lines += ["", "## Buckets", "", "| Bucket | block_count | percent_of_blocks | total_chunks |", "|---|---:|---:|---:|"]
    for label, row in report["buckets"].items():
        lines.append(f"| {label} | {row['block_count']} | {row['percent_of_blocks']:.4f} | {row['total_chunks']} |")
    lines += ["", "## By Block Type", "", "| block_type | blocks | chunks | avg | median | p95 | max | >1 | >=5 | >=10 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for block_type, row in report["by_block_type"].items():
        lines.append(f"| {block_type} | {row['block_count']} | {row['chunk_count']} | {row['average_chunks_per_block']:.4f} | {row['median_chunks_per_block']} | {row['p95_chunks_per_block']:.4f} | {row['maximum_chunks_per_block']} | {row['blocks_with_more_than_1_chunk']} | {row['blocks_with_at_least_5_chunks']} | {row['blocks_with_at_least_10_chunks']} |")
    lines += ["", "## Blocks Without Chunks", "", f"By block type: `{json.dumps(report['blocks_without_chunks']['by_block_type'], ensure_ascii=False, sort_keys=True)}`", "", f"By reason: `{json.dumps(report['blocks_without_chunks']['by_reason'], ensure_ascii=False, sort_keys=True)}`", "", "## Consistency", "", "| Check | Result |", "|---|---:|"]
    for key, value in report["consistency"].items():
        if key.endswith("_block_ids"):
            continue
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Top 30 Blocks", "", "Private text is intentionally omitted.", "", "| block_id | conversation_id | message_id | role | type | source_chars | chunks | token_min | token_avg | token_max | first_offset | last_offset |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["top_30_blocks"]:
        tokens = row["chunk_token_count"]
        lines.append(f"| {row['block_id']} | {row['conversation_id']} | {row['message_id']} | {row['role']} | {row['block_type']} | {row['source_character_length']} | {row['chunk_count']} | {tokens['min']} | {tokens['avg']:.2f} | {tokens['max']} | {row['first_source_offset']} | {row['last_source_offset']} |")
    return "\n".join(lines) + "\n"
