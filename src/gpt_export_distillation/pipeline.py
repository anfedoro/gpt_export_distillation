from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil

from .config import AppConfig
from .models import ChatDocument, ChatMetrics, InputBundle, MessageRow
from .nlp import normalize_token


URL_RE = re.compile(r'https?://[^\s)>"\']+')
TOKEN_RE = re.compile(r"[A-Za-z0-9А-Яа-яЁё]{2,}")
GENERIC_PROJECT_NAMES = {"common", "pinned"}
STOP_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "chat",
    "new",
    "project",
    "analysis",
    "assistant",
    "python",
    "rust",
    "linux",
    "windows",
    "memory",
    "model",
    "models",
    "data",
    "code",
    "about",
    "это",
    "как",
    "что",
    "для",
    "про",
    "или",
    "chatgpt",
    "assistant",
    "user",
    "using",
    "use",
    "based",
    "make",
    "need",
    "help",
    "problem",
    "issue",
    "error",
    "file",
    "files",
    "folder",
    "project_01",
    "project_02",
    "project_03",
    "project_04",
    "project_05",
    "project_06",
    "project_07",
    "project_08",
    "project_09",
    "проект",
    "чат",
    "тема",
    "нужно",
    "надо",
    "можно",
    "сделать",
    "ошибка",
    "файл",
    "файлы",
}


def slugify(value: str, fallback: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9а-яё]+", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or fallback[:12] or "chat")[:120]


def safe_file_name(value: str, fallback: str) -> str:
    text = value.strip().replace("/", "_").replace(":", "_")
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text or fallback


def ts_to_iso(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def title_tokens(title: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(title)]
    return [token for token in tokens if token not in STOP_TOKENS and len(token) >= 3]


def normalized_title_tokens(title: str, config: AppConfig | None) -> list[str]:
    tokens = title_tokens(title)
    if config is None:
        return tokens
    nlp_enabled = bool(config.nlp.enabled and config.nlp.naming_mode == "nlp")
    normalized = [normalize_token(token, enabled=nlp_enabled) for token in tokens]
    return [
        token for token in normalized if token not in STOP_TOKENS and len(token) >= 3
    ]


def candidate_phrases(tokens: list[str], max_ngram: int = 3) -> list[str]:
    phrases: list[str] = []
    for size in range(1, max_ngram + 1):
        for idx in range(len(tokens) - size + 1):
            chunk = tokens[idx : idx + size]
            if any(token in STOP_TOKENS for token in chunk):
                continue
            phrases.append(" ".join(chunk))
    return phrases


def row_signal_tokens(row: MessageRow, limit: int = 24) -> list[str]:
    compact = row.text.replace("\n", " ")
    tokens = title_tokens(compact)
    return tokens[:limit]


def format_phrase(phrase: str) -> str:
    parts = []
    for token in phrase.split():
        if token.isascii() and token.isalpha() and len(token) <= 4:
            parts.append(token.upper())
        else:
            parts.append(token.title())
    return " ".join(parts)


def proposed_group_name(group_name: str, items: list[tuple[str, ChatDocument]]) -> str | None:
    return choose_project_name(group_name, items, None)[0]


def choose_project_name(
    group_name: str,
    items: list[tuple[str, ChatDocument]],
    config: AppConfig | None,
) -> tuple[str | None, str]:
    if group_name.lower() in GENERIC_PROJECT_NAMES:
        return None, "none"
    naming_mode = (config.nlp.naming_mode if config else "basic").strip().lower()
    max_phrase_words = config.nlp.max_phrase_words if config else 3
    min_repeated_titles = config.nlp.min_repeated_titles if config else 2
    fill_all = config.nlp.fill_all_project_names if config else False
    title_phrase_counts: Counter[str] = Counter()
    title_token_counts: Counter[str] = Counter()
    raw_titles: list[str] = []
    for _, item in items:
        title = str(item.conversation.get("title") or "")
        if title.strip():
            raw_titles.append(title.strip())
        tokens = normalized_title_tokens(title, config)
        title_phrases = set(candidate_phrases(tokens, max_ngram=max_phrase_words))
        title_phrase_counts.update(title_phrases)
        title_token_counts.update(set(tokens))
    if not title_phrase_counts and not raw_titles:
        return None, "none"

    repeated_title_phrases = [
        (phrase, count)
        for phrase, count in title_phrase_counts.items()
        if count >= min_repeated_titles and len(phrase.split()) >= 2
    ]
    if repeated_title_phrases:
        phrase, _ = max(
            repeated_title_phrases,
            key=lambda item: (
                item[1],
                len(item[0].split()),
                item[0],
            ),
        )
        return format_phrase(phrase), "high"

    min_repeated_tokens = min_repeated_titles if naming_mode in {"auto", "nlp"} else 2
    top_title_tokens = [
        token for token, count in title_token_counts.most_common(4) if count >= min_repeated_tokens
    ]
    if len(top_title_tokens) >= 2:
        return format_phrase(" ".join(top_title_tokens[:2])), "medium"
    if len(top_title_tokens) == 1:
        return format_phrase(top_title_tokens[0]), "medium"

    if fill_all:
        fallback_tokens = list(title_token_counts.keys())
        if len(fallback_tokens) >= 2:
            return format_phrase(" ".join(fallback_tokens[:2])), "low"
        if len(fallback_tokens) == 1:
            return format_phrase(fallback_tokens[0]), "low"
        for raw_title in raw_titles:
            fallback_title_tokens = title_tokens(raw_title)
            if len(fallback_title_tokens) >= 2:
                return format_phrase(" ".join(fallback_title_tokens[:2])), "low"
            if len(fallback_title_tokens) == 1:
                return format_phrase(fallback_title_tokens[0]), "low"
    return None, "none"


def extract_attachment_ids(conversation: dict) -> tuple[str, ...]:
    attachment_ids: set[str] = set()
    mapping = conversation.get("mapping") or {}
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        metadata = message.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        attachments = metadata.get("attachments")
        if isinstance(attachments, list):
            for attachment in attachments:
                if isinstance(attachment, dict):
                    attachment_id = attachment.get("id")
                    if isinstance(attachment_id, str) and attachment_id:
                        attachment_ids.add(attachment_id)
    return tuple(sorted(attachment_ids))


def effective_workers(config: AppConfig, task_count: int) -> int:
    configured = config.performance.workers
    if configured and configured > 0:
        return min(configured, max(task_count, 1))
    cpu_count = os.cpu_count() or 4
    return min(max(cpu_count, 4), max(task_count, 1))


def flatten_text(conversation: dict) -> list[MessageRow]:
    rows: list[MessageRow] = []
    mapping = conversation.get("mapping") or {}
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        role = ((message.get("author") or {}).get("role") or "other").lower()
        content = message.get("content") or {}
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        chunks: list[str] = []
        for part in parts:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                # Keep structured payloads readable for future manual review.
                import json

                chunks.append(json.dumps(part, ensure_ascii=False, indent=2))
        text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
        if not text:
            continue
        timestamp = message.get("create_time")
        rows.append(
            MessageRow(
                role=role,
                text=text,
                timestamp=timestamp if isinstance(timestamp, (int, float)) else None,
                message_id=str(message.get("id") or ""),
            )
        )
    rows.sort(key=lambda row: (row.timestamp or 0, row.message_id))
    return rows


def compute_metrics(rows: list[MessageRow]) -> ChatMetrics:
    assistant_messages = sum(1 for row in rows if row.role == "assistant")
    user_messages = sum(1 for row in rows if row.role == "user")
    text_chars = sum(len(row.text) for row in rows)
    code_blocks = sum(row.text.count("```") // 2 for row in rows)
    urls = sum(len(URL_RE.findall(row.text)) for row in rows)
    return ChatMetrics(
        assistant_messages=assistant_messages,
        user_messages=user_messages,
        total_messages=len(rows),
        text_chars=text_chars,
        code_blocks=code_blocks,
        urls=urls,
    )


def build_document(
    conversation: dict,
    groups: dict[str, str],
    config: AppConfig,
    source_label: str,
) -> ChatDocument:
    rows = flatten_text(conversation)
    metrics = compute_metrics(rows)
    template_id = conversation.get("conversation_template_id")
    pinned = conversation.get("pinned_time") is not None
    if pinned and config.grouping.keep_pinned_separately:
        group_name = config.grouping.pinned_folder_name
    elif template_id:
        group_name = groups[str(template_id)]
    else:
        group_name = config.grouping.common_folder_name
    return ChatDocument(
        conversation=conversation,
        rows=rows,
        metrics=metrics,
        group_name=group_name,
        source_label=source_label,
        attachment_ids=extract_attachment_ids(conversation),
    )


def project_group_names(conversations: list[dict], prefix: str) -> dict[str, str]:
    template_ids = sorted(
        {
            str(item.get("conversation_template_id"))
            for item in conversations
            if item.get("conversation_template_id")
        }
    )
    return {template_id: f"{prefix}_{idx:02d}" for idx, template_id in enumerate(template_ids, 1)}


def is_potential_trash(document: ChatDocument, config: AppConfig, now_ts: float) -> bool:
    conversation = document.conversation
    if config.filters.apply_only_to_non_project_non_pinned and document.group_name != config.grouping.common_folder_name:
        return False
    create_time = conversation.get("create_time")
    update_time = conversation.get("update_time")
    ref_time = update_time if isinstance(update_time, (int, float)) else create_time
    if not isinstance(ref_time, (int, float)):
        return False
    is_old = ref_time < (now_ts - config.filters.old_days * 86400)
    return (
        is_old
        and document.metrics.assistant_messages
        <= config.filters.max_assistant_messages_for_old_common
    )


def build_documents(bundle: InputBundle, config: AppConfig) -> list[ChatDocument]:
    groups = project_group_names(bundle.conversations, config.grouping.project_prefix)
    overrides = config.grouping.project_name_overrides or {}
    groups.update({str(template_id): str(name) for template_id, name in overrides.items()})
    worker_count = effective_workers(config, len(bundle.conversations))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            executor.map(
                lambda conversation: build_document(
                    conversation,
                    groups,
                    config,
                    bundle.source.label,
                ),
                bundle.conversations,
            )
        )


def resolve_project_folder_names(
    written_index: dict[str, list[tuple[str, ChatDocument, bool]]],
    config: AppConfig,
) -> tuple[dict[str, str], list[str]]:
    group_folder_names: dict[str, str] = {
        config.grouping.common_folder_name: config.grouping.common_folder_name,
        config.grouping.pinned_folder_name: config.grouping.pinned_folder_name,
    }
    used_folder_names = {
        config.grouping.common_folder_name.lower(),
        config.grouping.pinned_folder_name.lower(),
    }
    summary_lines: list[str] = []
    for group_name, items in sorted(written_index.items()):
        if group_name in group_folder_names:
            continue
        useful_items = [(file_name, item) for file_name, item, trash in items if not trash]
        suggested, confidence = choose_project_name(group_name, useful_items, config)
        combined_name = f"{group_name} - {suggested}" if suggested else group_name
        base_folder_name = safe_file_name(combined_name, group_name)
        candidate = base_folder_name
        suffix = 2
        while candidate.lower() in used_folder_names:
            candidate = f"{base_folder_name} ({suffix})"
            suffix += 1
        group_folder_names[group_name] = candidate
        used_folder_names.add(candidate.lower())
        if suggested:
            summary_lines.append(
                f"- `{group_name}` -> `{candidate}` | confidence: `{confidence}`"
            )
        else:
            summary_lines.append(
                f"- `{group_name}` -> `{candidate}` | confidence: `none`"
            )
    return group_folder_names, summary_lines


def render_chat_markdown(document: ChatDocument) -> str:
    conv = document.conversation
    lines = [f"# {(conv.get('title') or 'Untitled Chat').strip()}", ""]
    lines.extend(
        [
            "## Metadata",
            "",
            f"- `id`: `{conv.get('id') or conv.get('conversation_id') or ''}`",
            f"- `conversation_template_id`: `{conv.get('conversation_template_id') or ''}`",
            f"- `source`: `{document.source_label}`",
            f"- `create_time_utc`: `{ts_to_iso(conv.get('create_time') if isinstance(conv.get('create_time'), (int, float)) else None)}`",
            f"- `update_time_utc`: `{ts_to_iso(conv.get('update_time') if isinstance(conv.get('update_time'), (int, float)) else None)}`",
            f"- `message_count`: `{document.metrics.total_messages}`",
            f"- `assistant_messages`: `{document.metrics.assistant_messages}`",
            f"- `user_messages`: `{document.metrics.user_messages}`",
            f"- `text_chars`: `{document.metrics.text_chars}`",
            f"- `estimated_code_blocks`: `{document.metrics.code_blocks}`",
            "",
            "## Conversation",
            "",
        ]
    )
    for idx, row in enumerate(document.rows, 1):
        lines.extend(
            [
                f"### {idx}. {row.role.upper()}",
                "",
                f"- `time_utc`: `{ts_to_iso(row.timestamp)}`",
                f"- `message_id`: `{row.message_id}`",
                "",
                row.text,
                "",
            ]
        )
    return "\n".join(lines)


def attachment_lines(bundle: InputBundle) -> list[str]:
    if not bundle.asset_file_names:
        return ["# Attachments", "", "No attachment filename mapping in export."]
    extension_counts = Counter()
    for name in bundle.asset_file_names.values():
        suffix = Path(name).suffix.lower() or "[no-ext]"
        extension_counts[suffix] += 1
    lines = ["# Attachments", "", f"Total mapped assets: `{len(bundle.asset_file_names)}`", "", "## By extension", ""]
    for suffix, count in sorted(extension_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{suffix}`: `{count}`")
    lines.extend(["", "## Asset names", ""])
    for name in sorted(bundle.asset_file_names.values()):
        lines.append(f"- {name}")
    return lines


def library_lines(bundle: InputBundle) -> list[str]:
    if not bundle.library_files:
        return ["# Library Files", "", "No library file metadata in export."]
    lines = ["# Library Files", "", f"Entries: `{len(bundle.library_files)}`", ""]
    for item in bundle.library_files:
        name = item.get("name") or item.get("file_name") or item.get("id") or "unknown"
        kind = item.get("mime_type") or item.get("type") or ""
        lines.append(f"- `{name}` {kind}".rstrip())
    return lines


def group_output_dir(
    output_root: Path,
    group_name: str,
    config: AppConfig,
    trash: bool,
    group_folder_names: dict[str, str],
) -> Path:
    if group_name == config.grouping.pinned_folder_name:
        return output_root / config.grouping.pinned_folder_name
    if group_name == config.grouping.common_folder_name:
        bucket = (
            config.grouping.potential_trash_folder_name
            if trash
            else config.grouping.useful_folder_name
        )
        return output_root / config.grouping.common_folder_name / bucket
    return output_root / config.grouping.projects_folder_name / group_folder_names[group_name]


def attachment_export_name(bundle: InputBundle, attachment_id: str) -> str | None:
    direct_name = f"{attachment_id}.dat"
    if (bundle.source.root_dir / direct_name).exists():
        return direct_name
    for export_name in bundle.asset_file_names.keys():
        if export_name == direct_name:
            return export_name
    return None


def attachment_display_name(bundle: InputBundle, export_name: str) -> str:
    return safe_file_name(bundle.asset_file_names.get(export_name, export_name), export_name)


def write_output(
    bundle: InputBundle,
    documents: list[ChatDocument],
    config: AppConfig,
    explicit_output_dir: str | None = None,
) -> Path:
    base_output_dir = explicit_output_dir or config.output.output_dir
    if base_output_dir:
        output_root = Path(base_output_dir).expanduser().resolve()
    else:
        output_root = bundle.source.output_dir / config.output.root_folder_name
    output_root.mkdir(parents=True, exist_ok=True)
    now_ts = datetime.now(timezone.utc).timestamp()
    kept: list[ChatDocument] = []
    trash_docs: list[ChatDocument] = []
    file_names_by_group: dict[str, set[str]] = defaultdict(set)
    written_index: dict[str, list[tuple[str, ChatDocument, bool]]] = defaultdict(list)
    document_plans: list[tuple[Path, ChatDocument, str, bool]] = []
    attachment_plans: list[tuple[Path, Path]] = []
    exported_attachments: dict[Path, set[str]] = defaultdict(set)

    for document in documents:
        trash = is_potential_trash(document, config, now_ts)
        title = str(document.conversation.get("title") or "")
        fallback = str(document.conversation.get("id") or document.conversation.get("conversation_id") or "chat")
        base_name = slugify(title, fallback)
        final_name = base_name
        suffix = 2
        while f"{final_name}.md" in file_names_by_group[document.group_name]:
            final_name = f"{base_name}_{suffix}"
            suffix += 1
        file_names_by_group[document.group_name].add(f"{final_name}.md")
        written_index[document.group_name].append((f"{final_name}.md", document, trash))
        document_plans.append((Path(f"{final_name}.md"), document, final_name, trash))
        if trash:
            trash_docs.append(document)
        else:
            kept.append(document)

    group_folder_names, proposed_mapping_lines = resolve_project_folder_names(
        written_index, config
    )

    planned_documents: list[tuple[Path, ChatDocument]] = []
    for relative_name, document, _, trash in document_plans:
        group_dir = group_output_dir(
            output_root,
            document.group_name,
            config,
            trash,
            group_folder_names,
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        planned_documents.append((group_dir / relative_name, document))
        if document.attachment_ids:
            attachments_dir = group_dir / "attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)
            for attachment_id in document.attachment_ids:
                export_name = attachment_export_name(bundle, attachment_id)
                if not export_name:
                    continue
                if export_name in exported_attachments[attachments_dir]:
                    continue
                src = bundle.source.root_dir / export_name
                if not src.exists():
                    continue
                display_name = attachment_display_name(bundle, export_name)
                dst = attachments_dir / display_name
                stem = dst.stem
                suffix = dst.suffix
                counter = 2
                while dst.exists() or any(planned_dst == dst for _, planned_dst in attachment_plans):
                    dst = attachments_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                attachment_plans.append((src, dst))
                exported_attachments[attachments_dir].add(export_name)

    worker_count = effective_workers(config, len(planned_documents) + len(attachment_plans))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(
            executor.map(
                lambda item: item[0].write_text(
                    render_chat_markdown(item[1]),
                    encoding="utf-8",
                ),
                planned_documents,
            )
        )
        list(executor.map(lambda item: shutil.copy2(item[0], item[1]), attachment_plans))

    summary_lines = [
        "# Export Summary",
        "",
        f"- `source`: `{bundle.source.label}`",
        f"- `kind`: `{bundle.source.kind}`",
        f"- `total_conversations`: `{len(documents)}`",
        f"- `useful`: `{len(kept)}`",
        f"- `potential_trash`: `{len(trash_docs)}`",
        "",
        "## Groups",
        "",
    ]
    useful_group_counts = Counter(document.group_name for document in kept)
    trash_group_counts = Counter(document.group_name for document in trash_docs)
    for group_name, count in sorted(useful_group_counts.items()):
        display_name = group_folder_names.get(group_name, group_name)
        summary_lines.append(f"- `useful/{display_name}`: `{count}`")
    for group_name, count in sorted(trash_group_counts.items()):
        display_name = group_folder_names.get(group_name, group_name)
        summary_lines.append(f"- `potential_trash/{display_name}`: `{count}`")
    proposed_lines = ["# Proposed Project Names", ""]
    proposed_lines.extend(proposed_mapping_lines or ["No project-like groups detected."])
    (output_root / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (output_root / "PROPOSED_PROJECT_NAMES.md").write_text(
        "\n".join(proposed_lines),
        encoding="utf-8",
    )

    for group_name, items in written_index.items():
        buckets = {
            "useful": [(file_name, item) for file_name, item, trash in items if not trash],
            "potential_trash": [(file_name, item) for file_name, item, trash in items if trash],
        }
        for bucket_name, bucket_items in buckets.items():
            if not bucket_items:
                continue
            group_dir = group_output_dir(
                output_root,
                group_name,
                config,
                trash=(bucket_name == "potential_trash"),
                group_folder_names=group_folder_names,
            )
            display_name = group_folder_names.get(group_name, group_name)
            index_lines = [f"# {display_name} Index", "", f"Total: `{len(bucket_items)}`", ""]
            for file_name, item in sorted(
                bucket_items,
                key=lambda row: (
                    str(row[1].conversation.get("title") or "").lower(),
                    row[0],
                ),
            ):
                title = str(item.conversation.get("title") or "Untitled Chat")
                index_lines.append(
                    f"- `{file_name}` | `{item.metrics.assistant_messages}` assistant | `{item.metrics.total_messages}` msgs | {title}"
                )
            (group_dir / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    if config.output.include_attachments_summary:
        (output_root / "ATTACHMENTS.md").write_text(
            "\n".join(attachment_lines(bundle)),
            encoding="utf-8",
        )
    if config.output.include_files_summary:
        (output_root / "FILES.md").write_text(
            "\n".join(library_lines(bundle)),
            encoding="utf-8",
        )
    return output_root
