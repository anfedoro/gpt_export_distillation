from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Mapping


@dataclass(frozen=True)
class InputConfig:
    search_dir: str = "."
    conversations_glob: str = "conversations-*.json"
    include_zip: bool = True
    zip_glob: str = "*.zip"


@dataclass(frozen=True)
class FilterConfig:
    old_days: int = 60
    max_assistant_messages_for_old_common: int = 9
    apply_only_to_non_project_non_pinned: bool = True


@dataclass(frozen=True)
class GroupingConfig:
    project_prefix: str = "Project"
    projects_folder_name: str = "Projects"
    common_folder_name: str = "Common"
    useful_folder_name: str = "useful"
    potential_trash_folder_name: str = "potential_trash"
    pinned_folder_name: str = "Pinned"
    keep_pinned_separately: bool = True
    project_name_overrides: Mapping[str, str] | None = None


@dataclass(frozen=True)
class OutputConfig:
    root_folder_name: str = "md_export"
    output_dir: str | None = None
    include_files_summary: bool = True
    include_attachments_summary: bool = True


@dataclass(frozen=True)
class PerformanceConfig:
    workers: int = 0


@dataclass(frozen=True)
class NlpConfig:
    enabled: bool = False
    naming_mode: str = "basic"
    max_phrase_words: int = 3
    min_repeated_titles: int = 2
    fill_all_project_names: bool = False


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    filters: FilterConfig
    grouping: GroupingConfig
    output: OutputConfig
    performance: PerformanceConfig
    nlp: NlpConfig


DEFAULT_CONFIG = AppConfig(
    input=InputConfig(),
    filters=FilterConfig(),
    grouping=GroupingConfig(),
    output=OutputConfig(),
    performance=PerformanceConfig(),
    nlp=NlpConfig(),
)


def load_config(config_path: Path | None) -> AppConfig:
    if config_path is None:
        return DEFAULT_CONFIG
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    input_cfg = raw.get("input", {})
    filter_cfg = raw.get("filters", {})
    grouping_cfg = raw.get("grouping", {})
    output_cfg = raw.get("output", {})
    performance_cfg = raw.get("performance", {})
    nlp_cfg = raw.get("nlp", {})

    return AppConfig(
        input=InputConfig(
            search_dir=input_cfg.get("search_dir", DEFAULT_CONFIG.input.search_dir),
            conversations_glob=input_cfg.get(
                "conversations_glob", DEFAULT_CONFIG.input.conversations_glob
            ),
            include_zip=input_cfg.get("include_zip", DEFAULT_CONFIG.input.include_zip),
            zip_glob=input_cfg.get("zip_glob", DEFAULT_CONFIG.input.zip_glob),
        ),
        filters=FilterConfig(
            old_days=filter_cfg.get("old_days", DEFAULT_CONFIG.filters.old_days),
            max_assistant_messages_for_old_common=filter_cfg.get(
                "max_assistant_messages_for_old_common",
                DEFAULT_CONFIG.filters.max_assistant_messages_for_old_common,
            ),
            apply_only_to_non_project_non_pinned=filter_cfg.get(
                "apply_only_to_non_project_non_pinned",
                DEFAULT_CONFIG.filters.apply_only_to_non_project_non_pinned,
            ),
        ),
        grouping=GroupingConfig(
            project_prefix=grouping_cfg.get(
                "project_prefix", DEFAULT_CONFIG.grouping.project_prefix
            ),
            projects_folder_name=grouping_cfg.get(
                "projects_folder_name", DEFAULT_CONFIG.grouping.projects_folder_name
            ),
            common_folder_name=grouping_cfg.get(
                "common_folder_name", DEFAULT_CONFIG.grouping.common_folder_name
            ),
            useful_folder_name=grouping_cfg.get(
                "useful_folder_name", DEFAULT_CONFIG.grouping.useful_folder_name
            ),
            potential_trash_folder_name=grouping_cfg.get(
                "potential_trash_folder_name",
                DEFAULT_CONFIG.grouping.potential_trash_folder_name,
            ),
            pinned_folder_name=grouping_cfg.get(
                "pinned_folder_name", DEFAULT_CONFIG.grouping.pinned_folder_name
            ),
            keep_pinned_separately=grouping_cfg.get(
                "keep_pinned_separately",
                DEFAULT_CONFIG.grouping.keep_pinned_separately,
            ),
            project_name_overrides=grouping_cfg.get("project_name_overrides"),
        ),
        output=OutputConfig(
            root_folder_name=output_cfg.get(
                "root_folder_name", DEFAULT_CONFIG.output.root_folder_name
            ),
            output_dir=output_cfg.get("output_dir", DEFAULT_CONFIG.output.output_dir),
            include_files_summary=output_cfg.get(
                "include_files_summary", DEFAULT_CONFIG.output.include_files_summary
            ),
            include_attachments_summary=output_cfg.get(
                "include_attachments_summary",
                DEFAULT_CONFIG.output.include_attachments_summary,
            ),
        ),
        performance=PerformanceConfig(
            workers=performance_cfg.get(
                "workers", DEFAULT_CONFIG.performance.workers
            ),
        ),
        nlp=NlpConfig(
            enabled=nlp_cfg.get(
                "enabled", DEFAULT_CONFIG.nlp.enabled
            ),
            naming_mode=nlp_cfg.get(
                "naming_mode", DEFAULT_CONFIG.nlp.naming_mode
            ),
            max_phrase_words=nlp_cfg.get(
                "max_phrase_words", DEFAULT_CONFIG.nlp.max_phrase_words
            ),
            min_repeated_titles=nlp_cfg.get(
                "min_repeated_titles", DEFAULT_CONFIG.nlp.min_repeated_titles
            ),
            fill_all_project_names=nlp_cfg.get(
                "fill_all_project_names",
                DEFAULT_CONFIG.nlp.fill_all_project_names,
            ),
        ),
    )
