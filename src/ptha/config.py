"""Versioned PTHA configuration and precedence handling."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from ptha.errors import ConfigurationError
from ptha.paths import PthaPaths, platform_paths

CONFIG_VERSION = 1
DEFAULT_CONFIG_TEXT = '''config_version = 1

[paths]
database = ""
working_dir = ""
model_cache = ""

[import]
keep_distilled_archive = false
include_low_interest = false
replace_existing = false

[models]
dense_model = "BAAI/bge-m3"
sparse_model = "BAAI/bge-m3"
dense_device = "auto"
sparse_device = "auto"
dense_dtype = "auto"
sparse_dtype = "auto"
sparse_top_k = 128
batch_size = 16

[retrieval]
candidate_pool = 500
default_output_tokens = 1800
max_output_tokens = 6000

[service]
request_timeout_seconds = 30
startup_timeout_seconds = 120
shutdown_timeout_seconds = 30
preload_models = true
max_request_bytes = 1048576
max_response_bytes = 16777216

[logging]
level = "info"
retain_days = 7
service_max_bytes = 10485760
service_backup_count = 3
'''


@dataclass(frozen=True)
class PthaConfig:
    config_file: Path
    paths: PthaPaths
    database: Path
    working_dir: Path | None = None
    model_cache: Path | None = None
    dense_model: str = "BAAI/bge-m3"
    sparse_model: str = "BAAI/bge-m3"
    dense_device: str = "auto"
    sparse_device: str = "auto"
    dense_dtype: str = "auto"
    sparse_dtype: str = "auto"
    sparse_top_k: int = 128
    batch_size: int = 16
    candidate_pool: int = 500
    default_output_tokens: int = 1800
    max_output_tokens: int = 6000
    request_timeout_seconds: int = 30
    startup_timeout_seconds: int = 120
    shutdown_timeout_seconds: int = 30
    service_max_bytes: int = 10_485_760
    service_backup_count: int = 3
    max_request_bytes: int = 1_048_576
    max_response_bytes: int = 16_777_216
    log_level: str = "info"
    import_options: Mapping[str, bool] = field(default_factory=dict)


def config_path(cli_path: str | Path | None = None) -> Path:
    value = cli_path or os.environ.get("PTHA_CONFIG")
    return Path(value).expanduser() if value else platform_paths().config_file


def load_config(path: Path | None = None, *, overrides: Mapping[str, Any] | None = None) -> PthaConfig:
    locations = platform_paths()
    selected = path or config_path()
    raw: dict[str, Any] = {}
    if selected.exists():
        try:
            with selected.open("rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"Cannot read PTHA configuration: {selected}") from exc
    version = raw.get("config_version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigurationError(f"Unsupported config_version: {version}. Expected: {CONFIG_VERSION}.")
    path_raw = raw.get("paths", {})
    models = raw.get("models", {})
    retrieval = raw.get("retrieval", {})
    service = raw.get("service", {})
    logging = raw.get("logging", {})
    import_raw = raw.get("import", {})
    database = _path(os.environ.get("PTHA_DB_PATH") or path_raw.get("database")) or locations.database
    model_cache = _path(os.environ.get("PTHA_MODEL_CACHE_DIR") or path_raw.get("model_cache"))
    dense_model = str(models.get("dense_model", PthaConfig.dense_model))
    sparse_model = str(models.get("sparse_model", PthaConfig.sparse_model))
    if sparse_model == "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1" and dense_model == "BAAI/bge-m3":
        sparse_model = dense_model
    cfg = PthaConfig(
        config_file=selected,
        paths=locations,
        database=database,
        working_dir=_path(path_raw.get("working_dir")),
        model_cache=model_cache,
        dense_model=dense_model,
        sparse_model=sparse_model,
        dense_device=os.environ.get("PTHA_DENSE_DEVICE", str(models.get("dense_device", "auto"))),
        sparse_device=os.environ.get("PTHA_SPARSE_DEVICE", str(models.get("sparse_device", "auto"))),
        dense_dtype=str(models.get("dense_dtype", "auto")),
        sparse_dtype=str(models.get("sparse_dtype", "auto")),
        sparse_top_k=int(models.get("sparse_top_k", 128)),
        batch_size=int(models.get("batch_size", 16)),
        candidate_pool=int(retrieval.get("candidate_pool", 500)),
        default_output_tokens=int(retrieval.get("default_output_tokens", 1800)),
        max_output_tokens=int(retrieval.get("max_output_tokens", 6000)),
        request_timeout_seconds=int(service.get("request_timeout_seconds", 30)),
        startup_timeout_seconds=int(service.get("startup_timeout_seconds", 120)),
        shutdown_timeout_seconds=int(service.get("shutdown_timeout_seconds", 30)),
        service_max_bytes=int(logging.get("service_max_bytes", 10_485_760)),
        service_backup_count=int(logging.get("service_backup_count", 3)),
        max_request_bytes=int(service.get("max_request_bytes", 1_048_576)),
        max_response_bytes=int(service.get("max_response_bytes", 16_777_216)),
        log_level=os.environ.get("PTHA_LOG_LEVEL", str(logging.get("level", "info"))),
        import_options={
            "keep_distilled_archive": bool(import_raw.get("keep_distilled_archive", False)),
            "include_low_interest": bool(import_raw.get("include_low_interest", False)),
            "replace_existing": bool(import_raw.get("replace_existing", False)),
        },
    )
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg = replace(cfg, **{key: value})
    return cfg


def _path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else None
