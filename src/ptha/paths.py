"""Platform-specific PTHA filesystem locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs


@dataclass(frozen=True)
class PthaPaths:
    config_dir: Path
    data_dir: Path
    cache_dir: Path
    state_dir: Path
    log_dir: Path
    runtime_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def database(self) -> Path:
        return self.data_dir / "ptha.db"

    @property
    def socket(self) -> Path:
        return self.runtime_dir / "ptha.sock"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "service.pid"

    @property
    def service_log(self) -> Path:
        return self.log_dir / "service.log"


def platform_paths() -> PthaPaths:
    dirs = PlatformDirs("ptha", appauthor=False, ensure_exists=False)
    runtime_override = os.environ.get("PTHA_RUNTIME_DIR")
    runtime = (Path(runtime_override).expanduser() if runtime_override else
               Path(dirs.user_runtime_dir) if dirs.user_runtime_dir else Path(dirs.user_state_dir) / "run")
    data_override = os.environ.get("PTHA_DATA_DIR")
    data_dir = Path(data_override).expanduser() if data_override else Path(dirs.user_data_dir)
    state = Path(dirs.user_state_dir)
    return PthaPaths(
        config_dir=Path(dirs.user_config_dir),
        data_dir=data_dir,
        cache_dir=Path(dirs.user_cache_dir),
        state_dir=state,
        log_dir=state / "logs",
        runtime_dir=runtime,
    )
