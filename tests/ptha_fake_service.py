"""Synthetic foreground service entry point used by lifecycle subprocess tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from ptha.config import load_config
from ptha.service import RetrievalService, configure_service_logging, install_signal_handlers


class SyntheticSession:
    def search_archive(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"mode": "focused", "items": [{"text": "synthetic"}]}

    def construct_archive_context(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"mode": "broad", "items": [{"text": "synthetic"}]}

    def close(self) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("service")
    parser.add_argument("run")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    configure_service_logging(config)
    service = RetrievalService(config, session_factory=lambda _config: SyntheticSession())
    install_signal_handlers(service)
    service.run()


if __name__ == "__main__":
    main()
