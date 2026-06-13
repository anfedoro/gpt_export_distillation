from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .loader import discover_inputs, load_bundle
from .pipeline import build_documents, write_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill ChatGPT export into Markdown-only folders."
    )
    parser.add_argument(
        "--config",
        default="gpt_export_distillation.toml",
        help="Path to TOML config. Defaults to ./gpt_export_distillation.toml if present.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional explicit input path: export zip, export directory, or conversations JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Overrides config output.output_dir/root folder placement.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path if config_path.exists() else None)
    input_paths = discover_inputs(config, args.input)
    if not input_paths:
        raise SystemExit("No inputs found. Check config input.search_dir and patterns.")

    for input_path in input_paths:
        bundle = load_bundle(input_path)
        documents = build_documents(bundle, config)
        output_root = write_output(bundle, documents, config, args.output_dir)
        print(
            f"{input_path.name}: conversations={len(documents)} -> {output_root}"
        )
