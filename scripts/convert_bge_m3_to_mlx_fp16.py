#!/usr/bin/env python3
"""Convert the official BGE-M3 checkpoint and heads to reproducible MLX FP16 artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ASSETS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
)
HEADS = ("sparse_linear", "colbert_linear")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Local BAAI/bge-m3 snapshot; downloads it when omitted.")
    parser.add_argument("--source-repo", default="BAAI/bge-m3")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload-repo")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = resolve_source(args.source, args.source_repo, args.source_revision)
    convert(source, output, source_repo=args.source_repo, source_revision=args.source_revision)
    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.upload_repo, repo_type="model", folder_path=output)
        print(api.repo_info(args.upload_repo, repo_type="model").sha)
    return 0


def resolve_source(local: str | None, repo: str, revision: str) -> Path:
    if local:
        path = Path(local).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, revision=revision))


def convert(source: Path, output: Path, *, source_repo: str, source_revision: str) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    backbone_path = source / "pytorch_model.bin"
    if backbone_path.is_file():
        raw: Any = torch.load(backbone_path, map_location="cpu", weights_only=True)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    elif (source / "model.safetensors").is_file():
        state = load_file(source / "model.safetensors", device="cpu")
    else:
        raise FileNotFoundError("Source checkpoint must contain pytorch_model.bin or model.safetensors.")
    if not isinstance(state, dict) or not state:
        raise ValueError("Backbone checkpoint does not contain a state_dict.")
    weights = {
        normalize_backbone_key(key): value.detach().cpu().to(torch.float16).contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and not key.endswith("position_ids")
    }
    save_file(weights, output / "model.safetensors")

    head_metadata: dict[str, dict[str, Any]] = {}
    for name in HEADS:
        source_path = source / f"{name}.pt"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        raw_head = torch.load(source_path, map_location="cpu", weights_only=True)
        head = raw_head.get("state_dict", raw_head) if isinstance(raw_head, dict) else raw_head
        if not isinstance(head, dict) or "weight" not in head or "bias" not in head:
            raise ValueError(f"{source_path.name} must contain weight and bias.")
        converted = {
            "weight": head["weight"].detach().cpu().to(torch.float16).contiguous(),
            "bias": head["bias"].detach().cpu().to(torch.float16).contiguous(),
        }
        validate_head(name, converted)
        save_file(converted, output / f"{name}.safetensors")
        head_metadata[name] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in converted.items()}

    for name in ASSETS:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, output / name)
    if (source / "1_Pooling").is_dir():
        shutil.copytree(source / "1_Pooling", output / "1_Pooling")
    metadata = {
        "schema_version": 1,
        "source_repository": source_repo,
        "source_revision": source_revision,
        "converted_at": datetime.now(UTC).isoformat(),
        "conversion_script": "scripts/convert_bge_m3_to_mlx_fp16.py",
        "target_runtime": "MLX",
        "target_dtype": "float16",
        "additional_training": False,
        "backbone_tensor_count": len(weights),
        "heads": head_metadata,
    }
    (output / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(model_card(metadata), encoding="utf-8")

    verified = load_file(output / "model.safetensors", device="cpu")
    if set(verified) != set(weights) or any(value.dtype != torch.float16 for value in verified.values()):
        raise RuntimeError("Saved backbone verification failed.")


def normalize_backbone_key(key: str) -> str:
    for prefix in ("module.", "model.", "roberta."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def validate_head(name: str, state: dict[str, Any]) -> None:
    weight = tuple(state["weight"].shape)
    bias = tuple(state["bias"].shape)
    if name == "sparse_linear" and weight not in {(1, 1024), (1024,)}:
        raise ValueError(f"Unexpected sparse weight shape: {weight}")
    if name == "sparse_linear" and bias not in {(1,), ()}:
        raise ValueError(f"Unexpected sparse bias shape: {bias}")
    if name == "colbert_linear" and (len(weight) != 2 or weight[1] != 1024):
        raise ValueError(f"Unexpected ColBERT weight shape: {weight}")


def model_card(metadata: dict[str, Any]) -> str:
    return f"""---
license: mit
library_name: mlx
base_model: {metadata['source_repository']}
---

# BGE-M3 MLX FP16

MLX FP16 format conversion of `{metadata['source_repository']}` at revision
`{metadata['source_revision']}`. Converted on {metadata['converted_at']} with
`{metadata['conversion_script']}`. The backbone, `sparse_linear`, and
`colbert_linear` weights all come from that source revision. No additional
training was performed.

Target runtime: MLX. Target dtype: FP16. The source model license is MIT; see
the upstream model card for its complete terms and attribution.
"""


if __name__ == "__main__":
    raise SystemExit(main())
