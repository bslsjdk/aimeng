#!/usr/bin/env python3
"""Export an AIMENG sparse-diffusion checkpoint to a portable JSON bundle for Android.

This is a weight/container conversion only. It does not imply an Android runtime
exists; the consumer must implement the documented forward pass exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def export_checkpoint(source: Path, destination: Path) -> dict:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    required = {"config", "stoi", "itos", "model_state"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")

    tensors = {}
    for name, tensor in checkpoint["model_state"].items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model_state[{name!r}] is not a tensor")
        tensors[name] = {
            "shape": list(tensor.shape),
            "values": tensor.detach().cpu().to(torch.float32).reshape(-1).tolist(),
        }

    config = checkpoint["config"]
    bundle = {
        "format": "aimeng-mobile-diffusion-json-v1",
        "source_checkpoint_format": checkpoint.get("format", "unknown"),
        "config": {
            key: config[key]
            for key in ("neurons", "width", "active_k", "fanout", "max_steps",
                         "context", "temperature")
            if key in config
        },
        "stoi": checkpoint["stoi"],
        "itos": checkpoint["itos"],
        "tensors": tensors,
        "training_source": checkpoint.get("source", "unknown"),
        "warning": checkpoint.get(
            "warning",
            "Portable weights do not prove Android runtime correctness or language ability.",
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "format": bundle["format"],
        "tensor_count": len(tensors),
        "vocab_size": len(bundle["itos"]),
        "neurons": bundle["config"].get("neurons"),
        "output_bytes": destination.stat().st_size,
        "training_source": bundle["training_source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
