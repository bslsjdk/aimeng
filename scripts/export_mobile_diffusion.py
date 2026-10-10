#!/usr/bin/env python3
"""Export an AIMENG sparse-diffusion checkpoint to JSON and canonical .aimg for Android."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import torch

MAGIC = b"AIMG"
AIMG_VERSION = 1
MAX_PACKED_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024


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
    json_bytes = (json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(json_bytes) > MAX_JSON_BYTES:
        raise ValueError(f"JSON bundle exceeds Android limit: {len(json_bytes)} bytes")
    destination.write_bytes(json_bytes)
    return {
        "format": bundle["format"],
        "tensor_count": len(tensors),
        "vocab_size": len(bundle["itos"]),
        "neurons": bundle["config"].get("neurons"),
        "output_bytes": len(json_bytes),
        "training_source": bundle["training_source"],
    }


def pack_aimg(json_path: Path, output_path: Path) -> dict:
    """Write Android AimengModelFormat's AIMG v1: magic, BE version/length, SHA-256, gzip JSON."""
    raw = json_path.read_bytes()
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"JSON payload size is outside 1..{MAX_JSON_BYTES} bytes")
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    packed = MAGIC + struct.pack(">II", AIMG_VERSION, len(raw)) + hashlib.sha256(raw).digest() + compressed
    if len(packed) > MAX_PACKED_BYTES:
        raise ValueError(f"compressed AIMG bundle exceeds Android limit: {len(packed)} bytes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(packed)
    # Verify the exact on-disk bytes before reporting success.
    if packed[:4] != MAGIC or struct.unpack(">II", packed[4:12]) != (AIMG_VERSION, len(raw)):
        raise RuntimeError("AIMG header self-check failed")
    if hashlib.sha256(gzip.decompress(packed[44:])).digest() != packed[12:44]:
        raise RuntimeError("AIMG payload hash self-check failed")
    return {
        "format": "AIMG",
        "version": AIMG_VERSION,
        "output_bytes": len(packed),
        "uncompressed_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="portable JSON bundle (kept for compatibility)")
    parser.add_argument("--binary-output", type=Path, help="canonical compressed Android .aimg import file")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    result = export_checkpoint(args.checkpoint, args.output)
    print(json.dumps(result, ensure_ascii=False))
    if args.binary_output:
        packed = pack_aimg(args.output, args.binary_output)
        print(json.dumps({"binary_output": str(args.binary_output), **packed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
