#!/usr/bin/env python3
"""Inspect a local GGUF without loading model weights into RAM or running inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    if hasattr(value, "contents"):
        try:
            return scalar(value.contents())
        except Exception:
            pass
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("AIMENG_MODEL_GGUF"),
                        help="GGUF path (defaults to AIMENG_MODEL_GGUF)")
    parser.add_argument("--output", default="runs/gguf_manifest.json",
                        help="JSON output path; default: runs/gguf_manifest.json")
    args = parser.parse_args()

    if not args.model:
        parser.error("Set AIMENG_MODEL_GGUF or pass --model /path/to/model.gguf")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if model_path.suffix.lower() != ".gguf":
        print("ERROR: expected a .gguf file; refusing to guess another format", file=sys.stderr)
        return 2

    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: missing dependency. Run: python -m pip install -r requirements.txt",
              file=sys.stderr)
        return 3

    try:
        # GGUFReader maps the file and reads metadata; this script does not allocate a
        # model-sized tensor copy or execute the network.
        reader = GGUFReader(str(model_path), mode="r")
        metadata = {}
        for key, field in reader.fields.items():
            try:
                metadata[key] = scalar(field.contents())
            except Exception:
                metadata[key] = str(field)
        tensors = []
        quant_types: dict[str, int] = {}
        total_tensor_bytes = 0
        for tensor in reader.tensors:
            shape = [int(dim) for dim in tensor.shape]
            tensor_type = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
            n_bytes = int(getattr(tensor, "n_bytes", 0))
            total_tensor_bytes += n_bytes
            quant_types[tensor_type] = quant_types.get(tensor_type, 0) + 1
            tensors.append({
                "name": str(tensor.name),
                "shape": shape,
                "type": tensor_type,
                "n_bytes": n_bytes,
            })
    except Exception as exc:
        print(f"ERROR: failed to parse GGUF: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    stat = model_path.stat()
    result = {
        "schema_version": 1,
        "model": {
            "filename": model_path.name,
            "local_path": str(model_path),
            "file_size_bytes": stat.st_size,
            "file_size_gib": round(stat.st_size / (1024 ** 3), 4),
            "sha256": sha256_file(model_path),
        },
        "gguf": {
            "metadata": metadata,
            "tensor_count": len(tensors),
            "tensor_storage_bytes_reported": total_tensor_bytes,
            "tensor_types": quant_types,
            "tensors": tensors,
        },
        "warning": (
            "File size and GGUF tensor storage are not runtime RAM measurements. "
            "Actual RSS/PSS, KV cache, backend buffers, and Android/NPU allocations "
            "must be measured during inference."
        ),
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "model": model_path.name,
        "file_size_gib": result["model"]["file_size_gib"],
        "sha256": result["model"]["sha256"],
        "tensor_count": len(tensors),
        "tensor_types": quant_types,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
