#!/usr/bin/env python3
"""Pack AIMENG's portable mobile JSON export into the canonical .aimg format.

Usage:
    python scripts/pack_mobile_aimg.py mobile_diffusion.json mobile_diffusion.aimg

The .aimg file is an inference bundle, not a PyTorch .pth checkpoint. Training
may use any backend, but the Android runtime consumes this portable format.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path

MAGIC = b"AIMG"
VERSION = 1
HEADER = struct.Struct(">4sII32s")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_PACKED_BYTES = 32 * 1024 * 1024
EXPECTED_FORMAT = "aimeng-mobile-diffusion-json-v1"


def pack(source: Path, destination: Path) -> None:
    raw = source.read_bytes()
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"source JSON must be between 1 byte and {MAX_JSON_BYTES} bytes")
    model = json.loads(raw.decode("utf-8"))
    if model.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"expected format={EXPECTED_FORMAT!r}")
    # Validate required top-level keys before making a phone bundle.
    for key in ("config", "stoi", "itos", "tensors"):
        if key not in model:
            raise ValueError(f"missing required model key: {key}")
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    packed = HEADER.pack(MAGIC, VERSION, len(raw), hashlib.sha256(raw).digest()) + compressed
    if len(packed) > MAX_PACKED_BYTES:
        raise ValueError(f"packed bundle is {len(packed)} bytes; limit is {MAX_PACKED_BYTES}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(packed)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(destination)
    print(f"AIMG v{VERSION}: {len(raw)} JSON bytes -> {len(packed)} packed bytes")
    print(f"SHA-256: {hashlib.sha256(raw).hexdigest()}")
    print(f"Output: {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: pack_mobile_aimg.py INPUT.json OUTPUT.aimg")
    pack(Path(sys.argv[1]), Path(sys.argv[2]))
