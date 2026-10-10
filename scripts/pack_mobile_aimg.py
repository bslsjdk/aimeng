#!/usr/bin/env python3
"""Pack an existing portable AIMENG JSON bundle into the Android .aimg format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.export_mobile_diffusion import pack_aimg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="existing mobile_diffusion.json bundle")
    parser.add_argument("--output", required=True, type=Path, help="destination .aimg file")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input bundle does not exist: {args.input}")
    try:
        bundle = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"input is not valid UTF-8 JSON: {exc}")
    if bundle.get("format") != "aimeng-mobile-diffusion-json-v1":
        parser.error("input JSON is not an AIMENG mobile diffusion bundle")
    if not isinstance(bundle.get("tensors"), dict) or not isinstance(bundle.get("itos"), list):
        parser.error("input AIMENG bundle is missing tensors or vocabulary")
    print(json.dumps(pack_aimg(args.input, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
