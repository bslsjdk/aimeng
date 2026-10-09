#!/usr/bin/env python3
"""Download the official Ornith-1.5-9B Q4_K_M GGUF into local model storage.

The multi-gigabyte model is intentionally never committed to Git. The Hugging Face
Hub client handles cache/resume and verifies repository metadata.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "ornith-ai/Ornith-1.5-9B-GGUF"
FILENAME = "Ornith-1.5-9B-Q4_K_M.gguf"
MIN_FREE_GIB = 8.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/ornith-1.5-9b-q4km",
                        help="Local output directory (default: models/ornith-1.5-9b-q4km)")
    parser.add_argument("--repo-id", default=REPO_ID,
                        help="Hugging Face repository (defaults to the official model repo)")
    parser.add_argument("--filename", default=FILENAME,
                        help="GGUF filename (defaults to Q4_K_M)")
    parser.add_argument("--allow-other-model", action="store_true",
                        help="Allow a non-default repository/file; use only when intentional")
    args = parser.parse_args()

    if not args.allow_other_model and (args.repo_id != REPO_ID or args.filename != FILENAME):
        parser.error("Non-default model rejected. Pass --allow-other-model to override explicitly.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir).free / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        print(
            f"ERROR: only {free_gib:.2f} GiB free in {output_dir}; "
            f"need at least {MIN_FREE_GIB:.1f} GiB for the ~5.8 GB GGUF plus cache/temp space.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: install dependencies with: python -m pip install -r requirements.txt",
              file=sys.stderr)
        return 3

    try:
        # hf_hub_download uses the Hub cache and supports resumable/cached downloads.
        cached_path = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="model",
        ))
        destination = output_dir / args.filename
        if cached_path.resolve() != destination.resolve():
            # Avoid a second multi-GB copy when possible; link first, then copy if unsupported.
            if destination.exists() and destination.stat().st_size == cached_path.stat().st_size:
                pass
            else:
                temporary = destination.with_suffix(destination.suffix + ".part")
                if temporary.exists():
                    temporary.unlink()
                try:
                    os.link(cached_path, temporary)
                except OSError:
                    shutil.copyfile(cached_path, temporary)
                temporary.replace(destination)
        else:
            destination = cached_path
    except Exception as exc:
        print(f"ERROR: download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    size = destination.stat().st_size
    if size < 1_000_000_000:
        print(f"ERROR: downloaded file is unexpectedly small ({size} bytes): {destination}",
              file=sys.stderr)
        return 5
    print(f"MODEL_PATH={destination}")
    print(f"MODEL_SIZE_BYTES={size}")
    print("NEXT_STEP=python scripts/inspect_gguf.py --model "
          + repr(str(destination)) + " --output runs/gguf_manifest.json")
    print("NOTE=Download complete does not mean training or Android memory validation is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
