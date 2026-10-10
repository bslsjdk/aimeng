#!/usr/bin/env python3
"""Create deterministic, record-level train/validation/test files from UTF-8 text.

Records are separated by one or more blank lines. Exact normalized duplicates are
removed BEFORE splitting, preventing the same record from appearing in multiple splits.
Raw corpus files and split outputs should remain in persistent storage, not Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_records(source: str, seed: int = 7, train_ratio: float = 0.90,
                  validation_ratio: float = 0.05):
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and sum to less than 1")
    raw_records = re.split(r"\n\s*\n+", source.replace("\r\n", "\n").replace("\r", "\n"))
    records = []
    seen = set()
    duplicates = 0
    for raw in raw_records:
        record = "\n".join(line.rstrip() for line in raw.strip().splitlines()).strip()
        if not record:
            continue
        key = sha256(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        records.append(record)
    if len(records) < 20:
        raise ValueError(f"Need at least 20 unique records for meaningful splits; found {len(records)}")
    random.Random(seed).shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * validation_ratio)
    # Keep all three splits non-empty and leave at least one record for test.
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    return {
        "train": records[:n_train],
        "validation": records[n_train:n_train+n_val],
        "test": records[n_train+n_val:],
    }, duplicates


def write_split(records, path: Path):
    text = "\n\n".join(records) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return {
        "path": str(path),
        "records": len(records),
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "sha256": sha256(text),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="UTF-8 corpus with records separated by blank lines")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--validation-ratio", type=float, default=0.05)
    args = p.parse_args()
    input_path = Path(args.input)
    source = input_path.read_text(encoding="utf-8")
    splits, duplicates = split_records(source, args.seed, args.train_ratio, args.validation_ratio)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "aimeng-record-splits-v1",
        "source_path": str(input_path),
        "source_sha256": sha256(source),
        "seed": args.seed,
        "split_method": "deduplicate exact normalized records before deterministic shuffle and split",
        "duplicates_removed_before_split": duplicates,
        "unique_records": sum(len(v) for v in splits.values()),
        "splits": {},
        "warning": "Record-level separation prevents exact-record overlap, but does not detect paraphrase/near-duplicate leakage. Review source license and quality separately.",
    }
    for name, records in splits.items():
        manifest["splits"][name] = write_split(records, out / f"{name}.txt")
    manifest_path = out / "split_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
