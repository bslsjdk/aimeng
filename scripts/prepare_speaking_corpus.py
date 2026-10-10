#!/usr/bin/env python3
"""Build a bounded Chinese speaking corpus from a public Hugging Face dataset.

The script streams records, normalizes them into user/assistant text, deduplicates
exact examples, and writes a plain UTF-8 corpus for the character-level AIMENG
prototype. It does not certify factual correctness. Review the source dataset's
license and restrictions before using the resulting model outside research.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def record_to_text(row: dict[str, Any]) -> str | None:
    instruction = row.get("instruction")
    input_text = row.get("input", "")
    output = row.get("output")
    if isinstance(instruction, str) and isinstance(output, str):
        user = instruction.strip()
        extra = input_text.strip() if isinstance(input_text, str) else ""
        answer = output.strip()
        if extra:
            user += "\n" + extra
        if user and answer:
            return f"用户：{user}\n助手：{answer}\n\n"
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role in {"user", "human"}:
                label = "用户"
            elif role in {"assistant", "gpt", "bot"}:
                label = "助手"
            elif role == "system":
                label = "系统"
            else:
                continue
            parts.append(f"{label}：{content.strip()}")
        if len(parts) >= 2:
            return "\n".join(parts) + "\n\n"
    return None


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare(dataset_name: str, split: str, output: Path, max_records: int,
            min_chars: int, max_chars: int, seed: int) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package before preparing the corpus") from exc

    stream = load_dataset(dataset_name, split=split, streaming=True)
    seen: set[str] = set()
    records: list[str] = []
    scanned = 0
    duplicates = 0
    rejected = 0
    for row in stream:
        scanned += 1
        if not isinstance(row, dict):
            rejected += 1
            continue
        text = record_to_text(row)
        if text is None:
            rejected += 1
            continue
        text = normalize(text)
        if len(text) < min_chars or len(text) > max_chars:
            rejected += 1
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        records.append(text)
        if len(records) >= max_records:
            break
    if not records:
        raise RuntimeError(f"No usable records from {dataset_name}; scanned={scanned}")
    random.Random(seed).shuffle(records)
    corpus = "\n\n".join(records) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(corpus, encoding="utf-8")
    temp.replace(output)
    manifest = {
        "format": "aimeng-speaking-corpus-v1",
        "dataset": dataset_name,
        "split": split,
        "records": len(records),
        "records_scanned": scanned,
        "duplicates_skipped": duplicates,
        "records_rejected": rejected,
        "characters": len(corpus),
        "utf8_bytes": len(corpus.encode("utf-8")),
        "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "seed": seed,
        "output": str(output),
        "warning": "Corpus formatting and deduplication do not verify factual correctness. Check dataset license/restrictions.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BelleGroup/train_0.5M_CN")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.max_records < 1 or args.min_chars < 1 or args.max_chars < args.min_chars:
        parser.error("record and character limits must be positive and consistent")
    prepare(args.dataset, args.split, args.output, args.max_records,
            args.min_chars, args.max_chars, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
