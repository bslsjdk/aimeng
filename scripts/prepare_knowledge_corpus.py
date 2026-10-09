"""Validate and deduplicate JSONL records distilled from a frozen teacher.

This script does not query the model or certify facts. Verification must come
from an external source, deterministic test, or human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED_STATUS = {"pending", "verified", "rejected"}
ALLOWED_SPLITS = {"unassigned", "train", "validation", "test"}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_record(row: dict[str, Any], line_no: int) -> list[str]:
    errors = []
    if row.get("schema_version") != "aimeng.knowledge.v1":
        errors.append(f"line {line_no}: schema_version must be aimeng.knowledge.v1")
    for key in ("topic", "content"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"line {line_no}: {key} must be a non-empty string")
    teacher = row.get("teacher")
    if not isinstance(teacher, dict) or not teacher.get("model") or not teacher.get("prompt_version"):
        errors.append(f"line {line_no}: teacher.model and teacher.prompt_version are required")
    verification = row.get("verification")
    if not isinstance(verification, dict) or verification.get("status") not in ALLOWED_STATUS:
        errors.append(f"line {line_no}: invalid verification.status")
    if isinstance(verification, dict) and verification.get("status") == "verified":
        if not verification.get("method") or not verification.get("verifier_version"):
            errors.append(f"line {line_no}: verified records require method and verifier_version")
        if not row.get("provenance"):
            errors.append(f"line {line_no}: verified records require non-empty provenance")
    if row.get("split", "unassigned") not in ALLOWED_SPLITS:
        errors.append(f"line {line_no}: invalid split")
    return errors


def load_and_deduplicate(source: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records, errors, seen = [], [], set()
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: each JSONL row must be an object")
            continue
        row_errors = validate_record(row, line_no)
        if row_errors:
            errors.extend(row_errors)
            continue
        content_key = canonical_hash({
            "topic": row["topic"].strip().casefold(),
            "content": " ".join(row["content"].split()).casefold(),
        })
        if content_key in seen:
            continue
        seen.add(content_key)
        row = dict(row)
        row["topic"], row["content"] = row["topic"].strip(), row["content"].strip()
        row["record_id"] = "sha256:" + content_key
        records.append(row)
    return records, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Teacher-produced JSONL records")
    parser.add_argument("--output", required=True, help="Validated/deduplicated JSONL output")
    parser.add_argument("--verified-only", action="store_true",
                        help="Export only records with verification.status=verified")
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        parser.error("input and output must be different files")
    records, errors = load_and_deduplicate(source)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 2
    if args.verified_only:
        records = [r for r in records if r["verification"]["status"] == "verified"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "ok": True, "output_records": len(records),
        "verified_records": sum(r["verification"]["status"] == "verified" for r in records),
        "output": str(target),
        "note": "Structural validation only; factual correctness is not established.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
