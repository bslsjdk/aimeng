"""Validate AIMENG SFT candidate JSONL without claiming factual correctness.

This is a structural and data-hygiene gate only. It does not verify truth,
license status, privacy, or whether a model has learned the content.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DIFFICULTIES = {"basic", "intermediate", "advanced"}
SPLITS = {"unassigned", "train", "validation", "test"}
STATUSES = {"pending", "verified", "rejected"}
ROLES = {"system", "user", "assistant"}


def validate(row: Any, line_no: int) -> list[str]:
    errors: list[str] = []
    prefix = f"line {line_no}"
    if not isinstance(row, dict):
        return [f"{prefix}: row must be a JSON object"]
    if row.get("schema_version") != "aimeng.sft_candidate.v1":
        errors.append(f"{prefix}: schema_version must be aimeng.sft_candidate.v1")
    for key in ("sample_id", "task_id", "topic", "expected_verifier"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"{prefix}: {key} must be a non-empty string")
    if row.get("difficulty") not in DIFFICULTIES:
        errors.append(f"{prefix}: invalid difficulty")
    if row.get("split", "unassigned") not in SPLITS:
        errors.append(f"{prefix}: invalid split")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"{prefix}: messages must be a non-empty list")
    else:
        roles = set()
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"{prefix}: messages[{i}] must be an object")
                continue
            if message.get("role") not in ROLES:
                errors.append(f"{prefix}: messages[{i}] has invalid role")
            else:
                roles.add(message["role"])
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append(f"{prefix}: messages[{i}].content must be non-empty")
        if "user" not in roles or "assistant" not in roles:
            errors.append(f"{prefix}: messages must include user and assistant roles")
    verification = row.get("verification")
    if not isinstance(verification, dict) or verification.get("status") not in STATUSES:
        errors.append(f"{prefix}: invalid verification.status")
    if isinstance(verification, dict) and verification.get("status") == "verified":
        if not verification.get("method") or not verification.get("evidence"):
            errors.append(f"{prefix}: verified records require method and evidence")
    eligible = row.get("training_eligible")
    if not isinstance(eligible, bool):
        errors.append(f"{prefix}: training_eligible must be boolean")
    if eligible is True:
        if not isinstance(verification, dict) or verification.get("status") != "verified":
            errors.append(f"{prefix}: training_eligible requires verified status")
        if row.get("split") not in {"train", "validation", "test"}:
            errors.append(f"{prefix}: eligible record must have an explicit split")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", help="Optional JSON report path")
    args = parser.parse_args()
    source = Path(args.input)
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    eligible = 0
    pending = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            count += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc.msg}")
                continue
            errors.extend(validate(row, line_no))
            if isinstance(row, dict):
                sample_id = row.get("sample_id")
                if isinstance(sample_id, str) and sample_id:
                    if sample_id in seen:
                        errors.append(f"line {line_no}: duplicate sample_id {sample_id}")
                    seen.add(sample_id)
                if row.get("training_eligible") is True:
                    eligible += 1
                verification = row.get("verification")
                if isinstance(verification, dict) and verification.get("status") == "pending":
                    pending += 1
    report = {
        "ok": not errors,
        "records": count,
        "pending": pending,
        "training_eligible": eligible,
        "errors": errors,
        "scope": "structure and duplicate IDs only; not factual verification",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
