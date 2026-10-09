"""Validate AIMENG trajectory JSONL and detect task-level split leakage.

Structural validation does not prove that targets are correct or legally usable.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SPLITS = {"unassigned", "train", "validation", "test"}
STATUSES = {"pass", "fail", "unknown", "not_run"}
SOURCE_TYPES = {"human_authored", "teacher_generated", "real_task", "public_dataset", "synthetic", "unknown"}
PRIVACY = {"passed", "not_required", "pending", "rejected"}
SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def validate_row(row, line_no):
    errors = []
    required = ("schema_version", "task_id", "sample_id", "task_family", "input", "outcome", "provenance", "split")
    for key in required:
        if key not in row:
            errors.append(f"line {line_no}: missing {key}")
    if row.get("schema_version") != "aimeng.trajectory.v1":
        errors.append(f"line {line_no}: schema_version must be aimeng.trajectory.v1")
    for key in ("task_id", "sample_id", "task_family"):
        if key in row and (not isinstance(row[key], str) or not row[key].strip()):
            errors.append(f"line {line_no}: {key} must be a non-empty string")
    inp = row.get("input")
    if not isinstance(inp, dict) or not isinstance(inp.get("prompt"), str) or not inp["prompt"].strip():
        errors.append(f"line {line_no}: input.prompt must be a non-empty string")
    outcome = row.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("status") not in STATUSES:
        errors.append(f"line {line_no}: invalid outcome.status")
    elif outcome.get("status") in {"pass", "fail"}:
        if not outcome.get("verifier") or not outcome.get("verifier_version"):
            errors.append(f"line {line_no}: pass/fail requires verifier and verifier_version")
        if not outcome.get("evidence"):
            errors.append(f"line {line_no}: pass/fail requires evidence")
    prov = row.get("provenance")
    if not isinstance(prov, dict) or prov.get("source_type") not in SOURCE_TYPES:
        errors.append(f"line {line_no}: invalid provenance.source_type")
    elif prov.get("privacy_review") not in PRIVACY:
        errors.append(f"line {line_no}: provenance.privacy_review must be explicit")
    if isinstance(prov, dict) and prov.get("teacher_artifact_sha256") is not None:
        if not isinstance(prov["teacher_artifact_sha256"], str) or not SHA256.fullmatch(prov["teacher_artifact_sha256"]):
            errors.append(f"line {line_no}: invalid teacher_artifact_sha256")
    if row.get("split") not in SPLITS:
        errors.append(f"line {line_no}: invalid split")
    return errors


def audit(path):
    errors, rows, sample_ids, task_splits = [], [], set(), {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: each JSONL line must be an object")
            continue
        errors.extend(validate_row(row, line_no))
        sid = row.get("sample_id")
        if isinstance(sid, str):
            if sid in sample_ids:
                errors.append(f"line {line_no}: duplicate sample_id {sid}")
            sample_ids.add(sid)
        task_id, split = row.get("task_id"), row.get("split")
        if isinstance(task_id, str) and split in {"train", "validation", "test"}:
            task_splits.setdefault(task_id, set()).add(split)
        rows.append(row)
    for task_id, splits in sorted(task_splits.items()):
        if len(splits) > 1:
            errors.append(f"task-level split leakage: task_id {task_id} occurs in {sorted(splits)}")
    counts = {key: 0 for key in ("train", "validation", "test", "unassigned")}
    statuses = {key: 0 for key in sorted(STATUSES)}
    for row in rows:
        if row.get("split") in counts:
            counts[row["split"]] += 1
        outcome = row.get("outcome")
        if isinstance(outcome, dict) and outcome.get("status") in statuses:
            statuses[outcome["status"]] += 1
    return {"ok": not errors, "records": len(rows), "split_counts": counts,
            "outcome_counts": statuses, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Trajectory JSONL")
    parser.add_argument("--report", help="Optional JSON report output path")
    args = parser.parse_args()
    report = audit(Path(args.input))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
