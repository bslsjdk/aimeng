"""Validate AIMENG idea-cycle JSONL without treating self-assessment as proof.

This checks schema-level invariants and promotion status only. It does not prove an
idea is true, novel, safe, or useful; that requires the referenced verifier/evidence.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STATUSES = {"speculative", "plausible", "supported", "verified_for_scope", "refuted", "unknown"}
RESULT_STATUSES = {"pass", "fail", "inconclusive", "unavailable"}
SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_row(row: object, line_no: int) -> list[str]:
    errors: list[str] = []
    prefix = f"line {line_no}: "
    if not isinstance(row, dict):
        return [prefix + "each JSONL line must be an object"]
    required = ("schema_version", "idea_id", "task_id", "statement", "assumptions",
                "predictions", "counterexamples_to_seek", "epistemic_status",
                "verification_plan", "provenance")
    for key in required:
        if key not in row:
            errors.append(prefix + f"missing {key}")
    if row.get("schema_version") != "aimeng.idea.v1":
        errors.append(prefix + "schema_version must be aimeng.idea.v1")
    for key in ("idea_id", "task_id", "statement"):
        if key in row and not _nonempty(row[key]):
            errors.append(prefix + f"{key} must be a non-empty string")
    for key in ("assumptions", "predictions", "counterexamples_to_seek", "evidence_refs", "result_refs"):
        if key in row and (not isinstance(row[key], list) or any(not isinstance(x, str) for x in row[key])):
            errors.append(prefix + f"{key} must be an array of strings")
    status = row.get("epistemic_status")
    if not isinstance(status, str) or status not in STATUSES:
        errors.append(prefix + "invalid epistemic_status")
    plan = row.get("verification_plan")
    if not isinstance(plan, dict) or not _nonempty(plan.get("method")):
        errors.append(prefix + "verification_plan.method must be a non-empty string")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(prefix + "provenance must be an object")
    else:
        for key in ("model_version", "policy_version"):
            if not _nonempty(provenance.get(key)):
                errors.append(prefix + f"provenance.{key} must be a non-empty string")
    result = row.get("verification_result")
    if result is not None:
        if not isinstance(result, dict):
            errors.append(prefix + "verification_result must be an object or null")
        else:
            if not isinstance(result.get("status"), str) or result.get("status") not in RESULT_STATUSES:
                errors.append(prefix + "invalid verification_result.status")
            for key in ("evidence_refs",):
                if key in result and (not isinstance(result[key], list) or any(not isinstance(x, str) for x in result[key])):
                    errors.append(prefix + f"verification_result.{key} must be an array of strings")
    if status == "verified_for_scope":
        if not isinstance(result, dict):
            errors.append(prefix + "verified_for_scope requires verification_result")
        else:
            if result.get("status") != "pass":
                errors.append(prefix + "verified_for_scope requires verification_result.status=pass")
            for key in ("verifier_id", "verifier_version"):
                if not _nonempty(result.get(key)):
                    errors.append(prefix + f"verified_for_scope requires non-empty {key}")
            if not isinstance(result.get("scope"), dict):
                errors.append(prefix + "verified_for_scope requires a scope object")
            if not isinstance(result.get("evidence_refs"), list) or not result["evidence_refs"] or any(not _nonempty(x) for x in result["evidence_refs"]):
                errors.append(prefix + "verified_for_scope requires non-empty evidence_refs")
    return errors


def audit(path: Path) -> dict:
    errors: list[str] = []
    rows = 0
    seen_ids: set[str] = set()
    task_ids: dict[str, set[str]] = {}
    counts = {status: 0 for status in sorted(STATUSES)}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc.msg}")
            continue
        row_errors = validate_row(row, line_no)
        errors.extend(row_errors)
        if not isinstance(row, dict):
            continue
        rows += 1
        idea_id, task_id, status = row.get("idea_id"), row.get("task_id"), row.get("epistemic_status")
        if isinstance(idea_id, str):
            if idea_id in seen_ids:
                errors.append(f"line {line_no}: duplicate idea_id {idea_id}")
            seen_ids.add(idea_id)
        if isinstance(task_id, str) and isinstance(status, str):
            task_ids.setdefault(task_id, set()).add(status)
            if status in counts:
                counts[status] += 1
    return {"ok": not errors, "records": rows, "status_counts": counts, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Idea-record JSONL")
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
