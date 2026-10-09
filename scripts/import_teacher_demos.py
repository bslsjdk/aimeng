"""Import black-box teacher generations into pending AIMENG SFT candidates.

Input schema (one JSON object per line):
{"schema_version":"aimeng.teacher_demo.v1","task_id":"...","topic":"...","difficulty":"basic|intermediate|advanced","prompt":"...","response":"...","teacher":{"model":"...","prompt_version":"..."},"provenance":{"source_ref":"optional"}}
All imported records remain pending, unassigned, and ineligible. This is not a verifier.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

DIFFICULTIES = {"basic", "intermediate", "advanced"}


def digest(value):
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def convert(row, line_no):
    if not isinstance(row, dict) or row.get("schema_version") != "aimeng.teacher_demo.v1":
        raise ValueError(f"line {line_no}: schema_version must be aimeng.teacher_demo.v1")
    for key in ("task_id", "topic", "prompt", "response"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"line {line_no}: {key} must be a non-empty string")
    if row.get("difficulty", "intermediate") not in DIFFICULTIES:
        raise ValueError(f"line {line_no}: invalid difficulty")
    teacher = row.get("teacher")
    if not isinstance(teacher, dict) or not isinstance(teacher.get("model"), str) or not teacher["model"].strip():
        raise ValueError(f"line {line_no}: teacher.model is required")
    if not isinstance(teacher.get("prompt_version"), str) or not teacher["prompt_version"].strip():
        raise ValueError(f"line {line_no}: teacher.prompt_version is required")
    fingerprint = digest({"task_id": row["task_id"], "prompt": row["prompt"].strip(), "response": row["response"].strip()})
    provenance = row.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError(f"line {line_no}: provenance must be an object")
    return {
        "schema_version": "aimeng.sft_candidate.v1",
        "sample_id": "teacher-" + fingerprint[:24],
        "task_id": row["task_id"].strip(),
        "topic": row["topic"].strip(),
        "difficulty": row.get("difficulty", "intermediate"),
        "messages": [
            {"role": "user", "content": row["prompt"].strip()},
            {"role": "assistant", "content": row["response"].strip()},
        ],
        "expected_verifier": row.get("expected_verifier") or "manual_review",
        "teacher": {
            "model": teacher["model"].strip(),
            "prompt_version": teacher["prompt_version"].strip(),
            "artifact_sha256": teacher.get("artifact_sha256"),
        },
        "provenance": {
            "source_type": "teacher_generated",
            "source_ref": provenance.get("source_ref"),
            "license_review": provenance.get("license_review", "pending"),
            "privacy_review": provenance.get("privacy_review", "pending"),
        },
        "verification": {"status": "pending", "method": None, "evidence": []},
        "split": "unassigned",
        "training_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        parser.error("input and output must be different files")
    rows, errors, seen = [], [], set()
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = convert(json.loads(raw), line_no)
            if row["sample_id"] in seen:
                continue
            seen.add(row["sample_id"])
            rows.append(row)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "imported": len(rows), "deduplicated": "exact task/prompt/response hash", "all_pending_and_ineligible": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
