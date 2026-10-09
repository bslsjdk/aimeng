"""Import teacher output into pending AIMENG SFT candidates.

Accepts either aimeng.teacher_demo.v1 (generic prompt/response) or the
aimeng.sft_candidate.v1 shape emitted by prompts/teacher_sft_batch_v1.md.
Regardless of input claims, imported data is reset to pending/unassigned/ineligible.
This script does not verify facts, code, provenance, or teacher identity.
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


def convert(row, line_no, teacher_model=None, prompt_version="teacher_sft_batch_v1"):
    if not isinstance(row, dict):
        raise ValueError(f"line {line_no}: each row must be a JSON object")
    schema = row.get("schema_version")
    if schema == "aimeng.teacher_demo.v1":
        for key in ("task_id", "topic", "prompt", "response"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"line {line_no}: {key} must be a non-empty string")
        messages = [{"role": "user", "content": row["prompt"].strip()},
                    {"role": "assistant", "content": row["response"].strip()}]
        model = (row.get("teacher") or {}).get("model") or teacher_model
        version = (row.get("teacher") or {}).get("prompt_version") or prompt_version
        difficulty = row.get("difficulty", "intermediate")
        topic, task_id = row["topic"].strip(), row["task_id"].strip()
        expected_verifier = row.get("expected_verifier") or "manual_review"
        provenance = row.get("provenance", {})
    elif schema == "aimeng.sft_candidate.v1":
        for key in ("sample_id", "task_id", "topic", "messages", "expected_verifier"):
            if key not in row:
                raise ValueError(f"line {line_no}: missing {key}")
        messages = row["messages"]
        if not isinstance(messages, list) or not messages or not any(isinstance(m, dict) and m.get("role") == "user" for m in messages) or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
            raise ValueError(f"line {line_no}: messages must contain user and end with assistant")
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"line {line_no}: invalid messages[{i}]")
        model = (row.get("teacher") or {}).get("model") or teacher_model
        version = (row.get("teacher") or {}).get("prompt_version") or prompt_version
        difficulty = row.get("difficulty", "intermediate")
        topic, task_id = row["topic"], row["task_id"]
        expected_verifier = row["expected_verifier"]
        provenance = row.get("provenance", {})
    else:
        raise ValueError(f"line {line_no}: unsupported schema_version {schema!r}")
    if not isinstance(topic, str) or not topic.strip() or not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"line {line_no}: topic and task_id must be non-empty strings")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"line {line_no}: invalid difficulty")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"line {line_no}: teacher model missing; pass --teacher-model")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"line {line_no}: prompt version missing")
    if not isinstance(provenance, dict):
        raise ValueError(f"line {line_no}: provenance must be an object")
    fingerprint = digest({"task_id": task_id.strip(), "messages": messages, "teacher_model": model.strip(), "prompt_version": version.strip()})
    return {
        "schema_version": "aimeng.sft_candidate.v1",
        "sample_id": "teacher-" + fingerprint[:24],
        "task_id": task_id.strip(),
        "topic": topic.strip(),
        "difficulty": difficulty,
        "messages": messages,
        "expected_verifier": str(expected_verifier).strip() or "manual_review",
        "teacher": {
            "model": model.strip(),
            "prompt_version": version.strip(),
            "artifact_sha256": (row.get("teacher") or {}).get("artifact_sha256") or (row.get("teacher") or {}).get("model_sha256"),
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
    parser.add_argument("--teacher-model", help="Required if teacher records do not contain teacher.model")
    parser.add_argument("--prompt-version", default="teacher_sft_batch_v1")
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        parser.error("input and output must be different files")
    rows, errors, seen = [], [], set()
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = convert(json.loads(raw), line_no, args.teacher_model, args.prompt_version)
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
    print(json.dumps({"ok": True, "imported": len(rows), "deduplicated": "content/task/model/prompt hash", "all_pending_and_ineligible": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
