"""Turn a structured verifier signal into a correction request for the core model.

This utility does not decide truth itself, call a model, change weights, or promote
memory. It preserves verifier scope and makes uncertain outcomes non-punitive.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

OUTCOMES = {"pass", "fail", "inconclusive", "unavailable"}
SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def validate_signal(signal: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(signal, dict):
        return ["signal must be a JSON object"]
    required = ("schema_version", "signal_id", "task_id", "target_artifact", "verifier",
                "outcome", "observed_at", "evidence_refs", "scope", "limitations")
    for key in required:
        if key not in signal:
            errors.append(f"missing {key}")
    if signal.get("schema_version") != "aimeng.learning_signal.v1":
        errors.append("schema_version must be aimeng.learning_signal.v1")
    for key in ("signal_id", "task_id", "observed_at"):
        if not isinstance(signal.get(key), str) or not signal[key].strip():
            errors.append(f"{key} must be a non-empty string")
    target = signal.get("target_artifact")
    if not isinstance(target, dict):
        errors.append("target_artifact must be an object")
    else:
        for key in ("ref", "kind"):
            if not isinstance(target.get(key), str) or not target[key].strip():
                errors.append(f"target_artifact.{key} must be a non-empty string")
        if not isinstance(target.get("sha256"), str) or not SHA256.fullmatch(target["sha256"]):
            errors.append("target_artifact.sha256 must be a 64-character SHA-256 hex string")
    verifier = signal.get("verifier")
    if not isinstance(verifier, dict):
        errors.append("verifier must be an object")
    else:
        for key in ("id", "version", "family"):
            if not isinstance(verifier.get(key), str) or not verifier[key].strip():
                errors.append(f"verifier.{key} must be a non-empty string")
    if not isinstance(signal.get("outcome"), str) or signal["outcome"] not in OUTCOMES:
        errors.append("invalid outcome")
    if not isinstance(signal.get("evidence_refs"), list) or any(not isinstance(x, str) or not x.strip() for x in signal["evidence_refs"]):
        errors.append("evidence_refs must be an array of non-empty strings")
    if not isinstance(signal.get("scope"), dict):
        errors.append("scope must be an object")
    if not isinstance(signal.get("limitations"), list) or any(not isinstance(x, str) for x in signal["limitations"]):
        errors.append("limitations must be an array of strings")
    if signal.get("outcome") == "fail":
        if not isinstance(signal.get("error_category"), str) or not signal["error_category"].strip():
            errors.append("fail outcome requires a non-empty error_category")
        if not isinstance(signal.get("evidence_refs"), list) or not signal["evidence_refs"]:
            errors.append("fail outcome requires at least one evidence reference")
    return errors


def build_feedback(signal: dict[str, Any]) -> dict[str, Any]:
    errors = validate_signal(signal)
    if errors:
        raise ValueError("; ".join(errors))
    outcome = signal["outcome"]
    if outcome == "fail":
        action = "revise_then_reverify"
        instruction = (
            "The verifier reported a scoped failure. Identify the smallest claim or step "
            "that could explain the failure. Do not merely restate the prior answer. "
            "Propose a correction, state what changed and why, then request a fresh "
            "verification run. Treat the evidence as applying only to the recorded scope."
        )
    elif outcome == "pass":
        action = "record_scoped_success"
        instruction = (
            "The verifier passed for this artifact and scope. Record what was tested and "
            "the limits of the result. Do not generalize this pass into proof of universal correctness."
        )
    else:
        action = "gather_more_evidence"
        instruction = (
            "The verifier could not establish pass or fail. Do not label the candidate wrong "
            "or correct. Identify the missing evidence or unavailable check and propose the "
            "lowest-cost next test."
        )
    return {
        "schema_version": "aimeng.correction_feedback.v1",
        "signal_id": signal["signal_id"],
        "task_id": signal["task_id"],
        "target_artifact": signal["target_artifact"],
        "verifier": signal["verifier"],
        "outcome": outcome,
        "action": action,
        "instruction_for_core_model": instruction,
        "preserved_error_category": signal.get("error_category"),
        "expected": signal.get("expected"),
        "observed": signal.get("observed"),
        "evidence_refs": signal["evidence_refs"],
        "scope": signal["scope"],
        "limitations": signal["limitations"],
        "weight_update_authorized": False,
        "memory_promotion_authorized": False
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One learning-signal JSON document")
    parser.add_argument("--output", help="Optional correction-feedback JSON output")
    args = parser.parse_args()
    try:
        signal = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = build_feedback(signal)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
