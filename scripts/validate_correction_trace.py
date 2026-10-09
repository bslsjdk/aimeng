"""Validate correction traces without treating model self-report as proof.

The validator checks trace consistency and artifact/signal binding. It does not
verify that referenced files exist, that a verifier is trustworthy, or that a
claimed SHA-256 matches bytes; those require a separate artifact store/verifier.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
OUTCOMES = {"pass", "fail", "inconclusive", "unavailable"}
STATUSES = {"pending", "verified_correction", "failed_correction", "inconclusive"}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def _valid_time(value: Any) -> bool:
    if not _nonempty(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except (ValueError, TypeError):
        return False


def _validate_artifact(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an artifact object")
        return
    for key in ("ref", "kind"):
        if not _nonempty(value.get(key)):
            errors.append(f"{label}.{key} must be a non-empty string")
    if not _valid_hash(value.get("sha256")):
        errors.append(f"{label}.sha256 must be a 64-character SHA-256 hex string")


def _validate_signal(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a signal object")
        return
    for key in ("ref",):
        if not _nonempty(value.get(key)):
            errors.append(f"{label}.{key} must be a non-empty string")
    if not _valid_hash(value.get("sha256")):
        errors.append(f"{label}.sha256 must be a 64-character SHA-256 hex string")
    if value.get("outcome") not in OUTCOMES:
        errors.append(f"{label}.outcome is invalid")
    if not _valid_hash(value.get("target_artifact_sha256")):
        errors.append(f"{label}.target_artifact_sha256 must be a SHA-256 hex string")
    if not isinstance(value.get("scope"), dict):
        errors.append(f"{label}.scope must be an object")
    refs = value.get("evidence_refs")
    if not isinstance(refs, list) or any(not _nonempty(ref) for ref in refs):
        errors.append(f"{label}.evidence_refs must be an array of non-empty strings")
    elif value.get("outcome") in {"pass", "fail"} and not refs:
        errors.append(f"{label} with {value.get('outcome')} requires evidence_refs")


def validate_trace(trace: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(trace, dict):
        return ["trace must be a JSON object"]

    required = (
        "schema_version", "trace_id", "task_id", "created_at", "updated_at",
        "initial_artifact", "initial_signal", "attribution_hypotheses",
        "correction_artifact", "recheck_signal", "status", "scope",
        "limitations", "provenance",
    )
    for key in required:
        if key not in trace:
            errors.append(f"missing {key}")
    if trace.get("schema_version") != "aimeng.correction_trace.v1":
        errors.append("schema_version must be aimeng.correction_trace.v1")
    for key in ("trace_id", "task_id"):
        if not _nonempty(trace.get(key)):
            errors.append(f"{key} must be a non-empty string")
    for key in ("created_at", "updated_at"):
        if not _valid_time(trace.get(key)):
            errors.append(f"{key} must be an ISO-8601 timestamp with timezone")
    if _valid_time(trace.get("created_at")) and _valid_time(trace.get("updated_at")):
        created = datetime.fromisoformat(trace["created_at"].replace("Z", "+00:00"))
        updated = datetime.fromisoformat(trace["updated_at"].replace("Z", "+00:00"))
        if updated < created:
            errors.append("updated_at must not precede created_at")

    _validate_artifact(trace.get("initial_artifact"), "initial_artifact", errors)
    _validate_signal(trace.get("initial_signal"), "initial_signal", errors)
    correction = trace.get("correction_artifact")
    if correction is not None:
        _validate_artifact(correction, "correction_artifact", errors)
    recheck = trace.get("recheck_signal")
    if recheck is not None:
        _validate_signal(recheck, "recheck_signal", errors)

    hypotheses = trace.get("attribution_hypotheses")
    if not isinstance(hypotheses, list):
        errors.append("attribution_hypotheses must be an array")
    else:
        for index, item in enumerate(hypotheses):
            label = f"attribution_hypotheses[{index}]"
            if not isinstance(item, dict) or not _nonempty(item.get("hypothesis")):
                errors.append(f"{label}.hypothesis must be a non-empty string")
                continue
            if item.get("confidence_label") not in {"low", "medium", "high"}:
                errors.append(f"{label}.confidence_label must be low, medium, or high")
            refs = item.get("evidence_refs")
            if not isinstance(refs, list) or any(not _nonempty(ref) for ref in refs):
                errors.append(f"{label}.evidence_refs must be an array of non-empty strings")
    if not isinstance(trace.get("scope"), dict):
        errors.append("scope must be an object")
    if not isinstance(trace.get("limitations"), list) or any(
        not isinstance(item, str) for item in trace.get("limitations", [])
    ):
        errors.append("limitations must be an array of strings")
    provenance = trace.get("provenance")
    if not isinstance(provenance, dict) or not _nonempty(provenance.get("producer")):
        errors.append("provenance.producer must be a non-empty string")
    elif not isinstance(provenance.get("recorded_from"), list) or any(
        not _nonempty(item) for item in provenance["recorded_from"]
    ):
        errors.append("provenance.recorded_from must be an array of non-empty strings")

    status = trace.get("status")
    if status not in STATUSES:
        errors.append("invalid status")
        return errors

    initial = trace.get("initial_signal")
    artifact = trace.get("initial_artifact")
    if isinstance(initial, dict) and isinstance(artifact, dict):
        if initial.get("target_artifact_sha256") != artifact.get("sha256"):
            errors.append("initial_signal must target the exact initial_artifact hash")

    if status == "pending":
        if recheck is not None:
            errors.append("pending trace must not contain a recheck_signal")
        if correction is not None and isinstance(artifact, dict):
            if correction.get("sha256") == artifact.get("sha256"):
                errors.append("correction_artifact must have a new hash, not the initial hash")

    elif status == "verified_correction":
        if not isinstance(initial, dict) or initial.get("outcome") != "fail":
            errors.append("verified_correction requires an initial fail signal")
        if not isinstance(artifact, dict) or not isinstance(correction, dict):
            errors.append("verified_correction requires initial and correction artifacts")
        elif correction.get("sha256") == artifact.get("sha256"):
            errors.append("correction_artifact hash must differ from initial_artifact hash")
        if not isinstance(recheck, dict) or recheck.get("outcome") != "pass":
            errors.append("verified_correction requires a passing recheck_signal")
        elif isinstance(correction, dict):
            if recheck.get("target_artifact_sha256") != correction.get("sha256"):
                errors.append("recheck_signal must target the correction_artifact hash")
            if isinstance(initial, dict) and recheck.get("scope") != initial.get("scope"):
                errors.append("recheck scope must match the initial signal scope")

    elif status == "failed_correction":
        if not isinstance(initial, dict) or initial.get("outcome") != "fail":
            errors.append("failed_correction requires an initial fail signal")
        if not isinstance(correction, dict) or not isinstance(artifact, dict):
            errors.append("failed_correction requires a correction artifact")
        elif correction.get("sha256") == artifact.get("sha256"):
            errors.append("correction_artifact hash must differ from initial_artifact hash")
        if not isinstance(recheck, dict) or recheck.get("outcome") != "fail":
            errors.append("failed_correction requires a failing recheck_signal")
        elif isinstance(correction, dict) and recheck.get("target_artifact_sha256") != correction.get("sha256"):
            errors.append("recheck_signal must target the correction_artifact hash")

    elif status == "inconclusive":
        outcomes = [
            initial.get("outcome") if isinstance(initial, dict) else None,
            recheck.get("outcome") if isinstance(recheck, dict) else None,
        ]
        if not any(outcome in {"inconclusive", "unavailable"} for outcome in outcomes):
            errors.append("inconclusive status requires an inconclusive or unavailable signal")
        if isinstance(recheck, dict) and isinstance(correction, dict):
            if recheck.get("target_artifact_sha256") != correction.get("sha256"):
                errors.append("recheck_signal must target the correction_artifact hash")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One correction-trace JSON document")
    args = parser.parse_args()
    try:
        trace = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 2
    errors = validate_trace(trace)
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
