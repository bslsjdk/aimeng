"""Validate a plastic-module manifest and reject unsafe promotion states."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STATUSES = {"draft", "training", "candidate", "accepted", "deprecated", "rejected"}
MODULE_TYPES = {"adapter", "subgraph", "future_neuron_graph"}
VALIDATION = {"pending", "pass", "fail", "inconclusive"}
SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def validate_manifest(row: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return ["manifest must be a JSON object"]
    required = (
        "schema_version", "module_id", "module_version", "status", "module_type",
        "base_model_id", "base_model_revision", "interface_signature",
        "input_contract", "output_contract", "task_families", "known_limits",
        "resource_cost", "training_data_fingerprint", "eval_suite_fingerprint",
        "validation_status", "validated_scope", "artifact_hashes",
    )
    for key in required:
        if key not in row:
            errors.append(f"missing {key}")
    if row.get("schema_version") != "aimeng.plastic_module.v1":
        errors.append("schema_version must be aimeng.plastic_module.v1")
    for key in ("module_id", "module_version", "base_model_id", "base_model_revision", "interface_signature",
                "training_data_fingerprint", "eval_suite_fingerprint"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(row.get("status"), str) or row.get("status") not in STATUSES:
        errors.append("invalid status")
    if not isinstance(row.get("module_type"), str) or row.get("module_type") not in MODULE_TYPES:
        errors.append("invalid module_type")
    if not isinstance(row.get("validation_status"), str) or row.get("validation_status") not in VALIDATION:
        errors.append("invalid validation_status")
    for key in ("input_contract", "output_contract", "resource_cost", "validated_scope"):
        if not isinstance(row.get(key), dict):
            errors.append(f"{key} must be an object")
    for key in ("task_families", "known_limits"):
        if not isinstance(row.get(key), list) or any(not isinstance(item, str) for item in row[key]):
            errors.append(f"{key} must be an array of strings")
    hashes = row.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        errors.append("artifact_hashes must be a non-empty object")
    elif any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes.values()):
        errors.append("every artifact_hashes value must be a 64-character SHA-256 hex string")
    if row.get("status") == "accepted":
        if row.get("validation_status") != "pass":
            errors.append("accepted module requires validation_status=pass")
        for key in ("eval_report_ref", "rollback_target"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"accepted module requires non-empty {key}")
        if not isinstance(row.get("validated_scope"), dict) or not row.get("validated_scope"):
            errors.append("accepted module requires a non-empty validated_scope")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Module manifest JSON")
    parser.add_argument("--report", help="Optional JSON report output path")
    args = parser.parse_args()
    path = Path(args.input)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_manifest(manifest)
    except json.JSONDecodeError as exc:
        errors = [f"invalid JSON: {exc.msg}"]
    report = {"ok": not errors, "errors": errors}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
