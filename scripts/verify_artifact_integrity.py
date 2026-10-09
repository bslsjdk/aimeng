"""Emit a learning signal from a file-integrity check.

This verifies byte identity only. A matching SHA-256 does not prove that file
contents are correct, safe, useful, or semantically valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    artifact_path: str,
    expected_sha256: str,
    task_id: str,
    signal_id: str,
    evidence_path: str,
) -> tuple[dict, dict]:
    if not SHA256.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character SHA-256 hex string")
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {artifact_path}")
    observed_hash = sha256_file(path)
    outcome = "pass" if observed_hash.lower() == expected_sha256.lower() else "fail"
    observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    evidence = {
        "schema_version": "aimeng.file_integrity_evidence.v1",
        "verifier_id": "aimeng-file-sha256",
        "verifier_version": "1.0",
        "artifact_ref": str(path),
        "expected_sha256": expected_sha256.lower(),
        "observed_sha256": observed_hash,
        "outcome": outcome,
        "checked_at": observed_at,
        "scope": {"check": "byte_identity_sha256", "file_size_bytes": path.stat().st_size},
        "limitations": [
            "This is an integrity check, not a semantic correctness or safety check.",
            "SHA-256 equality does not prove the expected file contents are desirable.",
        ],
    }
    signal = {
        "schema_version": "aimeng.learning_signal.v1",
        "signal_id": signal_id or f"signal-{uuid.uuid4().hex}",
        "task_id": task_id,
        "target_artifact": {
            "ref": str(path),
            "sha256": observed_hash,
            "kind": "file",
        },
        "verifier": {
            "id": "aimeng-file-sha256",
            "version": "1.0",
            "family": "other",
        },
        "outcome": outcome,
        "error_category": "artifact_hash_mismatch" if outcome == "fail" else None,
        "expected": expected_sha256.lower(),
        "observed": observed_hash,
        "severity": "medium" if outcome == "fail" else "low",
        "observed_at": observed_at,
        "evidence_refs": [evidence_path],
        "scope": evidence["scope"],
        "limitations": evidence["limitations"],
        "correlation_id": signal_id or None,
    }
    return evidence, signal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Path to the file to hash")
    parser.add_argument("--expected-sha256", required=True, help="Expected file SHA-256")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--signal-id", default="")
    parser.add_argument("--evidence-out", required=True, help="Evidence JSON output path")
    parser.add_argument("--signal-out", required=True, help="Learning signal JSON output path")
    args = parser.parse_args()
    try:
        evidence, signal = verify_file(
            args.artifact, args.expected_sha256, args.task_id, args.signal_id, args.evidence_out
        )
        for output_path, value in (
            (Path(args.evidence_out), evidence),
            (Path(args.signal_out), signal),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "outcome": signal["outcome"], "signal_out": args.signal_out}, ensure_ascii=False))
    return 0 if signal["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
