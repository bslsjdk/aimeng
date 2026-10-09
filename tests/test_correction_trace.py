import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_correction_trace", ROOT / "scripts" / "validate_correction_trace.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

INITIAL_HASH = "a" * 64
CORRECTION_HASH = "b" * 64
SIGNAL_HASH = "c" * 64
RECHECK_HASH = "d" * 64
SCOPE = {"suite": "toy-v1", "cases": 3}


def signal(ref, digest, outcome, target_hash):
    evidence = [f"reports/{ref}.json"] if outcome in {"pass", "fail"} else []
    return {
        "ref": f"signals/{ref}.json",
        "sha256": digest,
        "outcome": outcome,
        "target_artifact_sha256": target_hash,
        "scope": deepcopy(SCOPE),
        "evidence_refs": evidence,
    }


def trace(status="verified_correction"):
    return {
        "schema_version": "aimeng.correction_trace.v1",
        "trace_id": "trace-001",
        "task_id": "task-001",
        "created_at": "2026-10-09T10:00:00Z",
        "updated_at": "2026-10-09T10:05:00Z",
        "initial_artifact": {
            "ref": "artifacts/answer-v1.txt",
            "sha256": INITIAL_HASH,
            "kind": "answer",
        },
        "initial_signal": signal("initial-fail", SIGNAL_HASH, "fail", INITIAL_HASH),
        "attribution_hypotheses": [{
            "hypothesis": "A boundary condition may be missing.",
            "confidence_label": "low",
            "evidence_refs": ["reports/initial-fail.json"],
            "disconfirming_test": "Run the smallest boundary counterexample.",
        }],
        "correction_artifact": {
            "ref": "artifacts/answer-v2.txt",
            "sha256": CORRECTION_HASH,
            "kind": "answer",
        },
        "recheck_signal": signal("recheck-pass", RECHECK_HASH, "pass", CORRECTION_HASH),
        "status": status,
        "scope": deepcopy(SCOPE),
        "limitations": ["Only the named toy suite was checked."],
        "provenance": {
            "producer": "correction-trace-test",
            "recorded_from": ["signals/initial-fail.json", "signals/recheck-pass.json"],
            "reviewer": None,
        },
    }


class CorrectionTraceTests(unittest.TestCase):
    def test_verified_requires_pass_on_new_correction_hash(self):
        row = trace()
        self.assertEqual(module.validate_trace(row), [])

    def test_cannot_verify_without_recheck(self):
        row = trace()
        row["recheck_signal"] = None
        self.assertTrue(any("passing recheck_signal" in e for e in module.validate_trace(row)))

    def test_recheck_of_old_artifact_does_not_verify_correction(self):
        row = trace()
        row["recheck_signal"]["target_artifact_sha256"] = INITIAL_HASH
        self.assertTrue(any("correction_artifact hash" in e for e in module.validate_trace(row)))

    def test_same_artifact_hash_is_not_a_versioned_correction(self):
        row = trace()
        row["correction_artifact"]["sha256"] = INITIAL_HASH
        row["recheck_signal"]["target_artifact_sha256"] = INITIAL_HASH
        self.assertTrue(any("must differ" in e for e in module.validate_trace(row)))

    def test_inconclusive_signal_cannot_be_promoted_as_verified(self):
        row = trace()
        row["recheck_signal"]["outcome"] = "inconclusive"
        row["recheck_signal"]["evidence_refs"] = []
        self.assertTrue(any("passing recheck_signal" in e for e in module.validate_trace(row)))

    def test_inconclusive_status_accepts_unavailable_recheck(self):
        row = trace("inconclusive")
        row["recheck_signal"]["outcome"] = "unavailable"
        row["recheck_signal"]["evidence_refs"] = []
        self.assertEqual(module.validate_trace(row), [])

    def test_failed_correction_requires_failure_evidence(self):
        row = trace("failed_correction")
        row["recheck_signal"]["outcome"] = "fail"
        row["recheck_signal"]["evidence_refs"] = []
        self.assertTrue(any("requires evidence_refs" in e for e in module.validate_trace(row)))

    def test_initial_signal_must_bind_to_initial_artifact(self):
        row = trace()
        row["initial_signal"]["target_artifact_sha256"] = CORRECTION_HASH
        self.assertTrue(any("exact initial_artifact hash" in e for e in module.validate_trace(row)))

    def test_pending_trace_must_not_claim_recheck(self):
        row = trace("pending")
        self.assertTrue(any("must not contain a recheck_signal" in e for e in module.validate_trace(row)))

    def test_hypotheses_are_not_required_to_be_treated_as_facts(self):
        row = trace()
        row["attribution_hypotheses"][0]["confidence_label"] = "certain"
        self.assertTrue(any("confidence_label" in e for e in module.validate_trace(row)))


    def test_malformed_enum_values_return_validation_errors_not_crashes(self):
        row = trace()
        row["status"] = {"unexpected": "object"}
        errors = module.validate_trace(row)
        self.assertTrue(any("invalid status" in error for error in errors))

        row = trace()
        row["initial_signal"]["outcome"] = ["fail"]
        errors = module.validate_trace(row)
        self.assertTrue(any("initial_signal.outcome is invalid" in error for error in errors))

    def test_failed_correction_must_keep_same_declared_scope(self):
        row = trace("failed_correction")
        row["recheck_signal"]["outcome"] = "fail"
        row["recheck_signal"]["scope"] = {"suite": "different-suite"}
        errors = module.validate_trace(row)
        self.assertTrue(any("recheck scope must match" in error for error in errors))

    def test_inconclusive_recheck_without_correction_artifact_is_rejected(self):
        row = trace("inconclusive")
        row["correction_artifact"] = None
        row["recheck_signal"]["outcome"] = "unavailable"
        row["recheck_signal"]["evidence_refs"] = []
        errors = module.validate_trace(row)
        self.assertTrue(any("requires a correction_artifact" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
