import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "correction_feedback", ROOT / "scripts" / "build_correction_feedback.py"
)
feedback_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(feedback_module)


def signal(outcome="fail"):
    row = {
        "schema_version": "aimeng.learning_signal.v1",
        "signal_id": "signal-001",
        "task_id": "task-001",
        "target_artifact": {"ref": "runs/answer.txt", "sha256": "a" * 64, "kind": "answer"},
        "verifier": {"id": "toy-test", "version": "1.0", "family": "unit_test"},
        "outcome": outcome,
        "error_category": "assertion_mismatch" if outcome == "fail" else None,
        "expected": "42",
        "observed": "41",
        "severity": "medium" if outcome == "fail" else "info",
        "observed_at": "2026-10-09T12:00:00Z",
        "evidence_refs": ["runs/test-report.json"] if outcome == "fail" else [],
        "scope": {"suite": "toy-v1", "cases": 1},
        "limitations": ["Only one toy case was run."],
    }
    return row


class CorrectionFeedbackTests(unittest.TestCase):
    def test_failure_requests_revision_and_reverification(self):
        result = feedback_module.build_feedback(signal("fail"))
        self.assertEqual(result["action"], "revise_then_reverify")
        self.assertFalse(result["weight_update_authorized"])
        self.assertFalse(result["memory_promotion_authorized"])

    def test_inconclusive_does_not_label_answer_wrong(self):
        result = feedback_module.build_feedback(signal("inconclusive"))
        self.assertEqual(result["action"], "gather_more_evidence")
        self.assertIn("could not establish pass or fail", result["instruction_for_core_model"])

    def test_pass_is_scoped_not_universal(self):
        result = feedback_module.build_feedback(signal("pass"))
        self.assertEqual(result["action"], "record_scoped_success")
        self.assertIn("universal correctness", result["instruction_for_core_model"])

    def test_failure_without_evidence_is_rejected(self):
        row = signal("fail")
        row["evidence_refs"] = []
        with self.assertRaises(ValueError):
            feedback_module.build_feedback(row)

    def test_bad_hash_is_rejected(self):
        row = signal("fail")
        row["target_artifact"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            feedback_module.build_feedback(row)


if __name__ == "__main__":
    unittest.main()
