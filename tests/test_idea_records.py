import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "idea_validator", ROOT / "scripts" / "validate_idea_records.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def valid_idea():
    return {
        "schema_version": "aimeng.idea.v1",
        "idea_id": "idea-001",
        "task_id": "task-001",
        "statement": "A bounded cache may reduce repeated retrieval cost.",
        "novelty_axis": "resource trade-off",
        "assumptions": ["Repeated queries share useful context."],
        "mechanism": "Reuse a verified result while its version remains valid.",
        "predictions": ["Repeated identical queries use fewer retrieval calls."],
        "counterexamples_to_seek": ["A changed source invalidates the cached answer."],
        "evidence_refs": [],
        "risk_and_cost": {"risk": "stale result", "reversible": True},
        "epistemic_status": "speculative",
        "verification_plan": {"method": "Compare cached and uncached runs", "acceptance_criteria": ["Same answer", "Fewer retrieval calls"]},
        "verification_result": None,
        "result_refs": [],
        "provenance": {"model_version": "test-model", "policy_version": "idea-policy-v1"},
    }


class IdeaRecordTests(unittest.TestCase):
    def test_speculative_idea_can_exist_without_proof(self):
        self.assertEqual(validator.validate_row(valid_idea(), 1), [])

    def test_verified_status_requires_independent_verifier_and_evidence(self):
        row = valid_idea()
        row["epistemic_status"] = "verified_for_scope"
        self.assertTrue(any("requires verification_result" in e for e in validator.validate_row(row, 1)))

    def test_verified_status_rejects_empty_evidence(self):
        row = valid_idea()
        row["epistemic_status"] = "verified_for_scope"
        row["verification_result"] = {
            "status": "pass", "verifier_id": "unit-test", "verifier_version": "1",
            "scope": {"suite": "tiny"}, "evidence_refs": []
        }
        self.assertTrue(any("non-empty evidence_refs" in e for e in validator.validate_row(row, 1)))

    def test_verified_status_accepts_scoped_evidence(self):
        row = valid_idea()
        row["epistemic_status"] = "verified_for_scope"
        row["verification_result"] = {
            "status": "pass", "verifier_id": "unit-test", "verifier_version": "1",
            "scope": {"suite": "tiny", "cases": 3}, "evidence_refs": ["runs/test-report.json"]
        }
        self.assertEqual(validator.validate_row(row, 1), [])

    def test_audit_detects_duplicate_ids_and_invalid_json(self):
        row = valid_idea()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ideas.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n{bad json\n", encoding="utf-8")
            report = validator.audit(path)
        self.assertFalse(report["ok"])
        self.assertTrue(any("duplicate idea_id" in e for e in report["errors"]))
        self.assertTrue(any("invalid JSON" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
