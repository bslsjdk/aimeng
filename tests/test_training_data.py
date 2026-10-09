import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_training_data", ROOT / "scripts/validate_training_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainingDataTests(unittest.TestCase):
    def row(self, task_id="task-1", sample_id="sample-1", split="train"):
        return {
            "schema_version": "aimeng.trajectory.v1",
            "task_id": task_id,
            "sample_id": sample_id,
            "task_family": "reasoning",
            "input": {"prompt": "A small test prompt"},
            "target": {"answer": "candidate"},
            "outcome": {"status": "unknown", "verifier": None, "verifier_version": None, "evidence": []},
            "provenance": {"source_type": "human_authored", "privacy_review": "not_required"},
            "split": split,
        }

    def test_valid_unknown_sample_is_accepted(self):
        self.assertEqual(module.validate_row(self.row(), 1), [])

    def test_pass_requires_verifier_and_evidence(self):
        row = self.row()
        row["outcome"] = {"status": "pass", "verifier": None, "verifier_version": None, "evidence": []}
        errors = module.validate_row(row, 1)
        self.assertTrue(any("requires verifier" in e for e in errors))
        self.assertTrue(any("requires evidence" in e for e in errors))

    def test_task_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [self.row(split="train"), self.row(sample_id="sample-2", split="test")]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = module.audit(path)
        self.assertFalse(report["ok"])
        self.assertTrue(any("task-level split leakage" in e for e in report["errors"]))

    def test_duplicate_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(self.row()) + "\n" + json.dumps(self.row(task_id="task-2")) + "\n", encoding="utf-8")
            report = module.audit(path)
        self.assertFalse(report["ok"])
        self.assertTrue(any("duplicate sample_id" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
