import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_module("collector", "scripts/collect_budget_traces.py")
trainer = load_module("trainer", "scripts/train_budget_controller.py")
validator = load_module("validator", "scripts/validate_telemetry.py")


class CollectorTests(unittest.TestCase):
    def test_exact_match_verifier(self):
        task = {"verifier": "exact_match", "expected": "Mercury"}
        self.assertEqual(collector.verify_output(task, "Mercury\n")[:2], ("pass", 1.0))
        self.assertEqual(collector.verify_output(task, "Venus")[:2], ("fail", 0.0))

    def test_json_object_verifier_rejects_array(self):
        self.assertEqual(collector.verify_output({"verifier": "json_object"}, '{"ok":true}')[:2], ("pass", 1.0))
        self.assertEqual(collector.verify_output({"verifier": "json_object"}, '[1,2]')[:2], ("fail", 0.0))

    def test_unknown_verifier_stays_unknown(self):
        self.assertEqual(collector.verify_output({"verifier": "llm_judge"}, "probably correct")[:2], ("unknown", None))


class TrainingTargetTests(unittest.TestCase):
    def make_record(self, budget, quality, split="train"):
        return {
            "task_id": "task-1",
            "pair_group_id": "pair-1",
            "task": {"dataset_split": split, "task_family": "closed_fact",
                     "expected_output_type": "text", "input_tokens": 10},
            "environment": {"device_class": "colab_gpu", "accelerator": "CPU"},
            "budget": {"budget_id": budget + "-v1"},
            "result": {"status": "completed", "quality_label": quality},
        }

    def test_choose_lowest_passing_budget(self):
        rows = [
            self.make_record("small", "fail"),
            self.make_record("medium", "pass"),
            self.make_record("full", "pass"),
        ]
        examples, errors = trainer.group_records(rows)
        self.assertEqual(errors, [])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["target"], "medium")

    def test_unknown_does_not_become_training_target(self):
        rows = [
            self.make_record("small", "unknown"),
            self.make_record("medium", "unknown"),
            self.make_record("full", "unknown"),
        ]
        examples, errors = trainer.group_records(rows)
        self.assertEqual(errors, [])
        self.assertEqual(examples, [])


class TelemetryValidatorTests(unittest.TestCase):
    def base_record(self):
        return {
            "schema_version": "aimeng.telemetry.v1",
            "run_id": "run-1", "task_id": "task-1",
            "timestamp_utc": "2026-10-09T12:00:00Z",
            "model": {"name": "model.gguf", "backend": "llama.cpp"},
            "environment": {"device_class": "colab_gpu"},
            "task": {"task_family": "closed_fact", "dataset_split": "train"},
            "budget": {"budget_id": "small-v1", "execution_mode": "normal"},
            "result": {"status": "completed", "quality_label": "pass"},
            "performance": {"total_latency_ms": 100},
            "memory": {"measurement_method": "test"},
            "routing": {"controller_version": "test-v1", "verified_compute_skipped": False},
        }

    def test_valid_minimal_record(self):
        self.assertEqual(validator.validate_record(self.base_record(), 1), [])

    def test_invalid_quality_enum_rejected(self):
        row = self.base_record()
        row["result"]["quality_label"] = "probably"
        self.assertTrue(any("quality_label" in error for error in validator.validate_record(row, 1)))

    def test_simulation_cannot_claim_verified_skip(self):
        row = self.base_record()
        row["budget"]["execution_mode"] = "simulation"
        row["routing"]["verified_compute_skipped"] = True
        self.assertTrue(any("simulation cannot claim" in error for error in validator.validate_record(row, 1)))


if __name__ == "__main__":
    unittest.main()
