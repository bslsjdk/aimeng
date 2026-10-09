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
evaluator = load_module("evaluator", "scripts/evaluate_budget_controller.py")
adaptation = load_module("adaptation", "scripts/online_adaptation.py")


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

    def test_repeated_budget_runs_require_reliable_pass_rate(self):
        rows = [
            self.make_record("small", "pass"),
            self.make_record("small", "fail"),
            self.make_record("medium", "pass"),
            self.make_record("full", "pass"),
        ]
        examples, errors = trainer.group_records(rows)
        self.assertEqual(errors, [])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["target"], "medium")



class OnlineAdaptationTests(unittest.TestCase):
    def test_verified_failure_escalates_only_with_memory_headroom(self):
        decision = adaptation.decide_next_action({
            "budget": "small", "quality_label": "fail", "status": "completed",
            "whole_app_pss_mib": 2500,
        })
        self.assertEqual(decision["action"], "retry_with_higher_budget")
        self.assertEqual(decision["next_budget"], "medium")
        self.assertFalse(decision["persistent_update"])

    def test_unknown_feedback_never_becomes_retry_label(self):
        decision = adaptation.decide_next_action({
            "budget": "small", "quality_label": "unknown", "status": "completed",
            "whole_app_pss_mib": 2000,
        })
        self.assertEqual(decision["action"], "stop_for_reliable_feedback")
        self.assertIsNone(decision["next_budget"])

    def test_missing_memory_measurement_blocks_escalation(self):
        decision = adaptation.decide_next_action({
            "budget": "small", "quality_label": "fail", "status": "completed",
            "whole_app_pss_mib": None,
        })
        self.assertEqual(decision["action"], "stop_for_memory_measurement")

    def test_memory_headroom_gate_stops_before_hard_limit(self):
        decision = adaptation.decide_next_action({
            "budget": "small", "quality_label": "fail", "status": "completed",
            "whole_app_pss_mib": 3600,
        })
        self.assertEqual(decision["action"], "stop_for_memory_headroom")

    def test_hard_memory_limit_is_never_overridden_by_quality(self):
        decision = adaptation.decide_next_action({
            "budget": "small", "quality_label": "pass", "status": "completed",
            "whole_app_pss_mib": 4096,
        })
        self.assertEqual(decision["action"], "stop_and_flag_memory_violation")

    def test_full_budget_failure_stops(self):
        decision = adaptation.decide_next_action({
            "budget": "full", "quality_label": "fail", "status": "completed",
            "whole_app_pss_mib": 2500,
        })
        self.assertEqual(decision["action"], "stop_quality_failure")
        self.assertIsNone(decision["next_budget"])

    def test_timeout_does_not_escalate_budget(self):
        decision = adaptation.decide_next_action({
            "budget": "small", "quality_label": "unknown", "status": "timeout",
            "whole_app_pss_mib": 2000,
        })
        self.assertEqual(decision["action"], "stop_and_record_failure")
        self.assertIsNone(decision["next_budget"])


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

    def test_evaluator_counts_unverified_runs_as_non_passes(self):
        rows = [
            {"result": {"status": "completed", "quality_label": "pass"},
             "performance": {"total_latency_ms": 100},
             "memory": {"process_peak_pss_mib": 120}},
            {"result": {"status": "timeout", "quality_label": "unknown"},
             "performance": {"total_latency_ms": 500},
             "memory": {"process_peak_pss_mib": None}},
        ]
        report = evaluator.summarize_runs(rows)
        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["quality_pass_rate"], 0.5)
        self.assertEqual(report["mean_latency_ms"], 300.0)
        self.assertEqual(report["pss_measurements"], 1)


if __name__ == "__main__":
    unittest.main()
