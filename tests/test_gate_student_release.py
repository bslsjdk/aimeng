import unittest
from scripts.gate_student_release import evaluate

class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "suite_hash": "frozen-suite-sha",
            "heldout_success_rate": 0.60,
            "regression_success_rate": 0.90,
            "model_version": "baseline-v1",
        }
        self.candidate = {
            "suite_hash": "frozen-suite-sha",
            "heldout_success_rate": 0.64,
            "regression_success_rate": 0.895,
            "memory_measurement_scope": "android_app",
            "memory_measured": True,
            "peak_memory_mib": 3500,
            "model_version": "candidate-v2",
        }

    def test_promotes_only_when_all_gates_pass(self):
        result = evaluate(self.baseline, self.candidate, 0.01, 0.01, 3800)
        self.assertTrue(result["promote"])
        self.assertEqual(result["metrics"]["candidate_android_peak_mib"], 3500)

    def test_missing_android_measurement_blocks_promotion(self):
        candidate = dict(self.candidate, memory_measured=False)
        result = evaluate(self.baseline, candidate, 0.01, 0.01, 3800)
        self.assertFalse(result["promote"])
        self.assertTrue(any("whole-Android-app" in reason for reason in result["reasons"]))

    def test_memory_limit_blocks_promotion(self):
        candidate = dict(self.candidate, peak_memory_mib=3900)
        result = evaluate(self.baseline, candidate, 0.01, 0.01, 3800)
        self.assertFalse(result["promote"])

    def test_suite_mismatch_blocks_promotion(self):
        candidate = dict(self.candidate, suite_hash="different-suite")
        result = evaluate(self.baseline, candidate, 0.01, 0.01, 3800)
        self.assertFalse(result["promote"])

    def test_regression_blocks_promotion(self):
        candidate = dict(self.candidate, regression_success_rate=0.80)
        result = evaluate(self.baseline, candidate, 0.01, 0.01, 3800)
        self.assertFalse(result["promote"])

if __name__ == "__main__":
    unittest.main()
