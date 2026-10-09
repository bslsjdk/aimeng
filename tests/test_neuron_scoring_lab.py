import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "neuron_scoring_lab.py"
SPEC = importlib.util.spec_from_file_location("neuron_scoring_lab", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


class NeuronScoringLabTests(unittest.TestCase):
    def test_learning_reduces_held_out_error(self):
        result = lab.run_experiment(epochs=80)
        self.assertTrue(result["validation_mse_improved"])
        self.assertLess(result["final_validation_mse"], result["initial_validation_mse"])

    def test_score_is_based_on_ablation_and_cost(self):
        units = [
            lab.Unit("useful", 2.0, 1.0),
            lab.Unit("wrong", -2.0, 5.0),
        ]
        validation = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0)]
        records = lab.score_units(units, validation, compute_cost=0.0)
        self.assertEqual({r["unit_id"] for r in records}, {"useful", "wrong"})
        self.assertTrue(all("marginal_contribution" in r for r in records))

    def test_hysteresis_can_disable_and_reenable(self):
        unit = lab.Unit("u", 0.0, 0.0)
        lab.apply_score_policy([unit], [{"unit_id": "u", "score": -1.0}])
        self.assertFalse(unit.enabled)
        lab.apply_score_policy([unit], [{"unit_id": "u", "score": 1.0}])
        self.assertTrue(unit.enabled)

    def test_controller_never_sleeps_every_unit(self):
        units = [
            lab.Unit("a", 2.0, 1.0),
            lab.Unit("b", 2.0, 1.0),
        ]
        records = [
            {"unit_id": "a", "score": -1.0},
            {"unit_id": "b", "score": -1.0},
        ]
        changes = lab.apply_score_policy(units, records)
        self.assertTrue(any(unit.enabled for unit in units))
        self.assertTrue(any(change.get("reason") == "minimum_active_unit_safety_floor" for change in changes))

    def test_sleeping_unit_is_scored_as_candidate_admission(self):
        units = [lab.Unit("active", 2.0, 1.0), lab.Unit("sleeping", 2.0, 1.0, enabled=False)]
        validation = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0)]
        records = lab.score_units(units, validation, compute_cost=0.0)
        sleeping = next(item for item in records if item["unit_id"] == "sleeping")
        self.assertEqual(sleeping["comparison"], "admit_sleeping_unit")
        self.assertAlmostEqual(sleeping["marginal_contribution"], 0.0)

    def test_empty_evaluation_is_rejected(self):
        with self.assertRaises(ValueError):
            lab.evaluate([lab.Unit("u", 1.0, 0.0)], [])

    def test_trace_records_revision_and_state_changes(self):
        result = lab.run_experiment(epochs=3)
        cycles = [item for item in result["trace"] if item["event"] == "score_cycle"]
        self.assertEqual(len(cycles), 3)
        self.assertTrue(all("scores" in item and "state_changes" in item for item in cycles))
        updates = [item for item in result["trace"] if item["event"] == "parameter_update"]
        self.assertTrue(all(item["revision"] == 1 for item in updates))


if __name__ == "__main__":
    unittest.main()
