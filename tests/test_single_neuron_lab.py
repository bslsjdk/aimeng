import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "single_neuron_lab", ROOT / "scripts" / "single_neuron_lab.py"
)
lab_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab_module)


class SingleNeuronLabTests(unittest.TestCase):
    def test_training_improves_held_out_error(self):
        result = lab_module.run_demo(epochs=60, learning_rate=0.05)
        self.assertTrue(result["held_out_improved"])
        self.assertLess(result["held_out_mse_after"], 0.01)
        self.assertEqual(result["updates"], 300)

    def test_stable_identity_and_revision_required_for_replacement(self):
        base = lab_module.LinearNeuron("unit/a", 1, [0.0], 0.0)
        lab = lab_module.NeuronLab(base)
        validation = [([-1.0], -1.0), ([0.0], 1.0), ([1.0], 3.0)]
        candidate = lab_module.LinearNeuron("unit/a", 2, [2.0], 1.0)
        result = lab.replace(candidate, validation)
        self.assertTrue(result["accepted"])
        self.assertEqual(lab.neuron.unit_id, "unit/a")
        self.assertEqual(lab.neuron.revision, 2)

    def test_bad_candidate_is_rejected_and_baseline_kept(self):
        base = lab_module.LinearNeuron("unit/a", 1, [2.0], 1.0)
        lab = lab_module.NeuronLab(base)
        validation = [([-1.0], -1.0), ([0.0], 1.0), ([1.0], 3.0)]
        candidate = lab_module.LinearNeuron("unit/a", 2, [0.0], 0.0)
        result = lab.replace(candidate, validation)
        self.assertFalse(result["accepted"])
        self.assertEqual(lab.neuron.revision, 1)

    def test_disable_blocks_forward_execution(self):
        neuron = lab_module.LinearNeuron("unit/a", 1, [1.0], 0.0)
        lab = lab_module.NeuronLab(neuron)
        lab.set_enabled(False)
        with self.assertRaises(RuntimeError):
            lab.predict([2.0])
        self.assertEqual(lab.events[-1]["event_type"], "deactivated")

    def test_trace_is_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            lab = lab_module.NeuronLab(
                lab_module.LinearNeuron("unit/a", 1, [0.0], 0.0), trace_path=path
            )
            lab.predict([1.0])
            lab.train_one([1.0], 2.0, 0.1)
            rows = [__import__("json").loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["event_type"], "activated")
            self.assertEqual(rows[1]["event_type"], "parameter_update")
            self.assertNotEqual(rows[0]["event_id"], rows[1]["event_id"])


if __name__ == "__main__":
    unittest.main()
