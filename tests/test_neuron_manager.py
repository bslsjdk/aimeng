import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "neuron_manager.py"
SPEC = importlib.util.spec_from_file_location("neuron_manager", SCRIPT)
manager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manager
SPEC.loader.exec_module(manager)


class NeuronManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.registry = self.root / "registry.json"
        self.logs = self.root / "runs" / "manager.jsonl"
        self.a = self.artifacts / "n001.json"
        self.b = self.artifacts / "n003.json"
        self.a.write_text(json.dumps({
            "format": manager.ARTIFACT_FORMAT, "weights": [2.0], "bias": 1.0
        }), encoding="utf-8")
        self.b.write_text(json.dumps({
            "format": manager.ARTIFACT_FORMAT, "weights": [0.0], "bias": 0.0
        }), encoding="utf-8")
        neurons = []
        for unit_id, path in (("N-001", self.a), ("N-003", self.b)):
            neurons.append({
                "id": unit_id, "revision": 1, "artifact": str(path),
                "sha256": manager.sha256_file(path), "enabled": True, "quality": "unassessed"
            })
        self.registry.write_text(json.dumps({
            "schema_version": manager.REGISTRY_SCHEMA, "neurons": neurons
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, command, **kwargs):
        base = {"registry": str(self.registry), "root": str(self.root),
                "log": str(self.logs), "command": command}
        base.update(kwargs)
        return type("Args", (), base)()

    def test_import_registers_only_a_valid_real_artifact(self):
        fresh_registry = self.root / "new-registry.json"
        fresh_log = self.root / "new-runs" / "manager.jsonl"
        args = type("Args", (), {
            "registry": str(fresh_registry), "root": str(self.root), "log": str(fresh_log),
            "command": "import", "id": "N-010", "artifact": str(self.a), "revision": 1
        })()
        result = manager.run(args)
        self.assertTrue(result["ok"])
        data = json.loads(fresh_registry.read_text(encoding="utf-8"))
        self.assertEqual([n["id"] for n in data["neurons"]], ["N-010"])
        self.assertEqual(data["neurons"][0]["origin"], "user_import")
        self.assertEqual(data["neurons"][0]["sha256"], manager.sha256_file(self.a))

    def test_import_rejects_non_neuron_file(self):
        invalid = self.root / "not-a-neuron.json"
        invalid.write_text('{"hello":"world"}', encoding="utf-8")
        args = type("Args", (), {
            "registry": str(self.root / "registry-new.json"), "root": str(self.root),
            "log": str(self.logs), "command": "import", "id": "N-011",
            "artifact": str(invalid), "revision": 1
        })()
        result = manager.run(args)
        self.assertFalse(result["ok"])
        self.assertFalse((self.root / "registry-new.json").exists())

    def test_no_fake_neurons_when_registry_missing_and_failure_is_logged(self):
        self.registry.unlink()
        result = manager.run(self.args("list"))
        self.assertFalse(result["ok"])
        self.assertIn("不会自动生成示例神经元", result["errors"][0]["message"])
        row = json.loads(self.logs.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(row["level"], "ERROR")
        self.assertEqual(row["event"], "batch_list_failed")

    def test_unknown_selected_id_fails_without_partial_mutation(self):
        result = manager.run(self.args("disable", ids=["N-001", "N-999"]))
        self.assertFalse(result["ok"])
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertTrue(next(n for n in data["neurons"] if n["id"] == "N-001")["enabled"])

    def test_batch_enable_disable_uses_persistent_registry(self):
        off = manager.run(self.args("disable", ids=["N-001", "N-003"]))
        self.assertTrue(off["ok"])
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertFalse(any(n["enabled"] for n in data["neurons"]))
        on = manager.run(self.args("enable", ids=["N-003"]))
        self.assertTrue(on["ok"])
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertTrue(next(n for n in data["neurons"] if n["id"] == "N-003")["enabled"])

    def test_save_selected_copies_and_hashes_real_artifacts(self):
        output = self.root / "saved"
        result = manager.run(self.args("save", ids=["N-001", "N-003"], output=str(output)))
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["results"]), 2)
        for item in result["results"]:
            snapshot = Path(item["snapshot"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(manager.sha256_file(snapshot), item["sha256"])
            self.assertTrue((snapshot.parent / "manifest.json").is_file())

    def test_export_contains_only_selected_real_artifacts_and_manifest(self):
        output = self.root / "selected.zip"
        result = manager.run(self.args("export", ids=["N-001", "N-003"], output=str(output)))
        self.assertTrue(result["ok"])
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertIn("manifest.json", names)
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual({n["id"] for n in manifest["neurons"]}, {"N-001", "N-003"})
            self.assertEqual(len([n for n in names if n.endswith(".json") and n != "manifest.json"]), 2)

    def test_evaluation_uses_independent_dataset_and_persists_quality(self):
        dataset = self.root / "heldout.jsonl"
        dataset.write_text(
            '{"input":[-1.0],"target":-1.0}\n'
            '{"input":[0.0],"target":1.0}\n'
            '{"input":[1.0],"target":3.0}\n', encoding="utf-8")
        result = manager.run(self.args("evaluate", ids=["N-001", "N-003"],
                                       dataset=str(dataset), good_mse_threshold=0.05))
        self.assertTrue(result["ok"])
        scores = {row["id"]: row for row in result["results"]}
        self.assertEqual(scores["N-001"]["quality"], "excellent")
        self.assertEqual(scores["N-001"]["mse"], 0.0)
        self.assertEqual(scores["N-003"]["quality"], "candidate")
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual(next(n for n in data["neurons"] if n["id"] == "N-001")["quality"], "excellent")

    def test_changed_artifact_hash_is_rejected_and_logged(self):
        self.a.write_text('{"format":"aimeng.linear_neuron.v1","weights":[999],"bias":0}', encoding="utf-8")
        result = manager.run(self.args("disable", ids=["N-001"]))
        self.assertFalse(result["ok"])
        self.assertIn("SHA-256 不匹配", result["errors"][0]["message"])
        row = json.loads(self.logs.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(row["level"], "ERROR")

    def test_cli_prints_machine_readable_error_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            proc = subprocess.run([
                sys.executable, str(SCRIPT), "--registry", str(p / "missing.json"),
                "--root", str(p), "--log", str(p / "errors.jsonl"), "list"
            ], text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(json.loads(proc.stdout)["ok"])
            self.assertTrue((p / "errors.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
