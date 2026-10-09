import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plastic_module_validator", ROOT / "scripts" / "validate_plastic_module.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def manifest():
    return {
        "schema_version": "aimeng.plastic_module.v1",
        "module_id": "adapter-demo",
        "module_version": "0.1.0",
        "status": "candidate",
        "module_type": "adapter",
        "base_model_id": "model-test",
        "base_model_revision": "rev-001",
        "interface_signature": "layers.0-1:hidden=32",
        "input_contract": {"type": "token_ids"},
        "output_contract": {"type": "logits_delta"},
        "task_families": ["toy_reasoning"],
        "known_limits": ["Only tested on toy suite"],
        "resource_cost": {"artifact_bytes": 1024, "peak_ram_mib": None, "latency_ms": None},
        "training_data_fingerprint": "sha256:train",
        "eval_suite_fingerprint": "sha256:eval",
        "validation_status": "pending",
        "validated_scope": {},
        "artifact_hashes": {"weights.safetensors": "a" * 64},
    }


class PlasticModuleManifestTests(unittest.TestCase):
    def test_candidate_may_remain_pending(self):
        self.assertEqual(validator.validate_manifest(manifest()), [])

    def test_accepted_module_requires_passing_validation(self):
        row = manifest()
        row["status"] = "accepted"
        self.assertTrue(any("validation_status=pass" in e for e in validator.validate_manifest(row)))

    def test_accepted_module_requires_eval_and_rollback_refs(self):
        row = manifest()
        row.update(status="accepted", validation_status="pass", validated_scope={"suite": "v1"})
        errors = validator.validate_manifest(row)
        self.assertTrue(any("eval_report_ref" in e for e in errors))
        self.assertTrue(any("rollback_target" in e for e in errors))

    def test_accepted_module_passes_with_evidence_refs(self):
        row = manifest()
        row.update(
            status="accepted", validation_status="pass",
            validated_scope={"suite": "v1", "cases": 20},
            eval_report_ref="runs/eval-report.json",
            rollback_target="core-model@rev-001",
        )
        self.assertEqual(validator.validate_manifest(row), [])

    def test_bad_hash_is_rejected(self):
        row = manifest()
        row["artifact_hashes"]["weights.safetensors"] = "broken"
        self.assertTrue(any("SHA-256" in e for e in validator.validate_manifest(row)))


if __name__ == "__main__":
    unittest.main()
