import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_artifact_integrity", ROOT / "scripts" / "verify_artifact_integrity.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ArtifactIntegrityVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifact = self.root / "artifact.bin"
        self.artifact.write_bytes(b"aimeng-test-artifact\x00\x01")
        self.digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_matching_hash_emits_scoped_pass_signal(self):
        evidence, signal = module.verify_file(
            str(self.artifact), self.digest, "task-1", "signal-1", "reports/evidence.json"
        )
        self.assertEqual(evidence["outcome"], "pass")
        self.assertEqual(signal["outcome"], "pass")
        self.assertEqual(signal["target_artifact"]["sha256"], self.digest)
        self.assertEqual(signal["evidence_refs"], ["reports/evidence.json"])
        self.assertEqual(signal["verifier"]["id"], "aimeng-file-sha256")
        self.assertIn("not a semantic correctness", signal["limitations"][0])

    def test_mismatching_hash_emits_evidence_backed_fail_signal(self):
        wrong_hash = "0" * 64 if self.digest != "0" * 64 else "1" * 64
        evidence, signal = module.verify_file(
            str(self.artifact), wrong_hash, "task-2", "signal-2", "reports/failure.json"
        )
        self.assertEqual(evidence["outcome"], "fail")
        self.assertEqual(signal["outcome"], "fail")
        self.assertEqual(signal["error_category"], "artifact_hash_mismatch")
        self.assertTrue(signal["evidence_refs"])

    def test_malformed_expected_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            module.verify_file(str(self.artifact), "not-a-hash", "task-3", "signal-3", "e.json")

    def test_missing_artifact_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            module.verify_file(str(self.root / "missing.bin"), self.digest, "task-4", "signal-4", "e.json")


if __name__ == "__main__":
    unittest.main()
