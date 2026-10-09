import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workspace_mod = load_module("bounded_workspace", "scripts/bounded_workspace.py")
knowledge_mod = load_module("prepare_knowledge_corpus", "scripts/prepare_knowledge_corpus.py")


class BoundedWorkspaceTests(unittest.TestCase):
    def test_fixed_slots_and_lru_reuse(self):
        pool = workspace_mod.BoundedWorkspace(max_slots=2, max_bytes=100)
        pool.admit("a", "a.gguf", 40)
        pool.admit("b", "b.gguf", 40)
        pool.touch("a")
        result = pool.admit("c", "c.gguf", 40)
        self.assertEqual(result["evicted"], "b")
        self.assertEqual(set(pool.active_resources), {"a", "c"})
        self.assertEqual(len(pool.slots), 2)
        self.assertLessEqual(pool.used_bytes, 100)

    def test_rejects_resource_larger_than_budget(self):
        pool = workspace_mod.BoundedWorkspace(max_slots=2, max_bytes=100)
        with self.assertRaises(workspace_mod.BudgetExceeded):
            pool.admit("large", "large.bin", 101)

    def test_pinned_resources_are_not_evicted(self):
        pool = workspace_mod.BoundedWorkspace(max_slots=1, max_bytes=100)
        pool.admit("pinned", "p.bin", 50, pinned=True)
        with self.assertRaises(workspace_mod.NoEvictableSlot):
            pool.admit("next", "n.bin", 50)

    def test_snapshot_does_not_claim_ram_is_verified(self):
        pool = workspace_mod.BoundedWorkspace(max_slots=1, max_bytes=100)
        self.assertFalse(pool.snapshot()["ram_bound_verified"])

    def test_duplicate_id_with_conflicting_metadata_is_rejected(self):
        pool = workspace_mod.BoundedWorkspace(max_slots=1, max_bytes=100)
        pool.admit("same", "one.bin", 50)
        with self.assertRaises(ValueError):
            pool.admit("same", "two.bin", 50)


class KnowledgeCorpusTests(unittest.TestCase):
    def record(self):
        return {
            "schema_version": "aimeng.knowledge.v1",
            "topic": "python.exceptions",
            "content": "Use try/except to handle expected exceptions.",
            "teacher": {"model": "teacher-9b", "prompt_version": "extract-v1"},
            "provenance": [],
            "verification": {"status": "pending", "method": "manual", "verifier_version": None},
            "split": "unassigned",
        }

    def test_pending_record_is_structurally_valid(self):
        self.assertEqual(knowledge_mod.validate_record(self.record(), 1), [])

    def test_verified_requires_provenance_and_verifier_version(self):
        row = self.record()
        row["verification"] = {"status": "verified", "method": "manual", "verifier_version": "v1"}
        errors = knowledge_mod.validate_record(row, 1)
        self.assertTrue(any("provenance" in error for error in errors))

    def test_content_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            row1, row2 = self.record(), self.record()
            row2["content"] = "  USE TRY/EXCEPT to handle expected exceptions. "
            source.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
            rows, errors = knowledge_mod.load_and_deduplicate(source)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
