import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "idea_cycle_state", ROOT / "scripts" / "idea_cycle_state.py"
)
state_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state_module)


class IdeaCycleStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "task.json"
        self.task = state_module.IdeaCycleState.create(
            self.path, "task-001", {"max_ideas": 3}
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_transition_is_persisted_and_reloadable(self):
        self.task.transition("CONTEXT_READY", "context_retrieved")
        loaded = state_module.IdeaCycleState.load(self.path)
        self.assertEqual(loaded.state, "CONTEXT_READY")
        self.assertEqual(loaded.document["events"][-1]["reason"], "context_retrieved")

    def test_illegal_transition_is_rejected(self):
        with self.assertRaises(state_module.StateError):
            self.task.transition("DONE", "skip_everything")

    def test_existing_state_is_never_overwritten(self):
        with self.assertRaises(state_module.StateError):
            state_module.IdeaCycleState.create(self.path, "other-task")

    def test_recoverable_failure_resumes_previous_state(self):
        self.task.transition("CONTEXT_READY", "context_retrieved")
        self.task.fail_recoverably("temporary_io_error")
        loaded = state_module.IdeaCycleState.load(self.path)
        self.assertEqual(loaded.state, "FAILED_RECOVERABLE")
        self.assertEqual(loaded.resume(), "CONTEXT_READY")
        self.assertEqual(state_module.IdeaCycleState.load(self.path).state, "CONTEXT_READY")

    def test_budget_counter_cannot_exceed_maximum(self):
        self.assertEqual(self.task.increment("ideas_created", 2, maximum=3), 2)
        with self.assertRaises(state_module.StateError):
            self.task.increment("ideas_created", 2, maximum=3)

    def test_artifact_hash_is_checked(self):
        with self.assertRaises(state_module.StateError):
            self.task.record_artifact("runs/report.json", "not-a-hash", "verification_report")
        self.task.record_artifact("runs/report.json", "a" * 64, "verification_report")
        self.assertEqual(len(self.task.document["artifacts"]), 1)

    def test_terminal_state_cannot_be_reopened(self):
        self.task.transition("CONTEXT_READY", "context_retrieved")
        self.task.transition("EXPLORE", "selected_exploration")
        self.task.abort_for_budget("budget_exhausted")
        with self.assertRaises(state_module.StateError):
            self.task.transition("CONTEXT_READY", "try_again")


if __name__ == "__main__":
    unittest.main()
