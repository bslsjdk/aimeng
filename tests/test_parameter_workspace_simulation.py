import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "simulate_parameter_workspace", ROOT / "scripts" / "simulate_parameter_workspace.py"
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def plan():
    return {
        "budget_bytes": 100,
        "units": [
            {"unit_id": "base", "size_bytes": 40, "dependencies": []},
            {"unit_id": "math", "size_bytes": 30, "dependencies": ["base"]},
            {"unit_id": "vision", "size_bytes": 50, "dependencies": []},
        ],
        "actions": [],
    }


class ParameterWorkspaceSimulationTests(unittest.TestCase):
    def test_dependency_closed_load_stays_within_budget(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        workspace.load("math")
        self.assertEqual(workspace.resident_bytes, 70)
        self.assertEqual(workspace.units["base"].state, "resident")
        self.assertEqual(workspace.units["math"].state, "resident")

    def test_lru_evicts_unused_unit_to_make_room(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        workspace.load("math")
        workspace.load("vision")
        self.assertEqual(workspace.resident_bytes, 90)
        self.assertEqual(workspace.units["math"].state, "unloaded")
        self.assertEqual(workspace.units["base"].state, "resident")
        self.assertEqual(workspace.units["vision"].state, "resident")

    def test_load_failure_rolls_back_new_units(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        with self.assertRaises(module.WorkspaceError):
            workspace.load("math", fail_at="math")
        self.assertEqual(workspace.units["base"].state, "unloaded")
        self.assertEqual(workspace.units["math"].state, "unloaded")
        self.assertEqual(workspace.resident_bytes, 0)

    def test_in_use_units_are_not_evicted(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        workspace.load("math")
        workspace.begin_use("math")
        with self.assertRaises(module.WorkspaceError):
            workspace.load("vision")
        self.assertEqual(workspace.units["math"].state, "in_use")

    def test_use_requires_dependency_closure_resident(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        with self.assertRaises(module.WorkspaceError):
            workspace.begin_use("math")

    def test_dependency_cycle_is_rejected(self):
        value = {
            "budget_bytes": 100,
            "units": [
                {"unit_id": "a", "size_bytes": 10, "dependencies": ["b"]},
                {"unit_id": "b", "size_bytes": 10, "dependencies": ["a"]},
            ],
        }
        with self.assertRaises(module.WorkspaceError):
            module.BoundedWorkspace.from_plan(value)

    def test_duplicate_unit_is_rejected(self):
        value = {
            "budget_bytes": 100,
            "units": [
                {"unit_id": "a", "size_bytes": 10},
                {"unit_id": "a", "size_bytes": 20},
            ],
        }
        with self.assertRaises(module.WorkspaceError):
            module.BoundedWorkspace.from_plan(value)

    def test_unload_during_use_is_deferred(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        workspace.load("base")
        workspace.begin_use("base")
        workspace.unload("base")
        self.assertEqual(workspace.units["base"].state, "evict_pending")
        workspace.end_use("base")
        self.assertEqual(workspace.units["base"].state, "unloaded")
        self.assertEqual(workspace.resident_bytes, 0)

    def test_report_labels_simulation_and_limitations(self):
        workspace = module.BoundedWorkspace.from_plan(plan())
        workspace.load("base")
        report = workspace.run([])
        self.assertTrue(report["simulation_only"])
        self.assertEqual(report["peak_resident_bytes"], 40)
        self.assertTrue(any("No actual OS/device memory" in row for row in report["limitations"]))


if __name__ == "__main__":
    unittest.main()
