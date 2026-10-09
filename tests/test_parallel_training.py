import importlib.util
import json
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parallel_training", ROOT / "scripts/parallel_training.py")
parallel = importlib.util.module_from_spec(SPEC)
# dataclasses resolves the class's module through sys.modules during decoration.
# Register before exec_module so the test works in a clean Python process.
sys.modules[SPEC.name] = parallel
SPEC.loader.exec_module(parallel)

class PlanTests(unittest.TestCase):
    def test_valid_plan_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"memory_budget_mb":1000,"max_workers":2,"jobs":[
                {"id":"a","argv":["python","-c","print(1)"],"memory_mb":400,"output_dir":str(root/"a")},
                {"id":"b","argv":["python","-c","print(2)"],"memory_mb":500,"output_dir":str(root/"b")}]}), encoding="utf-8")
            jobs, budget, workers = parallel.load_plan(plan)
            self.assertEqual([job.job_id for job in jobs], ["a","b"])
            self.assertEqual((budget, workers), (1000,2))

    def test_rejects_budget_overrun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = root / "plan.json"
            plan.write_text(json.dumps({"memory_budget_mb":100,"jobs":[
                {"id":"a","argv":["python","-c","pass"],"memory_mb":101,"output_dir":str(root/"a")}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeds total budget"): parallel.load_plan(plan)

    def test_rejects_shared_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = str(root/"same"); plan = root / "plan.json"
            plan.write_text(json.dumps({"memory_budget_mb":1000,"jobs":[
                {"id":"a","argv":["python","-c","pass"],"memory_mb":100,"output_dir":output},
                {"id":"b","argv":["python","-c","pass"],"memory_mb":100,"output_dir":output}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not share output_dir"): parallel.load_plan(plan)

    def test_dry_run_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            job = parallel.Job("a", ("python","-c","raise SystemExit(99)"), 10, Path(directory)/"unused")
            report = parallel.run_plan([job],10,1,dry_run=True)
            self.assertEqual(report["status"],"dry_run")
            self.assertFalse((Path(directory)/"unused").exists())

    def test_runs_two_independent_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [parallel.Job("a",("python","-c","print('a')"),10,root/"a"),
                    parallel.Job("b",("python","-c","print('b')"),10,root/"b")]
            report = parallel.run_plan(jobs,20,2)
            self.assertEqual(report["status"],"passed")
            self.assertEqual({item["status"] for item in report["results"]},{"passed"})
            self.assertIn("a",(root/"a"/"training.log").read_text(encoding="utf-8"))
            self.assertIn("b",(root/"b"/"training.log").read_text(encoding="utf-8"))

    def test_parallel_toy_training_saves_independent_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = str(ROOT / "scripts" / "toy_train.py")
            jobs = [
                parallel.Job("reasoning", (sys.executable, trainer, "--task", "reasoning",
                    "--steps", "80", "--seed", "17", "--output", str(root / "model_reasoning")),
                    128, root / "logs_reasoning"),
                parallel.Job("tool_use", (sys.executable, trainer, "--task", "tool_use",
                    "--steps", "80", "--seed", "29", "--output", str(root / "model_tool_use")),
                    128, root / "logs_tool_use"),
            ]
            report = parallel.run_plan(jobs, 256, 2)
            self.assertEqual(report["status"], "passed")
            first = json.loads((root / "model_reasoning" / "checkpoint.json").read_text(encoding="utf-8"))
            second = json.loads((root / "model_tool_use" / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(first["task"], "reasoning")
            self.assertEqual(second["task"], "tool_use")
            self.assertLess(first["final_validation_loss"], first["initial_validation_loss"])
            self.assertLess(second["final_validation_loss"], second["initial_validation_loss"])
            self.assertNotEqual(first["weights"], second["weights"])

if __name__ == "__main__": unittest.main()
