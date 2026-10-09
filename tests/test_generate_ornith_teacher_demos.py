import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_ornith_teacher_demos import read_tasks, strip_reasoning


class TeacherDemoGeneratorTests(unittest.TestCase):
    def test_strip_reasoning_keeps_final_answer(self):
        self.assertEqual(
            strip_reasoning("<think>private reasoning</think>\nFinal: 42"),
            "Final: 42",
        )

    def test_strip_analysis_and_final_tags(self):
        self.assertEqual(
            strip_reasoning("<analysis>hidden</analysis><final>Done.</final>"),
            "Done.",
        )

    def test_read_tasks_accepts_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps({
                "task_id": "one", "topic": "test.topic", "prompt": "Answer",
            }) + "\n", encoding="utf-8")
            rows = read_tasks(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["difficulty"], "intermediate")

    def test_read_tasks_rejects_duplicate_task_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            row = {"task_id": "one", "topic": "test.topic", "prompt": "Answer"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate task_id"):
                read_tasks(path)


if __name__ == "__main__":
    unittest.main()
