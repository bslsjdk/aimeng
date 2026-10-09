import json
import tempfile
import unittest
from pathlib import Path
from scripts.evaluate_student_sft import verify, load_suite

class StudentEvalTests(unittest.TestCase):
    def test_exact_match_and_contains(self):
        self.assertTrue(verify({"verifier":"exact_match","expected":"42"}, "42\n"))
        self.assertFalse(verify({"verifier":"exact_match","expected":"42"}, "43"))
        self.assertTrue(verify({"verifier":"contains","expected":"ValueError"}, "catch ValueError here"))

    def test_json_object_rejects_array(self):
        self.assertTrue(verify({"verifier":"json_object"}, '{"ok":true}'))
        self.assertFalse(verify({"verifier":"json_object"}, '[1,2]'))

    def test_suite_schema_and_duplicate_task_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"suite.jsonl"
            row={"schema_version":"aimeng.eval_task.v1","task_id":"t1","prompt":"Q","expected":"A","verifier":"exact_match"}
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"duplicate task_id"):
                load_suite(path)

if __name__ == "__main__":
    unittest.main()
