import json, tempfile, unittest
from pathlib import Path
from scripts.train_student_sft import load_records

class GateTests(unittest.TestCase):
    def write(self, rows):
        f=tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",delete=False)
        for row in rows: f.write(json.dumps(row)+"\n")
        f.close(); self.addCleanup(Path(f.name).unlink, missing_ok=True); return Path(f.name)
    def row(self, sid, status="verified", eligible=True, split="train"):
        return {"sample_id":sid,"verification":{"status":status},"training_eligible":eligible,"split":split,"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}
    def test_pending_is_not_loaded(self):
        p=self.write([self.row("a",status="pending",eligible=False)])
        self.assertEqual(load_records(p,"train"),[])
    def test_only_requested_split_loaded(self):
        p=self.write([self.row("a",split="train"),self.row("b",split="validation")])
        self.assertEqual([x["sample_id"] for x in load_records(p,"train")],["a"])
        self.assertEqual([x["sample_id"] for x in load_records(p,"validation")],["b"])
    def test_duplicate_id_rejected(self):
        p=self.write([self.row("a"),self.row("a")])
        with self.assertRaises(ValueError): load_records(p,"train")
if __name__=="__main__": unittest.main()
