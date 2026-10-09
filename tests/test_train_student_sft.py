import json, tempfile, unittest
from pathlib import Path
from scripts.train_student_sft import load_records, load_all_records, encode_assistant_target
from scripts.import_teacher_demos import convert

class GateTests(unittest.TestCase):
    def write(self, rows):
        f=tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",delete=False)
        for row in rows: f.write(json.dumps(row)+"\n")
        f.close(); self.addCleanup(Path(f.name).unlink, missing_ok=True); return Path(f.name)
    def row(self, sid, status="verified", eligible=True, split="train", task_id=None):
        return {"schema_version":"aimeng.sft_candidate.v1","sample_id":sid,"task_id":task_id or sid,"topic":"test","difficulty":"basic","expected_verifier":"unit_test","verification":{"status":status,"method":"test_fixture","evidence":["unit-test-fixture"] if status=="verified" else []},"training_eligible":eligible,"split":split,"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}
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
    def test_task_split_leakage_rejected(self):
        p=self.write([self.row("a",split="train",task_id="same-task"),self.row("b",split="test",task_id="same-task")])
        with self.assertRaisesRegex(ValueError,"split leakage"): load_all_records(p)
    def test_eligible_record_requires_verification_evidence(self):
        row=self.row("a"); row["verification"]["evidence"]=[]
        p=self.write([row])
        with self.assertRaisesRegex(ValueError,"evidence"): load_records(p,"train")

    def test_prompt_schema_is_demoted_even_if_input_claims_verified(self):
        source={"schema_version":"aimeng.sft_candidate.v1","sample_id":"teacher-output-1","task_id":"t2","topic":"python.exceptions","difficulty":"basic","expected_verifier":"python_unit_test","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}],"verification":{"status":"verified","method":"teacher-self-check","evidence":["teacher said so"]},"split":"train","training_eligible":True}
        candidate=convert(source,2,teacher_model="Ornith-1.5-9B",prompt_version="teacher_sft_batch_v1")
        self.assertEqual(candidate["verification"]["status"],"pending")
        self.assertFalse(candidate["training_eligible"])
        self.assertEqual(candidate["split"],"unassigned")
\n    def test_teacher_import_never_auto_approves(self):
        candidate=convert({"schema_version":"aimeng.teacher_demo.v1","task_id":"t1","topic":"python","prompt":"Explain","response":"Answer","teacher":{"model":"Ornith-1.5-9B","prompt_version":"v1"}},1)
        self.assertEqual(candidate["verification"]["status"],"pending")
        self.assertFalse(candidate["training_eligible"])
        self.assertEqual(candidate["split"],"unassigned")

class MockTokenizer:
    chat_template = "mock"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered="".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt: rendered += "<assistant>"
        return rendered
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids":[ord(c) for c in text]}

class AssistantMaskTests(unittest.TestCase):
    def test_only_final_assistant_target_has_loss(self):
        row={"sample_id":"m1","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}
        encoded=encode_assistant_target(row,MockTokenizer(),512)
        masked=sum(x == -100 for x in encoded["labels"])
        self.assertEqual(masked,len("<user>Q<assistant>"))
        self.assertTrue(all(x == -100 for x in encoded["labels"][:masked]))
        self.assertTrue(all(x != -100 for x in encoded["labels"][masked:]))

if __name__=="__main__": unittest.main()
