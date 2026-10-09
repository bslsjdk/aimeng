import importlib.util, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("neuron_trace_validator",ROOT/"scripts"/"validate_neuron_activity_trace.py")
validator=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(validator)

def trace():
    return {
      "schema_version":"aimeng.neuron_activity_trace.v1","event_id":"event-001","run_id":"run-001",
      "timestamp":"2026-10-09T12:00:00Z","model_revision":"core@rev1","task_ref":"task-001",
      "unit_id":"core/mlp/block0/channel12","unit_revision":1,"event_type":"activated",
      "participation":{"selected":True,"reason_code":"router_selected","activation_count":1,"active_duration_ms":2.5},
      "measurements":{"latency_ms":2.5,"peak_memory_bytes":1024,"input_fingerprint":"sha256:in","output_fingerprint":"sha256:out","activation_summary_ref":None,"confidence_before":0.5,"confidence_after":0.7},
      "attribution":{"outcome_status":"unknown","evidence_ref":None,"causal_confidence":None,"counterfactual_run_ref":None,"review_notes":[]}
    }

class NeuronActivityTraceTests(unittest.TestCase):
    def test_valid_trace(self): self.assertEqual(validator.validate_trace(trace()),[])
    def test_helpful_claim_requires_evidence_and_confidence(self):
        row=trace(); row["attribution"]["outcome_status"]="helpful"; errors=validator.validate_trace(row)
        self.assertTrue(any("requires evidence_ref" in e for e in errors)); self.assertTrue(any("requires causal_confidence" in e for e in errors))
    def test_negative_cost_rejected(self):
        row=trace(); row["measurements"]["latency_ms"]=-1
        self.assertTrue(any("latency_ms" in e for e in validator.validate_trace(row)))
    def test_invalid_revision_rejected(self):
        row=trace(); row["unit_revision"]=0
        self.assertTrue(any("unit_revision" in e for e in validator.validate_trace(row)))

if __name__=="__main__": unittest.main()
