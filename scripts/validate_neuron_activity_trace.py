"""Validate AIMENG neural-unit activity records; not causal scientific truth."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

EVENTS = {"considered", "activated", "skipped", "deactivated", "replaced", "error", "evaluation"}
OUTCOMES = {"unknown", "helpful", "harmful", "neutral", "mixed", "not_measured"}

def validate_trace(row: Any) -> list[str]:
    if not isinstance(row, dict):
        return ["trace must be a JSON object"]
    errors: list[str] = []
    required = ("schema_version","event_id","run_id","timestamp","model_revision","task_ref","unit_id","unit_revision","event_type","participation","measurements","attribution")
    for key in required:
        if key not in row: errors.append(f"missing {key}")
    if row.get("schema_version") != "aimeng.neuron_activity_trace.v1": errors.append("invalid schema_version")
    for key in ("event_id","run_id","timestamp","model_revision","task_ref","unit_id"):
        if key in row and (not isinstance(row[key], str) or not row[key].strip()): errors.append(f"{key} must be a non-empty string")
    if "unit_revision" in row and (type(row["unit_revision"]) is not int or row["unit_revision"] < 1): errors.append("unit_revision must be an integer >= 1")
    if row.get("event_type") not in EVENTS: errors.append("invalid event_type")
    p = row.get("participation")
    if not isinstance(p, dict): errors.append("participation must be an object")
    else:
        if type(p.get("selected")) is not bool: errors.append("participation.selected must be boolean")
        allowed = {"router_selected","dependency_required","budget_limit","low_score","health_gate","manual_test","not_applicable"}
        if p.get("reason_code") not in allowed: errors.append("invalid participation.reason_code")
        for key in ("activation_count","active_duration_ms"):
            value=p.get(key)
            if value is not None and (type(value) not in (int,float) or value < 0): errors.append(f"participation.{key} must be null or non-negative")
    m=row.get("measurements")
    if not isinstance(m, dict): errors.append("measurements must be an object")
    else:
        for key in ("latency_ms","peak_memory_bytes"):
            value=m.get(key)
            if value is not None and (type(value) not in (int,float) or value < 0): errors.append(f"measurements.{key} must be null or non-negative")
    a=row.get("attribution")
    if not isinstance(a, dict): errors.append("attribution must be an object")
    else:
        if a.get("outcome_status") not in OUTCOMES: errors.append("invalid attribution.outcome_status")
        c=a.get("causal_confidence")
        if c is not None and (type(c) not in (int,float) or not 0 <= c <= 1): errors.append("attribution.causal_confidence must be null or between 0 and 1")
        if a.get("outcome_status") in {"helpful","harmful"}:
            if not isinstance(a.get("evidence_ref"),str) or not a["evidence_ref"].strip(): errors.append("helpful/harmful outcome requires evidence_ref")
            if c is None: errors.append("helpful/harmful outcome requires causal_confidence")
    return errors

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True); parser.add_argument("--report")
    args=parser.parse_args()
    try: errors=validate_trace(json.loads(Path(args.input).read_text(encoding="utf-8")))
    except (OSError,json.JSONDecodeError) as exc: errors=[f"cannot read valid JSON input: {exc}"]
    rendered=json.dumps({"ok":not errors,"errors":errors},ensure_ascii=False,indent=2)
    print(rendered)
    if args.report:
        target=Path(args.report); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(rendered+"\n",encoding="utf-8")
    return 0 if not errors else 2

if __name__ == "__main__": raise SystemExit(main())
