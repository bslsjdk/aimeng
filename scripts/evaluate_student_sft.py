"""Evaluate a compatible Hugging Face causal LM on frozen held-out/regression JSONL suites.

Input rows use schema aimeng.eval_task.v1:
{"schema_version":"aimeng.eval_task.v1","task_id":"...","prompt":"...","expected":"...","verifier":"exact_match|contains|json_object"}
This reports task quality only. Android memory must be measured separately on-device.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_suite(path: Path) -> list[dict[str, Any]]:
    rows, seen = [], set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict) or row.get("schema_version") != "aimeng.eval_task.v1":
            raise ValueError(f"{path}:{line_no}: expected aimeng.eval_task.v1 object")
        for key in ("task_id", "prompt", "verifier"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{line_no}: {key} must be a non-empty string")
        if row["task_id"] in seen:
            raise ValueError(f"{path}:{line_no}: duplicate task_id {row['task_id']}")
        seen.add(row["task_id"])
        if row["verifier"] not in {"exact_match", "contains", "json_object"}:
            raise ValueError(f"{path}:{line_no}: unsupported verifier {row['verifier']}")
        if row["verifier"] != "json_object" and (not isinstance(row.get("expected"), str) or not row["expected"]):
            raise ValueError(f"{path}:{line_no}: expected must be a non-empty string")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: evaluation suite is empty")
    return rows


def verify(row: dict[str, Any], output: str) -> bool:
    mode = row["verifier"]
    if mode == "exact_match":
        return output.strip() == row["expected"].strip()
    if mode == "contains":
        return row["expected"] in output
    if mode == "json_object":
        try:
            value = json.loads(output.strip())
            return isinstance(value, dict)
        except json.JSONDecodeError:
            return False
    return False


def evaluate_suite(model, tokenizer, rows, max_new_tokens: int) -> dict[str, Any]:
    import torch
    results = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = row["prompt"]
        inputs = tokenizer(rendered, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer_ids = generated[0][inputs["input_ids"].shape[-1]:]
        output = tokenizer.decode(answer_ids, skip_special_tokens=True)
        passed = verify(row, output)
        results.append({"task_id": row["task_id"], "passed": passed, "output": output})
    return {"tasks": len(results), "passed": sum(r["passed"] for r in results),
            "success_rate": sum(r["passed"] for r in results) / len(results), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--regression", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    heldout_path, regression_path = Path(args.heldout), Path(args.regression)
    heldout_rows, regression_rows = load_suite(heldout_path), load_suite(regression_path)
    # Task IDs must not overlap between the two evaluation sets.
    overlap = {r["task_id"] for r in heldout_rows} & {r["task_id"] for r in regression_rows}
    if overlap:
        raise SystemExit(f"heldout/regression task leakage: {sorted(overlap)[:10]}")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install compatible torch and transformers before model evaluation") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False)
    model.eval()
    heldout = evaluate_suite(model, tokenizer, heldout_rows, args.max_new_tokens)
    regression = evaluate_suite(model, tokenizer, regression_rows, args.max_new_tokens)
    suite_hash = hashlib.sha256(
        (file_sha256(heldout_path) + ":" + file_sha256(regression_path)).encode("ascii")
    ).hexdigest()
    report = {
        "schema_version": "aimeng.student_eval.v1",
        "model_version": args.model_version,
        "suite_hash": suite_hash,
        "heldout_success_rate": heldout["success_rate"],
        "regression_success_rate": regression["success_rate"],
        "heldout": heldout,
        "regression": regression,
        "memory_measurement_scope": None,
        "memory_measured": False,
        "peak_memory_mib": None,
        "note": "Quality evaluation only; promotion remains blocked until whole-Android-app memory is measured.",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model_version", "suite_hash", "heldout_success_rate", "regression_success_rate")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
