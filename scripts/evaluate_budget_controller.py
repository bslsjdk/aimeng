#!/usr/bin/env python3
"""Evaluate a trained budget controller on task IDs held out as dataset_split=test."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_budget_controller as trainer  # noqa: E402


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarize_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"runs": 0, "quality_pass_rate": None, "mean_latency_ms": None, "mean_peak_pss_mib": None}
    passes = sum(
        1 for row in records
        if row.get("result", {}).get("status") == "completed"
        and row.get("result", {}).get("quality_label") == "pass"
    )
    latencies = [
        float(row["performance"]["total_latency_ms"])
        for row in records
        if isinstance(row.get("performance", {}).get("total_latency_ms"), (int, float))
        and not isinstance(row.get("performance", {}).get("total_latency_ms"), bool)
    ]
    pss = [
        float(row["memory"]["process_peak_pss_mib"])
        for row in records
        if isinstance(row.get("memory", {}).get("process_peak_pss_mib"), (int, float))
        and not isinstance(row.get("memory", {}).get("process_peak_pss_mib"), bool)
    ]
    return {
        "runs": len(records),
        "quality_pass_rate": round(passes / len(records), 4),
        "mean_latency_ms": mean_or_none(latencies),
        "mean_peak_pss_mib": mean_or_none(pss),
        "latency_measurements": len(latencies),
        "pss_measurements": len(pss),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/telemetry.jsonl")
    parser.add_argument("--model-dir", default="outputs/budget_controller")
    parser.add_argument("--output", default="outputs/budget_controller/test_report.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    root = Path(args.model_dir)
    if not input_path.is_file() or not (root / "manifest.json").is_file() or not (root / "budget_controller.pt").is_file():
        print("ERROR: telemetry, manifest, or checkpoint missing.", file=sys.stderr)
        return 2
    try:
        import torch
        from torch import nn
        rows = trainer.read_jsonl(input_path)
        examples, errors = trainer.group_records(rows)
    except Exception as exc:
        print(f"ERROR: failed to load evaluation data: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("ERROR: telemetry integrity failed:", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        return 2

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    labels = manifest["labels"]
    vocab = manifest["categorical_vocab"]
    mean = float(manifest["input_tokens_log_mean"])
    std = float(manifest["input_tokens_log_std"]) or 1.0
    checkpoint = torch.load(root / "budget_controller.pt", map_location="cpu", weights_only=True)
    model = nn.Sequential(nn.Linear(int(checkpoint["input_dim"]), 32), nn.ReLU(), nn.Linear(32, len(labels)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_examples = [example for example in examples if example["split"] == "test"]
    if len(test_examples) < 3:
        print(f"STOP: only {len(test_examples)} eligible held-out test task-pairs; need at least 3.", file=sys.stderr)
        return 3

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id"))].append(row)

    predictions = []
    with torch.no_grad():
        for example in test_examples:
            features = trainer.encode(example, vocab, mean, std)
            logits = model(torch.tensor([features], dtype=torch.float32))[0]
            predicted = labels[int(logits.argmax().item())]
            predictions.append((example, predicted))

    correct = sum(predicted == example["target"] for example, predicted in predictions)
    majority = max(sum(example["target"] == label for example in test_examples) for label in labels)
    policy_runs: list[dict[str, Any]] = []
    full_runs: list[dict[str, Any]] = []
    for example, predicted in predictions:
        task_runs = by_task[example["task_id"]]
        policy_runs.extend(row for row in task_runs if trainer.budget_label(row) == predicted)
        full_runs.extend(row for row in task_runs if trainer.budget_label(row) == "full")

    report = {
        "schema_version": "aimeng.controller_eval.v1",
        "telemetry_file": str(input_path),
        "test_task_pairs": len(test_examples),
        "target_budget_accuracy": round(correct / len(test_examples), 4),
        "majority_target_baseline_accuracy": round(majority / len(test_examples), 4),
        "controller_selected_budget": summarize_runs(policy_runs),
        "fixed_full_budget_baseline": summarize_runs(full_runs),
        "warning": "Use the held-out metrics together. Do not deploy unless quality and resource gates pass on the target runtime; missing measurements are not zero.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT_WRITTEN={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
