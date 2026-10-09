#!/usr/bin/env python3
"""Train a small PyTorch budget classifier from paired, measured telemetry.

Training labels are derived only from completed, independently verified pass/fail runs.
For each task pair, the target is the lowest declared budget that passed. This does not
prove a backend really saves compute; the controller only learns budget selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BUDGET_ORDER = {"small": 0, "medium": 1, "full": 2}
LABELS = ["small", "medium", "full"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            rows.append(row)
    return rows


def budget_label(record: dict[str, Any]) -> str | None:
    budget_id = str(record.get("budget", {}).get("budget_id", "")).lower()
    # Avoid accidentally treating "full" as "small" etc.; use explicit token matching.
    for label in ("small", "medium", "full"):
        if re.search(rf"(^|[^a-z]){label}([^a-z]|$)", budget_id):
            return label
    return None


def group_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_by_task: dict[str, set[str]] = defaultdict(set)
    task_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        task_id = row.get("task_id")
        split = row.get("task", {}).get("dataset_split")
        if not task_id or split not in {"train", "validation", "test"}:
            errors.append("record missing task_id or valid dataset_split")
            continue
        split_by_task[str(task_id)].add(str(split))
        pair_id = row.get("pair_group_id")
        if not pair_id:
            errors.append(f"task {task_id} is missing pair_group_id")
        else:
            task_by_pair[str(pair_id)].add(str(task_id))
        by_task[str(task_id)].append(row)
    for task_id, splits in split_by_task.items():
        if len(splits) != 1:
            errors.append(f"task {task_id} leaks across splits: {sorted(splits)}")
    for pair_id, task_ids in task_by_pair.items():
        if len(task_ids) != 1:
            errors.append(f"pair_group_id {pair_id} is shared by multiple tasks: {sorted(task_ids)}")

    examples = []
    for task_id, records in by_task.items():
        if len({r.get("pair_group_id") for r in records}) != 1:
            errors.append(f"task {task_id} has inconsistent pair_group_id values")
            continue
        split = records[0].get("task", {}).get("dataset_split")
        task = records[0].get("task", {})
        env = records[0].get("environment", {})
        labels = [budget_label(r) for r in records]
        labels_seen = {label for label in labels if label}
        if labels_seen != set(LABELS) or any(label is None for label in labels):
            errors.append(f"task {task_id} must contain small, medium, and full budget runs; got {labels}")
            continue
        # Multiple repeats per budget are allowed. A budget is eligible only if every
        # run has an objective pass/fail label and its measured pass rate meets the gate.
        parser_pass_rate = 0.95
        eligible = []
        for label in LABELS:
            runs = [r for r in records if budget_label(r) == label]
            outcomes = []
            for run in runs:
                result = run.get("result", {})
                if result.get("status") != "completed" or result.get("quality_label") not in {"pass", "fail"}:
                    outcomes = []
                    break
                outcomes.append(result["quality_label"] == "pass")
            if outcomes and sum(outcomes) / len(outcomes) >= parser_pass_rate:
                eligible.append(label)
        if not eligible:
            # A task where no budget reliably passes supplies no safe target.
            continue
        target = min(eligible, key=lambda label: BUDGET_ORDER[label])
        perf = records[0].get("task", {})
        examples.append({
            "task_id": task_id,
            "split": split,
            "target": target,
            "task_family": str(task.get("task_family") or "unknown"),
            "expected_output_type": str(task.get("expected_output_type") or "unknown"),
            "device_class": str(env.get("device_class") or "unknown"),
            "accelerator": str(env.get("accelerator") or "unknown"),
            "input_tokens": max(0.0, float(task.get("input_tokens") or 0)),
        })
    return examples, errors


def make_vocab(examples: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, list[str]]:
    return {key: sorted({str(ex[key]) for ex in examples} | {"<UNK>"}) for key in keys}


def encode(ex: dict[str, Any], vocab: dict[str, list[str]], mean: float, std: float) -> list[float]:
    vector = [((math.log1p(ex["input_tokens"]) - mean) / std)]
    for key in ("task_family", "expected_output_type", "device_class", "accelerator"):
        categories = vocab[key]
        value = str(ex[key])
        vector.extend(1.0 if cat == (value if value in categories else "<UNK>") else 0.0 for cat in categories)
    return vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/telemetry.jsonl")
    parser.add_argument("--output-dir", default="outputs/budget_controller")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-train-pairs", type=int, default=12)
    args = parser.parse_args()
    try:
        import torch
        from torch import nn
    except ImportError:
        print("ERROR: PyTorch is required. In Colab, run: pip install torch", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: telemetry file not found: {input_path}", file=sys.stderr)
        return 2
    try:
        rows = read_jsonl(input_path)
        examples, errors = group_records(rows)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("ERROR: telemetry pairing/split integrity failed:", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        return 2

    train = [ex for ex in examples if ex["split"] == "train"]
    val = [ex for ex in examples if ex["split"] == "validation"]
    if len(train) < args.min_train_pairs:
        print(f"STOP: only {len(train)} eligible training task-pairs; need at least {args.min_train_pairs}.", file=sys.stderr)
        print("No synthetic examples will be created. Collect more independently verified paired tasks.", file=sys.stderr)
        return 3
    if len(val) < 3:
        print(f"STOP: only {len(val)} eligible validation task-pairs; need at least 3.", file=sys.stderr)
        return 3
    if len({ex["target"] for ex in train}) < 2:
        print("STOP: training targets contain fewer than two budget classes; a classifier would not learn a meaningful choice.", file=sys.stderr)
        return 3

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    keys = ("task_family", "expected_output_type", "device_class", "accelerator")
    vocab = make_vocab(train, keys)
    raw_log = [math.log1p(ex["input_tokens"]) for ex in train]
    mean = sum(raw_log) / len(raw_log)
    variance = sum((x - mean) ** 2 for x in raw_log) / max(1, len(raw_log))
    std = math.sqrt(variance) or 1.0
    x_train = torch.tensor([encode(ex, vocab, mean, std) for ex in train], dtype=torch.float32)
    y_train = torch.tensor([LABELS.index(ex["target"]) for ex in train], dtype=torch.long)
    x_val = torch.tensor([encode(ex, vocab, mean, std) for ex in val], dtype=torch.float32)
    y_val = torch.tensor([LABELS.index(ex["target"]) for ex in val], dtype=torch.long)

    model = nn.Sequential(nn.Linear(x_train.shape[1], 32), nn.ReLU(), nn.Linear(32, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        print("ERROR: no checkpoint produced", file=sys.stderr)
        return 4
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x_val).argmax(dim=1)
        accuracy = float((predictions == y_val).float().mean().item())
        baseline = max(sum(int(y == cls) for y in y_val) for cls in range(3)) / len(y_val)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "budget_controller.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": x_train.shape[1],
                "labels": LABELS}, model_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    excluded_pairs = max(0, len({str(row.get("task_id")) for row in rows}) - len(examples))
    manifest = {
        "schema_version": "aimeng.controller.v1",
        "model_type": "small_mlp_budget_classifier",
        "input_features": ["log1p(input_tokens)", "task_family", "expected_output_type", "device_class", "accelerator"],
        "categorical_vocab": vocab,
        "input_tokens_log_mean": mean,
        "input_tokens_log_std": std,
        "labels": LABELS,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_val,
        "validation_accuracy": accuracy,
        "validation_majority_class_baseline": baseline,
        "train_task_pairs": len(train),
        "validation_task_pairs": len(val),
        "excluded_unknown_or_failed_only_pairs": excluded_pairs,
        "telemetry_sha256": digest,
        "checkpoint_file": model_path.name,
        "warning": "Experimental classifier only. Do not deploy unless held-out quality, latency, memory, and fallback gates pass.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("VALIDATION CHECK: classifier trained; this is not evidence of real inference acceleration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
