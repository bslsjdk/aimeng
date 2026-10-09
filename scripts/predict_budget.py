#!/usr/bin/env python3
"""Run the experimental budget classifier trained by train_budget_controller.py."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/budget_controller")
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--expected-output-type", default="unknown")
    parser.add_argument("--device-class", default="colab_gpu")
    parser.add_argument("--accelerator", default="unknown")
    parser.add_argument("--input-tokens", type=float, required=True)
    args = parser.parse_args()
    if args.input_tokens < 0 or not math.isfinite(args.input_tokens):
        parser.error("--input-tokens must be a finite non-negative number")

    root = Path(args.model_dir)
    manifest_path = root / "manifest.json"
    checkpoint_path = root / "budget_controller.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        print("ERROR: controller manifest/checkpoint missing; train and validate first.", file=sys.stderr)
        return 2
    try:
        import torch
        from torch import nn
    except ImportError:
        print("ERROR: PyTorch required.", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = manifest["labels"]
    vocab = manifest["categorical_vocab"]
    mean = float(manifest["input_tokens_log_mean"])
    std = float(manifest["input_tokens_log_std"]) or 1.0
    log_tokens = math.log1p(args.input_tokens)
    vector = [(log_tokens - mean) / std]
    values = {
        "task_family": args.task_family,
        "expected_output_type": args.expected_output_type,
        "device_class": args.device_class,
        "accelerator": args.accelerator,
    }
    for key, value in values.items():
        categories = vocab[key]
        chosen = value if value in categories else "<UNK>"
        vector.extend(1.0 if category == chosen else 0.0 for category in categories)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = nn.Sequential(
        nn.Linear(int(checkpoint["input_dim"]), 32),
        nn.ReLU(),
        nn.Linear(32, len(labels)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([vector], dtype=torch.float32))[0]
        probabilities = torch.softmax(logits, dim=0).tolist()
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    print(json.dumps({
        "predicted_budget": ranked[0][0],
        "probabilities": {label: round(float(prob), 6) for label, prob in zip(labels, probabilities)},
        "ranked_candidates": [{"budget": label, "probability": round(float(prob), 6)} for label, prob in ranked],
        "warning": "Research output only. Apply quality gates, resource safety limits, and fallback logic before execution.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
