#!/usr/bin/env python3
"""Tiny dependency-free logistic-regression trainer for exercising AIMENG job orchestration.

This is an infrastructure smoke test, not a language model or evidence of general AI ability.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def make_example(rng: random.Random, task: str) -> tuple[float, float, float]:
    x1, x2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
    margin = x1 + x2 if task == "reasoning" else x1 - x2
    label = 1.0 if margin >= 0 else 0.0
    return x1, x2, label


def loss_for(weights: list[float], examples: list[tuple[float, float, float]]) -> float:
    total = 0.0
    for x1, x2, label in examples:
        p = sigmoid(weights[0] * x1 + weights[1] * x2 + weights[2])
        p = max(1e-9, min(1.0 - 1e-9, p))
        total -= label * math.log(p) + (1.0 - label) * math.log(1.0 - p)
    return total / max(1, len(examples))


def train(task: str, steps: int, seed: int, output: Path) -> dict:
    rng = random.Random(seed)
    train_data = [make_example(rng, task) for _ in range(512)]
    validation_rng = random.Random(seed + 1000003)
    validation_data = [make_example(validation_rng, task) for _ in range(256)]
    weights = [0.0, 0.0, 0.0]
    initial_loss = loss_for(weights, validation_data)
    learning_rate = 0.5
    for _ in range(steps):
        gradients = [0.0, 0.0, 0.0]
        for x1, x2, label in train_data:
            error = sigmoid(weights[0] * x1 + weights[1] * x2 + weights[2]) - label
            gradients[0] += error * x1
            gradients[1] += error * x2
            gradients[2] += error
        scale = learning_rate / len(train_data)
        weights = [w - scale * g for w, g in zip(weights, gradients)]
    final_loss = loss_for(weights, validation_data)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": "aimeng-toy-logreg-v0",
        "task": task,
        "seed": seed,
        "steps": steps,
        "training_examples": len(train_data),
        "validation_examples": len(validation_data),
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "weights": weights,
        "warning": "toy infrastructure smoke test; not a language model",
    }
    (output / "checkpoint.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("reasoning", "tool_use"), required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    result = train(args.task, args.steps, args.seed, args.output)
    print(json.dumps({k: result[k] for k in (
        "task", "steps", "initial_validation_loss", "final_validation_loss"
    )}, ensure_ascii=False))
    return 0 if result["final_validation_loss"] < result["initial_validation_loss"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
