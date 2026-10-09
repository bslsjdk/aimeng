#!/usr/bin/env python3
"""Deterministic toy lab for score-driven neuron participation.

This is a small educational experiment, not a biological-neuron simulator or LLM.
Each unit learns an affine predictor y = w*x + b. A controller estimates each
unit's marginal held-out contribution, penalizes compute cost, and applies
hysteresis before changing its participation state.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class Unit:
    unit_id: str
    w: float
    b: float
    learning_rate: float = 0.03
    enabled: bool = True
    revision: int = 1
    last_score: float = 0.0
    state: str = "active"

    def predict(self, x: float) -> float:
        return self.w * x + self.b

    def learn(self, samples: Sequence[tuple[float, float]]) -> float:
        """One deterministic batch-gradient update; returns pre-update MSE."""
        if not samples:
            raise ValueError("training samples must not be empty")
        loss = 0.0
        grad_w = 0.0
        grad_b = 0.0
        for x, target in samples:
            error = self.predict(x) - target
            loss += error * error
            grad_w += 2.0 * error * x
            grad_b += 2.0 * error
        scale = 1.0 / len(samples)
        self.w -= self.learning_rate * grad_w * scale
        self.b -= self.learning_rate * grad_b * scale
        return loss * scale


def mse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) != len(targets) or not targets:
        raise ValueError("predictions and targets must have equal nonzero length")
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)


def ensemble_predict(units: Sequence[Unit], xs: Sequence[float]) -> list[float]:
    active = [u for u in units if u.enabled]
    if not active:
        return [0.0 for _ in xs]
    return [sum(u.predict(x) for u in active) / len(active) for x in xs]


def evaluate(units: Sequence[Unit], samples: Sequence[tuple[float, float]]) -> float:
    if not samples:
        raise ValueError("evaluation samples must not be empty")
    xs = [x for x, _ in samples]
    ys = [y for _, y in samples]
    return mse(ensemble_predict(units, xs), ys)


def score_units(
    units: Sequence[Unit],
    validation: Sequence[tuple[float, float]],
    compute_cost: float = 0.002,
) -> list[dict]:
    """Estimate active-unit ablation or sleeping-unit admission value."""
    if not validation:
        raise ValueError("validation samples must not be empty")
    baseline = evaluate(units, validation)
    records = []
    for unit in units:
        was_enabled = unit.enabled
        if was_enabled:
            unit.enabled = False
            counterfactual_mse = evaluate(units, validation)
            unit.enabled = True
            contribution = counterfactual_mse - baseline
            comparison = "remove_active_unit"
        else:
            unit.enabled = True
            counterfactual_mse = evaluate(units, validation)
            unit.enabled = False
            contribution = baseline - counterfactual_mse
            comparison = "admit_sleeping_unit"
        score = contribution - compute_cost
        unit.last_score = score
        records.append({
            "unit_id": unit.unit_id,
            "revision": unit.revision,
            "enabled": was_enabled,
            "comparison": comparison,
            "baseline_mse": baseline,
            "counterfactual_mse": counterfactual_mse,
            "marginal_contribution": contribution,
            "compute_cost": compute_cost,
            "score": score,
        })
    return records

def apply_score_policy(
    units: Sequence[Unit],
    records: Sequence[dict],
    enter_threshold: float = 0.0001,
    exit_threshold: float = -0.0001,
) -> list[dict]:
    """Apply hysteresis while guaranteeing at least one active unit."""
    by_id = {record["unit_id"]: record for record in records}
    changes = []
    for unit in units:
        score = by_id[unit.unit_id]["score"]
        previous = unit.enabled
        if unit.enabled and score < exit_threshold:
            unit.enabled = False
            unit.state = "sleeping"
        elif not unit.enabled and score > enter_threshold:
            unit.enabled = True
            unit.state = "active"
        if previous != unit.enabled:
            changes.append({
                "unit_id": unit.unit_id,
                "revision": unit.revision,
                "from": "active" if previous else "sleeping",
                "to": unit.state,
                "score": score,
            })
    if units and not any(unit.enabled for unit in units):
        # Do not let simultaneous redundant-unit scores shut the whole system
        # down. Keep the highest-scoring candidate active for the next cycle.
        best = max(units, key=lambda item: by_id[item.unit_id]["score"])
        best.enabled = True
        best.state = "active"
        changes.append({
            "unit_id": best.unit_id,
            "revision": best.revision,
            "from": "sleeping",
            "to": "active",
            "score": by_id[best.unit_id]["score"],
            "reason": "minimum_active_unit_safety_floor",
        })
    return changes

def run_experiment(epochs: int = 80) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    train = [(-1.0, -1.0), (-0.5, 0.0), (0.0, 1.0), (0.5, 2.0), (1.0, 3.0)]
    validation = [(-0.8, -0.6), (-0.2, 0.6), (0.3, 1.6), (0.9, 2.8)]
    units = [
        Unit("unit-0", -0.8, 0.2, 0.03),
        Unit("unit-1", 0.1, -0.5, 0.03),
        Unit("unit-2", 1.8, 0.8, 0.02),
        Unit("unit-3", -1.5, 2.0, 0.01),
    ]
    trace = []
    initial_mse = evaluate(units, validation)
    for epoch in range(epochs):
        # In this lab, every candidate receives a training update, including
        # sleeping units. This is deliberate exploration, not an efficiency claim.
        for unit in units:
            pre_update_loss = unit.learn(train)
            trace.append({
                "event": "parameter_update",
                "epoch": epoch,
                "unit_id": unit.unit_id,
                "revision": unit.revision,
                "w": unit.w,
                "b": unit.b,
                "training_mse_before_update": pre_update_loss,
            })
        records = score_units(units, validation)
        changes = apply_score_policy(units, records)
        trace.append({
            "event": "score_cycle",
            "epoch": epoch,
            "scores": records,
            "state_changes": changes,
            "active_unit_ids": [u.unit_id for u in units if u.enabled],
            "validation_mse": evaluate(units, validation),
        })
    final_mse = evaluate(units, validation)
    return {
        "experiment": "score-driven-neuron-activity-v1",
        "task": "synthetic affine regression: target = 2*x + 1",
        "epochs": epochs,
        "initial_validation_mse": initial_mse,
        "final_validation_mse": final_mse,
        "validation_mse_improved": final_mse < initial_mse,
        "units": [asdict(u) for u in units],
        "final_scores": score_units(units, validation),
        "trace": trace,
        "limitations": [
            "toy affine predictors, not biological neurons or a language model",
            "marginal contribution depends on the current ensemble and validation set",
            "sleeping units still train in this lab to preserve exploration",
            "no claim that score-based gating improves general intelligence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--trace", type=Path, default=None)
    args = parser.parse_args()
    result = run_experiment(args.epochs)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("w", encoding="utf-8") as handle:
            for event in result["trace"]:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
    summary = {k: v for k, v in result.items() if k != "trace"}
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
