"""AIMENG single-neuron laboratory: deterministic local learning and safe replacement.

This is a tiny research demonstrator, not a language model and not a claim of
biological fidelity. It validates stable identity, local parameter updates,
activity traces, disable/replace controls, and held-out promotion gates.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class LinearNeuron:
    unit_id: str
    revision: int
    weights: list[float]
    bias: float = 0.0
    enabled: bool = True
    update_count: int = 0

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("unit_id must be non-empty")
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        if not self.weights or not all(math.isfinite(v) for v in self.weights):
            raise ValueError("weights must be a non-empty finite list")
        if not math.isfinite(self.bias):
            raise ValueError("bias must be finite")

    def forward(self, inputs: Sequence[float]) -> float:
        if not self.enabled:
            raise RuntimeError(f"unit {self.unit_id} is disabled")
        if len(inputs) != len(self.weights):
            raise ValueError("input dimension does not match weight dimension")
        if not all(math.isfinite(v) for v in inputs):
            raise ValueError("inputs must be finite")
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def learn_from_target(self, inputs: Sequence[float], target: float, lr: float) -> float:
        """One supervised delta-rule update; returns pre-update squared error."""
        if not self.enabled:
            raise RuntimeError(f"unit {self.unit_id} is disabled")
        if not math.isfinite(target) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("target must be finite and lr must be positive finite")
        prediction = self.forward(inputs)
        error = target - prediction
        loss = error * error
        # For a linear output and half-squared loss, this is the exact local gradient.
        self.weights = [w + lr * error * x for w, x in zip(self.weights, inputs)]
        self.bias += lr * error
        self.update_count += 1
        return loss

    def snapshot(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "revision": self.revision,
            "weights": list(self.weights),
            "bias": self.bias,
            "enabled": self.enabled,
            "update_count": self.update_count,
        }


@dataclass
class NeuronLab:
    neuron: LinearNeuron
    trace_path: Path | None = None
    run_id: str = "single-neuron-lab"
    event_counter: int = 0
    events: list[dict] = field(default_factory=list)

    def _trace(self, event_type: str, **details) -> None:
        self.event_counter += 1
        row = {
            "schema_version": "aimeng.neuron_lab_trace.v1",
            "event_id": f"{self.run_id}:{self.event_counter}",
            "run_id": self.run_id,
            "timestamp_unix": time.time(),
            "unit_id": self.neuron.unit_id,
            "unit_revision": self.neuron.revision,
            "event_type": event_type,
            "enabled": self.neuron.enabled,
            "weights": list(self.neuron.weights),
            "bias": self.neuron.bias,
            **details,
        }
        self.events.append(row)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def predict(self, inputs: Sequence[float]) -> float:
        started = time.perf_counter()
        result = self.neuron.forward(inputs)
        self._trace("activated", input=list(inputs), output=result,
                    latency_ms=(time.perf_counter() - started) * 1000)
        return result

    def train_one(self, inputs: Sequence[float], target: float, lr: float) -> float:
        before = self.neuron.snapshot()
        loss = self.neuron.learn_from_target(inputs, target, lr)
        self._trace("parameter_update", input=list(inputs), target=target,
                    pre_update_squared_error=loss, learning_rate=lr,
                    previous_revision=before["revision"])
        return loss

    def set_enabled(self, enabled: bool) -> None:
        self.neuron.enabled = bool(enabled)
        self._trace("activated" if enabled else "deactivated", reason="manual_ablation")

    def replace(self, candidate: LinearNeuron, validation: Iterable[tuple[Sequence[float], float]],
                *, min_improvement: float = 0.0) -> dict:
        """Promote a compatible candidate only if held-out MSE improves."""
        if candidate.unit_id != self.neuron.unit_id:
            raise ValueError("replacement must preserve stable unit_id")
        if candidate.revision <= self.neuron.revision:
            raise ValueError("replacement revision must increase")
        if len(candidate.weights) != len(self.neuron.weights):
            raise ValueError("replacement input dimension mismatch")
        rows = list(validation)
        if not rows:
            raise ValueError("validation set must not be empty")
        baseline = mse(self.neuron, rows)
        candidate.enabled = True
        candidate_score = mse(candidate, rows)
        accepted = candidate_score + min_improvement < baseline
        old_revision = self.neuron.revision
        if accepted:
            self.neuron = candidate
        self._trace("replaced" if accepted else "replacement_rejected",
                    previous_revision=old_revision,
                    candidate_revision=candidate.revision,
                    baseline_mse=baseline,
                    candidate_mse=candidate_score,
                    accepted=accepted)
        return {
            "accepted": accepted,
            "baseline_mse": baseline,
            "candidate_mse": candidate_score,
            "active_revision": self.neuron.revision,
        }


def mse(neuron: LinearNeuron, rows: Iterable[tuple[Sequence[float], float]]) -> float:
    examples = list(rows)
    if not examples:
        raise ValueError("evaluation set must not be empty")
    was_enabled = neuron.enabled
    neuron.enabled = True
    try:
        return sum((neuron.forward(x) - y) ** 2 for x, y in examples) / len(examples)
    finally:
        neuron.enabled = was_enabled


def run_demo(epochs: int = 60, learning_rate: float = 0.05,
             trace_path: Path | None = None) -> dict:
    train = [([-2.0], -3.0), ([-1.0], -1.0), ([0.0], 1.0),
             ([1.0], 3.0), ([2.0], 5.0)]  # y = 2x + 1
    held_out = [([-1.5], -2.0), ([0.5], 2.0), ([1.5], 4.0)]
    neuron = LinearNeuron("toy/linear-neuron-0", 1, [0.0], 0.0)
    lab = NeuronLab(neuron, trace_path=trace_path)
    before = mse(neuron, held_out)
    for _ in range(epochs):
        for x, y in train:
            lab.train_one(x, y, learning_rate)
    after = mse(lab.neuron, held_out)
    return {
        "experiment": "single_neuron_supervised_delta_rule",
        "is_language_model": False,
        "epochs": epochs,
        "updates": lab.neuron.update_count,
        "held_out_mse_before": before,
        "held_out_mse_after": after,
        "learned_weights": lab.neuron.weights,
        "learned_bias": lab.neuron.bias,
        "held_out_improved": after < before,
        "trace_events_in_memory": len(lab.events),
        "trace_path": str(trace_path) if trace_path else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--trace", type=Path, default=Path("runs/single-neuron-trace.jsonl"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    print(json.dumps(run_demo(args.epochs, args.learning_rate, args.trace),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
