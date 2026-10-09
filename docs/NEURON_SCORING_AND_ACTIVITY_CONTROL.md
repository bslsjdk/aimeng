# Score-Driven Neuron Activity Lab

## Purpose

This is the next toy experiment after the single-neuron lab. It tests whether independently trained units can be evaluated and scheduled using an explicit score, instead of treating activation as proof of usefulness.

Run:

```bash
python -m unittest discover -s tests -p 'test_neuron_scoring_lab.py'
python scripts/neuron_scoring_lab.py --epochs 80 --trace /tmp/neuron-score-trace.jsonl
```

## Per-unit score

For an active unit, the lab temporarily removes it and measures the held-out mean squared error (MSE) change. For a sleeping unit, it temporarily admits the candidate and measures whether the ensemble improves:

```text
active unit:   contribution = MSE(without_unit) - MSE(current_ensemble)
sleeping unit: contribution = MSE(current_ensemble) - MSE(with_candidate)
score = contribution - compute_cost
```

A positive contribution means the measured counterfactual change favors that unit in the current validation setup. It does **not** prove universal usefulness or causal value outside this measured setup. The cost term makes the policy consider whether the measured benefit justifies compute. A safety floor keeps at least one unit active, so redundant negative scores cannot shut the entire system down in one cycle.

The controller uses separate enter and exit thresholds (hysteresis) to reduce rapid state flipping. Score, activation state, parameter-update events, unit identity, and revision are logged separately.

## Important experimental limitation

Sleeping units still receive training updates in this lab. That is intentional: it gives candidates a chance to improve and re-enter instead of permanently starving them. Therefore this version tests score-based participation decisions, not actual compute savings. A later experiment can compare:
- A: all units always active;
- B: score-gated active units;
- C: score-gated units with periodic exploration;
- D: random gating with the same compute budget.

All variants must use identical data splits and comparable compute budgets. Report held-out loss, compute steps, state changes, and stability. Do not declare success merely because units became inactive.

## Not implemented or claimed

- No biological neuron or brain simulation.
- No language-model capability.
- No automatic structural growth or unit replacement.
- No guarantee that scoring improves learning.
- No mobile runtime or Android RAM measurements.

This lab is a deterministic educational model. Results should guide the next experiment, not be generalized beyond its scope.
