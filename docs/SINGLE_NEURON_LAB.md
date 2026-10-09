# Single-Neuron Laboratory: Phase 0

Status: first executable experiment. This is a deterministic toy learner, not an LLM, not a biological brain simulation, and not proof of bidirectional learning.

## Why start with one neuron?

Before introducing multiple units, verify the smallest unit contract end to end: stable identity, revision, forward computation, a documented local update rule, activity/update traces, enable/disable ablation, and guarded replacement. This catches interface and evidence problems before the graph becomes complicated.

## What this experiment implements

- A scalar-output linear neuron: y = w·x + b.
- Supervised delta-rule updates using an externally supplied target.
- An append-only JSONL trace of forward activations, parameter updates, ablation, and replacement decisions.
- A replacement gate that preserves unit_id, requires a higher immutable revision, checks compatible input dimension, and promotes only when held-out mean-squared error improves.
- A deterministic synthetic regression task y = 2x + 1.

Run:

```bash
python scripts/single_neuron_lab.py --epochs 60 --learning-rate 0.05 --trace runs/single-neuron-trace.jsonl
python -m unittest discover -s tests -p 'test_single_neuron_lab.py'
```

The script uses only the Python standard library. The demo should reduce held-out error and approach weight 2, bias 1. A passing toy test validates only this tiny experiment.

## What it does not prove

A single neuron has no meaningful multi-layer backward pathway. The target error is supplied externally, and the local delta rule is mathematically equivalent to the gradient for this simple linear regression loss. It does not implement simultaneous feedforward/feedback inference, biological plasticity, autonomous error discovery, or training of the AIMENG language model.

## Next phases and gates

1. **Phase 0, this lab:** single-unit learning, trace, ablation, replacement, held-out gate.
2. **Phase 1:** two or more units with explicit forward and feedback connections; compare iterative predictive-coding-style state correction against a baseline on a deterministic task.
3. **Phase 2:** small network with local parameter updates and controlled feedback; compare with standard backpropagation using the same data and parameter budget.
4. **Phase 3:** candidate unit training, causal ablation/counterfactual tests, regression checks, immutable artifacts and rollback.
5. **Phase 4:** adapt a compatible subgraph/adapter in a real trainable model checkpoint. Do not attempt arbitrary scalar-neuron hot-swap until tensor mappings and backend behavior are proven.

A candidate is not considered helpful merely because it was active or because its own confidence increased. Report measured outcomes, keep unknown results unknown, and promote only after independent validation.

## Resource constraints

This script is intended for desktop/CI toy tests and does not establish Android runtime usage. Any future Android integration must count the entire application peak RAM and stay below 4096 MiB, with a target release gate below 3800 MiB.
