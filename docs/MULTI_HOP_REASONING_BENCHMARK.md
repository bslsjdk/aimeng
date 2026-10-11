# Multi-hop reasoning benchmark

This benchmark is a first probe of the hypothesis that recurrent concept-cloud
state refinement helps on tasks with dependencies spanning an entire sequence.

## Task

Starting from state 0, process every input token in order:

```text
state = 0
for token in sequence:
    state = (3 * state + token + 1) % vocab
answer = state
```

Unlike the successor-token task, the answer depends on a sequence of state
updates. Unlike a last-two-token sum, early tokens can continue to affect the
final state. This is still a synthetic rule-following task, not natural-language
understanding or proof of general reasoning.

## Fast self-test

From the repository root:

```bash
python scripts/benchmark_recurrent_reasoning.py --self-test
```

The self-test caps training to three steps and uses a small dataset so it can
check that the whole benchmark path runs without a long training session.

## Full comparison

```bash
python scripts/benchmark_recurrent_reasoning.py --steps 300 --train-size 2048 --test-size 512 --batch-size 64 --seq-len 8 --seeds 17,23,41 --device cpu --threads 1
```

For every seed, the script trains separate concept-cloud models with 1, 2, 3,
and 4 cycles, plus two GRU baselines, on the same training examples and update
step budget. One GRU uses the same width; the other chooses a width that most
closely matches the concept-cloud parameter count. All use the same optimizer. Training uses the
configured sequence length only. Evaluation reports both held-out examples at
the training length and extrapolation to sequences 4 and 8 tokens longer, with
no further training. It also reports router recall, throughput, parameter
count, training loss movement, and wall-clock training time as JSON.

## Interpretation rules

- Count routing misses as failures in concept-cloud accuracy; router recall is
  reported separately.
- More cycles are not automatically better. Compare accuracy gains against
  end-to-end throughput and training time.
- During training, the model currently adds the gold target cloud to token
  scoring; at evaluation it must route without that hint. This teacher-forcing
  mismatch can make training easier than inference, so route-free accuracy and
  router recall are critical and should be improved in a later training revision.
- The GRU is a useful baseline, not a straw man. Same examples and update steps
  do not guarantee identical FLOPs, so this is not yet a matched-compute result.
  The parameter-matched GRU reduces parameter-count unfairness, while the
  same-width GRU preserves a capacity-width reference.
- The target rule is algorithmic and synthetic. Passing it only establishes
  performance on this task; it does not establish language understanding,
  broad reasoning, or superiority over other model families.
- A stronger follow-up should include compositional generalization splits,
  multiple task families, matched parameter/FLOP budgets, and real-text tests.
