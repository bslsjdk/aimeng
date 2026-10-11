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
and 4 cycles and a GRU baseline on the same training examples, using the same
embedding width, optimizer, and number of update steps. It reports held-out
accuracy, router recall, throughput, parameter count, training loss movement,
and wall-clock training time as JSON.

## Interpretation rules

- Count routing misses as failures in concept-cloud accuracy; router recall is
  reported separately.
- More cycles are not automatically better. Compare accuracy gains against
  end-to-end throughput and training time.
- The GRU is a useful baseline, not a straw man. Same examples and update steps
  do not guarantee identical FLOPs, so this is not yet a matched-compute result.
- The target rule is algorithmic and synthetic. Passing it only establishes
  performance on this task; it does not establish language understanding,
  broad reasoning, or superiority over other model families.
- A stronger follow-up should include compositional generalization splits,
  multiple task families, matched parameter/FLOP budgets, and real-text tests.
