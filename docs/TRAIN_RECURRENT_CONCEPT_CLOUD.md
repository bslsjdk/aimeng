# Terminal training and quick comparison

The prototype can be trained and compared from a shell; no Android app or GUI is required.

## 1. Fast correctness smoke test

Run from the repository root:

```bash
python scripts/experiment_recurrent_concept_cloud.py --self-test
python -m unittest tests.test_recurrent_concept_cloud
```

## 2. Train and compare with a GRU baseline

```bash
python scripts/train_recurrent_concept_cloud.py --steps 180 --train-size 2048 --test-size 512 --batch-size 64 --device cpu
```

For a faster smoke run, use `--steps 20 --train-size 512 --test-size 128 --batch-size 16`.
For CUDA, use `--device auto`; CPU is the portable default.

By default, the script uses an easy deterministic successor-token task (`--task successor`) to check whether learning works at all. Use `--task context-sum` for a harder rule depending on the last two tokens. Both systems receive the same generated data and training steps. The script prints JSON containing training-loss movement, test accuracy/loss, throughput, router recall@k, parameter counts, and route-conditional accuracy.

## Interpretation rules

- These synthetic rules are pipeline sanity checks, **not** evidence of general language intelligence.
- Router recall@k is mandatory: the token head is allowed to score only proposed concept clouds at inference.
- Routing misses count as failures in overall concept-cloud token accuracy; conditional accuracy is reported separately.
- The prototype still allocates dense vocabulary logits and uses Python loops for cloud scoring. Its sparse mask is not proof of lower latency or memory use.
- A credible claim requires repeated seeds, held-out real text, matched training compute, model-size and peak-RAM measurements, and real task evaluation. Keep the current result labelled exploratory until those tests pass.


## Latest smoke-test snapshot (2026-10-11)

The eight unit tests passed. With seed 17, 600 steps, batch size 64, 4,096 training examples and 1,024 test examples on CPU, the successor-token task produced:

| Metric | Recurrent concept-cloud | GRU baseline |
|---|---:|---:|
| Parameters | 5,592 | 3,744 |
| Test accuracy | 94.14% | 100.00% |
| Test throughput | 19,694 examples/s | 366,653 examples/s |
| Training time | 4.77 s | 0.83 s |
| Router recall@3 | 99.80% | Not applicable |

Vectorized candidate scoring improved this prototype's measured throughput from roughly 4,241 to 19,694 examples/s on the same synthetic configuration (about 4.6x), but the GRU baseline remains about 18.6x faster and is more accurate. Treat these as one small synthetic CPU result, not proof of language-model superiority.
