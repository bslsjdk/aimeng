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

The script trains the recurrent concept-cloud model and a GRU baseline on the same deterministic synthetic next-token task, then prints JSON containing training-loss movement, test accuracy/loss, throughput, router recall@k, parameter counts, and route-conditional accuracy.

## Interpretation rules

- This synthetic rule is a pipeline sanity check, **not** evidence of general language intelligence.
- Router recall@k is mandatory: the token head is allowed to score only proposed concept clouds at inference.
- Routing misses count as failures in overall concept-cloud token accuracy; conditional accuracy is reported separately.
- The prototype still allocates dense vocabulary logits and uses Python loops for cloud scoring. Its sparse mask is not proof of lower latency or memory use.
- A credible claim requires repeated seeds, held-out real text, matched training compute, model-size and peak-RAM measurements, and real task evaluation. Keep the current result labelled exploratory until those tests pass.
