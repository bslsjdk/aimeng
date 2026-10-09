# AIMENG Student SFT Training Harness

## Purpose

This is the first explicit causal-language-model SFT entry point. It does not pick a model architecture for the project and does not claim that a model has already been trained. It accepts a compatible local Hugging Face model/tokenizer and deliberately refuses to train if the selected data file has no reviewed records for either the train or validation split.

## Data gate

A record is eligible only when `training_eligible=true`, `verification.status=verified`, and its split is explicitly `train` or `validation`. Test records are never included in optimizer updates. Structural validation alone is not sufficient to mark content verified; evidence must be supplied by an independent review or suitable automated verifier.

## Example

First inspect eligibility without loading model weights:

```bash
python scripts/train_student_sft.py --data data/approved/sft.jsonl --model /path/to/compatible-model --output runs/sft-smoke --dry-run
```

Then train only after the approved dataset and dependencies are ready:

```bash
python scripts/train_student_sft.py --data data/approved/sft.jsonl --model /path/to/compatible-model --output runs/sft-001 --epochs 1 --batch-size 1 --grad-accum 8 --max-length 1024
```

Pin dependency versions and model revision for reproducible runs. This starter uses the model's chat template and standard causal-LM loss; it does not yet mask user/system tokens from loss. Before serious training, add assistant-only loss masking, held-out test evaluation, source/task-cluster split enforcement, dataset fingerprints, memory/throughput telemetry, and a model-specific smoke test. The simple harness is a starting gate, not a production trainer.

## First experiment acceptance

- Data validator passes and every eligible row has evidence.
- Train and validation are non-empty and have no source/task-cluster leakage.
- Baseline checkpoint is preserved.
- Log dataset fingerprint, code revision, model revision, config, train/eval loss, wall time, peak memory, and test metrics.
- Do not claim capability improvement from loss alone; require an independent held-out task suite.
