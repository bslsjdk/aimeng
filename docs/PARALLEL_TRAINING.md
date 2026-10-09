# Parallel training: independent experts first

## Decision

AIMENG will experiment with **bounded concurrent training of independent jobs** before attempting distributed updates to one shared model. Parallelism is a scheduling capability, not a claim that the current repository already has a trainable student language model.

Three designs must not be confused:

1. **Independent models or specialists in parallel.** Each process owns its model, optimizer, checkpoint, logs, and output directory. Jobs may train on separate tasks/domains and can later be compared, routed as experts, or used to produce distillation data. Their weights must not be blindly concatenated or averaged.
2. **One model split across devices (distributed training).** Workers must exchange gradients or parameter updates using a framework such as PyTorch DistributedDataParallel/FSDP or another suitable system. This requires compatible hardware, communication, synchronization, checkpointing, and failure handling. Starting several copies does not implement it.
3. **One shared model trained by concurrent independent optimizers.** Do not do this. Without explicit synchronization, workers race or drift and the result is not a coherent training run.

## Initial implementation

The script scripts/parallel_training.py runs independent command-line jobs concurrently. It requires positive per-job memory estimates and unique output directories, limits simultaneous jobs and the sum of declared estimates, launches argument arrays without a shell, records each job's output in its own training.log, and can write a JSON report. Use --dry-run to validate a plan without launching processes.

Example:

    python scripts/parallel_training.py --plan examples/parallel-training-plan.example.json --dry-run

python scripts/parallel_training.py --plan examples/parallel-training-plan.example.json --report runs/parallel/report.json

The example plan now invokes scripts/toy_train.py, a dependency-free logistic-regression smoke test that trains two different tiny models and writes separate checkpoints. Run it to validate the scheduler end to end. It is deliberately not a language model; replace it with real model-training entry points only after the smoke test passes.

## Resource and correctness limits

- memory_mb is a declared estimate used by the scheduler. It is **not** an OS-enforced RAM/VRAM limit and cannot prevent OOM if estimates are wrong.
- Concurrent jobs on one GPU share VRAM and compute bandwidth. More workers can make every job slower or cause OOM. Begin with two small jobs and measure actual peak RAM/VRAM, throughput, and time-to-quality.
- The Android whole-app peak ceiling of 4096 MiB is a deployment constraint. Training concurrency must be benchmarked on the training host separately.
- Every run needs its own seed, dataset snapshot/hash, configuration, code revision, and isolated checkpoint path. Compare against a sequential baseline on the same data and compute budget.
- Do not average specialist weights unless architecture, tokenizer/vocabulary, parameter alignment, and the merge method explicitly support it. Prefer held-out evaluation, expert routing, or distillation as the first integration paths.
- Exit code zero only proves a process exited successfully. Model quality still requires independent validation and test sets.

## Acceptance experiment

1. Establish a sequential baseline for two small independent jobs.
2. Run the same jobs concurrently under a declared budget that fits measured host RAM/VRAM.
3. Compare total wall-clock time, per-job time, peak RAM/VRAM, throughput, failures, and validation quality.
4. Repeat with different seeds and keep artifacts separate.
5. Enable concurrency by default only if it improves time-to-quality without violating memory or quality gates.

## Current status

The scheduler is a prototype for independent process-level jobs. It does not implement gradient synchronization, automatic expert merging, GPU memory isolation, or a full student-language-model trainer. No speedup or successful training result is claimed until the experiment is actually run.
