# Ornith 1.5 9B -> AIMENG Student: Distillation First, Self-Learning Second

## Decision

Use Ornith 1.5 9B as a temporary teacher/baseline to transfer useful behavior into the AIMENG student, then start a gated self-learning loop. This is a sensible staged experiment, not a guarantee that the student inherits the teacher's full intelligence. The target is measurable transfer: the student must solve held-out tasks and use knowledge in new contexts, not merely imitate phrasing.

## Important distinction: what kind of 9B artifact is available?

Before implementation, inspect the exact local artifact, tokenizer, model license, and inference backend.

- A quantized GGUF or other inference-only package usually supports generating teacher outputs, but does not by itself provide a normal trainable checkpoint.
- If the teacher runtime exposes token logits, soft targets may be possible only if the student output vocabulary and training pipeline support a compatible mapping. Different tokenizers or a custom student architecture make direct logit distillation non-trivial.
- If only generated text is available, use response/trajectory distillation: teacher-generated demonstrations become candidate SFT targets after independent review. This is the most portable first path for a structurally different AIMENG student.
- Never commit the 9B model weights to Git. Record model revision and SHA-256 in local experiment manifests; do not fabricate a hash.

## Stage A: teacher capability baseline

Build a stable task suite before distillation. Include factual questions with sourced answers, reasoning and multi-step tasks, coding tasks with executable tests, debugging, instruction/constraint following, uncertainty handling, and tasks requiring knowledge transfer to new surface forms. Record prompt, model revision, decoding settings, outputs, verifier, latency, and failure category. Freeze a held-out test partition before generating training examples.

## Stage B: create candidate demonstrations

Generate multiple task families, not just short Q/A facts:

1. Direct knowledge with scope, prerequisites, and exceptions.
2. Application tasks where the model must select and use a concept.
3. Multi-step plans with observable intermediate outputs and result checks.
4. Code/debugging examples with tests or reproducible expected behavior.
5. Counterexamples and near-neighbor tasks where the correct procedure differs.
6. Error -> diagnosis -> verified correction examples.
7. Ambiguous or underspecified requests where the correct behavior is to state assumptions, verify, or ask for missing information.
8. Tool-selection and result-verification traces where the environment permits reproducible execution.

Each record should carry sample/task IDs, topic, difficulty, teacher/model revision, prompt-template version, source/provenance, expected verifier, verification status, license/privacy review, and split. Teacher-generated candidates start as pending and ineligible. Teacher confidence or repeated agreement between samples is not independent verification.

## Stage C: verify and split before training

- Code: run unit/integration tests in a controlled environment.
- Math or deterministic transformations: execute an independent checker where possible.
- Factual claims: preserve trustworthy source evidence and check applicability/version.
- Open-ended tasks without a reliable judge: mark uncertain or require human review; do not silently label as ground truth.
- Deduplicate exact and near-duplicate examples. Split by task family/source cluster, not just random rows, to reduce leakage.
- Never use the held-out test set to prompt the teacher, filter candidate targets, or tune the training recipe.

## Stage D: train the student

Start with response-level supervised fine-tuning (SFT) because it works with text outputs and does not require matching teacher/student logits. Train only reviewed eligible examples. For conversational data, render the actual student template and mask loss to the intended assistant target tokens; do not accidentally optimize on user/system text unless explicitly intended by the objective.

Run a small end-to-end smoke test first. Then compare feasible student capacities under a declared training budget. Track effective tokens, unique examples, source distribution, train/eval loss, held-out task success, transfer performance, old-capability retention, wall time, throughput, and peak memory. Larger parameter count is not automatically better; insufficient tokens or optimization can leave capacity unused.

If the custom architecture is not a conventional causal language model, define its tensor/module interfaces and a task-level imitation objective first. Do not force a standard Transformers trainer onto an incompatible architecture and call it native training.

## Stage E: self-learning after baseline distillation

1. Freeze a known-good student checkpoint and evaluate it.
2. Run diagnostics on new tasks; cluster failures by missing knowledge, wrong procedure, failed retrieval, execution bug, or poor uncertainty calibration.
3. Ask the teacher to generate targeted candidates only for these failures.
4. Independently verify, deduplicate, and review candidates; retain provenance.
5. Train a new checkpoint with a mixture of new reviewed examples and replay examples from previous capabilities.
6. Evaluate on the hidden transfer suite, a regression suite, and resource constraints.
7. Promote only if predeclared thresholds pass; otherwise keep the old checkpoint and inspect the failure.

Do not automatically train on the student's own unverified answers. That can reinforce mistakes and cause drift or forgetting. Keep stable, frequently reused procedures in weights; keep rapidly changing or source-sensitive facts in versioned external memory/retrieval.

## Acceptance gates

A distillation run is accepted only if:

- the exact teacher artifact and license are recorded;
- the training corpus has provenance, independent verification status, deduplication, and leakage-safe splits;
- the student training entry point actually runs and saves a restorable checkpoint;
- evaluation includes held-out task use/transfer, not loss alone;
- the new checkpoint is compared against the untrained/student baseline and the teacher baseline;
- old capability regression, calibration, latency, and memory are reported;
- the previous checkpoint remains available for rollback.

## Current status

This is an implementation plan. The existing candidate JSONL batches are small and pending; they are not enough to claim broad 9B-to-student transfer. The current generic SFT harness is only a starting point and must be adapted to the actual AIMENG architecture, assistant-only loss masking, model/tokenizer interface, and independent evaluation before a serious run. No 9B distillation or student training is claimed complete by this document.
