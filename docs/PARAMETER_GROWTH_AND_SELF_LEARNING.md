# Parameter Growth + Self-Learning: Knowledge Must Become Usable

## Design objective

AIMENG should not optimize for parameter count or recall of training phrases alone. The target is a student whose parameters internalize reusable representations and procedures, then apply them to new tasks. Parameter growth is a capacity decision; self-learning is a data, optimization, evaluation, and control loop. Neither substitutes for the other.

## The three capabilities to train together

1. **Capacity:** choose a model size that can be trained within the available compute and data budget. Larger parameter count is not evidence of higher capability unless controlled experiments demonstrate it.
2. **Internalization:** use diverse, high-quality supervised examples, explanations where appropriate, counterexamples, corrections, and varied task forms. Track effective training tokens, duplicate rates, topic coverage, and source distribution.
3. **Retrieval and transfer:** train the model to identify relevant concepts, state assumptions, select procedures, execute them, check results, and adapt to changed surface forms. Evaluate on held-out task families, not paraphrases of training items.

## Self-learning loop

1. Evaluate the current student on a fixed training-diagnostic suite and a hidden held-out suite.
2. Cluster failures: missing knowledge, wrong procedure, poor retrieval, invalid tool/code use, uncertainty calibration, or regression.
3. Generate targeted candidate examples from the teacher. Each example records topic, task family, difficulty, source/provenance, expected verifier, and generation/model version.
4. Validate candidates independently. Use executable tests for code/math where possible, trusted sources for factual claims, and human review when needed. Teacher confidence alone is not evidence.
5. Deduplicate and split by source and task family. Keep the held-out test suite out of candidate generation and training.
6. Train a new checkpoint from a preserved baseline, with reproducible config and resource limits.
7. Compare old and new checkpoints on unseen transfer tasks, knowledge-use tasks, calibration, old-capability retention, latency, and peak RAM.
8. Promote only when predeclared gates pass. Otherwise keep the previous checkpoint and inspect regressions.

## How to teach the model to use knowledge

Include task patterns, not just isolated fact cards:

- Question -> identify the relevant concept -> answer with conditions and evidence.
- Problem -> plan -> execute -> test result -> correct mistakes.
- Similar-looking tasks with different correct procedures, to teach discrimination.
- Counterexamples and boundary conditions, to prevent overgeneralization.
- New surface forms and composition of multiple concepts, to test transfer.
- Unknown or underspecified questions, to teach when to ask, verify, or abstain.
- Error traces followed by a verified correction, to teach debugging rather than merely outputting the final answer.

Where process traces are included, they must be useful, concise, verifiable task steps, not fabricated claims that a teacher's explanation is automatically correct.

## Parameter and data scaling experiment

Do not pick a large parameter count by intuition alone. Run controlled experiments at multiple feasible sizes while holding the data snapshot, tokenizer, evaluation suite, and training budget definition stable. Record:

- parameter count and architecture/config revision;
- unique and effective training tokens, epochs, deduplication and source statistics;
- training loss and held-out loss;
- held-out knowledge-use, transfer, reasoning, and coding results;
- old-capability regression and uncertainty calibration;
- wall time, throughput, accelerator memory, peak RAM, and failure rate.

Select the size that yields the best measured capability under the compute budget. More parameters with too few effective tokens or too little optimization can underperform a smaller, adequately trained model.

## Memory versus weights

Parameters are not a reliable editable database. Knowledge that changes frequently, must be cited, or must be precisely updated belongs in versioned external memory/retrieval. Stable reusable patterns can be learned into parameters through reviewed training. Keep the two mechanisms complementary: retrieval provides current evidence; weights provide learned generalization and procedures.

## Safety and resource gates

- Never train directly on unverified self-generated answers.
- Never use the held-out test suite to generate training data or tune prompts.
- Preserve baseline checkpoints and support rollback.
- Measure runtime memory on the actual target device; the project's Android inference budget is a hard peak-RAM limit of 4096 MiB, with headroom required. Training-time resource needs are separate and must be measured on the training host.
- A lower loss is not sufficient for release. Require measurable transfer gains and no unacceptable regression.

## Current state

This document is a design contract, not evidence that the loop is implemented or that any student has been trained. The initial SFT entry point and teacher candidate batches are scaffolding. Next implementation steps are assistant-only loss masking, dataset fingerprinting and leakage checks, held-out capability evaluation, telemetry, and a small reproducible end-to-end training run before scaling.


## Human-like idea generation and bounded plasticity

The broader implementation contract is now specified in [Human-like Thinking and Plasticity](HUMAN_LIKE_THINKING_AND_PLASTICITY.md). It defines the unified-core-model design, speculative idea lifecycle, evidence levels, independent verification, task-state machine, temporary plasticity trials, reusable module manifest, rollback rules, tests, and staged experiments.

Important implementation boundary: the current self-learning loop and any future idea-cycle code do not prove dynamic neuron growth. Implement the software-level idea/verification/experience loop first; test adapters as isolated candidate artifacts next; only then investigate real dynamic subgraph or neuron/connection growth when the actual backend supports it. Never overwrite stable core weights directly from an unverified task result.
