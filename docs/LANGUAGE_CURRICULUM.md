# AIMENG language curriculum: requirements before claiming language ability

## Honest status

The current `DiffusionTextModel` is a character-level next-character predictor with a bounded vocabulary (default 2,048 characters), a context embedding/pooling stage, top-k node activation, several rounds of graph message passing, and a learned halt head. It is a research prototype. It is **not yet a demonstrated reasoning engine**.

The default Colab corpus of up to 50,000 Chinese instruction/answer examples is only a starter dataset. It is not enough to teach broad Chinese, English, factual knowledge, robust dialogue, or general reasoning from scratch. A 500-step run is a pipeline check, not meaningful full training.

## What "diffusion" currently means

In this code, diffusion means repeated message passing along a fixed bidirectional ring graph. Each step updates selected node states and emits a next-character prediction. The code does not yet implement a demonstrated internal planner, a reliable multi-step reasoning objective, tool use, explicit working memory, or a validated self-correction loop. A learned halt probability only chooses a mixture/termination behavior; it does not itself create thought.

To test whether message passing helps, compare against:
1. A parameter-matched non-diffusion character model.
2. The same diffusion model with one propagation step.
3. The full multi-step model.
4. A shuffled-context or ablated-routing control.

Use the same data split, token budget, parameter budget where practical, and seeds. Report held-out loss, task scores, wall time, peak memory, and confidence across multiple seeds. If multi-step propagation does not outperform the controls on held-out tasks, do not call it a reasoning benefit.

## Curriculum layers

Treat the curriculum as multiple datasets with provenance, not one giant undifferentiated text file. Each source must have a reviewed license/terms, version, language tags, preprocessing record, deduplication report, and SHA-256. Keep train/validation/test splits separated by document or source before concatenation. The Colab notebook now runs `scripts/prepare_language_splits.py` to deduplicate exact records before a deterministic 90/5/5 record split, save split hashes/counts, train against `train.txt`, validate against `validation.txt`, and reserve `test.txt` for final evaluation. This does not catch paraphrases or validate source licenses.

| Stage | Material required | What it teaches / tests | Current status |
|---|---|---|---|
| 0. Pipeline smoke | Small Chinese instruction set, up to 50k examples | Confirms downloading, encoding, training, checkpointing and generation execute | Notebook can prepare this; not sufficient |
| 1. Language form | Broad, clean Chinese prose from licensed sources: encyclopedic articles, public-domain works, explanations, news with compatible rights; add English only if multilingual learning is a goal | Character/word patterns, punctuation, sentence structure, discourse | Not yet assembled or audited |
| 2. Instruction following | Diverse user requests with high-quality answers, multi-turn dialogue, summarization, rewriting, extraction and format constraints | Mapping requests to appropriate responses | Starter subset only; needs quality checks and diversity |
| 3. Grounded knowledge | Source-linked factual passages and questions, retrieval-grounded answers, date/version metadata | Answering from evidence rather than memorizing unsupported claims | Not yet assembled |
| 4. Reasoning and procedures | Carefully verified arithmetic, logic, multi-step transformations, planning, code/algorithm tasks where appropriate, with intermediate steps and answer validators | Explicit multi-step task behavior | Not yet assembled; labels need independent validation |
| 5. Safety and refusal | Benign/ambiguous requests, privacy protection, uncertainty, refusal examples for disallowed or harmful requests | Boundaries, calibrated uncertainty, safe behavior | Not yet assembled |
| 6. Evaluation only | Held-out document-level language modeling data and task-specific tests never used for training | Measures generalization and regression | Split-generation pipeline added; test set is reserved, but task-specific benchmark suite and human review are still missing |

Do not blindly combine datasets just because they are large. Remove duplicates and boilerplate, filter broken/empty examples, check language distribution, sample-review answer quality, and keep validation/test records out of training. Data quantity cannot compensate for systematically wrong labels.

## Practical scale and compute discipline

There is no single data size that guarantees language competence; capacity, architecture, optimizer, token budget, data quality and evaluation all matter. The following are **planning checkpoints, not promises of sufficiency**:

- Starter: 50k instruction records to validate the pipeline.
- Data audit: 100k reviewed/deduplicated examples across several task types, plus held-out evaluation. This is a quality/coverage checkpoint, not enough to claim broad language mastery.
- Curriculum pilot: increase to several hundred thousand or more permitted records across language modeling and instruction tasks only after the starter shows real held-out improvement and reasonable throughput.
- Broader capability: requires substantially more diverse text/tokens, reliable labels, separate reasoning/safety data, repeated evaluations, and likely an architecture/tokenizer upgrade. Do not choose a final scale before measuring examples/second, loss curves, RAM/VRAM and quality.

Use Colab to run small, resumable experiments and measure the real throughput. Do not start a huge download or 50k-step run until the 20-step preflight, checkpoint-resume test, validation split, and baseline comparison are sound. Keep raw corpora and large checkpoints in persistent Drive/object storage; put dataset manifests, source/license notes, hashes, configs, metrics, and small samples in Git.

## Architecture blockers to resolve before serious curriculum training

1. **Character-level representation:** A 2,048-character cap maps rare characters to `<unk>` and makes long sequences inefficient. Evaluate a byte/subword tokenizer and explicitly test Chinese coverage before scaling.
2. **Weak context representation:** Current context uses position-scaled embeddings followed by mean pooling. Position is represented, but the model still compresses the entire context to one vector before routing. Compare against a causal sequence baseline with the same data.
3. **Graph definition:** Neighbors are a fixed ring. Learned edge logits weight existing edges, but do not discover arbitrary graph connectivity. Validate whether graph message passing helps over a simpler baseline.
4. **Objective:** Next-character cross entropy teaches local continuation, not verified reasoning or factual truth. Add task-specific supervised objectives only after reliable data and evaluation exist.
5. **Validation:** The trainer still has a legacy contiguous 90/10 fallback when `--validation-text` is omitted. The Colab path now generates record-level train/validation/test files, fits vocabulary on training text only, and logs split hashes. Exact deduplication does not catch paraphrases, and test-set scoring is not yet wired into the notebook, so capability claims remain blocked.
6. **Resource measurement:** Colab GPU memory is not Android memory. The eventual phone runtime must independently remain below the project's 4 GiB peak RAM limit.

## Acceptance gates

Do not describe the model as having learned language or thought until all of these are reported:
- Dataset manifest, license review, corpus hashes, deduplication stats and split counts.
- A learning curve over enough optimizer steps to show stable held-out improvement.
- Generated samples on unseen prompts, scored with a predefined rubric rather than cherry-picked.
- Separate test scores for language continuation, instruction following, knowledge grounding, multi-step tasks and safety.
- Matched baseline and ablation comparisons demonstrating whether multi-step diffusion contributes.
- Repeatability across at least three seeds for any headline claim.
- Actual throughput and memory measurements, plus a persistent checkpoint/recovery test.

Until then, call the work a training pipeline and architecture experiment, not a completed language model or thinking brain.
