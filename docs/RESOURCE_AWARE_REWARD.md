# Resource-aware reward experiment (proposal)

## Purpose

Test whether a controller can learn when extra compute and memory are worth spending to solve a difficult task. This is an evaluation proposal, not a change to the model architecture or a claim that the policy has learned.

## Reward principle

Reward independently verified task success, add a difficulty bonus only for verified successes on tasks that a fixed reference baseline repeatedly fails, and subtract normalized resource costs:

`score = success * (base_reward + difficulty_bonus) - memory_weight * memory_ratio - time_weight * time_ratio`

Where:
- `success` is 1 only when an independent validator passes; otherwise it is 0.
- `difficulty_bonus` is bounded and derived from a held-out baseline success rate, not from the candidate model's self-assessment. For example, `max_bonus * (1 - baseline_success_rate)`.
- `memory_ratio` is measured peak memory divided by a declared task budget.
- `time_ratio` is measured elapsed time divided by a declared time budget.
- The weights and caps must be chosen on a validation set and frozen before the final test.

This formula is only a starting point. Score weights must not allow a difficult-task bonus to override hard safety or resource limits. If a run exceeds a hard memory/compute limit, mark it as a constraint violation and reject it regardless of reward. For mobile deployment, whole-app peak memory must remain below 4096 MiB with safety headroom.

## Difficulty and correctness controls

1. Freeze the reference model/version, prompt, tools, retry policy, and resource budget.
2. Estimate baseline difficulty from repeated runs or a set of reference models; a single failure is not enough.
3. Use task-specific independent validators where possible. Unverifiable answers are `unknown`, not successful.
4. Split tasks by source/task family into train, validation, and test so near-duplicates do not leak across splits.
5. Record correctness, baseline pass rate, peak RAM/VRAM, elapsed time, tokens/steps, timeout/OOM, seed, code revision, and task/data hashes.
6. Compare against simple baselines: correctness-only reward, resource-only penalty, and the combined reward.

## Acceptance criteria

The combined policy is useful only if it improves held-out task success per unit of measured resource or time without degrading overall quality or violating hard limits. Report raw quality and resource metrics separately; never report only the composite score. Repeat across seeds and task families.

## Current status

Design proposal only. The current parallel scheduler runs independent processes and uses declared memory estimates; it does not enforce OS-level memory isolation, synchronize gradients, or automatically learn this reward. The toy logistic-regression example tests orchestration only, not language-model reasoning or this reward policy.
