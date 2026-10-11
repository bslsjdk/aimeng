# AIMENG-MC curriculum training guide

## What runs where

- Dataset generation and evaluation-gate inspection use only Python's standard library.
- The recurrent policy trainer requires PyTorch and should run in GitHub Actions or another training host, not be assumed to work in an Android terminal.
- The current controller receives a structured 405-cell voxel observation (9×5×9), relative target offset and a four-way heading bucket. It is not image-based vision. The curriculum generator varies the easy walking stage's start position, heading and target distance, while keeping its only unlocked actions as walk-forward and stop. The controller now has a separate geometry-to-action path for relative target offset, forward/lateral alignment, distance and the at-goal signal, alongside the recurrent hidden state.
- The trained checkpoint is a cloud-side `.pt` research checkpoint. It is not an AIMG package, not an Android app model, and not connected to live Bedrock action control yet.

## Recommended first run on a phone

Open the repository on GitHub, choose **Actions → AIMENG-MC curriculum training → Run workflow**. Start with:

- Maximum stage: `walking`
- Episodes per stage: `1000`
- Epochs per stage: `15`
- Initial recurrent hidden units: `1024`
- Maximum hidden units: `2048`
- Automatic expansion: enabled

The workflow is designed to train stages in order, save a checkpoint after each stage, replay earlier-stage data, re-evaluate earlier skills, and stop if the progression gate declines to unlock the next stage. It uploads checkpoint and metrics artifacts after the run. For a cheap code/test check, push/PR CI uses only a 20-episode, 2-epoch smoke run and a 16-unit hidden state; that smoke run is **not** a meaningful learning benchmark.

The workflow input `max_stage` lets you select the highest level to attempt. If the policy has not mastered walking, the workflow stops at walking even if a harder target was selected.

## Continue a stage manually

Generate one stage's data:

```bash
python scripts/generate_aimeng_mc_curriculum.py --stage walking --episodes 1000 --seed 12345 --output data/walking.jsonl
```

Train a fresh model on a host with PyTorch installed:

```bash
python scripts/train_aimeng_mc_curriculum.py --stage walking --data data/walking.jsonl --epochs 5 --hidden-size 1024 --max-hidden-size 2048 --auto-expand --output checkpoints/walking.pt --report reports/walking.json
```

For the next stage, create a new dataset and load the previous checkpoint while replaying all earlier training datasets. Pass the previous report so historical mastery scores are retained:

```bash
python scripts/generate_aimeng_mc_curriculum.py --stage turning --episodes 1000 --seed 22345 --output data/turning.jsonl
python scripts/train_aimeng_mc_curriculum.py --stage turning --data data/turning.jsonl --resume checkpoints/walking.pt --prior-report reports/walking.json --replay-data data/walking.jsonl --epochs 5 --auto-expand --max-hidden-size 2048 --output checkpoints/turning.pt --report reports/turning.json
python scripts/generate_aimeng_mc_curriculum.py --metrics reports/turning.json
```

The model expands only when `--auto-expand` is enabled and the **current stage's** training and validation action accuracy both remain low, have a small gap, and validation has plateaued for multiple epochs. Expansion doubles the hidden state up to `--max-hidden-size`, transfers old recurrent parameters so the immediate logits stay almost unchanged, and creates a fresh optimizer. Replaying prior stages reduces forgetting, but the measured retention score still decides whether progression is allowed.

## What the report means

The data is split by whole map/episode seed, never adjacent frames: 75% train, 10% validation, 10% final test, and 5% retention audit. With 1,000 episodes this yields 750 training, 100 validation, 100 final-test, and 50 retention episodes per stage.

The trainer reports two different metrics:

- **Teacher-action accuracy/recall**: how often the policy predicts the teacher's action on labeled observations, with per-action recall exposing rare-action failures.
- **Episode success rate**: whether the model actually navigates a reconstructed symbolic course and emits stop at the goal within the step budget.

Checkpoint selection and stage progression use validation episode success, while the final test split is reserved for reporting final results. Before unlocking the next stage, require at least 90% validation success over at least 100 episodes. After learning a new stage, previously learned skills must still achieve at least 85% success over at least 50 separate retention-audit episodes each. Otherwise, repeat training or investigate forgetting/data quality before adding difficulty.

These are grid-world results. The teacher and rollout evaluator share the same symbolic movement rules, so this validates implementation consistency and learning in that simulator, not real Minecraft skill. Actual Bedrock jump distances, collision, momentum, turn timing, special jump abilities, and action delivery must be calibrated against the target game/version. In particular, super jump and double jump require a custom mechanic unless the chosen map/mod already provides one.

## Memory constraints

All stages retain the same 405 voxel IDs and one-step structured observation, so a harder stage does not automatically increase model input size. The JSONL generator streams records and the trainer indexes episode byte ranges instead of loading the entire dataset into memory. More episodes mainly cost disk space and training time.

For this trainer's dense GRU policy, parameter storage is dominated by the recurrent matrix and scales roughly quadratically with hidden units. Approximate float32 **weight-only** storage is around 16 MB at 1024 units and 55 MB at 2048 units; these are estimates for this prototype, not measured phone RAM. Runtime also needs the embedding, encoder, buffers, framework overhead, and hidden state. Training requires additional gradients and optimizer state. Every deployed Android build must still be profiled end-to-end and stay under 4 GiB RAM.

## Deployment is a separate milestone

The repository's AIMG packer only understands the existing portable mobile format; it cannot package this new PyTorch architecture as-is. Before this policy can control a phone or a real Bedrock world, implement and test a compatible exporter/runtime controller, map its high-level actions to verified game controls, and measure actual latency, task success, and peak RAM. Do not report synthetic rollout success as live Minecraft parkour ability.


## Action imbalance and curriculum action masking

The first 1,000-episode walking benchmark exposed a real problem: raw per-step action accuracy was about 83%, but only 23 of 100 held-out episodes reached the goal and stopped. The model often learned the majority action (walking) better than the rare but essential stop decision. Expanding 1,024 to 2,048 hidden units did not by itself solve that failure.

The trainer now uses inverse-frequency class weights for actions in the training split, reports per-action recall, masks locked future-stage actions during training and evaluation, and selects checkpoints primarily by **closed-loop validation episode success**. The controller has a separate geometry-to-action head for goal offset, forward/lateral alignment, distance and the at-goal flag, alongside the recurrent hidden state. In the walking stage only `walk_forward` and `stop` are allowed; turning and jump variants are unlocked at later stages.

The first fixed-start prototype scored 23% closed-loop test success, showing that raw action accuracy can be misleading. A later 1,000-episode run with randomized starts/headings/distances and the geometry-head policy achieved **100/100 validation episodes and 98/100 final-test episodes**, using 1,024 hidden units without expanding the model. These are separate prototype revisions rather than a controlled architecture ablation, but the newest run is a successful result for the synthetic straight-walking task. No claim is made about live Bedrock yet.

The current benchmark workflow then tries the turning stage only when walking passes its gate, carries forward the walking checkpoint, replays walking samples, and audits walking retention on a separate map split. No stage advances merely because action accuracy or hidden-unit count rose.
