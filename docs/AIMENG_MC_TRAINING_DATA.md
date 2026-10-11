# AIMENG-MC training data plan

Status: training/data specification. No dataset has been collected yet; all quantities below are targets for a first experiment, not achieved results.

## Do we need real player recordings?

No, not for the first stages. Use three data sources in order:

1. **Automatically labelled synthetic/controlled states**: create small voxel grids with known block IDs, player position, target location, obstacles, and a computed shortest path. These teach coordinate handling and basic navigation without needing Minecraft running.
2. **Add-On observations in a controlled Bedrock test world**: record actual block IDs and player coordinates, then compare them against hand-checked known locations. This validates that the adapter reports the real game state correctly.
3. **Optional player demonstrations and live interaction**: record state/action/result sequences from a consenting player, or have the controller act in a test world and learn from verified outcomes. Player data is useful for natural movement and task strategy, but it is not a prerequisite. Do not collect usernames, chat, or unrelated personal information; record only task-relevant game state and actions.

Synthetic data can teach geometry and action mapping, but cannot alone prove performance in the real game. Bedrock observations with block IDs are structured-state input, not vision.

## Training stages and proposed initial data sizes

| Stage | Task | Initial target |
|---|---|---:|
| A | Locate a named block and predict its relative direction/distance | 20,000 labelled observations |
| B | Mark walkable/blocked cells and predict a short path | 50,000 states with teacher actions |
| C | Predict action outcomes (move, turn, interact, no-op) | 30,000 state/action/result examples |
| D | Collect a small number of target blocks in controlled maps | 5,000 short episodes |
| E | Closed-loop evaluation on unseen maps/seeds | at least 100 episodes per model and seed set |

These are practical starting targets to be revised based on learning curves and generation speed. Split by map layout/seed before generating derived windows so near-identical states cannot leak across train and test. Keep validation and test sets fixed.

## One JSONL record per observation/action step

Store each record as a UTF-8 JSON object on one line. Use stable IDs and explicit missing values. Suggested schema:

\`\`\`json
{
  "schema": "aimeng-mc-step-v1",
  "episode_id": "train-map-00017-episode-0003",
  "step": 12,
  "source": "synthetic",
  "map_seed": 17003,
  "observation": {
    "player_cell": [4, 1, 6],
    "yaw_bucket": 1,
    "goal": {"type": "find_block", "block_type": "minecraft:oak_log"},
    "grid_size": [9, 5, 9],
    "cells": [
      {"offset": [0, 0, 0], "block_type": "minecraft:air"},
      {"offset": [1, 0, 0], "block_type": "minecraft:stone"}
    ]
  },
  "teacher_action": {
    "buttons": ["forward"],
    "duration_ticks": 2
  },
  "outcome": {
    "result": "success",
    "player_moved": true,
    "target_changed": false,
    "stuck": false
  },
  "labels": {
    "target_direction": [1, 0, -1],
    "target_distance": 5.0,
    "action_success": true
  }
}
\`\`\`

This is an abbreviated format example, not a complete 405-cell grid. In real records, include all sampled cells or a lossless sparse encoding with a documented default value. Never interpret omitted/unloaded cells as air.

## How to obtain teacher actions

- For synthetic grid worlds, use breadth-first search or A* on the known walkable map to generate a shortest-path teacher. Randomize maps, obstacles, target locations, and spawn locations.
- For a controlled Bedrock world, first validate block and player coordinates. Then generate simple demonstrations from known target positions, or record manual movement and verify the resulting state.
- For interaction tasks, labels must be based on the observed result (e.g., the target block changed or inventory count changed), not merely on which button was pressed.
- Preserve failed actions. They teach the outcome model and stuck detection; do not silently discard all failures.
- Avoid teaching the model that one fixed coordinate always means a particular block or action. Randomize world coordinates and use player-relative offsets.

## Split and quality checks

- Split by entire map seeds/layout families and episodes, not by randomly shuffling adjacent frames.
- Deduplicate observations or near-identical frames across splits.
- Check every action label against the next state; reject contradictory records.
- Include blocked movement, turning, no-op, unknown block IDs, missing/unloaded cells, and target-not-visible cases.
- Track class balance and success/failure balance.
- Version the schema, generator, map seeds, and action semantics.
- Keep a small manually inspected golden set for checking the Add-On adapter.

## Training objective

Train separate heads so each has a clear measurable task:

1. block/target relative direction and distance;
2. walkable-cell/path-action prediction;
3. next-state/action-effect prediction;
4. action selection for the current goal;
5. optional compressed event report for the language model.

Use supervised learning first, because its labels are testable. Then perform closed-loop fine-tuning from verified rewards/outcomes with a replay buffer. A reward should reflect task progress and completion while penalizing collisions, stuck loops, and invalid actions. Evaluate on held-out maps throughout to detect overfitting and forgetting.

## Required baselines and reporting

Compare AIMENG's proposed neuron-group/concept-routing design against rule-based shortest path, matched-size RNN, and matched-size GRU controllers. Match training data, map seeds, action interface, and interaction budget. Report completion rate, steps, stuck/collision rate, action-effect accuracy, parameter count, training compute/time, end-to-end latency, and peak RAM. Repeat across seeds.

Do not call synthetic-grid success “Minecraft mastery”. Do not call structured block-ID input “visual recognition”. Do not claim continual learning until weights actually update from new verified experiences and a held-out regression test shows what was retained or forgotten.

## Android constraint

The complete runtime must stay below 4 GB RAM. Keep observations bounded, encode only when needed, and measure the whole process on the actual device. The current AIMG wrapper does not automatically export a new model architecture; Android inference and matching export code must be implemented and tested separately.
