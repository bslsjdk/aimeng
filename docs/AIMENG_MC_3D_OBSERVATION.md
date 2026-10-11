# AIMENG-MC: 3D observation protocol

Status: design draft; not yet validated against a specific Bedrock version or a live device.

## Goal

A Bedrock behavior pack should provide structured world observations to AIMENG so the neural controller does not have to infer block identity from screenshots. A small language model consumes compressed summaries and issues high-level goals; AIMENG uses the structured observation to act and report outcomes.

This document defines the data contract, not a claim that the game-to-model transport or player-input injection already works.

## Separation of responsibilities

1. **Bedrock Add-On / observation adapter**: reads only data exposed by the selected stable Script API; samples a bounded 3D region and player state.
2. **Transport adapter**: transfers observations to the controller. This must be selected and tested separately; behavior packs must not be assumed to have arbitrary socket/network access.
3. **AIMENG controller**: maintains recurrent state, selects low-level actions, predicts outcomes, and emits compact event reports.
4. **Language model**: receives compressed reports and sends goals, constraints, and completion conditions.
5. **Input adapter**: converts controller actions to keyboard/gamepad events where supported. The Add-On itself must not be assumed to synthesize operating-system keyboard/gamepad events.

## Initial local 3D grid

Start with a small player-centred grid rather than scanning the whole world:

- X: 9 cells
- Y: 5 cells
- Z: 9 cells
- Total: 405 voxel cells per observation
- Initial range: 4 blocks horizontally in each direction and 2 blocks vertically in each direction, with coordinates defined relative to the player's floored block position.
- Record player position and rotation separately, so relative grid cells can be transformed into world coordinates.

The dimensions are experimental defaults, not proven optimal. Profile the observation cost before increasing the grid.

Each cell should contain:
- integer dx, dy, dz relative to the anchor;
- stable block_type identifier string (for example, minecraft:oak_log) or a reserved unknown/unavailable value;
- small derived flags when safely available: is_solid, is_liquid, is_air, is_passable.
Do not collapse distinct block types into one class if their behavior differs for the task. Keep the original type identifier in the observation; a learned embedding can map identifiers to vectors inside AIMENG.

## Player and task state

Include only fields available from the chosen API and validate each field before use:

- observation sequence number and game tick/time marker;
- player world position (x/y/z) and rotation (yaw/pitch);
- dimension identifier;
- health and selected hotbar slot only if the required component/property is available;
- active high-level goal and action currently being attempted;
- last action outcome: succeeded / failed / unknown;
- changed blocks and inventory deltas only when they can be observed reliably.

Missing data must be represented explicitly as unknown, not silently fabricated.

## Example message

\`\`\`json
{
  "schema": "aimeng-mc-observation-v1",
  "sequence": 42,
  "anchor": {"x": 120, "y": 64, "z": -35},
  "player": {
    "position": {"x": 120.5, "y": 64.0, "z": -34.5},
    "rotation": {"yaw": 90.0, "pitch": 0.0},
    "dimension": "overworld"
  },
  "grid": {
    "size": [9, 5, 9],
    "origin_offset": [-4, -2, -4],
    "cells": [
      {"offset": [0, 0, 0], "block_type": "minecraft:air"}
    ]
  },
  "last_action": {"id": "move_forward", "result": "unknown"},
  "goal": {"id": "collect_oak_logs", "target_count": 10}
}
\`\`\`

The example shows the schema shape only; the one-cell array is abbreviated and is not a complete 405-cell observation.

## Feeding the network

Do not feed raw strings directly into the recurrent core. Use an encoder:

1. Map each block identifier to a stable vocabulary index; reserve indices for unknown and unavailable.
2. Embed each block type and combine it with the cell's relative position and derived flags.
3. Encode the 3D grid into a compact spatial feature vector (a small 3D CNN or a simpler shared per-cell encoder plus pooling can be compared).
4. Combine spatial features with player pose, recent actions, outcomes, and the current goal.
5. Feed the combined vector to AIMENG's recurrent state.
6. Produce two outputs: low-level action signals and a structured event/status report for the language model.

A baseline using direct structured block IDs should be measured against a screenshot-only vision baseline if visual robustness is a target. Success with block IDs does not prove screenshot-based recognition.

## Training plan

1. **Observation validation**: confirm block IDs and positions against a controlled test world.
2. **Spatial tasks**: predict target direction, traversable cells, and short path actions.
3. **Action-effect prediction**: predict whether an action will move the player, change a target block, or produce no effect.
4. **Closed-loop control**: use demonstrations first, then feedback/reward updates; keep training and test maps separate.
5. **Language-model loop**: measure task success with compressed reports versus full structured state.
6. **Minecraft evaluation**: compare AIMENG with rule-based, RNN, and GRU baselines on the same tasks, seeds, and interaction budget.

Track success rate, steps to completion, collisions/stuck events, invalid actions, report size, end-to-end latency, parameter count, compute, and peak RAM. Android runtime RAM must remain below the project's 4 GB limit.

## Compatibility and safety constraints

- Pin a specific Bedrock version and stable @minecraft/server API version before implementing the pack. Verify every API used against that version.
- Keep the sampled region bounded and throttle observations; do not scan the entire world each tick.
- Handle unloaded/out-of-bounds/unavailable blocks explicitly.
- Use a separate, explicitly tested transport mechanism between the Add-On and AIMENG. Do not assume a behavior pack can open arbitrary sockets or write files.
- Keep observation and action injection as separate adapters. Reading the world successfully does not mean the Add-On can generate real keyboard/gamepad input.
- The existing AIMENG Android runtime format does not automatically support a new recurrent architecture; a compatible exporter/runtime implementation and device profiling will be needed before mobile deployment.
