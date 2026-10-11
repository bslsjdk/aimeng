# AIMENG-MC model scaling proposal

Status: proposed experiment sizes, not measured winners. This proposal supersedes the idea that a tiny randomly initialized recurrent core alone should solve the full Minecraft task.

## Why scale alone is not enough

Minecraft control requires spatial perception, persistent state, action selection, and prediction of action outcomes. A larger randomly initialized network does not acquire those skills by existing. It needs a supervised curriculum and closed-loop feedback, with held-out maps and seeds.

The 3D Add-On observation protocol is an oracle-like structured input: it can expose block identifiers directly. Results from this input must be labelled **structured-state control**, not visual recognition.

## Proposed modular model

1. **Block embedding**: stable vocabulary IDs mapped to learned vectors; reserve unknown/unavailable IDs.
2. **Spatial encoder**: small 3D CNN over the bounded local voxel grid, outputting a spatial feature vector.
3. **State fusion**: combine spatial features with player pose, goal embedding, recent action, and previous outcome.
4. **Recurrent core**: GRU/RNN state retains useful history across observations.
5. **Action head**: discrete movement/turn/use/attack/stop outputs and optional action duration.
6. **Outcome/report heads**: predict movement success, target interaction success, and a compact structured report for the language model.

Keep these modules separable so that the spatial encoder can be evaluated independently and the recurrent core can be swapped for a matched GRU baseline.

## Capacity sweep

Run all sizes on the same tasks, map seeds, observation cadence, training examples, and interaction budget.

| Variant | Spatial channels | Recurrent hidden size | Purpose |
|---|---:|---:|---|
| S | 16 / 32 | 256 | Debugging and cheap baseline |
| M | 16 / 32 / 64 | 512 | Recommended first serious model |
| L | 32 / 64 / 96 | 1024 | Capacity test after M is stable |

These are candidate sizes, not claims about an optimum. Start with M for task development, but keep S as the control. Only spend resources on L if M's learning curves show underfitting rather than bad observations, reward design, or training bugs.

## Training curriculum

1. **Spatial probes**: locate a named block relative to the player; classify traversable cells; estimate target direction and distance.
2. **Short-horizon movement**: reach a marked location in bounded maps, with collision and stuck feedback.
3. **Interaction**: aim at and interact with a target block; predict whether the action should succeed.
4. **Multi-step collection**: collect a small target set while avoiding obstacles and keeping state across turns.
5. **Language-model handoff**: compare full structured observations with compressed reports; score end-to-end task success, not report quality alone.
6. **Continual correction**: use verified outcomes as labelled updates; retain a replay buffer and held-out evaluation to reduce catastrophic forgetting.

Use scripted demonstrations or a deterministic teacher for early stages, then feedback/reward learning. Random exploration alone is a poor teacher for a large action space.

## Required comparisons

- Rule-based controller.
- Small RNN and GRU with approximately matched trainable parameters.
- AIMENG recurrent/concept-cloud core with its spatial encoder.
- If claiming vision capability, a screenshot-only encoder baseline.

Keep training and evaluation maps/seeds separate. Report success rate, median steps, stuck/collision rate, invalid actions, action-effect prediction accuracy, parameter count, training compute/time, end-to-end latency, and peak runtime RAM. Repeat over multiple seeds and publish raw results.

## Android budget

The hard runtime RAM limit is below 4 GB. The model weights for a 1–5 million parameter FP32 model alone occupy roughly 4–20 MB, but activations, game state, Java/Kotlin runtime, model runtime, buffers, and the game itself add memory. Measure the complete process on the target phone; parameter storage estimates are not a RAM guarantee.

Do not choose a model solely by parameter count. Measure full observation-to-action latency and task completion. Keep the local voxel window bounded and avoid scanning or encoding it more often than necessary.

## Implementation gates

- Verify the Bedrock Script API can read the required player and block data in the exact target game version.
- Build a bounded observation adapter and test coordinates/block IDs in a controlled world.
- Validate the observation transport and action injection independently.
- Implement a trainable encoder + recurrent core in the training environment.
- Export only after defining a portable schema and implementing matching Android inference math. The current AIMG v1 wrapper is a delivery format for the existing portable schema, not automatic support for a new architecture.
- Do not claim real Minecraft competence until the model completes held-out in-game tasks through the complete observation/action loop.
