# AIMENG Live Parkour Bridge for Minecraft Bedrock

This is the first in-game integration scaffold. It observes a real Bedrock player entity and applies motion impulses so Minecraft physics, gravity, and collision resolve movement.

## Honest status

- Implemented here: behavior-pack manifest, Script API event bridge, live position/velocity/rotation/ground-support observations, movement impulses, waypoint completion/timeout, status and stop commands.
- Current controller: a deterministic heuristic used only to smoke-test the in-game observation/action loop.
- Not implemented yet: importing the trained PyTorch checkpoint, converting its GRU/embedding/geometry-head weights to a Bedrock JavaScript inference format, exact model parity, online learning, or measured in-game success rate.
- This is not teleport-based navigation. The heuristic is not AIMENG's learned policy, so its results must not be presented as neural-network performance.

## Requirements

- Minecraft Bedrock with Script API support, minimum engine version 1.21.90.
- Enable the behavior pack for a test world. If the game requires it, enable Beta APIs / Script API experiments for that version.
- The manifest pins @minecraft/server 2.0.0. If the installed game does not provide that module version, update the dependency to a version supported by that exact game release.

## Quick smoke test

1. Copy this folder as a behavior pack into the Bedrock development behavior-packs directory, or package the folder as a .mcpack.
2. Activate AIMENG Live Parkour Bridge in a disposable flat test world.
3. Stand on a broad, flat platform with at least 10 blocks of clear space in the direction you face.
4. Run /scriptevent aimeng:start. The bridge fixes a waypoint about 8 blocks ahead and begins applying small movement impulses.
5. Run /scriptevent aimeng:status to inspect action count and state.
6. Run /scriptevent aimeng:stop to stop the controller.

The current policy follows a waypoint and attempts to jump if the support block ahead is absent. Test on flat ground first; this first iteration is a bridge smoke test, not a parkour solver.

## Next integration step: learned policy

The training model's inputs are a 9×5×9 voxel tensor, yaw bucket, and 3D goal offset. The current bridge emits raw nearby support/position/velocity data, so a calibrated observation encoder must map real Bedrock observations into the exact training schema. Then export the trained PyTorch model into a compact JSON weight format and implement matching embedding + encoder + GRUCell + linear and geometry heads in JavaScript. Compare Python and JavaScript logits on identical fixtures before allowing the learned model to control a live world.
