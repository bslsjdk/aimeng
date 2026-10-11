# AIMENG Parkour Agent (Bedrock)

This pack adds a separate humanoid test entity that can be spawned in a single-player world. It does not take over the human player's controls.

## Install
Import the combined .mcaddon package and activate both the AIMENG Parkour Agent behavior pack and its resource pack. The behavior pack depends on the resource pack UUID included here.

## Commands
- `/summon aimeng:parkour_agent ~ ~ ~` creates a visible agent.
- `/scriptevent aimeng:agent_start` starts the test controller for the nearest agent within 32 blocks.
- `/scriptevent aimeng:agent_status` reports the nearest agent's state.
- `/scriptevent aimeng:agent_stop` stops the nearest agent.

The current controller is a heuristic smoke test, not the trained AIMENG neural network. It moves a real entity under Minecraft physics and follows a short waypoint. Learned model export, exact observation encoding, and in-game success-rate measurements remain future work.

## Notes
- Intended for Minecraft Bedrock with Script API support and engine 1.21.90 or compatible.
- If the texture path `textures/entity/steve` is not available to custom entities in your game version, replace it with a texture included in a resource pack.
- Always test in a copy of a world first.
