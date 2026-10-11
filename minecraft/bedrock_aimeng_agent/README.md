# AIMENG Parkour Agent (Bedrock)

This pack adds a separate humanoid test entity that can be spawned in a single-player world. It does not take over the human player's controls.

## Install
Import the combined .mcaddon package and activate both the AIMENG Parkour Agent behavior pack and its resource pack. The behavior pack depends on the resource pack UUID included here.

## Commands
- `/summon aimeng:parkour_agent ~ ~ ~` creates a visible agent.
- `/scriptevent aimeng:agent_goto 120 65 -30` sends the nearest agent to an explicit world coordinate.
- `/scriptevent aimeng:agent_goto ~ ~ ~10` sends it 10 blocks forward along the world Z axis from its current position.
- `/scriptevent aimeng:agent_start` starts a short 8-block waypoint test.
- `/scriptevent aimeng:agent_status` reports the nearest agent's state and goal coordinates.
- `/scriptevent aimeng:agent_stop` stops the nearest agent.

Coordinate tokens accept absolute numbers or relative syntax such as `~` and `~5`. The target entity is the nearest agent within 32 blocks of the player.

The current controller is a heuristic smoke test, not the trained AIMENG neural network. It moves a real entity under Minecraft physics. Learned model export, exact observation encoding, and in-game success-rate measurements remain future work.

## Notes
- Intended for Minecraft Bedrock with Script API support and engine 1.21.90 or compatible.
- If the texture path `textures/entity/steve` is not available to custom entities in your game version, replace it with a texture included in a resource pack.
- Always test in a copy of a world first.
