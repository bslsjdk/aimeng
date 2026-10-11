#!/usr/bin/env python3
"""Generate labelled, deterministic AIMENG-MC navigation data using only Python's standard library.

This is a synthetic voxel-grid teacher dataset, not a Minecraft capture and not a trained model.
The teacher uses BFS over (x, z, yaw); the student sees only a bounded 9x5x9 voxel observation
and the relative waypoint. Use held-out seeds/maps for evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Iterable

# Voxel IDs are stable within schema aimeng-mc-step-v1.
AIR = 0
GRASS = 1
STONE = 2
DIRT = 3
OAK_LOG = 4
LEAVES = 5
UNKNOWN = 6
BLOCK_NAMES = {
    AIR: "minecraft:air",
    GRASS: "minecraft:grass_block",
    STONE: "minecraft:stone",
    DIRT: "minecraft:dirt",
    OAK_LOG: "minecraft:oak_log",
    LEAVES: "minecraft:oak_leaves",
    UNKNOWN: "unknown",
}
# yaw buckets: 0=north (-z), 1=east (+x), 2=south (+z), 3=west (-x)
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTIONS = ("forward", "turn_left", "turn_right", "stop")
GRID_X, GRID_Y, GRID_Z = 9, 5, 9
RADIUS = 4


def bfs_teacher(
    walkable: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int], yaw: int
) -> list[tuple[tuple[int, int], int, str]]:
    """Return (position, yaw, action) steps along a shortest route, or [] if unreachable."""
    initial = (start[0], start[1], yaw % 4)
    queue = deque([initial])
    parent: dict[tuple[int, int, int], tuple[tuple[int, int, int], str] | None] = {initial: None}
    final = None
    while queue:
        state = queue.popleft()
        x, z, facing = state
        if (x, z) == goal:
            final = state
            break
        dx, dz = DIRS[facing]
        candidates = (
            ((x + dx, z + dz, facing), "forward"),
            ((x, z, (facing - 1) % 4), "turn_left"),
            ((x, z, (facing + 1) % 4), "turn_right"),
        )
        for nxt, action in candidates:
            if action == "forward" and (nxt[0], nxt[1]) not in walkable:
                continue
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    if final is None:
        return []
    reversed_steps = []
    cur = final
    while parent[cur] is not None:
        prev, action = parent[cur]  # type: ignore[misc]
        reversed_steps.append(((prev[0], prev[1]), prev[2], action))
        cur = prev
    reversed_steps.reverse()
    reversed_steps.append(((goal[0], goal[1]), final[2], "stop"))
    return reversed_steps


def make_world(rng: random.Random, obstacle_rate: float = 0.18):
    """Create a 9x9 walkable map with random solid obstacles and a reachable nearby waypoint."""
    coords = [(x, z) for z in range(GRID_Z) for x in range(GRID_X)]
    for _ in range(100):
        start = (GRID_X // 2, GRID_Z // 2)
        goal = (rng.randrange(GRID_X), rng.randrange(GRID_Z))
        if max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) > RADIUS:
            continue
        blocked = {
            p for p in coords
            if p != start and p != goal and rng.random() < obstacle_rate
        }
        walkable = set(coords) - blocked
        yaw = rng.randrange(4)
        route = bfs_teacher(walkable, start, goal, yaw)
        if route:
            # Obstacles get varied solid identities to discourage a single-ID shortcut.
            solids = {p: rng.choice((STONE, DIRT, OAK_LOG, LEAVES)) for p in blocked}
            return walkable, blocked, solids, start, goal, yaw, route
    # Guaranteed fallback: empty map, with a nearby goal.
    goal = (GRID_X // 2 + 1, GRID_Z // 2)
    walkable = set(coords)
    return walkable, set(), {}, (GRID_X // 2, GRID_Z // 2), goal, 0, bfs_teacher(
        walkable, (GRID_X // 2, GRID_Z // 2), goal, 0
    )


def observe(
    x: int, z: int, yaw: int, goal: tuple[int, int],
    blocked: set[tuple[int, int]], solids: dict[tuple[int, int], int],
    source: str, map_seed: int, episode_id: str, step: int,
) -> dict:
    """Build a player-centred 9x5x9 voxel observation, flattened y,z,x."""
    voxels = []
    for y in range(GRID_Y):
        for lz in range(-RADIUS, RADIUS + 1):
            for lx in range(-RADIUS, RADIUS + 1):
                wx, wz = x + lx, z + lz
                outside_map = not (0 <= wx < GRID_X and 0 <= wz < GRID_Z)
                if y == 0:
                    voxels.append(GRASS)
                elif y in (1, 2) and (outside_map or (wx, wz) in blocked):
                    voxels.append(solids.get((wx, wz), STONE))
                else:
                    voxels.append(AIR)
    return {
        "schema": "aimeng-mc-step-v1",
        "episode_id": episode_id,
        "step": step,
        "source": source,
        "map_seed": map_seed,
        "observation": {
            "grid_size": [GRID_X, GRID_Y, GRID_Z],
            "voxel_order": "y,z,x",
            "block_vocab": BLOCK_NAMES,
            "voxels": voxels,
            "player_offset": [0, 1, 0],
            "yaw_bucket": yaw,
            "goal_offset": [goal[0] - x, 0, goal[1] - z],
            "goal_type": "waypoint",
        },
    }


def generate_records(seed: int, episodes: int, obstacle_rate: float = 0.18) -> Iterable[dict]:
    rng = random.Random(seed)
    for episode in range(episodes):
        map_seed = rng.randrange(0, 2**31)
        world_rng = random.Random(map_seed)
        walkable, blocked, solids, start, goal, yaw, route = make_world(world_rng, obstacle_rate)
        # Each route state becomes one supervised sample. Train/test separation must be by map seed.
        for step, (position, facing, action) in enumerate(route):
            x, z = position
            record = observe(
                x, z, facing, goal, blocked, solids, "synthetic", map_seed,
                f"seed-{map_seed}-episode-{episode}", step,
            )
            record["teacher_action"] = {"action": action, "duration_ticks": 1}
            record["labels"] = {
                "target_direction_xz": [goal[0] - x, goal[1] - z],
                "at_goal": position == goal,
                "action_success": True if action in ("turn_left", "turn_right", "stop") else
                    ((x + DIRS[facing][0], z + DIRS[facing][1]) in walkable),
            }
            yield record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run fast schema/reproducibility checks and exit")
    parser.add_argument("--episodes", type=int, default=1000, help="number of random maps/episodes")
    parser.add_argument("--seed", type=int, default=12345, help="reproducible master seed")
    parser.add_argument("--obstacle-rate", type=float, default=0.18, help="solid obstacle probability")
    parser.add_argument("--output", type=Path, default=Path("aimeng_mc_train.jsonl"))
    args = parser.parse_args()
    if args.self_test:
        records = list(generate_records(123, 5))
        assert records, "generator produced no records"
        assert all(len(r["observation"]["voxels"]) == GRID_X * GRID_Y * GRID_Z for r in records)
        assert all(r["teacher_action"]["action"] in ACTIONS for r in records)
        assert all(r["labels"]["at_goal"] == (r["teacher_action"]["action"] == "stop") for r in records)
        assert records == list(generate_records(123, 5)), "same seed did not reproduce identical records"
        print(json.dumps({"self_test": "passed", "records_checked": len(records), "voxel_cells_per_record": GRID_X * GRID_Y * GRID_Z}))
        return 0
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if not 0.0 <= args.obstacle_rate <= 0.45:
        parser.error("--obstacle-rate must be between 0 and 0.45")

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for record in generate_records(args.seed, args.episodes, args.obstacle_rate):
            out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({
        "output": str(args.output),
        "master_seed": args.seed,
        "episodes_requested": args.episodes,
        "records_written": count,
        "schema": "aimeng-mc-step-v1",
        "note": "synthetic waypoint navigation only; not Minecraft capture or model training",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
