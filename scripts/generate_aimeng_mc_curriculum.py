#!/usr/bin/env python3
"""Generate synthetic AIMENG-MC parkour curricula with deterministic teacher actions.

Uses only Python's standard library. This is a symbolic grid simulator, not Minecraft
physics and not evidence that these jump macros are available in vanilla Bedrock.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

GRID_X, GRID_Y, GRID_Z = 9, 5, 9
RADIUS = 4
WORLD_SIZE = 15
START = (7, 7)
# yaw buckets: 0=north, 1=east, 2=south, 3=west
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
AIR, GRASS, STONE, DIRT, OAK_LOG, LEAVES, UNKNOWN = range(7)
BLOCK_NAMES = {
    AIR: "minecraft:air",
    GRASS: "minecraft:grass_block",
    STONE: "minecraft:stone",
    DIRT: "minecraft:dirt",
    OAK_LOG: "minecraft:oak_log",
    LEAVES: "minecraft:oak_leaves",
    UNKNOWN: "unknown_unloaded_or_outside",
}
ACTIONS = (
    "walk_forward", "turn_left", "turn_right", "jump_forward", "gap_jump_forward",
    "rotating_jump_left", "rotating_jump_right", "super_jump_forward",
    "double_jump_forward", "stop",
)


@dataclass(frozen=True)
class Stage:
    name: str
    index: int
    obstacle_rate: float = 0.0
    gap_distance: int = 0
    jump_action: str | None = None
    rotating_jump: bool = False
    special_mechanic: bool = False
    description: str = ""


STAGES = (
    Stage("walking", 0, description="Flat ground, straight walking and stopping."),
    Stage("turning", 1, description="Flat ground, controlled left/right turns and route following."),
    Stage("obstacle_avoidance", 2, obstacle_rate=0.12,
          description="Walk around a blocking obstacle without jumping."),
    Stage("small_jump", 3, gap_distance=2, jump_action="jump_forward",
          description="A single one-cell gap; two-cell jump macro."),
    Stage("gap_jump", 4, gap_distance=3, jump_action="gap_jump_forward",
          description="A wider two-cell gap; three-cell jump macro."),
    Stage("rotating_jump", 5, gap_distance=2, rotating_jump=True, special_mechanic=True,
          description="Jump across a gap while rotating the landing heading by 90 degrees."),
    Stage("super_jump", 6, gap_distance=4, jump_action="super_jump_forward", special_mechanic=True,
          description="A three-cell gap; four-cell extended-jump macro."),
    Stage("double_jump", 7, gap_distance=5, jump_action="double_jump_forward", special_mechanic=True,
          description="A four-cell gap; five-cell double-jump macro requiring a custom mechanic."),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
MIN_STAGE_EVAL_EPISODES = 100
MIN_RETENTION_EPISODES = 50
MASTERY_SUCCESS_RATE = 0.90
RETENTION_SUCCESS_RATE = 0.85


def _in_world(pos: tuple[int, int]) -> bool:
    return 0 <= pos[0] < WORLD_SIZE and 0 <= pos[1] < WORLD_SIZE


def _route_neighbors(
    state: tuple[int, int, int],
    platforms: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
    stage: Stage,
):
    x, z, yaw = state
    dx, dz = DIRS[yaw]
    forward = (x + dx, z + dz)
    if forward in platforms and forward not in obstacles:
        yield (forward[0], forward[1], yaw), "walk_forward"

    yield (x, z, (yaw - 1) % 4), "turn_left"
    yield (x, z, (yaw + 1) % 4), "turn_right"

    jump_specs: list[tuple[int, str, int]] = []
    if stage.rotating_jump:
        jump_specs = [(2, "rotating_jump_left", -1), (2, "rotating_jump_right", 1)]
    elif stage.gap_distance and stage.jump_action:
        jump_specs = [(stage.gap_distance, stage.jump_action, 0)]

    for distance, action, yaw_delta in jump_specs:
        land = (x + dx * distance, z + dz * distance)
        if land not in platforms or land in obstacles:
            continue
        if not _in_world(land):
            continue
        # The synthetic teacher treats jump macros as high-level transitions. Solid blocks
        # along the arc are rejected; gaps may be crossed, but landing must be supported.
        clear = True
        for offset in range(1, distance):
            mid = (x + dx * offset, z + dz * offset)
            if not _in_world(mid) or mid in obstacles:
                clear = False
                break
        if clear:
            yield (land[0], land[1], (yaw + yaw_delta) % 4), action


def shortest_teacher_route(
    start: tuple[int, int], goal: tuple[int, int], start_yaw: int,
    platforms: set[tuple[int, int]], obstacles: dict[tuple[int, int], int], stage: Stage,
) -> list[tuple[tuple[int, int, int], str, tuple[int, int, int]]]:
    """BFS shortest route. Each tuple is (state-before, action, state-after)."""
    initial = (start[0], start[1], start_yaw % 4)
    queue = [initial]
    cursor = 0
    parent: dict[tuple[int, int, int], tuple[tuple[int, int, int], str] | None] = {initial: None}
    final = None
    while cursor < len(queue):
        state = queue[cursor]
        cursor += 1
        if state[:2] == goal:
            final = state
            break
        for nxt, action in _route_neighbors(state, platforms, obstacles, stage):
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    if final is None:
        return []
    reversed_steps = []
    current = final
    while parent[current] is not None:
        previous, action = parent[current]  # type: ignore[misc]
        reversed_steps.append((previous, action, current))
        current = previous
    reversed_steps.reverse()
    reversed_steps.append((final, "stop", final))
    return reversed_steps


def make_course(stage: Stage, seed: int):
    """Create a solvable seeded world; jump stages contain a compulsory gap band."""
    rng = random.Random(seed)
    platforms = {(x, z) for x in range(WORLD_SIZE) for z in range(WORLD_SIZE)}
    obstacles: dict[tuple[int, int], int] = {}
    start_x, start_z = START
    start = START
    jump_distance = stage.gap_distance

    if jump_distance:
        # Removing complete columns prevents the agent from walking around the gap.
        for x in range(start_x + 1, start_x + jump_distance):
            for z in range(WORLD_SIZE):
                platforms.discard((x, z))
        landing_x = start_x + jump_distance
        goal = (min(WORLD_SIZE - 2, landing_x + 1), max(2, min(WORLD_SIZE - 3, start_z + rng.choice((-2, -1, 0, 1, 2)))))
        start_yaw = 1  # face east, across the compulsory gap
        # Reserve a clear launch/landing corridor; random props live away from this band.
        safe_x = set(range(start_x - 1, goal[0] + 1))
        for pos in list(platforms):
            x, z = pos
            if x in safe_x and abs(z - start_z) <= 2:
                continue
            if rng.random() < stage.obstacle_rate:
                obstacles[pos] = rng.choice((1, 1, 2))
    elif stage.name == "walking":
        distance = rng.randrange(2, 6)
        goal = (start_x + distance, start_z)
        start_yaw = 1
    elif stage.name == "turning":
        goal = (start_x + 4, start_z - 3)
        start_yaw = 1
    else:
        goal = (start_x + 5, start_z)
        start_yaw = 1
        # A compulsory block on the straight line forces a safe turn and detour.
        obstacles[(start_x + 2, start_z)] = 1
        safe_corridor = {
            (x, z)
            for x in range(start_x, goal[0] + 1)
            for z in (start_z - 1, start_z, start_z + 1)
            if _in_world((x, z))
        }
        for pos in list(platforms):
            if pos in safe_corridor or pos in obstacles:
                continue
            if rng.random() < stage.obstacle_rate:
                obstacles[pos] = rng.choice((1, 1, 2))

    # Retry a few deterministic per-episode seeds if random props happen to block the route.
    route = shortest_teacher_route(start, goal, start_yaw, platforms, obstacles, stage)
    if not route:
        return make_course(stage, seed + 1)
    if stage.gap_distance and not any(action in {
        "jump_forward", "gap_jump_forward", "rotating_jump_left", "rotating_jump_right",
        "super_jump_forward", "double_jump_forward",
    } for _, action, _ in route):
        return make_course(stage, seed + 1)
    if stage.name == "obstacle_avoidance" and not any(
        action in ("turn_left", "turn_right") for _, action, _ in route
    ):
        return make_course(stage, seed + 1)
    if stage.name == "turning" and not any(
        action in ("turn_left", "turn_right") for _, action, _ in route
    ):
        return make_course(stage, seed + 1)
    return platforms, obstacles, start, goal, start_yaw, route


def make_observation(
    pos: tuple[int, int], yaw: int, goal: tuple[int, int],
    platforms: set[tuple[int, int]], obstacles: dict[tuple[int, int], int],
    stage: Stage, map_seed: int, episode_id: str, step: int, split: str,
) -> dict:
    x, z = pos
    voxels = []
    for y in range(GRID_Y):
        for lz in range(-RADIUS, RADIUS + 1):
            for lx in range(-RADIUS, RADIUS + 1):
                wx, wz = x + lx, z + lz
                world_pos = (wx, wz)
                if not _in_world(world_pos):
                    voxels.append(UNKNOWN)
                elif y == 0:
                    voxels.append(GRASS if world_pos in platforms else AIR)
                elif world_pos in obstacles and y <= obstacles[world_pos]:
                    voxels.append(STONE if obstacles[world_pos] == 1 else OAK_LOG)
                else:
                    voxels.append(AIR)
    return {
        "schema": "aimeng-mc-curriculum-step-v1",
        "episode_id": episode_id,
        "step": step,
        "map_seed": map_seed,
        "split": split,
        "curriculum_stage": {"name": stage.name, "index": stage.index},
        "observation": {
            "grid_size": [GRID_X, GRID_Y, GRID_Z],
            "voxel_order": "y,z,x",
            "block_vocab": BLOCK_NAMES,
            "voxels": voxels,
            "player_offset": [0, 1, 0],
            "yaw_bucket": yaw,
            "goal_offset": [goal[0] - x, 0, goal[1] - z],
            "goal_type": "parkour_waypoint",
        },
        "teacher_action": None,
        "labels": None,
        "data_contract_note": "Synthetic grid only; do not treat this as calibrated Bedrock physics.",
    }


def generate_records(stage_name: str, seed: int, episodes: int) -> Iterable[dict]:
    if stage_name not in STAGE_BY_NAME:
        raise ValueError(f"unknown stage: {stage_name}")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    stage = STAGE_BY_NAME[stage_name]
    master_rng = random.Random(seed)
    for episode_index in range(episodes):
        map_seed = master_rng.randrange(0, 2**31)
        platforms, obstacles, start, goal, start_yaw, route = make_course(stage, map_seed)
        episode_id = f"{stage.name}-seed-{map_seed}-ep-{episode_index}"
        # Split by whole episode/map seed, never by adjacent frames.
        split = "test" if episode_index % 10 == 9 else ("validation" if episode_index % 10 == 8 else "train")
        for step, (state, action, next_state) in enumerate(route):
            x, z, yaw = state
            nx, nz, next_yaw = next_state
            record = make_observation(
                (x, z), yaw, goal, platforms, obstacles, stage,
                map_seed, episode_id, step, split,
            )
            yaw_delta = (next_yaw - yaw) % 4
            if yaw_delta == 3:
                yaw_delta = -1
            record["teacher_action"] = {
                "action": action,
                "is_jump_macro": "jump" in action,
                "control_horizon": "one_high_level_action",
            }
            record["labels"] = {
                "at_goal": state[:2] == goal,
                "action_success": True,
                "expected_displacement_xz": [nx - x, nz - z],
                "expected_yaw_delta_quarters": yaw_delta,
                "remaining_manhattan_distance": abs(goal[0] - x) + abs(goal[1] - z),
            }
            yield record


def recommend_curriculum(metrics_path: Path) -> dict:
    """Recommend one stage from exported held-out success/retention metrics.

    Expected JSON:
      {"stages": {"walking": {"success_rate": .96, "episodes": 100,
        "retention_success_rate": .92, "retention_episodes": 50}, ...}}
    """
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    stage_metrics = data.get("stages", data)
    passed = []
    for index, stage in enumerate(STAGES):
        metric = stage_metrics.get(stage.name, {})
        episodes = int(metric.get("episodes", 0))
        success = float(metric.get("success_rate", 0.0))
        if episodes < MIN_STAGE_EVAL_EPISODES:
            return {
                "recommended_stage": stage.name,
                "unlocked_through": passed[-1] if passed else None,
                "decision": "collect_more_evaluation",
                "reason": f"{stage.name} needs at least {MIN_STAGE_EVAL_EPISODES} held-out episodes",
                "mastery_threshold": MASTERY_SUCCESS_RATE,
                "retention_threshold": RETENTION_SUCCESS_RATE,
            }
        if success < MASTERY_SUCCESS_RATE:
            return {
                "recommended_stage": stage.name,
                "unlocked_through": passed[-1] if passed else None,
                "decision": "continue_current_stage",
                "reason": f"success_rate {success:.3f} is below {MASTERY_SUCCESS_RATE:.2f}",
                "mastery_threshold": MASTERY_SUCCESS_RATE,
                "retention_threshold": RETENTION_SUCCESS_RATE,
            }
        passed.append(stage.name)
        # Before advancing beyond a new stage, require evidence that previous skills remain.
        if index > 0:
            for prior in STAGES[:index]:
                previous = stage_metrics.get(prior.name, {})
                n_retention = int(previous.get("retention_episodes", 0))
                retention = float(previous.get("retention_success_rate", 0.0))
                if n_retention < MIN_RETENTION_EPISODES or retention < RETENTION_SUCCESS_RATE:
                    return {
                        "recommended_stage": stage.name,
                        "unlocked_through": passed[-1],
                        "decision": "retest_previous_skills",
                        "reason": (
                            f"before unlocking the next stage, retest {prior.name}: "
                            f"need >= {MIN_RETENTION_EPISODES} episodes and >= "
                            f"{RETENTION_SUCCESS_RATE:.2f} retention success"
                        ),
                        "mastery_threshold": MASTERY_SUCCESS_RATE,
                        "retention_threshold": RETENTION_SUCCESS_RATE,
                    }
    return {
        "recommended_stage": None,
        "unlocked_through": passed[-1] if passed else None,
        "decision": "all_stages_mastered",
        "reason": "all stage mastery and previous-skill retention gates passed",
        "mastery_threshold": MASTERY_SUCCESS_RATE,
        "retention_threshold": RETENTION_SUCCESS_RATE,
    }


def self_test() -> dict:
    checks = {}
    total_records = 0
    for stage in STAGES:
        records = list(generate_records(stage.name, 101 + stage.index, 2))
        assert records, f"{stage.name}: no records"
        assert all(len(r["observation"]["voxels"]) == 405 for r in records)
        assert all(r["teacher_action"]["action"] in ACTIONS for r in records)
        assert all(r["labels"]["at_goal"] == (r["teacher_action"]["action"] == "stop") for r in records)
        assert records == list(generate_records(stage.name, 101 + stage.index, 2))
        episode_ids = {r["episode_id"] for r in records}
        assert all(len({r["split"] for r in records if r["episode_id"] == eid}) == 1 for eid in episode_ids)
        actions = {r["teacher_action"]["action"] for r in records}
        if stage.name == "walking":
            assert actions <= {"walk_forward", "stop"}
        if stage.name in ("turning", "obstacle_avoidance"):
            assert actions & {"turn_left", "turn_right"}
        if stage.gap_distance:
            required = {"rotating_jump_left", "rotating_jump_right"} if stage.rotating_jump else {stage.jump_action}
            assert actions & required, f"{stage.name}: required mechanic not taught: {required}"
        checks[stage.name] = {"records": len(records), "actions": sorted(actions)}
        total_records += len(records)

    # The gate must not advance on missing/weak held-out evidence.
    temporary = {
        "stages": {
            "walking": {"success_rate": 0.95, "episodes": 100},
            "turning": {"success_rate": 0.60, "episodes": 100},
        }
    }
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "metrics.json"
        path.write_text(json.dumps(temporary), encoding="utf-8")
        gate = recommend_curriculum(path)
        assert gate["recommended_stage"] == "turning"
        assert gate["decision"] == "continue_current_stage"
    return {"self_test": "passed", "stages_checked": len(STAGES), "records_checked": total_records, "stage_checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stage", choices=tuple(STAGE_BY_NAME), default="walking")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("aimeng_mc_curriculum.jsonl"))
    parser.add_argument("--metrics", type=Path, help="JSON stage evaluation report; print next curriculum recommendation")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return 0
    if args.metrics:
        print(json.dumps(recommend_curriculum(args.metrics), ensure_ascii=False))
        return 0
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for record in generate_records(args.stage, args.seed, args.episodes):
            out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            count += 1
    stage = STAGE_BY_NAME[args.stage]
    print(json.dumps({
        "output": str(args.output),
        "stage": stage.name,
        "stage_index": stage.index,
        "episodes_requested": args.episodes,
        "records_written": count,
        "master_seed": args.seed,
        "grid_size": [GRID_X, GRID_Y, GRID_Z],
        "voxel_cells_per_record": GRID_X * GRID_Y * GRID_Z,
        "special_mechanic": stage.special_mechanic,
        "note": "synthetic curriculum only; special jump macros need live Bedrock calibration",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
