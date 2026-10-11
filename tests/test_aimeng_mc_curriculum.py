import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_aimeng_mc_curriculum import (
    ACTIONS,
    STAGES,
    generate_records,
    recommend_curriculum,
)


class MinecraftCurriculumTests(unittest.TestCase):
    def test_walking_stage_varies_heading_and_goal_distance(self):
        records = list(generate_records("walking", 20261011, 100))
        starts = {}
        for record in records:
            if record["step"] == 0:
                starts[record["episode_id"]] = (
                    record["observation"]["yaw_bucket"],
                    tuple(record["observation"]["goal_offset"]),
                )
        headings = {value[0] for value in starts.values()}
        goal_offsets = {value[1] for value in starts.values()}
        self.assertEqual(headings, {0, 1, 2, 3})
        self.assertGreaterEqual(len(goal_offsets), 12)

    def test_all_stages_generate_valid_schema_and_repeat_by_seed(self):
        for stage in STAGES:
            first = list(generate_records(stage.name, 900 + stage.index, 2))
            second = list(generate_records(stage.name, 900 + stage.index, 2))
            self.assertEqual(first, second, stage.name)
            self.assertTrue(first)
            self.assertTrue(all(len(row["observation"]["voxels"]) == 405 for row in first))
            self.assertTrue(all(row["teacher_action"]["action"] in ACTIONS for row in first))
            self.assertTrue(all(row["labels"]["at_goal"] == (row["teacher_action"]["action"] == "stop") for row in first))

    def test_turning_and_obstacle_maps_are_not_near_duplicates(self):
        turning = [r for r in generate_records("turning", 20261012, 100) if r["step"] == 0]
        obstacles = [r for r in generate_records("obstacle_avoidance", 20261013, 100) if r["step"] == 0]
        turning_layouts = {
            (r["observation"]["yaw_bucket"], tuple(r["observation"]["goal_offset"]))
            for r in turning
        }
        obstacle_layouts = {
            (r["observation"]["yaw_bucket"], tuple(r["observation"]["goal_offset"]), tuple(r["observation"]["voxels"]))
            for r in obstacles
        }
        self.assertGreaterEqual(len(turning_layouts), 50)
        self.assertEqual({r["observation"]["yaw_bucket"] for r in turning}, {0, 1, 2, 3})
        self.assertGreaterEqual(len(obstacle_layouts), 50)
        self.assertTrue(all(
            any(row["teacher_action"]["action"] in ("turn_left", "turn_right") for row in generate_records(stage_name, 303 + index, 2))
            for index, stage_name in enumerate(("turning", "obstacle_avoidance"))
        ))

    def test_curriculum_contains_required_jump_mechanic(self):
        for stage in STAGES:
            if not stage.gap_distance:
                continue
            rows = list(generate_records(stage.name, 321 + stage.index, 2))
            actions = {row["teacher_action"]["action"] for row in rows}
            expected = (
                {"rotating_jump_left", "rotating_jump_right"}
                if stage.rotating_jump else {stage.jump_action}
            )
            self.assertTrue(actions & expected, f"{stage.name} lacks {expected}")

    def test_split_is_constant_within_an_episode(self):
        rows = list(generate_records("obstacle_avoidance", 1234, 20))
        episode_splits = {}
        for row in rows:
            episode_splits.setdefault(row["episode_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in episode_splits.values()))

    def test_gate_does_not_advance_without_mastery(self):
        data = {"stages": {
            "walking": {"success_rate": 0.95, "episodes": 100},
            "turning": {"success_rate": 0.72, "episodes": 100},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = recommend_curriculum(path)
        self.assertEqual(result["recommended_stage"], "turning")
        self.assertEqual(result["decision"], "continue_current_stage")

    def test_gate_requires_previous_skill_retention(self):
        data = {"stages": {
            "walking": {"success_rate": 0.96, "episodes": 100},
            "turning": {"success_rate": 0.92, "episodes": 100},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = recommend_curriculum(path)
        self.assertEqual(result["decision"], "retest_previous_skills")
        self.assertEqual(result["recommended_stage"], "turning")

    def test_gate_unlocks_next_stage_after_mastery_and_retention(self):
        data = {"stages": {
            "walking": {"success_rate": 0.96, "episodes": 100,
                        "retention_success_rate": 0.92, "retention_episodes": 50},
            "turning": {"success_rate": 0.93, "episodes": 100,
                        "retention_success_rate": 0.90, "retention_episodes": 50},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = recommend_curriculum(path)
        self.assertEqual(result["recommended_stage"], "obstacle_avoidance")
        self.assertEqual(result["decision"], "collect_more_evaluation")


if __name__ == "__main__":
    unittest.main()
