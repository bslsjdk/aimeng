import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.generate_aimeng_mc_curriculum import generate_records
from scripts.train_aimeng_mc_curriculum import (
    ACTION_TO_ID,
    EpisodeDataset,
    EpisodeFile,
    ParkourGRUPolicy,
    allowed_actions_for_stage,
    compute_class_weights,
    mask_action_logits,
    parameter_count,
    estimated_weight_mib,
    collate_episodes,
    evaluate_rollouts,
)


class ParkourTrainingTests(unittest.TestCase):
    def test_inverse_frequency_weights_balance_rare_stop_action(self):
        from collections import Counter
        from types import SimpleNamespace

        source = SimpleNamespace(action_counts={
            "train": Counter({"walk_forward": 90, "stop": 10})
        })
        weights = compute_class_weights([source], torch.device("cpu"))
        walk_weight = float(weights[ACTION_TO_ID["walk_forward"]].item())
        stop_weight = float(weights[ACTION_TO_ID["stop"]].item())
        self.assertAlmostEqual(stop_weight / walk_weight, 9.0, places=5)
        self.assertAlmostEqual((walk_weight * 90 + stop_weight * 10) / 100, 1.0, places=5)
        self.assertEqual(float(weights[ACTION_TO_ID["turn_left"]].item()), 1.0)

    def test_action_space_unlocks_by_curriculum_stage(self):
        walking = set(allowed_actions_for_stage("walking"))
        turning = set(allowed_actions_for_stage("turning"))
        small_jump = set(allowed_actions_for_stage("small_jump"))
        rotating = set(allowed_actions_for_stage("rotating_jump"))
        double_jump = set(allowed_actions_for_stage("double_jump"))
        self.assertEqual(walking, {"walk_forward", "stop"})
        self.assertTrue({"turn_left", "turn_right"}.issubset(turning))
        self.assertNotIn("jump_forward", turning)
        self.assertIn("jump_forward", small_jump)
        self.assertIn("rotating_jump_left", rotating)
        self.assertIn("double_jump_forward", double_jump)
        self.assertTrue(walking.issubset(turning))
        self.assertTrue(turning.issubset(small_jump))
        self.assertTrue(small_jump.issubset(rotating))
        self.assertTrue(rotating.issubset(double_jump))

    def test_locked_actions_cannot_win_logits(self):
        logits = torch.full((10,), -3.0)
        logits[1] = 100.0  # turn_left is not unlocked during walking
        logits[9] = 1.0    # stop
        masked = mask_action_logits(logits, "walking")
        self.assertEqual(int(masked.argmax().item()), 9)
        self.assertTrue(torch.isneginf(masked[1]))
        self.assertFalse(torch.isneginf(masked[0]))
        self.assertFalse(torch.isneginf(masked[9]))

    def test_parameter_and_weight_memory_estimates(self):
        model = ParkourGRUPolicy(hidden_size=8, encoder_width=16)
        params = parameter_count(model)
        weight_mib = estimated_weight_mib(model)
        self.assertEqual(params, sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertGreater(params, 0)
        self.assertAlmostEqual(weight_mib, params * 4 / (1024 * 1024), places=6)

    def test_model_forward_shapes(self):
        model = ParkourGRUPolicy(hidden_size=8, encoder_width=16)
        voxels = torch.zeros((2, 3, 405), dtype=torch.long)
        goals = torch.zeros((2, 3, 3), dtype=torch.float32)
        yaws = torch.zeros((2, 3), dtype=torch.long)
        logits, hidden = model(voxels, goals, yaws)
        self.assertEqual(tuple(logits.shape), (2, 3, 10))
        self.assertEqual(tuple(hidden.shape), (2, 8))

    def test_hidden_expansion_preserves_logits(self):
        torch.manual_seed(4)
        model = ParkourGRUPolicy(hidden_size=8, encoder_width=16).eval()
        voxels = torch.randint(0, 7, (2, 4, 405))
        goals = torch.randn(2, 4, 3)
        yaws = torch.randint(0, 4, (2, 4))
        old_logits, old_hidden = model(voxels, goals, yaws)
        widened = model.expand_hidden(16, seed=5).eval()
        new_logits, new_hidden = widened(voxels, goals, yaws)
        torch.testing.assert_close(new_logits, old_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new_hidden[:, :8], old_hidden, rtol=1e-5, atol=1e-6)

    def test_episode_index_uses_whole_episode_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walking.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in generate_records("walking", 222, 20):
                    handle.write(json.dumps(record) + "\n")
            source = EpisodeFile(path)
            train = EpisodeDataset(source, "train")
            validation = EpisodeDataset(source, "validation")
            test = EpisodeDataset(source, "test")
            retention = EpisodeDataset(source, "retention")
            self.assertEqual((len(train), len(validation), len(test), len(retention)), (15, 2, 2, 1))
            for split_name, dataset in (("train", train), ("validation", validation), ("test", test), ("retention", retention)):
                for i in range(len(dataset)):
                    rows = dataset[i]
                    self.assertTrue(rows)
                    self.assertTrue(all(row["split"] == split_name for row in rows))
                    self.assertEqual(len({row["episode_id"] for row in rows}), 1)

    def test_collate_pads_variable_length_episodes(self):
        episodes = [
            list(generate_records("walking", 11, 1)),
            list(generate_records("walking", 12, 1)),
        ]
        batch = collate_episodes(episodes)
        self.assertEqual(batch["voxels"].shape[0], 2)
        self.assertEqual(batch["voxels"].shape[-1], 405)
        self.assertEqual(batch["actions"].shape[0], 2)

    def test_oracle_rule_solves_walking_courses(self):
        class OracleWalkPolicy(torch.nn.Module):
            def forward(self, voxels, goals, yaws, hidden=None):
                batch, steps, _ = goals.shape
                logits = goals.new_full((batch, steps, len(ACTION_TO_ID)), -10.0)
                at_goal = (goals[..., 0] == 0) & (goals[..., 2] == 0)
                logits[..., ACTION_TO_ID["walk_forward"]] = torch.where(at_goal, -10.0, 10.0)
                logits[..., ACTION_TO_ID["stop"]] = torch.where(at_goal, 10.0, -10.0)
                next_hidden = hidden if hidden is not None else goals.new_zeros((batch, 1))
                return logits, next_hidden

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walking.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in generate_records("walking", 555, 100):
                    handle.write(json.dumps(record) + "\n")
            source = EpisodeFile(path)
            test = EpisodeDataset(source, "test")
            result = evaluate_rollouts(OracleWalkPolicy(), test, torch.device("cpu"))
        self.assertEqual(result["episodes"], 10)
        self.assertEqual(result["success_rate"], 1.0)

    def test_rollout_evaluator_returns_episode_success_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walking.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in generate_records("walking", 333, 20):
                    handle.write(json.dumps(record) + "\n")
            source = EpisodeFile(path)
            test = EpisodeDataset(source, "test")
            model = ParkourGRUPolicy(hidden_size=8, encoder_width=16)
            result = evaluate_rollouts(model, test, torch.device("cpu"))
            self.assertEqual(result["episodes"], 2)
            self.assertGreaterEqual(result["success_rate"], 0.0)
            self.assertLessEqual(result["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
