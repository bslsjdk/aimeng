import unittest

import torch

from scripts.experiment_recurrent_concept_cloud import RecurrentConceptCloudNet


class RecurrentConceptCloudTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        assignment = [i // 4 for i in range(16)]
        self.model = RecurrentConceptCloudNet(
            vocab_size=16,
            token_cloud=assignment,
            width=8,
            neurons=16,
            active_k=4,
            fanout=3,
            candidate_clouds=2,
            cycles=3,
        )

    def test_multiple_cycles_keep_shape_and_candidate_hypotheses(self):
        x = torch.randint(0, 16, (3, 7))
        out = self.model(x)
        self.assertEqual(tuple(out["cycle_cloud_logits"].shape), (3, 3, 4))
        self.assertEqual(tuple(out["cycle_token_logits"].shape), (3, 3, 16))
        self.assertEqual(tuple(out["selected_clouds"].shape), (3, 2))
        self.assertTrue(torch.isfinite(out["cloud_logits"]).all())

    def test_training_loss_backpropagates(self):
        x = torch.randint(0, 16, (3, 7))
        y = torch.tensor([1, 6, 14])
        out = self.model(x, target=y)
        losses = self.model.loss(out, y)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(self.model.cloud_head.weight.grad)
        self.assertTrue(torch.isfinite(self.model.cloud_head.weight.grad).all())

    def test_active_neuron_budget_is_bounded(self):
        x = torch.randint(0, 16, (2, 5))
        out = self.model(x)
        self.assertTrue((out["active_counts"] <= 4).all())
        self.assertTrue((out["active_counts"] >= 1).all())

    def test_target_cloud_routing_status_is_reported(self):
        x = torch.randint(0, 16, (2, 5))
        y = torch.tensor([2, 13])
        out = self.model(x, target=y)
        self.assertEqual(tuple(out["target_routed"].shape), (2,))
        self.assertTrue(out["target_routed"].dtype == torch.bool)

    def test_training_does_not_inject_unrouted_gold_cloud(self):
        # Force the router to choose clouds 0 and 1. Targets are in clouds 2
        # and 3, so their token logits must remain masked during training too.
        with torch.no_grad():
            self.model.cloud_head.weight.zero_()
            self.model.cloud_head.bias.copy_(torch.tensor([100.0, 90.0, 0.0, -1.0]))
        self.model.candidate_clouds = 2
        x = torch.randint(0, 16, (2, 5))
        y = torch.tensor([8, 13])
        out = self.model(x, target=y)
        self.assertFalse(out["target_routed"].any())
        floor = torch.finfo(out["logits"].dtype).min
        self.assertTrue((out["logits"][torch.arange(2), y] == floor).all())
        losses = self.model.loss(out, y)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(self.model.cloud_head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
