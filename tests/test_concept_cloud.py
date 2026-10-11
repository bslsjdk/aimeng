import unittest

import torch

from scripts.experiment_concept_cloud import ConceptCloudLM


class ConceptCloudTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.cloud_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.model = ConceptCloudLM(
            vocab_size=12, cloud_ids=self.cloud_ids, width=8, top_k_clouds=1
        )

    def test_inference_scores_only_selected_clouds(self):
        x = torch.randint(0, 12, (4, 5))
        out = self.model(x)
        for row, selected in enumerate(out["selected_clouds"]):
            expected = torch.tensor(self.cloud_ids) == int(selected[0])
            self.assertTrue(torch.equal(out["routed_mask"][row].cpu(), expected))
            self.assertEqual(int(out["routed_token_count"][row]), 3)
            self.assertTrue(torch.isfinite(out["logits"][row, expected]).all())

    def test_teacher_forced_training_has_finite_gradients(self):
        x = torch.randint(0, 12, (4, 5))
        y = torch.tensor([0, 4, 8, 10])
        out = self.model(x, target=y)
        losses = self.model.loss(out, y)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(self.model.router.weight.grad)
        self.assertTrue(torch.isfinite(self.model.router.weight.grad).all())

    def test_rejects_missing_cloud_ids(self):
        with self.assertRaises(ValueError):
            ConceptCloudLM(4, [0, 0, 2, 2], width=8)

    def test_rejects_bad_assignment_length(self):
        with self.assertRaises(ValueError):
            ConceptCloudLM(4, [0, 1, 1], width=8)


if __name__ == "__main__":
    unittest.main()
