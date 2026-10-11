import unittest

import torch

from scripts.dynamic_expansion import CapacityMonitor, ExpandableGRU


class ExpandableGRUTests(unittest.TestCase):
    def test_expansion_preserves_outputs(self):
        torch.manual_seed(7)
        model = ExpandableGRU(5, 8, 3).eval()
        inputs = torch.randn(4, 6, 5)
        old_logits, old_hidden = model(inputs)
        widened = model.expand_hidden(16, seed=11).eval()
        new_logits, new_hidden = widened(inputs)
        self.assertEqual(widened.hidden_size, 16)
        torch.testing.assert_close(new_logits, old_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new_hidden[:, :8], old_hidden, rtol=1e-5, atol=1e-6)

    def test_expanded_units_can_learn(self):
        torch.manual_seed(3)
        model = ExpandableGRU(4, 4, 2)
        widened = model.expand_hidden(8, seed=5)
        optimizer = torch.optim.SGD(widened.parameters(), lr=0.05)
        new_rows = torch.cat([widened.cell.weight_ih[4:8], widened.cell.weight_ih[12:16], widened.cell.weight_ih[20:24]])
        initial_new_rows = new_rows.detach().clone()
        x = torch.randn(3, 5, 4)
        target = torch.randn(3, 5, 2)
        for _ in range(3):
            optimizer.zero_grad()
            logits, _ = widened(x)
            loss = torch.nn.functional.mse_loss(logits, target)
            loss.backward()
            optimizer.step()
        self.assertGreater(widened.readout.weight[:, 4:].abs().sum().item(), 0.0)
        new_rows_after = torch.cat([widened.cell.weight_ih[4:8], widened.cell.weight_ih[12:16], widened.cell.weight_ih[20:24]])
        self.assertGreater((new_rows_after - initial_new_rows).abs().sum().item(), 0.0)

    def test_old_hidden_state_can_be_padded(self):
        model = ExpandableGRU(2, 3, 1)
        widened = model.expand_hidden(6)
        logits, hidden = widened(torch.ones(1, 2, 2), torch.zeros(1, 3))
        self.assertEqual(tuple(logits.shape), (1, 2, 1))
        self.assertEqual(tuple(hidden.shape), (1, 6))

    def test_expansion_rejects_shrinking(self):
        model = ExpandableGRU(2, 4, 1)
        with self.assertRaises(ValueError):
            model.expand_hidden(4)

    def test_capacity_monitor_does_not_expand_for_overfit(self):
        monitor = CapacityMonitor(patience=2, target_score=0.95, max_generalization_gap=0.10)
        self.assertFalse(monitor.update(0.90, 0.60, 64).expand)
        decision = monitor.update(0.90, 0.60, 64)
        self.assertFalse(decision.expand)
        self.assertIn("generalization gap", decision.reason)

    def test_capacity_monitor_expands_after_plateau(self):
        monitor = CapacityMonitor(patience=2, target_score=0.95)
        self.assertFalse(monitor.update(0.70, 0.68, 64).expand)
        self.assertFalse(monitor.update(0.70, 0.69, 64).expand)
        self.assertFalse(monitor.update(0.70, 0.69, 64).expand)
        decision = monitor.update(0.70, 0.69, 64)
        self.assertTrue(decision.expand)
        self.assertEqual(decision.suggested_size, 128)


if __name__ == "__main__":
    unittest.main()
