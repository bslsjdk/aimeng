import unittest

import torch

from scripts.benchmark_recurrent_reasoning import make_data


class MultiHopBenchmarkTests(unittest.TestCase):
    def test_labels_match_full_sequence_recurrence(self):
        vocab = 32
        x, y = make_data(24, vocab, 8, 17, "cpu")
        expected = torch.zeros(len(x), dtype=torch.long)
        for position in range(x.shape[1]):
            expected = (3 * expected + x[:, position] + 1) % vocab
        self.assertTrue(torch.equal(y, expected))

    def test_additive_rule_labels_match_full_sequence_sum(self):
        vocab = 32
        x, y = make_data(24, vocab, 8, 17, "cpu", rule="additive")
        expected = torch.zeros(len(x), dtype=torch.long)
        for position in range(x.shape[1]):
            expected = (expected + x[:, position] + 1) % vocab
        self.assertTrue(torch.equal(y, expected))

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            make_data(2, 32, 8, 17, "cpu", rule="unknown")

    def test_early_tokens_can_change_final_answer(self):
        vocab = 32
        first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        second = torch.tensor([[2, 2, 3, 4]], dtype=torch.long)

        def answer(sequence):
            state = 0
            for token in sequence[0].tolist():
                state = (3 * state + token + 1) % vocab
            return state

        self.assertNotEqual(answer(first), answer(second))

    def test_longer_sequence_labels_are_valid(self):
        vocab = 32
        x, y = make_data(12, vocab, 16, 23, "cpu")
        self.assertEqual(tuple(x.shape), (12, 16))
        self.assertEqual(tuple(y.shape), (12,))
        self.assertTrue(((y >= 0) & (y < vocab)).all())


if __name__ == "__main__":
    unittest.main()
