import unittest
import torch
from scripts.train_diffusion_text import DiffusionTextModel, make_vocab, encode, SMOKE_TEXT

class DiffusionTextTests(unittest.TestCase):
    def test_vocab_roundtrip_and_model_shapes(self):
        stoi, itos = make_vocab(SMOKE_TEXT)
        ids = encode("AIMENG", stoi)
        self.assertEqual(len(ids), 6)
        model = DiffusionTextModel(len(itos), neurons=32, width=8, active_k=4, fanout=4, max_steps=3)
        edges = {(src, int(dst)) for src, row in enumerate(model.neighbors.tolist()) for dst in row}
        self.assertTrue(all((dst, src) in edges for src, dst in edges), "fanout-4 graph must propagate bidirectionally")
        out = model(torch.tensor([ids, ids]), early_stop=False)
        self.assertEqual(tuple(out["logits"].shape), (2, len(itos)))
        self.assertGreaterEqual(out["steps_used"], 1)
        self.assertLessEqual(out["steps_used"], 3)
        self.assertGreaterEqual(float(out["expected_steps"].min().detach()), 2.0)
        self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_backward_updates_parameters(self):
        stoi, itos = make_vocab(SMOKE_TEXT)
        model = DiffusionTextModel(len(itos), neurons=32, width=8, active_k=4, fanout=4, max_steps=2)
        x = torch.randint(0, len(itos), (4, 8)); y = torch.randint(0, len(itos), (4,))
        before = model.decoder.weight.detach().clone()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], y) + 0.001 * out["expected_steps"].mean()
        loss.backward()
        self.assertIsNotNone(model.decoder.weight.grad)
        torch.optim.AdamW(model.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, model.decoder.weight.detach()))

if __name__ == "__main__": unittest.main()
