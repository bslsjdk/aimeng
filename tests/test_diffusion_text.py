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

    def test_order_information_is_not_discarded(self):
        stoi, itos = make_vocab("我喜欢你你喜欢我")
        model = DiffusionTextModel(len(itos), neurons=32, width=8, active_k=4, fanout=4, max_steps=2)
        a = torch.tensor([[stoi[c] for c in "我喜欢你"]], dtype=torch.long)
        b = torch.tensor([[stoi[c] for c in "你喜欢我"]], dtype=torch.long)
        with torch.no_grad():
            pa = model.position_embedding(torch.arange(a.shape[1]))[None, :, :]
            pb = model.position_embedding(torch.arange(b.shape[1]))[None, :, :]
            ca = model.context_proj((model.embedding(a) * (1.0 + pa)).mean(dim=1))
            cb = model.context_proj((model.embedding(b) * (1.0 + pb)).mean(dim=1))
        self.assertFalse(torch.allclose(ca, cb), "ordered sequences must have distinguishable context vectors")

    def test_vocab_is_bounded_for_mobile(self):
        stoi, itos = make_vocab("天地玄黄宇宙洪荒" * 1000, max_vocab=5)
        self.assertEqual(len(itos), 5)
        self.assertEqual(itos[0], "<unk>")

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
