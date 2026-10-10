import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.train_diffusion_text import DiffusionTextModel, make_vocab, encode, SMOKE_TEXT
from scripts.prepare_language_splits import split_records

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

    def test_periodic_checkpoint_can_resume_training(self):
        from scripts.train_diffusion_text import train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus.txt"
            corpus.write_text(SMOKE_TEXT * 10, encoding="utf-8")
            output = root / "run"

            def args(steps, resume):
                return SimpleNamespace(
                    text=str(corpus), context=8, vocab_size=64, seed=7,
                    neurons=16, width=4, active_k=2, fanout=2, max_steps=1,
                    lr=0.002, step_penalty=0.001, batch_size=1, steps=steps,
                    log_every=100, checkpoint_every=1, memory_stop_mib=100000.0,
                    output=str(output), resume=resume,
                )

            first = train(args(2, False))
            self.assertEqual(first["steps_completed"], 2)
            self.assertTrue((output / "training_state.pt").is_file())
            self.assertTrue((output / "diffusion_checkpoint.pt").is_file())

            resumed = train(args(3, True))
            self.assertEqual(resumed["steps_completed"], 3)
            state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["completed_steps"], 3)
            self.assertEqual(state["format"], "aimeng-resumable-training-state-v1")


    def test_record_splits_are_disjoint_and_deduplicated(self):
        records = [f"问题：样本{i}\\n回答：答案{i}" for i in range(40)]
        source = "\\n\\n".join(records + [records[3], records[7]]) + "\\n"
        splits, duplicates = split_records(source, seed=11)
        self.assertEqual(duplicates, 2)
        self.assertEqual(sum(len(rows) for rows in splits.values()), 40)
        sets = {name: set(rows) for name, rows in splits.items()}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_record_split_rejects_too_small_input(self):
        with self.assertRaises(ValueError):
            split_records("\\n\\n".join(f"r{i}" for i in range(5)))

    def test_vocab_can_be_fitted_without_validation_only_characters(self):
        stoi, itos = make_vocab("中文训练语料", max_vocab=32)
        self.assertNotIn("罕", stoi)
        self.assertEqual(encode("中文罕见字", stoi)[2], 0)

if __name__ == "__main__": unittest.main()
