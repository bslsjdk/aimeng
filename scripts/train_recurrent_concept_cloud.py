"""Train and compare the recurrent concept-cloud prototype from a terminal.

This is a small synthetic next-token sanity benchmark, not a language-model result.
It reports router recall separately because inference cannot see the gold cloud.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Support both `python scripts/train_recurrent_concept_cloud.py` and `python -m ...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_recurrent_concept_cloud import RecurrentConceptCloudNet


class GRUBaseline(nn.Module):
    def __init__(self, vocab: int, width: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.head(h[-1])


def make_data(n: int, vocab: int, seq_len: int, seed: int, device, task: str):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(vocab, (n, seq_len), generator=g)
    # A deterministic, context-dependent next-token rule. The same split and
    # generated examples are used for both systems.
    y = (x[:, -1] + 1) % vocab if task == "successor" else (x[:, -1] + x[:, -2]) % vocab
    return x.to(device), y.to(device)


def evaluate_cloud(model, x, y, batch_size):
    model.eval()
    correct = routed = count = 0
    loss_sum = 0.0
    start = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            xb, yb = x[lo:lo + batch_size], y[lo:lo + batch_size]
            out = model(xb)
            hit = out["target_routed"] if "target_routed" in out else (
                model.token_cloud[yb, None] == out["selected_clouds"]
            ).any(-1)
            # Masked full-vocabulary logits only permit valid argmax when the
            # correct cloud was routed. Count routing failures as failures.
            pred = out["logits"].argmax(-1)
            correct += int(((pred == yb) & hit).sum())
            routed += int(hit.sum())
            count += len(yb)
            valid = out["logits"][torch.arange(len(yb), device=yb.device), yb]
            finite = torch.isfinite(valid) & (valid > torch.finfo(valid.dtype).min / 2)
            if finite.any():
                loss_sum += float(F.cross_entropy(out["logits"][finite], yb[finite], reduction="sum"))
            loss_sum += int((~finite).sum()) * 30.0
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "token_accuracy": correct / count,
        "router_recall_at_k": routed / count,
        "conditional_token_accuracy_given_routed": correct / max(routed, 1),
        "token_loss_with_route_misses_penalized": loss_sum / count,
        "examples_per_second": count / elapsed,
    }


def evaluate_baseline(model, x, y, batch_size):
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    start = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            xb, yb = x[lo:lo + batch_size], y[lo:lo + batch_size]
            logits = model(xb)
            loss_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
            correct += int((logits.argmax(-1) == yb).sum())
            count += len(yb)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {"token_accuracy": correct / count, "token_loss": loss_sum / count,
            "examples_per_second": count / elapsed}


def train_one(model, train_x, train_y, steps, batch_size, lr, seed, cloud=False):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        ids = [rng.randrange(len(train_x)) for _ in range(batch_size)]
        xb, yb = train_x[ids], train_y[ids]
        opt.zero_grad(set_to_none=True)
        if cloud:
            out = model(xb, target=yb)
            parts = model.loss(out, yb)
            loss = parts["loss"]
        else:
            loss = F.cross_entropy(model(xb), yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    window = max(1, min(20, len(losses) // 3))
    return {"first_window_steps": window,
            "first_window_mean_loss": sum(losses[:window]) / window,
            "last_window_steps": window,
            "last_window_mean_loss": sum(losses[-window:]) / window,
            "training_seconds": time.perf_counter() - started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=4096)
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--clouds", type=int, default=8)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--neurons", type=int, default=48)
    p.add_argument("--active-k", type=int, default=6)
    p.add_argument("--candidate-clouds", type=int, default=3)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--task", choices=["successor", "context-sum"], default="successor")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.vocab % args.clouds:
        p.error("steps/batch-size must be positive and vocab must be divisible by clouds")
    if args.threads < 1:
        p.error("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_x, train_y = make_data(args.train_size, args.vocab, args.seq_len, args.seed, device, args.task)
    test_x, test_y = make_data(args.test_size, args.vocab, args.seq_len, args.seed + 1, device, args.task)
    assignment = [i // (args.vocab // args.clouds) for i in range(args.vocab)]
    cloud = RecurrentConceptCloudNet(
        args.vocab, assignment, width=args.width, neurons=args.neurons,
        active_k=args.active_k, candidate_clouds=args.candidate_clouds,
        cycles=args.cycles,
    ).to(device)
    baseline = GRUBaseline(args.vocab, args.width).to(device)
    cloud_params = sum(q.numel() for q in cloud.parameters())
    base_params = sum(q.numel() for q in baseline.parameters())
    cloud_train = train_one(cloud, train_x, train_y, args.steps, args.batch_size, 1e-3, args.seed, True)
    base_train = train_one(baseline, train_x, train_y, args.steps, args.batch_size, 1e-3, args.seed, False)
    results = {
        "benchmark": f"synthetic_{args.task}_next_token_v1",
        "warning": "Sanity benchmark only; does not prove language ability or general superiority.",
        "config": vars(args) | {"device_resolved": device},
        "recurrent_concept_cloud": {
            "parameters": cloud_params, "train": cloud_train,
            "test": evaluate_cloud(cloud, test_x, test_y, args.batch_size),
            "active_neurons_per_cycle": args.active_k,
            "candidate_clouds": args.candidate_clouds,
        },
        "gru_baseline": {"parameters": base_params, "train": base_train,
                         "test": evaluate_baseline(baseline, test_x, test_y, args.batch_size)},
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
