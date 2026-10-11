"""Multi-hop reasoning benchmark for AIMENG recurrent concept-cloud networks.

This benchmark asks models to compute a rolling modular state over the ENTIRE
input sequence: state <- (3 * state + token + 1) % vocab. It is deliberately
harder than successor-token and last-two-token sum tasks because early tokens
must influence the final answer through several dependent updates.

It is a synthetic compositional-reasoning probe, not a natural-language claim.
It compares the concept-cloud model at 1/2/3/4 recurrent cycles with a GRU
baseline using the same generated examples, seed, width, and optimizer steps.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from scripts.experiment_recurrent_concept_cloud import RecurrentConceptCloudNet


class GRUBaseline(nn.Module):
    def __init__(self, vocab: int, width: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)

    def forward(self, x):
        _, hidden = self.gru(self.embedding(x))
        return self.head(hidden[-1])


def make_data(n: int, vocab: int, seq_len: int, seed: int, device):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(vocab, (n, seq_len), generator=generator)
    # Sequential composition: every token can affect all later state updates.
    y = torch.zeros(n, dtype=torch.long)
    for position in range(seq_len):
        y = (3 * y + x[:, position] + 1) % vocab
    return x.to(device), y.to(device)


def train(model, x, y, steps: int, batch_size: int, seed: int, cloud: bool):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first, last = [], []
    started = time.perf_counter()
    for step in range(steps):
        indices = [rng.randrange(len(x)) for _ in range(batch_size)]
        xb, yb = x[indices], y[indices]
        optimizer.zero_grad(set_to_none=True)
        if cloud:
            output = model(xb, target=yb)
            loss = model.loss(output, yb)["loss"]
        else:
            loss = F.cross_entropy(model(xb), yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        if step < min(20, steps):
            first.append(value)
        if step >= max(0, steps - min(20, steps)):
            last.append(value)
    return {
        "steps": steps,
        "initial_mean_loss": sum(first) / max(1, len(first)),
        "final_mean_loss": sum(last) / max(1, len(last)),
        "training_seconds": time.perf_counter() - started,
    }


def evaluate_cloud(model, x, y, batch_size: int, cycles: int):
    model.eval()
    correct = routed = count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            xb, yb = x[lo:lo + batch_size], y[lo:lo + batch_size]
            output = model(xb, cycles=cycles)
            hit = (model.token_cloud[yb, None] == output["selected_clouds"]).any(-1)
            prediction = output["logits"].argmax(-1)
            correct += int(((prediction == yb) & hit).sum())
            routed += int(hit.sum())
            count += len(yb)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "accuracy": correct / max(count, 1),
        "router_recall_at_k": routed / max(count, 1),
        "examples_per_second": count / elapsed,
    }


def evaluate_gru(model, x, y, batch_size: int):
    model.eval()
    correct = count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            xb, yb = x[lo:lo + batch_size], y[lo:lo + batch_size]
            correct += int((model(xb).argmax(-1) == yb).sum())
            count += len(yb)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {"accuracy": correct / max(count, 1), "examples_per_second": count / elapsed}


def run_seed(args, seed: int, device: str):
    train_x, train_y = make_data(args.train_size, args.vocab, args.seq_len, seed, device)
    test_x, test_y = make_data(args.test_size, args.vocab, args.seq_len, seed + 100_003, device)
    assignment = [i // (args.vocab // args.clouds) for i in range(args.vocab)]
    cycle_results = {}
    cloud_train_results = {}
    # Train a separately initialized model for each cycle count so the comparison
    # does not give one setting weights trained under a different recurrence depth.
    for cycles in (1, 2, 3, 4):
        torch.manual_seed(seed)
        cloud = RecurrentConceptCloudNet(
            args.vocab, assignment, width=args.width, neurons=args.neurons,
            active_k=args.active_k, candidate_clouds=args.candidate_clouds,
            cycles=cycles,
        ).to(device)
        cloud_train_results[str(cycles)] = train(
            cloud, train_x, train_y, args.steps, args.batch_size, seed, cloud=True
        )
        cycle_results[str(cycles)] = {
            "parameters": sum(p.numel() for p in cloud.parameters()),
            "train": cloud_train_results[str(cycles)],
            "test": evaluate_cloud(cloud, test_x, test_y, args.batch_size, cycles),
        }
    torch.manual_seed(seed)
    baseline = GRUBaseline(args.vocab, args.width).to(device)
    baseline_train = train(
        baseline, train_x, train_y, args.steps, args.batch_size, seed, cloud=False
    )
    return {
        "seed": seed,
        "concept_cloud_by_cycles": cycle_results,
        "gru": {
            "parameters": sum(p.numel() for p in baseline.parameters()),
            "train": baseline_train,
            "test": evaluate_gru(baseline, test_x, test_y, args.batch_size),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--clouds", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--neurons", type=int, default=48)
    parser.add_argument("--active-k", type=int, default=6)
    parser.add_argument("--candidate-clouds", type=int, default=3)
    parser.add_argument("--seeds", default="17,23,41")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.train_size < 1 or args.test_size < 1:
        parser.error("steps, batch-size, train-size and test-size must be positive")
    if args.clouds < 1 or args.vocab < 8 or args.vocab % args.clouds:
        parser.error("clouds must be >=1; vocab must be >=8 and divisible by clouds")
    if args.seq_len < 3:
        parser.error("seq-len must be >=3 for a multi-hop task")
    if args.threads < 1:
        parser.error("threads must be >=1")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.self_test:
        # Keep the self-test fast enough for CI and constrained terminals.
        args.steps = min(args.steps, 3)
        args.train_size = min(args.train_size, 64)
        args.test_size = min(args.test_size, 32)
        args.batch_size = min(args.batch_size, 16)
        args.seq_len = max(args.seq_len, 3)
    seeds = [17] if args.self_test else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("provide at least one seed")
    results = {
        "benchmark": "synthetic_multi_hop_modular_state_v1",
        "self_test_status": "PASS" if args.self_test else None,
        "rule": "state=0; for each token: state=(3*state+token+1)%vocab",
        "warning": "Synthetic compositional task only; not proof of natural-language reasoning.",
        "config": vars(args) | {"device_resolved": device, "seeds_used": seeds},
        "runs": [],
    }
    for seed in seeds:
        results["runs"].append(run_seed(args, seed, device))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
