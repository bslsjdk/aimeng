"""Operator-level CPU profile for recurrent concept-cloud inference or training."""
from __future__ import annotations

import argparse
import json
import time

import torch

from scripts.experiment_recurrent_concept_cloud import RecurrentConceptCloudNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "training"), default="inference")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--neurons", type=int, default=48)
    parser.add_argument("--active-k", type=int, default=6)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.iterations, args.threads, args.cycles,
           args.vocab, args.width, args.neurons, args.active_k) < 1:
        parser.error("all sizes, iterations, threads and cycles must be positive")
    if args.active_k > args.neurons:
        parser.error("active-k cannot exceed neurons")

    torch.set_num_threads(args.threads)
    assignment = [i // 4 for i in range(args.vocab)]
    torch.manual_seed(17)
    model = RecurrentConceptCloudNet(
        args.vocab, assignment, width=args.width, neurons=args.neurons,
        active_k=args.active_k, candidate_clouds=min(3, args.vocab // 4),
        cycles=args.cycles,
    )
    model.train(args.mode == "training")
    x = torch.randint(args.vocab, (args.batch_size, args.seq_len))
    y = torch.randint(args.vocab, (args.batch_size,))

    def step() -> None:
        if args.mode == "training":
            model.zero_grad(set_to_none=True)
            output = model(x, target=y)
            losses = model.loss(output, y)
            losses["loss"].backward()
        else:
            with torch.inference_mode():
                model(x)

    for _ in range(args.warmup):
        step()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        step()
        samples.append(time.perf_counter() - start)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(min(10, args.iterations)):
            step()

    ordered = sorted(samples)
    mean_seconds = sum(samples) / len(samples)
    events = sorted(
        prof.key_averages(),
        key=lambda event: event.self_cpu_time_total,
        reverse=True,
    )[:20]
    result = {
        "benchmark": "recurrent_concept_cloud_operator_profile_v2",
        "note": "CPU-only diagnostic; profiler timings include instrumentation overhead and are not Android measurements. Training includes forward, loss, backward, and gradient reset.",
        "config": vars(args) | {
            "candidate_clouds": min(3, args.vocab // 4),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "step_seconds": {
            "mean": mean_seconds,
            "median": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "examples_per_second": args.batch_size / mean_seconds,
            "steps_per_second": 1.0 / mean_seconds,
        },
        "top_operators": [
            {
                "operator": event.key,
                "self_cpu_time_us": round(event.self_cpu_time_total, 3),
                "cpu_time_us": round(event.cpu_time_total, 3),
                "calls": event.count,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "input_shapes": event.input_shapes[:3],
            }
            for event in events
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
