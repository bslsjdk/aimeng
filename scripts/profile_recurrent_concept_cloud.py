"""Operator-level CPU profile for the recurrent concept-cloud inference path.

This is an inference-only diagnostic: it does not train models or claim that
operator-level profiler timings equal end-to-end Android latency.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from scripts.experiment_recurrent_concept_cloud import RecurrentConceptCloudNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.iterations, args.threads, args.cycles) < 1:
        parser.error("batch-size, seq-len, iterations, threads and cycles must be positive")
    torch.set_num_threads(args.threads)
    vocab, width, neurons, active_k = 32, 16, 48, 6
    assignment = [i // 4 for i in range(vocab)]
    torch.manual_seed(17)
    model = RecurrentConceptCloudNet(
        vocab, assignment, width=width, neurons=neurons,
        active_k=active_k, candidate_clouds=3, cycles=args.cycles,
    ).eval()
    x = torch.randint(vocab, (args.batch_size, args.seq_len))
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model(x)
            samples.append(time.perf_counter() - start)
        # Profile a separate short window so profiler overhead does not pollute
        # the ordinary end-to-end latency samples above.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(min(10, args.iterations)):
                model(x)
    ordered = sorted(samples)
    result = {
        "benchmark": "recurrent_concept_cloud_operator_profile_v1",
        "note": "CPU-only diagnostic; profiler operator times include instrumentation overhead and are not Android measurements.",
        "config": vars(args) | {
            "vocab": vocab, "width": width, "neurons": neurons,
            "active_k": active_k, "candidate_clouds": 3,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "latency_seconds": {
            "mean": sum(samples) / len(samples),
            "median": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "examples_per_second": args.batch_size / (sum(samples) / len(samples)),
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
            for event in prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20
            ) and sorted(
                prof.key_averages(),
                key=lambda event: event.self_cpu_time_total,
                reverse=True,
            )[:20]
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
