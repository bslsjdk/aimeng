#!/usr/bin/env python3
"""Evaluate a saved AIMENG checkpoint on a sealed, record-level test corpus.

Reports character next-step loss, perplexity, OOV rate, and a same-checkpoint
one-step-vs-full-diffusion ablation. This is not a semantic/reasoning benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.train_diffusion_text import DiffusionTextModel, encode


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_checkpoint(checkpoint_path: str, test_path: str, limit: int = 512,
                        batch_size: int = 1, device_name: str | None = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    text = Path(test_path).read_text(encoding="utf-8")
    ids = encode(text, stoi)
    context = int(config["context"])
    if len(ids) <= context + 1:
        raise ValueError(f"Test corpus needs more than context+1 characters ({context + 2} minimum)")
    if limit < 1 or batch_size < 1:
        raise ValueError("limit and batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DiffusionTextModel(
        len(itos), int(config["neurons"]), int(config["width"]),
        int(config["active_k"]), int(config["fanout"]), int(config["max_steps"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    starts = list(range(min(len(ids) - context - 1, limit)))
    one_sum = full_sum = 0.0
    seen = 0
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            chunk = starts[offset:offset + batch_size]
            x = torch.tensor([ids[s:s + context] for s in chunk], dtype=torch.long, device=device)
            y = torch.tensor([ids[s + context] for s in chunk], dtype=torch.long, device=device)
            out = model(x, early_stop=False)
            one_sum += F.cross_entropy(out["step_logits"][0], y, reduction="sum").item()
            full_sum += F.cross_entropy(out["logits"], y, reduction="sum").item()
            seen += len(chunk)
    one_loss, full_loss = one_sum / seen, full_sum / seen
    oov_count = sum(1 for ch in text if ch not in stoi)
    return {
        "format": "aimeng-heldout-test-report-v1",
        "checkpoint": str(checkpoint_path),
        "test_path": str(test_path),
        "test_sha256": sha256(text),
        "test_characters": len(text),
        "evaluated_next_character_positions": seen,
        "oov_character_rate": oov_count / max(1, len(text)),
        "vocabulary_size": len(itos),
        "device": str(device),
        "one_step_loss": one_loss,
        "one_step_perplexity": math.exp(min(20.0, one_loss)),
        "full_diffusion_loss": full_loss,
        "full_diffusion_perplexity": math.exp(min(20.0, full_loss)),
        "full_minus_one_step_loss": full_loss - one_loss,
        "interpretation": "Negative full_minus_one_step_loss means multi-step output improved next-character loss on this held-out corpus only. It does not prove semantic understanding or reasoning.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-text", required=True)
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default=None, help="optional cpu/cuda override")
    p.add_argument("--output", default=None, help="optional JSON report path")
    args = p.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.test_text, args.limit, args.batch_size, args.device)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(rendered, encoding="utf-8")
        tmp.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
