"""Concept-cloud next-token routing prototype.

This is an architecture probe, not a claim that semantic clustering is solved.
Cloud assignments must come from a train-only clustering step and must preserve
distinct senses. The router scores clouds; token scores are computed only inside
the selected cloud(s). During training, the gold cloud can be teacher-forced while
a separate router loss teaches context-to-cloud prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConceptCloudLM(nn.Module):
    """GRU context encoder + cloud router + per-cloud token scoring."""

    def __init__(
        self,
        vocab_size: int,
        cloud_ids: Sequence[int],
        width: int = 64,
        top_k_clouds: int = 1,
    ) -> None:
        super().__init__()
        if len(cloud_ids) != vocab_size:
            raise ValueError("cloud_ids must contain one assignment per vocabulary item")
        if vocab_size < 2 or width < 4:
            raise ValueError("vocab_size must be >=2 and width >=4")
        ids = torch.as_tensor(cloud_ids, dtype=torch.long)
        if int(ids.min()) < 0:
            raise ValueError("cloud IDs must be non-negative")
        unique_ids = torch.unique(ids, sorted=True)
        expected_ids = torch.arange(int(ids.max()) + 1, dtype=torch.long)
        if not torch.equal(unique_ids, expected_ids):
            raise ValueError("cloud IDs must be contiguous and every cloud must contain a token")
        self.num_clouds = int(ids.max()) + 1
        if self.num_clouds < 1 or self.num_clouds > vocab_size:
            raise ValueError("invalid number of clouds")
        if top_k_clouds < 1 or top_k_clouds > self.num_clouds:
            raise ValueError("top_k_clouds must be between 1 and num_clouds")
        self.vocab_size = vocab_size
        self.top_k_clouds = top_k_clouds
        self.register_buffer("token_cloud", ids)
        members = [torch.where(ids == c)[0] for c in range(self.num_clouds)]
        max_members = max(int(x.numel()) for x in members)
        member_table = torch.full((self.num_clouds, max_members), -1, dtype=torch.long)
        for c, row in enumerate(members):
            member_table[c, : row.numel()] = row
        self.register_buffer("cloud_members", member_table)
        self.embedding = nn.Embedding(vocab_size, width)
        self.context_encoder = nn.GRU(width, width, batch_first=True)
        self.router = nn.Linear(width, self.num_clouds)
        self.token_bias = nn.Parameter(torch.zeros(vocab_size))
        self.width = width

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
        top_k_clouds: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("context must have shape [batch, sequence]")
        _, hidden = self.context_encoder(self.embedding(context))
        state = hidden[-1]
        cloud_logits = self.router(state)
        k = self.top_k_clouds if top_k_clouds is None else top_k_clouds
        if not 1 <= k <= self.num_clouds:
            raise ValueError("invalid top_k_clouds")

        # Teacher forcing guarantees a usable local-token loss while router learns
        # to predict the target cloud. Inference routes only to top-k predicted clouds.
        if target is not None:
            if target.ndim != 1 or target.shape[0] != context.shape[0]:
                raise ValueError("target must have shape [batch]")
            selected = self.token_cloud[target].unsqueeze(1)
        else:
            selected = cloud_logits.topk(k, dim=-1).indices

        dense_logits = torch.full(
            (context.shape[0], self.vocab_size),
            torch.finfo(state.dtype).min,
            device=state.device,
            dtype=state.dtype,
        )
        routed_mask = torch.zeros(
            (context.shape[0], self.vocab_size), device=state.device, dtype=torch.bool
        )
        # Only calculate token scores for tokens belonging to selected clouds.
        for b in range(context.shape[0]):
            token_parts = []
            for cloud in selected[b].unique().tolist():
                members = self.cloud_members[cloud]
                members = members[members >= 0]
                if members.numel() == 0:
                    continue
                scores = self.embedding(members) @ state[b] / (self.width ** 0.5)
                scores = scores + self.token_bias[members]
                dense_logits[b, members] = scores
                routed_mask[b, members] = True
                token_parts.append(members)
        return {
            "logits": dense_logits,
            "cloud_logits": cloud_logits,
            "selected_clouds": selected,
            "routed_mask": routed_mask,
            "routed_token_count": routed_mask.sum(dim=-1),
        }

    def loss(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        router_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        gold_cloud = self.token_cloud[target]
        local_loss = F.cross_entropy(output["logits"], target)
        router_loss = F.cross_entropy(output["cloud_logits"], gold_cloud)
        return {
            "loss": local_loss + router_weight * router_loss,
            "token_loss": local_loss,
            "router_loss": router_loss,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--clouds", type=int, default=16)
    parser.add_argument("--top-k-clouds", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    if args.self_test:
        torch.manual_seed(7)
        vocab = max(args.vocab_size, args.clouds)
        cloud_ids = [i % args.clouds for i in range(vocab)]
        model = ConceptCloudLM(vocab, cloud_ids, args.width, args.top_k_clouds)
        x = torch.randint(0, vocab, (8, 12))
        y = torch.randint(0, vocab, (8,))
        out = model(x, target=y)
        losses = model.loss(out, y)
        losses["loss"].backward()
        infer = model(x)
        active = infer["routed_token_count"].float().mean().item()
        total = sum(p.numel() for p in model.parameters())
        print(json.dumps({
            "status": "PASS",
            "vocab_size": vocab,
            "num_clouds": model.num_clouds,
            "top_k_clouds": args.top_k_clouds,
            "mean_routed_tokens": active,
            "full_vocab_size": vocab,
            "parameter_count": total,
            "train_loss_finite": bool(torch.isfinite(losses["loss"]).item()),
            "inference_masks_unrouted_tokens": bool((~infer["routed_mask"]).any().item()),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
