"""Iterative recurrent concept-cloud network prototype for AIMENG.

Unlike a Transformer stack, this model repeatedly updates a persistent neuron state.
At each cycle it predicts multiple candidate concept clouds, activates only a
bounded set of neurons, exchanges messages over a sparse graph, and revises its
concept/token predictions. This is a research prototype, not a validated model.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentConceptCloudNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_cloud: Sequence[int],
        width: int = 32,
        neurons: int = 64,
        active_k: int = 8,
        fanout: int = 4,
        candidate_clouds: int = 3,
        cycles: int = 4,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or len(token_cloud) != vocab_size:
            raise ValueError("token_cloud must assign every vocabulary token")
        if width < 4 or neurons < 8:
            raise ValueError("width must be >=4 and neurons >=8")
        if not 1 <= active_k <= neurons:
            raise ValueError("active_k must be in [1, neurons]")
        if not 1 <= fanout < neurons:
            raise ValueError("fanout must be in [1, neurons-1]")
        if cycles < 1:
            raise ValueError("cycles must be >=1")
        ids = torch.tensor(token_cloud, dtype=torch.long)
        unique = torch.unique(ids, sorted=True)
        expected = torch.arange(int(ids.max()) + 1)
        if int(ids.min()) < 0 or not torch.equal(unique, expected):
            raise ValueError("cloud IDs must be non-negative, contiguous and non-empty")
        n_clouds = int(ids.max()) + 1
        if not 1 <= candidate_clouds <= n_clouds:
            raise ValueError("candidate_clouds must be in [1, number of clouds]")
        self.vocab_size, self.num_clouds = vocab_size, n_clouds
        self.width, self.neurons = width, neurons
        self.active_k, self.fanout = active_k, fanout
        self.candidate_clouds, self.cycles = candidate_clouds, cycles
        self.register_buffer("token_cloud", ids)
        members = [torch.where(ids == c)[0] for c in range(n_clouds)]
        max_members = max(x.numel() for x in members)
        table = torch.full((n_clouds, max_members), -1, dtype=torch.long)
        for c, row in enumerate(members):
            table[c, :row.numel()] = row
        self.register_buffer("cloud_members", table)

        self.embedding = nn.Embedding(vocab_size, width)
        self.context_encoder = nn.GRU(width, width, batch_first=True)
        self.neuron_base = nn.Parameter(torch.randn(neurons, width) * 0.05)
        self.context_seed = nn.Linear(width, width)
        self.message = nn.Linear(width, width, bias=False)
        self.update_gate = nn.Linear(width * 2, width)
        self.update_candidate = nn.Linear(width * 2, width)
        self.cloud_query = nn.Linear(width, width, bias=False)
        self.cloud_embedding = nn.Parameter(torch.randn(n_clouds, width) * 0.05)
        self.cloud_head = nn.Linear(width, n_clouds)
        self.token_bias = nn.Parameter(torch.zeros(vocab_size))
        self.register_buffer("neighbors", self._make_graph(neurons, fanout))

    @staticmethod
    def _make_graph(n: int, fanout: int) -> torch.Tensor:
        rows = []
        for i in range(n):
            row = []
            d = 1
            while len(row) < fanout:
                row.append((i - d) % n)
                if len(row) < fanout:
                    row.append((i + d) % n)
                d += 1
            rows.append(row[:fanout])
        return torch.tensor(rows, dtype=torch.long)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
        cycles: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("context must have shape [batch, sequence]")
        n_cycles = self.cycles if cycles is None else cycles
        if n_cycles < 1:
            raise ValueError("cycles must be >=1")
        _, hidden = self.context_encoder(self.embedding(context))
        ctx = hidden[-1]
        batch = context.shape[0]

        # Persistent per-call network state: the same neuron-state tensor is
        # updated across cycles rather than replaced by independent layers.
        initial_scores = ctx @ self.neuron_base.T / math.sqrt(self.width)
        active_ids = initial_scores.topk(self.active_k, dim=-1).indices
        state = torch.tanh(self.neuron_base.unsqueeze(0) + self.context_seed(ctx).unsqueeze(1))
        active_mask = torch.zeros(batch, self.neurons, dtype=torch.bool, device=context.device)
        active_mask.scatter_(1, active_ids, True)
        state = state * active_mask.unsqueeze(-1)
        cycle_cloud_logits = []
        cycle_token_logits = []
        cycle_active_counts = []

        for _ in range(n_cycles):
            # Multi-hypothesis prediction: keep several clouds alive at once.
            pooled = state.sum(dim=1) / active_mask.sum(dim=1, keepdim=True).clamp_min(1)
            cloud_logits = self.cloud_head(pooled + ctx)
            candidate_ids = cloud_logits.topk(self.candidate_clouds, dim=-1).indices
            cycle_cloud_logits.append(cloud_logits)

            # Score tokens only in the union of currently proposed clouds.
            token_logits = torch.full(
                (batch, self.vocab_size), torch.finfo(ctx.dtype).min,
                device=context.device, dtype=ctx.dtype,
            )
            for b in range(batch):
                chosen = candidate_ids[b]
                if target is not None:
                    # Teacher-force the gold cloud into token scoring during training;
                    # router recall is measured separately from its own top-k guesses.
                    chosen = torch.cat([chosen, self.token_cloud[target[b]].view(1)])
                chosen = chosen.unique()
                token_ids = torch.cat([self.cloud_members[c][self.cloud_members[c] >= 0] for c in chosen])
                token_ids = token_ids.unique()
                scores = self.embedding(token_ids) @ (self.cloud_query(pooled[b] + ctx[b])) / math.sqrt(self.width)
                token_logits[b, token_ids] = scores + self.token_bias[token_ids]
            cycle_token_logits.append(token_logits)
            cycle_active_counts.append(active_mask.sum(dim=-1))

            # Sparse message passing from active source neurons to their graph
            # neighbors. Inactive neurons do not send messages this cycle.
            src = state
            neighbor_ids = self.neighbors
            messages = self.message(src[:, :, None, :].expand(-1, -1, self.fanout, -1))
            valid_edges = active_mask[:, :, None].expand(-1, -1, self.fanout)
            next_state = torch.zeros_like(state)
            next_count = torch.zeros(batch, self.neurons, 1, device=context.device, dtype=ctx.dtype)
            dst = neighbor_ids.unsqueeze(0).expand(batch, -1, -1)
            offsets = torch.arange(batch, device=context.device)[:, None, None] * self.neurons
            flat_dst = (dst + offsets).reshape(-1)
            flat_msg = (messages * valid_edges.unsqueeze(-1)).reshape(-1, self.width)
            next_state.view(batch * self.neurons, self.width).index_add_(0, flat_dst, flat_msg)
            next_count.view(batch * self.neurons, 1).index_add_(
                0, flat_dst, valid_edges.reshape(-1, 1).to(ctx.dtype)
            )
            neighbor_state = next_state / next_count.clamp_min(1.0)
            combined = torch.cat([state, neighbor_state], dim=-1)
            gate = torch.sigmoid(self.update_gate(combined))
            proposal = torch.tanh(self.update_candidate(combined))
            revised = gate * state + (1.0 - gate) * proposal
            # Reactivate a bounded set using context + current multi-cloud belief.
            cloud_belief = torch.softmax(cloud_logits, dim=-1) @ self.cloud_embedding
            routed_context = self.context_seed(ctx + cloud_belief)
            neuron_scores = (self.neuron_base.unsqueeze(0) * routed_context.unsqueeze(1)).sum(dim=-1)
            next_ids = neuron_scores.topk(self.active_k, dim=-1).indices
            next_mask = torch.zeros_like(active_mask)
            next_mask.scatter_(1, next_ids, True)
            state = revised * next_mask.unsqueeze(-1)
            active_mask = next_mask

        # Final cycle is the refined prediction. During training the target may
        # be unreachable if its cloud is not proposed, so report that explicitly.
        final_logits = cycle_token_logits[-1]
        result = {
            "logits": final_logits,
            "cloud_logits": cycle_cloud_logits[-1],
            "cycle_cloud_logits": torch.stack(cycle_cloud_logits, dim=1),
            "cycle_token_logits": torch.stack(cycle_token_logits, dim=1),
            "active_counts": torch.stack(cycle_active_counts, dim=1),
            "selected_clouds": cycle_cloud_logits[-1].topk(self.candidate_clouds, dim=-1).indices,
        }
        if target is not None:
            result["target_cloud"] = self.token_cloud[target]
            result["target_routed"] = (self.token_cloud[target, None] == result["selected_clouds"]).any(dim=-1)
        return result

    def loss(self, output: dict[str, torch.Tensor], target: torch.Tensor, router_weight: float = 1.0) -> dict[str, torch.Tensor]:
        token_loss = F.cross_entropy(output["logits"], target)
        cloud_loss = F.cross_entropy(output["cloud_logits"], self.token_cloud[target])
        # Intermediate cycle supervision teaches successive predictions to improve.
        intermediate = output["cycle_cloud_logits"]
        aux = torch.stack([
            F.cross_entropy(intermediate[:, i], self.token_cloud[target])
            for i in range(intermediate.shape[1])
        ]).mean()
        total = token_loss + router_weight * cloud_loss + 0.25 * aux
        return {"loss": total, "token_loss": token_loss, "cloud_loss": cloud_loss, "cycle_aux_loss": aux}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument("--clouds", type=int, default=12)
    parser.add_argument("--candidate-clouds", type=int, default=3)
    parser.add_argument("--neurons", type=int, default=48)
    parser.add_argument("--active-k", type=int, default=6)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if args.self_test:
        torch.manual_seed(17)
        vocab = max(args.vocab_size, args.clouds)
        assignment = [i % args.clouds for i in range(vocab)]
        model = RecurrentConceptCloudNet(
            vocab, assignment, args.width, args.neurons, args.active_k,
            candidate_clouds=args.candidate_clouds, cycles=args.cycles,
        )
        x = torch.randint(0, vocab, (4, 10))
        y = torch.randint(0, vocab, (4,))
        out = model(x, target=y)
        losses = model.loss(out, y)
        losses["loss"].backward()
        print(json.dumps({
            "status": "PASS",
            "cycles": args.cycles,
            "candidate_clouds": args.candidate_clouds,
            "target_cloud_routed_rate": out["target_routed"].float().mean().item(),
            "mean_active_neurons": out["active_counts"].float().mean().item(),
            "mean_scored_tokens_per_example": (out["logits"] > torch.finfo(out["logits"].dtype).min / 2).sum(-1).float().mean().item(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "loss_finite": bool(torch.isfinite(losses["loss"]).item()),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
