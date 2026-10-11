"""Expandable GRU core for cloud-side AIMENG experiments.

This module is a research prototype. It preserves the old network's function at the
moment of widening by zeroing connections from new hidden units into the old subnetwork
and zero-initializing new hidden-to-output weights. Rebuild the optimizer after expansion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


class ExpandableGRU(nn.Module):
    """A GRUCell unrolled over [batch, time, input] sequences with a linear readout."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("all dimensions must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: Tensor, hidden: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape [batch, time, input_size]")
        batch = inputs.shape[0]
        if hidden is None:
            hidden = inputs.new_zeros((batch, self.hidden_size))
        elif hidden.ndim != 2 or hidden.shape[0] != batch or hidden.shape[1] > self.hidden_size:
            raise ValueError("hidden must have shape [batch, <=hidden_size]")
        elif hidden.shape[1] < self.hidden_size:
            # An old checkpoint's hidden state can be padded with zeros after expansion.
            hidden = torch.cat(
                [hidden, hidden.new_zeros((batch, self.hidden_size - hidden.shape[1]))], dim=-1
            )
        outputs = []
        for t in range(inputs.shape[1]):
            hidden = self.cell(inputs[:, t, :], hidden)
            outputs.append(self.readout(hidden))
        if not outputs:
            return inputs.new_empty((batch, 0, self.output_size)), hidden
        return torch.stack(outputs, dim=1), hidden

    def expand_hidden(self, new_size: int, seed: int = 0) -> "ExpandableGRU":
        """Return a wider model while preserving the old model's outputs initially.

        This function does not mutate the original model. After expansion, create a fresh
        optimizer: existing optimizer state tensors have the old parameter shapes.
        """
        old_size = self.hidden_size
        if new_size <= old_size:
            raise ValueError(f"new_size ({new_size}) must exceed current size ({old_size})")
        device = self.cell.weight_ih.device
        dtype = self.cell.weight_ih.dtype
        # Isolate seeded initialization from the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            widened = ExpandableGRU(self.input_size, new_size, self.output_size).to(
                device=device, dtype=dtype
            )

        old_cell = self.cell
        new_cell = widened.cell
        with torch.no_grad():
            # GRU gate rows are stored as reset, update, new. Preserve each old gate block.
            for gate in range(3):
                old_rows = slice(gate * old_size, (gate + 1) * old_size)
                new_rows = slice(gate * new_size, gate * new_size + old_size)
                new_cell.weight_ih[new_rows].copy_(old_cell.weight_ih[old_rows])
                new_cell.weight_hh[new_rows, :old_size].copy_(
                    old_cell.weight_hh[old_rows, :old_size]
                )
                # All new-to-old recurrent connections remain zero so old hidden states match.
                new_cell.weight_hh[new_rows, old_size:].zero_()
                new_cell.bias_ih[new_rows].copy_(old_cell.bias_ih[old_rows])
                new_cell.bias_hh[new_rows].copy_(old_cell.bias_hh[old_rows])

            widened.readout.weight[:, :old_size].copy_(self.readout.weight)
            widened.readout.weight[:, old_size:].zero_()
            widened.readout.bias.copy_(self.readout.bias)
        return widened


@dataclass
class CapacityDecision:
    expand: bool
    reason: str
    suggested_size: int


class CapacityMonitor:
    """Conservative rule: expand only for persistent underfit, not a generalization gap."""

    def __init__(
        self,
        patience: int = 4,
        target_score: float = 0.95,
        max_generalization_gap: float = 0.10,
        min_delta: float = 0.002,
    ) -> None:
        if patience < 2:
            raise ValueError("patience must be at least 2")
        self.patience = patience
        self.target_score = target_score
        self.max_generalization_gap = max_generalization_gap
        self.min_delta = min_delta
        self._best_val = float("-inf")
        self._stale = 0

    def update(self, train_score: float, val_score: float, current_size: int) -> CapacityDecision:
        if current_size < 1:
            raise ValueError("current_size must be positive")
        if not (0.0 <= train_score <= 1.0 and 0.0 <= val_score <= 1.0):
            raise ValueError("scores must be in [0, 1]")
        # Plateau is measured on held-out validation performance, not training fit.
        improved = val_score > self._best_val + self.min_delta
        if improved:
            self._best_val = val_score
            self._stale = 0
        else:
            self._stale += 1

        gap = train_score - val_score
        if val_score >= self.target_score:
            return CapacityDecision(False, "validation target reached; keep current size", current_size)
        if train_score >= self.target_score:
            return CapacityDecision(
                False, "training target reached; investigate generalization/data before widening", current_size
            )
        if gap > self.max_generalization_gap:
            return CapacityDecision(
                False, "generalization gap is too large; improve data/regularization first", current_size
            )
        if self._stale < self.patience:
            return CapacityDecision(False, "validation has not plateaued long enough", current_size)
        # Grow gradually; never jump directly from a tiny network to a huge one.
        return CapacityDecision(True, "persistent underfit with small train/validation gap", current_size * 2)
