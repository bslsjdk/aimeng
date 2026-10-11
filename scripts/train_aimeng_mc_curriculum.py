#!/usr/bin/env python3
"""Train and evaluate a recurrent policy on AIMENG-MC synthetic parkour curricula.

Cloud-side PyTorch research prototype. It does not capture Minecraft or export an Android
AIMG model. Episodes are indexed from JSONL so the entire dataset is not loaded into RAM.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from scripts.generate_aimeng_mc_curriculum import (
    ACTIONS, BLOCK_NAMES, GRID_X, GRID_Y, GRID_Z, RADIUS, STAGE_BY_NAME, STAGES,
    WORLD_SIZE, _route_neighbors, make_course, make_observation,
)

ACTION_TO_ID = {name: i for i, name in enumerate(ACTIONS)}
PAD_ACTION = -100
VOXEL_COUNT = GRID_X * GRID_Y * GRID_Z
EMBED_DIM = 8
STATIC_FEATURE_SIZE = 11
ENCODER_WIDTH = 128
MAX_ROLLOUT_STEPS = 64


def allowed_actions_for_stage(stage_name: str) -> tuple[str, ...]:
    """Expose only mechanics unlocked by the current curriculum stage."""
    stage = STAGE_BY_NAME[stage_name]
    allowed = ["walk_forward", "stop"]
    if stage.index >= STAGE_BY_NAME["turning"].index:
        allowed.extend(["turn_left", "turn_right"])
    if stage.index >= STAGE_BY_NAME["small_jump"].index:
        allowed.append("jump_forward")
    if stage.index >= STAGE_BY_NAME["gap_jump"].index:
        allowed.append("gap_jump_forward")
    if stage.index >= STAGE_BY_NAME["rotating_jump"].index:
        allowed.extend(["rotating_jump_left", "rotating_jump_right"])
    if stage.index >= STAGE_BY_NAME["super_jump"].index:
        allowed.append("super_jump_forward")
    if stage.index >= STAGE_BY_NAME["double_jump"].index:
        allowed.append("double_jump_forward")
    return tuple(allowed)


def mask_action_logits(logits: Tensor, stage_name: str) -> Tensor:
    """Prevent locked future-stage actions from competing during training or evaluation."""
    allowed_ids = {ACTION_TO_ID[name] for name in allowed_actions_for_stage(stage_name)}
    if len(allowed_ids) == len(ACTIONS):
        return logits
    masked = logits.clone()
    for action_id in range(len(ACTIONS)):
        if action_id not in allowed_ids:
            masked[..., action_id] = float("-inf")
    return masked


class EpisodeFile:
    """Byte-offset index for JSONL episodes; keeps only per-episode metadata in RAM."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[dict[str, Any]] = []
        self.action_counts = {"train": Counter(), "validation": Counter(), "test": Counter(), "retention": Counter()}
        current_id = None
        current_start = 0
        current_split = None
        current_stage = None
        current_seed = None
        current_end = 0
        with self.path.open("rb") as handle:
            while True:
                line_start = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_end = handle.tell()
                record = json.loads(raw_line)
                episode_id = record["episode_id"]
                split = record.get("split")
                stage = record.get("curriculum_stage", {}).get("name")
                map_seed = int(record["map_seed"])
                if split in self.action_counts:
                    self.action_counts[split][record["teacher_action"]["action"]] += 1
                if current_id is None:
                    current_id, current_start = episode_id, line_start
                    current_split, current_stage, current_seed = split, stage, map_seed
                elif episode_id != current_id:
                    self.entries.append({
                        "episode_id": current_id, "start": current_start, "end": current_end,
                        "split": current_split, "stage": current_stage, "map_seed": current_seed,
                    })
                    current_id, current_start = episode_id, line_start
                    current_split, current_stage, current_seed = split, stage, map_seed
                elif (split, stage, map_seed) != (current_split, current_stage, current_seed):
                    raise ValueError(f"inconsistent episode metadata in {self.path}: {episode_id}")
                current_end = line_end
        if current_id is not None:
            self.entries.append({
                "episode_id": current_id, "start": current_start, "end": current_end,
                "split": current_split, "stage": current_stage, "map_seed": current_seed,
            })
        if not self.entries:
            raise ValueError(f"no episodes found in {self.path}")
        stage_names = {entry["stage"] for entry in self.entries}
        if len(stage_names) != 1 or None in stage_names:
            raise ValueError(f"one curriculum stage per file is required; found {stage_names}")
        self.stage_name = next(iter(stage_names))
        map_seeds = [entry["map_seed"] for entry in self.entries]
        if len(map_seeds) != len(set(map_seeds)):
            raise ValueError(f"duplicate map seeds in {self.path}; regenerate with the current generator")

    def read(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        with self.path.open("rb") as handle:
            handle.seek(entry["start"])
            raw = handle.read(entry["end"] - entry["start"])
        return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


class EpisodeDataset(Dataset):
    def __init__(self, source: EpisodeFile, split: str):
        self.source = source
        self.entries = [entry for entry in source.entries if entry["split"] == split]
        self.split = split

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return self.source.read(self.entries[index])


def collate_episodes(episodes: list[list[dict[str, Any]]]) -> dict[str, Tensor]:
    if not episodes:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(episodes)
    max_steps = max(len(episode) for episode in episodes)
    voxels = torch.zeros((batch_size, max_steps, VOXEL_COUNT), dtype=torch.long)
    goals = torch.zeros((batch_size, max_steps, 3), dtype=torch.float32)
    yaws = torch.zeros((batch_size, max_steps), dtype=torch.long)
    actions = torch.full((batch_size, max_steps), PAD_ACTION, dtype=torch.long)
    for row_index, episode in enumerate(episodes):
        count = len(episode)
        voxels[row_index, :count] = torch.tensor(
            [row["observation"]["voxels"] for row in episode], dtype=torch.long
        )
        goals[row_index, :count] = torch.tensor(
            [row["observation"]["goal_offset"] for row in episode], dtype=torch.float32
        )
        yaws[row_index, :count] = torch.tensor(
            [row["observation"]["yaw_bucket"] for row in episode], dtype=torch.long
        )
        try:
            actions[row_index, :count] = torch.tensor(
                [ACTION_TO_ID[row["teacher_action"]["action"]] for row in episode], dtype=torch.long
            )
        except KeyError as exc:
            raise ValueError(f"unknown teacher action {exc.args[0]!r}") from exc
    return {"voxels": voxels, "goals": goals, "yaws": yaws, "actions": actions}


class ParkourGRUPolicy(nn.Module):
    """Voxel embeddings, recurrent control state, and a fixed parkour action vocabulary."""

    def __init__(self, hidden_size: int = 1024, encoder_width: int = ENCODER_WIDTH):
        super().__init__()
        if hidden_size < 1 or encoder_width < 1:
            raise ValueError("hidden_size and encoder_width must be positive")
        self.hidden_size = hidden_size
        self.encoder_width = encoder_width
        self.block_embedding = nn.Embedding(len(BLOCK_NAMES), EMBED_DIM)
        feature_size = VOXEL_COUNT * EMBED_DIM + STATIC_FEATURE_SIZE
        self.encoder = nn.Sequential(nn.Linear(feature_size, encoder_width), nn.ReLU())
        self.cell = nn.GRUCell(encoder_width, hidden_size)
        # A separate geometry head gives goal alignment and "at goal" features a clean,
        # short gradient path instead of making them compete with hundreds of recurrent units.
        self.readout = nn.Linear(hidden_size, len(ACTIONS))
        self.geometry_head = nn.Linear(STATIC_FEATURE_SIZE, len(ACTIONS), bias=False)

    def forward(
        self, voxels: Tensor, goals: Tensor, yaws: Tensor, hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if voxels.ndim != 3 or voxels.shape[-1] != VOXEL_COUNT:
            raise ValueError(f"voxels must have shape [batch,time,{VOXEL_COUNT}]")
        batch, steps, _ = voxels.shape
        if goals.shape != (batch, steps, 3) or yaws.shape != (batch, steps):
            raise ValueError("goals/yaws must align with [batch,time]")
        if hidden is None:
            hidden = torch.zeros((batch, self.hidden_size), device=voxels.device, dtype=torch.float32)
        elif hidden.shape[0] != batch or hidden.shape[1] > self.hidden_size:
            raise ValueError("hidden state shape is incompatible with this model")
        elif hidden.shape[1] < self.hidden_size:
            hidden = torch.cat(
                [hidden, hidden.new_zeros((batch, self.hidden_size - hidden.shape[1]))], dim=-1
            )
        embedded = self.block_embedding(voxels).flatten(start_dim=2)
        normalized_goals = goals / float(WORLD_SIZE // 2)
        yaw_features = F.one_hot(yaws.clamp(min=0, max=3), num_classes=4).to(embedded.dtype)
        dx = normalized_goals[..., 0]
        dz = normalized_goals[..., 2]
        forward_x = yaw_features[..., 1] - yaw_features[..., 3]
        forward_z = yaw_features[..., 2] - yaw_features[..., 0]
        right_x = yaw_features[..., 0] - yaw_features[..., 2]
        right_z = yaw_features[..., 1] - yaw_features[..., 3]
        forward_delta = (dx * forward_x + dz * forward_z).unsqueeze(-1)
        lateral_delta = (dx * right_x + dz * right_z).unsqueeze(-1)
        manhattan_distance = (dx.abs() + dz.abs()).unsqueeze(-1)
        at_goal = ((dx == 0) & (dz == 0)).to(embedded.dtype).unsqueeze(-1)
        static_features = torch.cat([
            normalized_goals.to(embedded.dtype), yaw_features,
            forward_delta, lateral_delta, manhattan_distance, at_goal,
        ], dim=-1)
        features = torch.cat([embedded, static_features], dim=-1)
        encoded = self.encoder(features)
        logits = []
        for step in range(steps):
            hidden = self.cell(encoded[:, step, :], hidden)
            logits.append(self.readout(hidden) + self.geometry_head(static_features[:, step, :]))
        if not logits:
            return voxels.new_empty((batch, 0, len(ACTIONS)), dtype=torch.float32), hidden
        return torch.stack(logits, dim=1), hidden

    def expand_hidden(self, new_size: int, seed: int = 0) -> "ParkourGRUPolicy":
        """Widen hidden state while preserving existing logits at expansion time."""
        old_size = self.hidden_size
        if new_size <= old_size:
            raise ValueError("new_size must be larger than the current hidden size")
        device = self.cell.weight_ih.device
        dtype = self.cell.weight_ih.dtype
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            widened = ParkourGRUPolicy(new_size, self.encoder_width).to(device=device, dtype=dtype)
        with torch.no_grad():
            widened.block_embedding.load_state_dict(self.block_embedding.state_dict())
            widened.encoder.load_state_dict(self.encoder.state_dict())
            for gate in range(3):
                old_rows = slice(gate * old_size, (gate + 1) * old_size)
                new_rows = slice(gate * new_size, gate * new_size + old_size)
                widened.cell.weight_ih[new_rows].copy_(self.cell.weight_ih[old_rows])
                widened.cell.weight_hh[new_rows, :old_size].copy_(
                    self.cell.weight_hh[old_rows, :old_size]
                )
                widened.cell.weight_hh[new_rows, old_size:].zero_()
                widened.cell.bias_ih[new_rows].copy_(self.cell.bias_ih[old_rows])
                widened.cell.bias_hh[new_rows].copy_(self.cell.bias_hh[old_rows])
            # Zero new recurrent-to-action weights so old logits are unchanged, and
            # copy the parallel geometry head exactly.
            widened.readout.weight[:, :old_size].copy_(self.readout.weight)
            widened.readout.weight[:, old_size:].zero_()
            widened.readout.bias.copy_(self.readout.bias)
            widened.geometry_head.load_state_dict(self.geometry_head.state_dict())
        return widened


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def accuracy_counts(logits: Tensor, targets: Tensor) -> tuple[int, int]:
    mask = targets != PAD_ACTION
    if not mask.any():
        return 0, 0
    correct = int((logits.argmax(dim=-1)[mask] == targets[mask]).sum().item())
    return correct, int(mask.sum().item())


def compute_class_weights(sources: list[EpisodeFile], device: torch.device) -> Tensor:
    counts = Counter()
    for source in sources:
        counts.update(source.action_counts["train"])
    values = torch.tensor([counts.get(action, 0) for action in ACTIONS], dtype=torch.float32, device=device)
    present = values > 0
    weights = torch.ones_like(values)
    if present.any():
        mean_count = values[present].mean()
        # Inverse frequency gives every action present in this stage comparable total
        # training weight. This is deliberate for stop/jump/turn decisions that are rare but
        # essential for completing a route. Actions absent in this stage remain weight 1.
        weights[present] = mean_count / values[present]
        weighted_mean = (weights[present] * values[present]).sum() / values[present].sum()
        weights[present] /= weighted_mean.clamp_min(1e-8)
    return weights


def evaluate_action_accuracy(
    model: ParkourGRUPolicy, loader: DataLoader, device: torch.device,
    class_weights: Tensor | None = None,
    stage_name: str | None = None,
) -> tuple[float, float, int, dict[str, float]]:
    model.eval()
    correct_total = count_total = 0
    loss_total = loss_denominator = 0.0
    per_class_correct = [0] * len(ACTIONS)
    per_class_count = [0] * len(ACTIONS)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits, _ = model(batch["voxels"], batch["goals"], batch["yaws"])
            if stage_name is not None:
                logits = mask_action_logits(logits, stage_name)
            targets = batch["actions"]
            mask = targets != PAD_ACTION
            correct, count = accuracy_counts(logits, targets)
            correct_total += correct
            count_total += count
            if count:
                selected_targets = targets[mask]
                loss_sum = F.cross_entropy(
                    logits[mask], selected_targets, weight=class_weights, reduction="sum"
                )
                loss_total += float(loss_sum.item())
                if class_weights is None:
                    loss_denominator += count
                else:
                    loss_denominator += float(class_weights[selected_targets].sum().item())
                predictions = logits.argmax(dim=-1)
                for action_id in range(len(ACTIONS)):
                    class_mask = targets == action_id
                    class_count = int(class_mask.sum().item())
                    if class_count:
                        per_class_count[action_id] += class_count
                        per_class_correct[action_id] += int(
                            (predictions[class_mask] == action_id).sum().item()
                        )
    recalls = {
        ACTIONS[index]: per_class_correct[index] / per_class_count[index]
        for index in range(len(ACTIONS)) if per_class_count[index]
    }
    if count_total == 0:
        return 0.0, 0.0, 0, recalls
    return loss_total / max(loss_denominator, 1e-8), correct_total / count_total, count_total, recalls


def train_one_epoch(
    model: ParkourGRUPolicy, loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device, class_weights: Tensor, stage_name: str,
) -> tuple[float, float]:
    model.train()
    loss_total = loss_denominator = 0.0
    correct_total = count_total = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch["voxels"], batch["goals"], batch["yaws"])
        logits = mask_action_logits(logits, stage_name)
        targets = batch["actions"]
        mask = targets != PAD_ACTION
        if not mask.any():
            continue
        selected_targets = targets[mask]
        loss = F.cross_entropy(logits[mask], selected_targets, weight=class_weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        count = int(mask.sum().item())
        denominator = float(class_weights[selected_targets].sum().item())
        loss_total += float(loss.detach().item()) * denominator
        loss_denominator += denominator
        correct, _ = accuracy_counts(logits.detach(), targets)
        correct_total += correct
        count_total += count
    return loss_total / max(loss_denominator, 1e-8), correct_total / max(1, count_total)


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters without allocating additional model tensors."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimated_weight_mib(model: nn.Module) -> float:
    """Approximate fp32 parameter storage only; excludes activations/framework/runtime overhead."""
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024.0 * 1024.0)


def clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def evaluate_rollouts(
    model: ParkourGRUPolicy, dataset: EpisodeDataset, device: torch.device,
) -> dict[str, Any]:
    """Roll out policy on reconstructed held-out symbolic courses."""
    if not len(dataset):
        return {"success_rate": 0.0, "successes": 0, "episodes": 0}
    model.eval()
    successes = 0
    with torch.no_grad():
        for episode_index in range(len(dataset)):
            rows = dataset[episode_index]
            first = rows[0]
            stage = STAGE_BY_NAME[first["curriculum_stage"]["name"]]
            map_seed = int(first["map_seed"])
            platforms, obstacles, start, goal, start_yaw, _ = make_course(stage, map_seed)
            state = (start[0], start[1], start_yaw)
            hidden = None
            solved = False
            episode_id = f"rollout-{stage.name}-{map_seed}"
            for step in range(MAX_ROLLOUT_STEPS):
                x, z, yaw = state
                record = make_observation(
                    (x, z), yaw, goal, platforms, obstacles, stage,
                    map_seed, episode_id, step, "test",
                )
                observation = record["observation"]
                voxels = torch.tensor(observation["voxels"], dtype=torch.long, device=device).view(1, 1, -1)
                goals = torch.tensor(observation["goal_offset"], dtype=torch.float32, device=device).view(1, 1, 3)
                yaws = torch.tensor([[yaw]], dtype=torch.long, device=device)
                logits, hidden = model(voxels, goals, yaws, hidden)
                masked_logits = mask_action_logits(logits[0, 0], stage.name)
                action = ACTIONS[int(masked_logits.argmax().item())]
                if action == "stop":
                    solved = state[:2] == goal
                    break
                legal_transitions = {
                    candidate_action: next_state
                    for next_state, candidate_action in _route_neighbors(
                        state, platforms, obstacles, stage
                    )
                }
                if action in legal_transitions:
                    state = legal_transitions[action]
                # Invalid actions consume a step and leave the agent in place; policy may recover.
            successes += int(solved)
    return {"success_rate": successes / len(dataset), "successes": successes, "episodes": len(dataset)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="current-stage JSONL dataset")
    parser.add_argument("--stage", choices=tuple(STAGE_BY_NAME), required=True)
    parser.add_argument("--output", type=Path, default=Path("aimeng_mc_policy.pt"))
    parser.add_argument("--report", type=Path, help="metrics JSON path; default: <output>.metrics.json")
    parser.add_argument("--resume", type=Path, help="previous curriculum checkpoint")
    parser.add_argument("--replay-data", type=Path, action="append", default=[],
                        help="previous-stage JSONL replay data; may be repeated")
    parser.add_argument("--prior-report", type=Path, help="prior stage metrics JSON to update")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=1024, help="used for a fresh model")
    parser.add_argument("--auto-expand", action="store_true",
                        help="double hidden size on persistent underfit with a small train/validation gap")
    parser.add_argument("--max-hidden-size", type=int, default=2048)
    parser.add_argument("--expand-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_size < 1 or args.max_hidden_size < 1:
        parser.error("epochs, batch size, and hidden sizes must be positive")
    if args.hidden_size > 4096 or args.max_hidden_size > 4096:
        parser.error("until end-to-end memory is benchmarked, hidden sizes are capped at 4096")
    if args.max_hidden_size < args.hidden_size:
        parser.error("--max-hidden-size must be >= --hidden-size")
    if args.expand_patience < 2 or args.threads < 1:
        parser.error("--expand-patience must be >= 2 and --threads must be positive")

    stage = STAGE_BY_NAME[args.stage]
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    current_file = EpisodeFile(args.data)
    if current_file.stage_name != stage.name:
        parser.error(f"--data contains {current_file.stage_name}, not requested stage {stage.name}")
    replay_files = [EpisodeFile(path) for path in args.replay_data]
    for replay in replay_files:
        if STAGE_BY_NAME[replay.stage_name].index >= stage.index:
            parser.error(f"replay stage {replay.stage_name} must precede current stage {stage.name}")

    current_train_dataset = EpisodeDataset(current_file, "train")
    train_datasets = [current_train_dataset]
    train_datasets.extend(EpisodeDataset(source, "train") for source in replay_files)
    train_datasets = [dataset for dataset in train_datasets if len(dataset)]
    val_dataset = EpisodeDataset(current_file, "validation")
    test_dataset = EpisodeDataset(current_file, "test")
    if not len(current_train_dataset):
        parser.error("current stage has no training episodes; generate at least 2 episodes")
    if not train_datasets:
        parser.error("no training episodes; generate more episodes")
    if not len(val_dataset):
        parser.error("no validation episodes; generate at least 9 episodes")
    if not len(test_dataset):
        parser.error("no held-out test episodes; generate at least 10 episodes")

    combined_train = ConcatDataset(train_datasets)
    sampler = None
    shuffle = True
    if len(train_datasets) > 1:
        weights = []
        for dataset in train_datasets:
            weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double), num_samples=sum(len(dataset) for dataset in train_datasets),
            replacement=True, generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
    train_loader = DataLoader(
        combined_train, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        collate_fn=collate_episodes,
    )
    train_eval_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_episodes)
    current_train_eval_loader = DataLoader(current_train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_episodes)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_episodes)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("format") != "aimeng-mc-curriculum-checkpoint-v3":
            parser.error("unsupported checkpoint format")
        if checkpoint.get("action_vocab") != list(ACTIONS):
            parser.error("checkpoint action vocabulary does not match this code version")
        checkpoint_stage = checkpoint.get("stage")
        if checkpoint_stage not in STAGE_BY_NAME or STAGE_BY_NAME[checkpoint_stage].index > stage.index:
            parser.error("resume checkpoint must come from the current or an earlier curriculum stage")
        model = ParkourGRUPolicy(int(checkpoint["hidden_size"]), int(checkpoint["encoder_width"])).to(device)
        if model.hidden_size > args.max_hidden_size:
            parser.error("resume checkpoint hidden size exceeds --max-hidden-size")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = ParkourGRUPolicy(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # The previous run showed oscillating left/right recall at 3e-4. Start more
    # conservatively and reduce LR when validation loss stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=0.001,
        threshold_mode="rel", min_lr=1e-5,
    )
    class_weights = compute_class_weights([current_file, *replay_files], device)

    best_val_accuracy = -1.0
    best_val_episode_success = -1.0
    best_state = clone_state_dict(model)
    best_hidden_size = model.hidden_size
    stale_epochs = 0
    history = []
    expansion_events = []
    for epoch in range(args.epochs):
        train_loss, combined_train_accuracy = train_one_epoch(
            model, train_loader, optimizer, device, class_weights, stage.name
        )
        current_train_loss, current_train_accuracy, current_train_count, current_train_recalls = evaluate_action_accuracy(
            model, current_train_eval_loader, device, class_weights, stage.name
        )
        val_loss, val_accuracy, val_count, val_recalls = evaluate_action_accuracy(
            model, val_loader, device, class_weights, stage.name
        )
        current_train_macro_recall = sum(current_train_recalls.values()) / max(1, len(current_train_recalls))
        val_macro_recall = sum(val_recalls.values()) / max(1, len(val_recalls))
        val_rollout = evaluate_rollouts(model, val_dataset, device)
        scheduler.step(val_loss)
        improved = (
            val_rollout["success_rate"] > best_val_episode_success + 1e-9
            or (abs(val_rollout["success_rate"] - best_val_episode_success) <= 1e-9
                and val_accuracy > best_val_accuracy + 0.002)
        )
        if improved:
            best_val_episode_success = val_rollout["success_rate"]
            best_val_accuracy = val_accuracy
            best_state = clone_state_dict(model)
            best_hidden_size = model.hidden_size
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch + 1, "combined_train_loss": train_loss,
            "combined_train_action_accuracy": combined_train_accuracy,
            "current_stage_train_loss": current_train_loss,
            "current_stage_train_action_accuracy": current_train_accuracy,
            "current_stage_train_steps": current_train_count,
            "current_stage_train_action_recall": current_train_recalls,
            "current_stage_train_macro_recall": current_train_macro_recall,
            "validation_loss": val_loss,
            "validation_action_accuracy": val_accuracy, "validation_steps": val_count,
            "validation_action_recall": val_recalls,
            "validation_macro_recall": val_macro_recall,
            "validation_episode_success_rate": val_rollout["success_rate"],
            "validation_episode_successes": val_rollout["successes"],
            "validation_rollout_episodes": val_rollout["episodes"],
            "hidden_size": model.hidden_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(json.dumps(history[-1]), flush=True)

        can_expand = (
            args.auto_expand and epoch < args.epochs - 1
            and model.hidden_size < args.max_hidden_size
            and stale_epochs >= args.expand_patience
            and current_train_macro_recall < 0.95 and val_macro_recall < 0.90
            and abs(current_train_macro_recall - val_macro_recall) <= 0.10
        )
        if can_expand:
            old_size = model.hidden_size
            new_size = min(old_size * 2, args.max_hidden_size)
            model = model.expand_hidden(new_size, seed=args.seed + epoch + 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, threshold=0.001,
                threshold_mode="rel", min_lr=1e-5,
            )
            # Widen the best checkpoint as well as the live model. Expansion preserves
            # old logits, so the previous best policy remains available at the new width.
            # Do not reset its validation score: doing so can discard a much better
            # pre-expansion policy merely because the current epoch triggered widening.
            best_model = ParkourGRUPolicy(old_size, model.encoder_width).to(device)
            best_model.load_state_dict(best_state)
            best_model = best_model.expand_hidden(new_size, seed=args.seed + epoch + 1)
            best_state = clone_state_dict(best_model)
            best_hidden_size = new_size
            stale_epochs = 0
            expansion_events.append({
                "epoch": epoch + 1, "from": old_size, "to": new_size,
                "reason": "validation plateau with persistent underfit and small generalization gap",
            })
            print(json.dumps({"event": "hidden_size_expanded", **expansion_events[-1]}), flush=True)

    if model.hidden_size != best_hidden_size:
        raise RuntimeError("best checkpoint hidden size does not match final model")
    model.load_state_dict(best_state)
    train_loss, combined_train_accuracy, train_steps, combined_train_recalls = evaluate_action_accuracy(
        model, train_eval_loader, device, class_weights, stage.name
    )
    current_train_loss, current_train_accuracy, current_train_steps, current_train_recalls = evaluate_action_accuracy(
        model, current_train_eval_loader, device, class_weights, stage.name
    )
    val_loss, val_accuracy, val_steps, val_recalls = evaluate_action_accuracy(
        model, val_loader, device, class_weights, stage.name
    )
    test_loss, test_accuracy, test_steps, test_recalls = evaluate_action_accuracy(
        model, test_loader, device, class_weights, stage.name
    )
    current_validation_rollout = evaluate_rollouts(model, val_dataset, device)
    current_rollout = evaluate_rollouts(model, test_dataset, device)

    prior_metrics: dict[str, Any] = {}
    if args.prior_report:
        prior_data = json.loads(args.prior_report.read_text(encoding="utf-8"))
        prior_metrics = dict(prior_data.get("stages", prior_data))
    for prior_file in replay_files:
        prior_stage_name = prior_file.stage_name
        prior_retention = EpisodeDataset(prior_file, "retention")
        retention = evaluate_rollouts(model, prior_retention, device)
        metric = dict(prior_metrics.get(prior_stage_name, {}))
        metric["retention_success_rate"] = retention["success_rate"]
        metric["retention_episodes"] = retention["episodes"]
        prior_metrics[prior_stage_name] = metric

    prior_metrics[stage.name] = {
        **prior_metrics.get(stage.name, {}),
        # Progression is based on validation episodes. Final test episodes stay separate.
        "success_rate": current_validation_rollout["success_rate"],
        "episodes": current_validation_rollout["episodes"],
        "test_success_rate": current_rollout["success_rate"],
        "test_episodes": current_rollout["episodes"],
        "teacher_action_accuracy": test_accuracy,
        "teacher_action_steps": test_steps,
        "teacher_action_recall": test_recalls,
        "train_teacher_action_accuracy": current_train_accuracy,
        "validation_teacher_action_accuracy": val_accuracy,
        "validation_episode_success_rate": current_validation_rollout["success_rate"],
    }
    report = {
        "format": "aimeng-mc-curriculum-metrics-v1",
        "stage": stage.name,
        "hidden_size": model.hidden_size,
        "encoder_width": model.encoder_width,
        "trainable_parameters": parameter_count(model),
        "fp32_weight_only_mib_estimate": round(estimated_weight_mib(model), 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "train_episodes": len(combined_train),
        "validation_episodes": len(val_dataset),
        "test_episodes": len(test_dataset),
        "combined_train_action_accuracy": combined_train_accuracy,
        "current_stage_train_action_accuracy": current_train_accuracy,
        "current_stage_train_action_steps": current_train_steps,
        "current_stage_train_action_recall": current_train_recalls,
        "combined_train_action_recall": combined_train_recalls,
        "validation_action_accuracy": val_accuracy,
        "validation_action_recall": val_recalls,
        "validation_episode_success_rate": current_validation_rollout["success_rate"],
        "test_action_accuracy": test_accuracy,
        "test_action_recall": test_recalls,
        "class_weights": {ACTIONS[i]: float(class_weights[i].item()) for i in range(len(ACTIONS))},
        "test_episode_success_rate": current_rollout["success_rate"],
        "test_episode_successes": current_rollout["successes"],
        "test_episode_count": current_rollout["episodes"],
        "expansion_events": expansion_events,
        "history": history,
        "stages": prior_metrics,
        "note": "Synthetic grid-world rollouts only; not live Bedrock parkour competence.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "aimeng-mc-curriculum-checkpoint-v3",
        "stage": stage.name,
        "hidden_size": model.hidden_size,
        "encoder_width": model.encoder_width,
        "action_vocab": list(ACTIONS),
        "model_state_dict": model.state_dict(),
        "seed": args.seed,
        "grid_size": [GRID_X, GRID_Y, GRID_Z],
    }, args.output)
    report_path = args.report or args.output.with_suffix(".metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(args.output), "report": str(report_path), "stage": stage.name,
        "hidden_size": model.hidden_size, "trainable_parameters": parameter_count(model),
        "fp32_weight_only_mib_estimate": round(estimated_weight_mib(model), 3),
        "test_teacher_action_accuracy": test_accuracy,
        "validation_episode_success_rate": current_validation_rollout["success_rate"],
        "test_episode_success_rate": current_rollout["success_rate"],
        "test_episodes": current_rollout["episodes"], "expansion_events": expansion_events,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
