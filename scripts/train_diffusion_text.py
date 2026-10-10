#!/usr/bin/env python3
"""Train a small sparse-diffusion character model. Smoke data is NOT language evidence."""
from __future__ import annotations
import argparse, json, math, random, time, resource
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SMOKE_TEXT = (
    "AIMENG learns by spreading signals through a sparse neuron graph.\n"
    "Useful neurons pass messages; inactive neurons keep their saved state.\n"
    "The network updates its state repeatedly and stops when more work is not useful.\n"
    "输入进入网络后，相关神经元开始扩散信息。\n"
    "没有参与当前任务的神经元保持休眠，需要时再被唤醒。\n"
    "网络通过训练学习预测字符，但小规模实验不代表通用智能。\n"
) * 100

class DiffusionTextModel(nn.Module):
    """Sparse message passing with bounded active nodes and a learned halt head."""
    def __init__(self, vocab_size: int, neurons: int = 128, width: int = 32,
                 active_k: int = 8, fanout: int = 8, max_steps: int = 6,
                 energy_decay: float = 0.9):
        super().__init__()
        if neurons < 8 or width < 4 or not 1 <= active_k <= neurons or not 1 <= fanout < neurons:
            raise ValueError("invalid graph dimensions")
        self.neurons, self.width, self.active_k = neurons, width, active_k
        self.fanout, self.max_steps, self.energy_decay = fanout, max_steps, energy_decay
        self.embedding = nn.Embedding(vocab_size, width)
        # Position embeddings are essential: a mean of token embeddings alone
        # cannot distinguish sequences such as "我喜欢你" and "你喜欢我".
        self.position_embedding = nn.Embedding(512, width)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.node_embedding = nn.Parameter(torch.randn(neurons, width) * 0.08)
        self.context_proj = nn.Linear(width, width)
        self.self_proj = nn.Linear(width, width, bias=False)
        self.message_proj = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width * 2, width)
        self.decoder = nn.Linear(width, vocab_size)
        self.halt_head = nn.Linear(width, 1)
        # Deterministic bidirectional ring graph: each node can send waves both ways.
        # Even fanout gives symmetric +/- offsets; odd fanout adds one extra forward edge.
        rows = []
        for i in range(neurons):
            row = []
            for distance in range(1, fanout // 2 + 1):
                row.extend(((i - distance) % neurons, (i + distance) % neurons))
            if fanout % 2:
                row.append((i + fanout // 2 + 1) % neurons)
            rows.append(row)
        self.register_buffer("neighbors", torch.tensor(rows, dtype=torch.long))
        self.edge_logits = nn.Parameter(torch.zeros(neurons, fanout))

    def forward(self, tokens: torch.Tensor, early_stop: bool = False,
                halt_threshold: float = 0.80, convergence_threshold: float = 0.025,
                min_steps: int = 2):
        batch = tokens.shape[0]
        if tokens.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError("input context exceeds positional embedding limit")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        token_embeddings = self.embedding(tokens)
        positional = self.position_embedding(positions)[None, :, :]
        # Multiplicative interaction binds each token to its position before pooling;
        # adding position vectors then averaging would cancel order information.
        ordered_embeddings = token_embeddings * (1.0 + positional)
        context = torch.tanh(self.context_proj(ordered_embeddings.mean(dim=1)))
        route_scores = context @ self.node_embedding.T / math.sqrt(self.width)
        seed_ids = route_scores.topk(self.active_k, dim=1).indices
        seed_state = torch.tanh(context[:, None, :] + self.node_embedding[seed_ids])
        state = torch.zeros(batch, self.neurons, self.width, device=tokens.device, dtype=context.dtype)
        state = state.scatter(1, seed_ids[..., None].expand(-1, -1, self.width), seed_state)
        energy = torch.zeros(batch, self.neurons, device=tokens.device, dtype=context.dtype)
        energy = energy.scatter(1, seed_ids, torch.ones_like(seed_ids, dtype=context.dtype))
        edge_weights = torch.softmax(self.edge_logits, dim=-1)
        logits_by_step, halts, deltas = [], [], []
        batch_offsets = torch.arange(batch, device=tokens.device)[:, None, None] * self.neurons
        for step in range(self.max_steps):
            active_ids = energy.topk(self.active_k, dim=1).indices
            src_state = state.gather(1, active_ids[..., None].expand(-1, -1, self.width))
            dst_ids = self.neighbors[active_ids]
            weights = edge_weights[active_ids]
            edge_messages = src_state[:, :, None, :] * weights[..., None]
            flat_dst = (dst_ids + batch_offsets).reshape(-1)
            flat_msg = edge_messages.reshape(-1, self.width)
            msg_flat = torch.zeros(batch * self.neurons, self.width, device=tokens.device, dtype=context.dtype)
            msg_flat = msg_flat.index_add(0, flat_dst, flat_msg)
            messages = msg_flat.reshape(batch, self.neurons, self.width)
            edge_energy = edge_messages.abs().mean(dim=-1)
            energy_flat = torch.zeros(batch * self.neurons, device=tokens.device, dtype=context.dtype)
            energy_flat = energy_flat.index_add(0, flat_dst, edge_energy.reshape(-1))
            energy = energy * self.energy_decay + energy_flat.reshape(batch, self.neurons)
            # Only selected destination nodes receive learned state updates.
            candidate_ids = energy.topk(self.active_k, dim=1).indices
            old = state.gather(1, candidate_ids[..., None].expand(-1, -1, self.width))
            incoming = messages.gather(1, candidate_ids[..., None].expand(-1, -1, self.width))
            proposal = torch.tanh(self.self_proj(old) + self.message_proj(incoming))
            gate = torch.sigmoid(self.gate(torch.cat([old, incoming], dim=-1)))
            updated = gate * proposal + (1.0 - gate) * old
            state = state.scatter(1, candidate_ids[..., None].expand(-1, -1, self.width), updated)
            delta = (updated - old).abs().mean(dim=(1, 2))
            deltas.append(delta)
            pooled = (state * energy[..., None]).sum(dim=1) / energy.sum(dim=1, keepdim=True).clamp_min(1e-6)
            logits_by_step.append(self.decoder(pooled))
            halt = torch.sigmoid(self.halt_head(pooled)).squeeze(-1)
            if step + 1 < min_steps:
                halt = torch.zeros_like(halt)
            halts.append(halt)
            if early_stop and step + 1 >= min_steps and bool(torch.all((halt >= halt_threshold) & (delta <= convergence_threshold))):
                break
        survival = torch.ones(batch, device=tokens.device, dtype=context.dtype)
        mixed = torch.zeros_like(logits_by_step[0])
        expected_steps = torch.zeros(batch, device=tokens.device, dtype=context.dtype)
        for logits, halt in zip(logits_by_step, halts):
            mixed = mixed + (survival * halt)[:, None] * logits
            expected_steps = expected_steps + survival
            survival = survival * (1.0 - halt)
        mixed = mixed + survival[:, None] * logits_by_step[-1]
        return {"logits": mixed, "step_logits": logits_by_step, "halt_probs": halts,
                "expected_steps": expected_steps, "steps_used": len(logits_by_step),
                "mean_delta": torch.stack(deltas).mean().detach()}

def make_vocab(text: str, max_vocab: int = 2048):
    # Bound vocabulary to the Android runtime's current 2048-token safety limit.
    # Keep frequent Chinese characters and role markers; rare characters map to <unk>.
    from collections import Counter
    if max_vocab < 2:
        raise ValueError("max_vocab must be at least 2")
    counts = Counter(text)
    chars = [ch for ch, _ in counts.most_common(max_vocab - 1)]
    itos = ["<unk>"] + chars
    return {ch: i + 1 for i, ch in enumerate(chars)}, itos

def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi.get(ch, 0) for ch in text]

def batches(ids: list[int], context: int, batch_size: int, steps: int, seed: int):
    rng = random.Random(seed)
    max_start = len(ids) - context - 1
    if max_start < 1:
        raise ValueError(f"text needs at least context_length+2 characters ({context + 2})")
    for _ in range(steps):
        starts = [rng.randrange(max_start) for _ in range(batch_size)]
        x = torch.tensor([ids[s:s+context] for s in starts], dtype=torch.long)
        y = torch.tensor([ids[s+context] for s in starts], dtype=torch.long)
        yield x, y

def evaluate(model, ids, context, limit=512, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    if len(ids) <= context + 1:
        return float("nan")
    starts = list(range(min(len(ids)-context-1, limit)))
    total = 0.0
    with torch.no_grad():
        for offset in range(0, len(starts), 64):
            ss = starts[offset:offset+64]
            x = torch.tensor([ids[s:s+context] for s in ss], dtype=torch.long, device=device)
            y = torch.tensor([ids[s+context] for s in ss], dtype=torch.long, device=device)
            total += F.cross_entropy(model(x, early_stop=False)["logits"], y, reduction="sum").item()
    return total / len(starts)

def process_memory_mib():
    """Return current PSS-or-RSS MiB and process peak RSS MiB on Linux."""
    pss = None
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                pss = float(line.split()[1]) / 1024.0
                break
    except (OSError, ValueError, IndexError):
        pass
    rss = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = float(line.split()[1]) / 1024.0
                break
    except (OSError, ValueError, IndexError):
        pass
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    if pss is not None:
        return pss, "PSS", peak_rss
    if rss is not None:
        return rss, "RSS_FALLBACK", peak_rss
    return peak_rss, "PEAK_RSS_FALLBACK", peak_rss

def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    source = Path(args.text).read_text(encoding="utf-8") if args.text else SMOKE_TEXT
    if len(source) < args.context + 20:
        raise ValueError("corpus is too short for the chosen context length")
    cut = max(args.context + 2, int(len(source) * 0.9))
    train_text, val_text = source[:cut], source[cut:]
    stoi, itos = make_vocab(source, args.vocab_size)
    train_ids, val_ids = encode(train_text, stoi), encode(val_text, stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTextModel(len(itos), args.neurons, args.width, args.active_k, args.fanout, args.max_steps).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps({"event":"device_selected", "device":str(device), "gpu":torch.cuda.get_device_name(0) if device.type == "cuda" else None}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    initial_loss = evaluate(model, val_ids, args.context, device=device)
    started = time.time(); model.train(); last_loss = float("nan"); completed_steps = 0
    memory_stopped = False; peak_observed_mib = 0.0; memory_metric = "unknown"
    for step, (x, y) in enumerate(batches(train_ids, args.context, args.batch_size, args.steps, args.seed + 1), 1):
        if step % 25 == 0 or step == 1:
            current_mib, memory_metric, _ = process_memory_mib()
            peak_observed_mib = max(peak_observed_mib, current_mib)
            if current_mib >= args.memory_stop_mib:
                memory_stopped = True
                print(json.dumps({"event":"safe_stop_memory_limit", "step":step, "measured_mib":round(current_mib,1), "metric":memory_metric, "limit_mib":args.memory_stop_mib}), flush=True)
                break
        completed_steps = step
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, early_stop=False)
        task_loss = F.cross_entropy(out["logits"], y)
        loss = task_loss + args.step_penalty * out["expected_steps"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        last_loss = float(task_loss.detach())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step":step, "steps":args.steps, "train_loss":round(last_loss,5),
                              "mean_expected_steps":round(float(out["expected_steps"].mean().detach()),3)}, ensure_ascii=False), flush=True)
    final_loss = evaluate(model, val_ids, args.context, device=device)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format":"aimeng-sparse-diffusion-char-v1", "config":vars(args), "stoi":stoi, "itos":itos,
                  "model_state":model.state_dict(), "initial_validation_loss":initial_loss,
                  "final_validation_loss":final_loss, "train_loss_last":last_loss,
                  "validation_characters":len(val_ids), "source":"user_corpus" if args.text else "built_in_smoke_corpus",
                  "warning":"A smoke corpus validates code execution only; it is not evidence of general language ability."}
    torch.save(checkpoint, out_dir / "diffusion_checkpoint.pt")
    current_mib, final_metric, peak_rss_mib = process_memory_mib()
    peak_observed_mib = max(peak_observed_mib, current_mib)
    report = {"format":"aimeng-diffusion-training-report-v1", "steps_requested":args.steps, "steps_completed":completed_steps,
              "neurons":args.neurons, "hidden_width":args.width, "active_top_k":args.active_k,
              "fanout":args.fanout, "max_diffusion_steps":args.max_steps, "vocab_size":len(itos),
              "source":checkpoint["source"], "initial_validation_loss":initial_loss,
              "final_validation_loss":final_loss, "train_loss_last":last_loss,
              "checkpoint":"diffusion_checkpoint.pt", "elapsed_seconds":round(time.time()-started,2),
              "memory_metric":final_metric, "current_memory_mib":round(current_mib,1),
              "peak_observed_memory_mib":round(peak_observed_mib,1), "process_peak_rss_mib":round(peak_rss_mib,1),
              "memory_stop_limit_mib":args.memory_stop_mib, "stopped_for_memory":memory_stopped,
              "device":str(device), "gpu_name":torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "gpu_peak_allocated_mib":round(torch.cuda.max_memory_allocated(device)/1024**2,1) if device.type == "cuda" else None,
              "status":"safe_stopped_memory" if memory_stopped else ("loss_improved" if final_loss < initial_loss else "needs_investigation"),
              "warning":checkpoint["warning"]}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report

def generate_from_checkpoint(checkpoint_path: str, prompt: str, count: int, temperature: float = 0.8):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    model = DiffusionTextModel(len(itos), int(cfg["neurons"]), int(cfg["width"]),
                               int(cfg["active_k"]), int(cfg["fanout"]), int(cfg["max_steps"]))
    model.load_state_dict(ckpt["model_state"]); model.eval()
    generated = list(prompt)
    with torch.no_grad():
        for _ in range(max(0, count)):
            context = generated[-int(cfg["context"]):] or [" "]
            ids = [stoi.get(ch, 0) for ch in context]
            out = model(torch.tensor([ids], dtype=torch.long), early_stop=True)
            probs = torch.softmax(out["logits"][0] / max(0.05, temperature), dim=-1)
            # ID 0 is the fallback for out-of-vocabulary characters, not printable text.
            # Never emit the literal string "<unk>" into user-facing generations.
            if probs.numel() > 0:
                probs[0] = 0
                probs = probs / probs.sum().clamp_min(1e-12)
            next_id = int(torch.multinomial(probs, 1).item())
            generated.append(itos[next_id] if 0 < next_id < len(itos) else "")
    result = "".join(generated)
    print(json.dumps({"prompt":prompt, "generated_text":result, "generated_characters":count,
                      "checkpoint_source":ckpt.get("source","unknown"), "warning":ckpt.get("warning","")}, ensure_ascii=False), flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", type=str, default=None, help="UTF-8 corpus file; omitted means synthetic smoke data only")
    p.add_argument("--checkpoint", type=str, default=None, help="load a checkpoint for terminal-only text generation")
    p.add_argument("--prompt", type=str, default="", help="prompt used with --checkpoint")
    p.add_argument("--generate-tokens", type=int, default=120, help="characters to generate with --checkpoint")
    p.add_argument("--temperature", type=float, default=0.8, help="sampling temperature for generation")
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--context", type=int, default=32); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=2048, help="maximum vocabulary including <unk>; Android currently supports up to 2048")
    p.add_argument("--neurons", type=int, default=128); p.add_argument("--width", type=int, default=32)
    p.add_argument("--active-k", type=int, default=8); p.add_argument("--fanout", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=6); p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--step-penalty", type=float, default=0.001); p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--memory-stop-mib", type=float, default=2560.0, help="stop and checkpoint at this PSS/RSS MiB threshold")
    p.add_argument("--output", type=str, default="runs/diffusion-smoke")
    args=p.parse_args()
    if args.checkpoint:
        generate_from_checkpoint(args.checkpoint, args.prompt, args.generate_tokens, args.temperature)
        return 0
    if args.steps < 1 or args.context < 2 or args.context > 512 or args.vocab_size < 2 or args.vocab_size > 2048 or args.batch_size < 1 or args.max_steps < 1 or args.log_every < 1:
        p.error("steps/batch-size/max-steps/log-every must be positive; context must be 2..512 and vocab-size 2..2048")
    report=train(args)
    return 0 if math.isfinite(report["final_validation_loss"]) else 1
if __name__ == "__main__": raise SystemExit(main())
