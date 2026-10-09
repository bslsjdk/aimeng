"""Gated, reproducible response-level SFT for AIMENG candidate JSONL.

Safety properties:
- only verified + explicitly eligible records enter training/evaluation;
- assistant-target-only loss (fail closed if template tokenization is ambiguous);
- task_id cannot cross train/validation/test;
- test split is audited but never loaded by the optimizer;
- data/config hashes and a run manifest are saved beside checkpoints.
This trains a compatible Hugging Face causal LM, not an arbitrary custom AIMENG
architecture or an inference-only GGUF/MLX artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "aimeng.sft_candidate.v1"
SPLITS = {"train", "validation", "test"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    task_splits: dict[str, set[str]] = {}
    for line_no, raw in enumerate(path.open(encoding="utf-8"), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError(f"{path}:{line_no}: missing/duplicate sample_id {sid!r}")
        seen.add(sid)
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}:{line_no}: unsupported schema_version")
        verification = row.get("verification")
        if not isinstance(verification, dict):
            raise ValueError(f"{path}:{line_no}: verification must be an object")
        eligible = row.get("training_eligible") is True
        if eligible:
            if verification.get("status") != "verified":
                raise ValueError(f"{path}:{line_no}: eligible record is not verified")
            if not isinstance(verification.get("method"), str) or not verification["method"].strip():
                raise ValueError(f"{path}:{line_no}: verified eligible record needs verification.method")
            evidence = verification.get("evidence")
            if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) and x.strip() for x in evidence):
                raise ValueError(f"{path}:{line_no}: verified eligible record needs non-empty string evidence")
            split = row.get("split")
            if split not in SPLITS:
                raise ValueError(f"{path}:{line_no}: eligible record needs train/validation/test split")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path}:{line_no}: eligible record needs task_id")
            task_splits.setdefault(task_id, set()).add(split)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_no}: messages must contain user and assistant turns")
            if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
                raise ValueError(f"{path}:{line_no}: messages need a user turn")
            if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant" or not isinstance(messages[-1].get("content"), str) or not messages[-1]["content"].strip():
                raise ValueError(f"{path}:{line_no}: final message must be a non-empty assistant target")
            for i, message in enumerate(messages):
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError(f"{path}:{line_no}: invalid messages[{i}]")
        rows.append(row)
    leaked = {task: sorted(splits) for task, splits in task_splits.items() if len(splits) > 1}
    if leaked:
        preview = list(leaked.items())[:10]
        raise ValueError(f"task_id split leakage detected (first entries): {preview}")
    return rows


def load_records(path: Path, split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return [
        row for row in load_all_records(path)
        if row.get("training_eligible") is True
        and row.get("verification", {}).get("status") == "verified"
        and row.get("split") == split
    ]


def encode_assistant_target(row: dict[str, Any], tokenizer, max_length: int) -> dict[str, list[int]]:
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    if not getattr(tokenizer, "chat_template", None) or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer has no chat_template; refusing to guess the student conversation format")
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, truncation=False)["input_ids"]
    if not full_ids or len(prefix_ids) >= len(full_ids):
        raise ValueError(f"{row.get('sample_id')}: empty target or incompatible chat-template prefix")
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"{row.get('sample_id')}: tokenizer template prefix is not token-stable; "
            "cannot safely mask assistant-only loss"
        )
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        if len(prefix_ids) >= max_length:
            raise ValueError(f"{row.get('sample_id')}: max_length truncates the entire assistant target")
        # Keep the most recent context and all available target tokens.
        full_ids = full_ids[overflow:]
        prefix_ids = prefix_ids[overflow:] if overflow < len(prefix_ids) else []
        # Left-truncating can change token boundaries. Fail closed rather than silently mask wrong tokens.
        if prefix_ids and full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"{row.get('sample_id')}: truncation made assistant target boundary ambiguous")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    if not any(label != -100 for label in labels):
        raise ValueError(f"{row.get('sample_id')}: no assistant target tokens remain after masking")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class AssistantOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")
        input_ids, masks, labels = [], [], []
        for item in features:
            amount = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * amount)
            masks.append(item["attention_mask"] + [0] * amount)
            labels.append(item["labels"] + [-100] * amount)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_run_manifest(data_path: Path, model: str, args, train, val, test_count: int) -> dict[str, Any]:
    return {
        "schema_version": "aimeng.sft_run.v1",
        "model_reference": model,
        "data_path_name": data_path.name,
        "data_sha256": sha256_file(data_path),
        "train_sample_ids": [r["sample_id"] for r in train],
        "validation_sample_ids": [r["sample_id"] for r in val],
        "test_records_reserved_not_optimized": test_count,
        "config": {
            "epochs": args.epochs, "learning_rate": args.lr, "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum, "max_length": args.max_length,
            "seed": args.seed, "gradient_checkpointing": args.gradient_checkpointing,
        },
        "runtime": {"python": platform.python_version()},
        "claim_scope": "SFT run metadata; does not establish task competence or deployment RAM compliance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, help="Trainable HF model ID or local checkpoint; not a GGUF file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Audit eligibility/splits/config without loading model weights")
    args = parser.parse_args()

    data_path = Path(args.data)
    all_rows = load_all_records(data_path)
    train = [r for r in all_rows if r.get("training_eligible") is True and r.get("verification", {}).get("status") == "verified" and r.get("split") == "train"]
    val = [r for r in all_rows if r.get("training_eligible") is True and r.get("verification", {}).get("status") == "verified" and r.get("split") == "validation"]
    test = [r for r in all_rows if r.get("training_eligible") is True and r.get("verification", {}).get("status") == "verified" and r.get("split") == "test"]
    if not train:
        raise SystemExit("Refusing to train: zero verified, training-eligible train records")
    if not val:
        raise SystemExit("Refusing to train: zero verified, training-eligible validation records")
    if args.epochs <= 0 or args.lr <= 0 or args.batch_size < 1 or args.grad_accum < 1 or args.max_length < 16:
        raise SystemExit("Invalid training hyperparameters")
    manifest = build_run_manifest(data_path, args.model, args, train, val, len(test))
    print(json.dumps({
        "ok": True, "train_records": len(train), "validation_records": len(val),
        "test_records_reserved": len(test), "data_sha256": manifest["data_sha256"],
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Install compatible torch, transformers, and datasets.") from exc
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("Tokenizer has neither pad_token nor eos_token; refusing unsafe padding")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False)
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise SystemExit("Selected model does not support gradient checkpointing")
        model.gradient_checkpointing_enable()

    def convert(rows):
        features = [encode_assistant_target(row, tokenizer, args.max_length) for row in rows]
        return Dataset.from_list(features)

    train_ds, val_ds = convert(train), convert(val)
    trainer_parameters = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in trainer_parameters else "evaluation_strategy"
    training_kwargs = dict(
        output_dir=args.output, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, save_strategy="epoch",
        logging_steps=10, report_to="none", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
        seed=args.seed,
    )
    training_kwargs[eval_key] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aimeng_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    summary = {
        **manifest,
        "train_metrics": result.metrics,
        "evaluation_metrics": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
    }
    (output / "aimeng_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "global_step": trainer.state.global_step, "eval_metrics": metrics}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
