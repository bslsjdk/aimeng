#!/usr/bin/env python3
"""Generate pending AIMENG teacher demonstrations with Ornith-1.5-9B on a CUDA GPU.

This is an inference-only data generator. It never trains on its own output and
never marks generated records verified or training-eligible.
Input JSONL rows: {task_id, topic, prompt, difficulty?, expected_verifier?}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "ornith-ai/Ornith-1.5-9B"
PROMPT_VERSION = "aimeng_ornith_teacher_gpu_v1"


def read_tasks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        for key in ("task_id", "topic", "prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{line_no}: {key} must be a non-empty string")
        if row["task_id"] in seen:
            raise ValueError(f"{path}:{line_no}: duplicate task_id {row['task_id']!r}")
        seen.add(row["task_id"])
        difficulty = row.get("difficulty", "intermediate")
        if difficulty not in {"basic", "intermediate", "advanced"}:
            raise ValueError(f"{path}:{line_no}: invalid difficulty")
        rows.append({**row, "difficulty": difficulty})
    if not rows:
        raise ValueError(f"{path}: no tasks found")
    return rows


def strip_reasoning(answer: str) -> str:
    # Keep only the user-facing answer; never save hidden reasoning traces.
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"<analysis>.*?</analysis>", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"^\s*<final>\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s*</final>\s*$", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="JSONL task prompts; one task per row")
    parser.add_argument("--output", required=True, help="New JSONL output file; existing files are refused")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Transformers-compatible Ornith checkpoint, not MLX")
    parser.add_argument("--revision", default="main", help="Model repository revision to record/use")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep at 1 for free-GPU memory safety")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 means all tasks")
    parser.add_argument("--trust-remote-code", action="store_true", help="Explicitly allow model repository code")
    args = parser.parse_args()

    task_path, output_path = Path(args.tasks).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if not task_path.is_file():
        parser.error(f"tasks file not found: {task_path}")
    if task_path == output_path:
        parser.error("output must not overwrite the tasks file")
    if output_path.exists():
        parser.error(f"output already exists; refusing to overwrite: {output_path}")
    if args.max_new_tokens < 1 or args.max_input_tokens < 16 or args.batch_size != 1 or args.limit < 0:
        parser.error("invalid limits; batch-size must remain 1 for the first free-GPU run")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing teacher dependencies. In a CUDA GPU notebook install requirements-teacher-gpu.txt "
            "after confirming the notebook has a GPU runtime."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available. Select a GPU runtime; CPU fallback is intentionally disabled.")

    tasks = read_tasks(task_path)
    if args.limit:
        tasks = tasks[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_hash = hashlib.sha256(f"{args.model}@{args.revision}".encode("utf-8")).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("x", encoding="utf-8") as out:
        for index, task in enumerate(tasks, 1):
            user_prompt = (
                "Answer the user's task accurately and directly. Provide the useful final answer only; "
                "do not include private chain-of-thought or hidden analysis. If the task is underspecified, "
                "state the uncertainty. Do not claim tests, browsing, or verification you did not perform.\n\n"
                + task["prompt"].strip()
            )
            messages = [{"role": "user", "content": user_prompt}]
            if getattr(tokenizer, "chat_template", None):
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                rendered = user_prompt
            encoded = tokenizer(
                rendered, return_tensors="pt", truncation=True,
                max_length=args.max_input_tokens
            )
            input_device = model.get_input_embeddings().weight.device
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded["input_ids"].shape[-1]:]
            answer = strip_reasoning(tokenizer.decode(new_tokens, skip_special_tokens=True))
            if not answer:
                raise RuntimeError(f"Task {task['task_id']} produced an empty final answer; stopping.")
            record = {
                "schema_version": "aimeng.teacher_demo.v1",
                "task_id": task["task_id"],
                "topic": task["topic"],
                "difficulty": task["difficulty"],
                "prompt": task["prompt"].strip(),
                "response": answer,
                "expected_verifier": task.get("expected_verifier", "independent_review_required"),
                "teacher": {
                    "model": args.model,
                    "revision": args.revision,
                    "model_reference_sha256": model_hash,
                    "prompt_version": PROMPT_VERSION,
                    "artifact_sha256": None,
                },
                "provenance": {
                    "source_type": "gpu_teacher_generation",
                    "source_ref": str(task_path),
                    "license_review": "pending",
                    "privacy_review": "pending",
                },
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "quantization": "bitsandbytes-nf4-4bit",
                },
                "verification": {"status": "pending", "method": None, "evidence": []},
                "split": "unassigned",
                "training_eligible": False,
            }
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            count += 1
            print(f"[{index}/{len(tasks)}] wrote pending candidate: {task['task_id']}", flush=True)
    print(json.dumps({
        "ok": True,
        "generated": count,
        "output": str(output_path),
        "all_pending_and_ineligible": True,
        "next_step": "independently verify, deduplicate, assign leakage-safe splits, then run the student pipeline",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
