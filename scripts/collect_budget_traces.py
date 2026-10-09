#!/usr/bin/env python3
"""Run paired budget experiments through an existing llama.cpp llama-cli binary.

This is a measurement harness, not a model builder. It does not claim GPU use unless
the supplied llama-cli build was configured for the intended accelerator. Exact/contains/
JSON verifiers are intentionally conservative; unsupported tasks stay quality=unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUDGETS = (
    {"budget_id": "small-v1", "n_ctx": 1024, "max_tokens": 128, "threads": 4},
    {"budget_id": "medium-v1", "n_ctx": 2048, "max_tokens": 256, "threads": 4},
    {"budget_id": "full-v1", "n_ctx": 4096, "max_tokens": 512, "threads": 4},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_pss_mib(pid: int) -> float | None:
    # Linux smaps_rollup is available on many Linux runtimes; missing permission/data is null.
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^Pss:\s+(\d+)\s+kB$", text, re.MULTILINE)
        return round(int(match.group(1)) / 1024, 2) if match else None
    except (OSError, ValueError):
        return None


def verify_output(task: dict[str, Any], output: str) -> tuple[str, float | None, str]:
    verifier = task.get("verifier", "unknown")
    expected = task.get("expected")
    if verifier == "exact_match" and isinstance(expected, str):
        actual = output.strip()
        target = expected.strip()
        passed = actual == target
        return ("pass" if passed else "fail", 1.0 if passed else 0.0, "exact_match_v1")
    if verifier == "contains" and isinstance(expected, str):
        passed = expected.casefold() in output.casefold()
        return ("pass" if passed else "fail", 1.0 if passed else 0.0, "contains_v1")
    if verifier == "json_object":
        try:
            value = json.loads(output.strip())
            passed = isinstance(value, dict)
            return ("pass" if passed else "fail", 1.0 if passed else 0.0, "json_object_v1")
        except (json.JSONDecodeError, TypeError):
            return "fail", 0.0, "json_object_v1"
    return "unknown", None, "none"


def run_one(cli: str, model_path: str, task: dict[str, Any], budget: dict[str, Any],
            timeout_s: int, device_class: str, accelerator: str | None) -> dict[str, Any]:
    run_id = f"run-{uuid.uuid4().hex[:16]}"
    pair_id = str(task.get("pair_group_id") or f"pair-{task['task_id']}")
    cmd = [
        cli, "-m", model_path, "-p", str(task["prompt"]),
        "-c", str(budget["n_ctx"]), "-n", str(budget["max_tokens"]),
        "-t", str(budget["threads"]), "--no-display-prompt", "--no-warmup",
    ]
    start = time.monotonic()
    pss_peak: float | None = None
    output = ""
    error_text = ""
    status = "completed"
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace")
        deadline = start + timeout_s
        while process.poll() is None:
            current_pss = read_pss_mib(process.pid)
            if current_pss is not None:
                pss_peak = max(pss_peak or 0.0, current_pss)
            if time.monotonic() >= deadline:
                process.kill()
                status = "timeout"
                break
            time.sleep(0.1)
        stdout, stderr = process.communicate(timeout=5)
        output = stdout or ""
        error_text = (stderr or "")[-2000:]
        if status == "completed" and process.returncode != 0:
            status = "failed"
    except Exception as exc:
        status = "failed"
        error_text = f"{type(exc).__name__}: {exc}"
        if process and process.poll() is None:
            process.kill()
            try:
                process.communicate(timeout=5)
            except Exception:
                pass

    latency_ms = round((time.monotonic() - start) * 1000, 2)
    if status == "completed":
        quality_label, quality_score, verifier = verify_output(task, output)
    else:
        quality_label, quality_score, verifier = "unknown", None, "none"

    # Output token count is deliberately null: whitespace tokenization is not model tokenization.
    return {
        "schema_version": "aimeng.telemetry.v1",
        "run_id": run_id,
        "task_id": str(task["task_id"]),
        "pair_group_id": pair_id,
        "timestamp_utc": utc_now(),
        "model": {
            "name": Path(model_path).name,
            "sha256": task.get("model_sha256"),
            "backend": "llama.cpp/llama-cli",
            "backend_version": os.environ.get("AIMENG_BACKEND_VERSION"),
        },
        "environment": {
            "device_class": device_class,
            "accelerator": accelerator,
            "gpu_vram_total_mib": None,
            "gpu_peak_allocated_mib": None,
            "gpu_peak_reserved_mib": None,
            "android_ram_limit_mib": 4096,
            "device_profile_id": os.environ.get("AIMENG_DEVICE_PROFILE_ID", "unconfigured"),
        },
        "task": {
            "task_family": task.get("task_family", "unknown"),
            "dataset_split": task.get("dataset_split", "train"),
            "prompt_template_id": task.get("prompt_template_id", "raw-prompt-v1"),
            "input_tokens": task.get("input_tokens"),
            "expected_output_type": task.get("expected_output_type", "unknown"),
            "privacy_safe_features": task.get("privacy_safe_features", {}),
        },
        "budget": {
            "budget_id": budget["budget_id"],
            "n_ctx_requested": budget["n_ctx"],
            "n_ctx_actual": None,
            "n_batch_requested": None,
            "n_batch_actual": None,
            "max_tokens": budget["max_tokens"],
            "threads": budget["threads"],
            "concurrency": 1,
            "kv_cache_type_k": None,
            "kv_cache_type_v": None,
            "decode_profile": "default",
            "execution_mode": "normal",
            "unsupported_features": ["actual_context_batch_and_kv_cache_not_exposed_by_this_harness"],
        },
        "result": {
            "status": status,
            "quality_label": quality_label,
            "quality_score": quality_score,
            "quality_verifier": verifier,
            "failure_reason": error_text or None,
            "output_tokens": None,
        },
        "performance": {
            "ttft_ms": None,
            "total_latency_ms": latency_ms,
            "tokens_per_second": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
        },
        "memory": {
            "process_peak_rss_mib": None,
            "process_peak_pss_mib": pss_peak,
            "backend_buffer_peak_mib": None,
            "kv_cache_estimated_mib": None,
            "gpu_peak_allocated_mib": None,
            "gpu_peak_reserved_mib": None,
            "measurement_method": "sampled_linux_proc_smaps_rollup" if pss_peak is not None else "pss_unavailable",
        },
        "routing": {
            "controller_version": "fixed-budget-collector-v1",
            "decision_reason": "paired_budget_sweep",
            "requested_channel_groups": None,
            "verified_compute_skipped": False,
            "fallback_stage": 0,
            "fallback_reason": None,
        },
        "provenance": {
            "config_sha256": None,
            "dataset_snapshot_sha256": None,
            "random_seed": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the local GGUF file")
    parser.add_argument("--tasks", required=True, help="JSONL task file; one task per line")
    parser.add_argument("--output", default="data/telemetry.jsonl")
    parser.add_argument("--llama-cli", default=os.environ.get("LLAMA_CLI", "llama-cli"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device-class", default="colab_gpu")
    parser.add_argument("--accelerator", default="unknown",
                        help="Declared build/runtime accelerator, e.g. CUDA-T4 or CPU; do not guess")
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test task limit")
    args = parser.parse_args()

    if not Path(args.model).is_file():
        parser.error(f"GGUF not found: {args.model}")
    if not Path(args.tasks).is_file():
        parser.error(f"task file not found: {args.tasks}")
    cli = shutil.which(args.llama_cli) if not Path(args.llama_cli).is_file() else args.llama_cli
    if not cli:
        parser.error("llama-cli not found. Build/install a compatible llama.cpp backend first.")

    tasks: list[dict[str, Any]] = []
    with Path(args.tasks).open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                task = json.loads(raw)
            except json.JSONDecodeError as exc:
                parser.error(f"invalid JSON at task line {line_no}: {exc.msg}")
            if not isinstance(task, dict) or not task.get("task_id") or not task.get("prompt"):
                parser.error(f"task line {line_no} needs task_id and prompt")
            tasks.append(task)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        parser.error("no tasks to run")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    budget_set = [dict(b, threads=args.threads) for b in BUDGETS]
    with output_path.open("a", encoding="utf-8") as out:
        for index, task in enumerate(tasks, 1):
            for budget in budget_set:
                record = run_one(cli, args.model, task, budget, args.timeout,
                                 args.device_class, args.accelerator)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{index}/{len(tasks)}] task={task['task_id']} budget={budget['budget_id']} "
                      f"status={record['result']['status']} quality={record['result']['quality_label']} "
                      f"latency_ms={record['performance']['total_latency_ms']}")
    print(f"APPENDED {len(tasks) * len(budget_set)} records to {output_path}")
    print("Note: requested context is recorded; actual context, TTFT, GPU memory, and token throughput are null unless measured by another backend instrument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
