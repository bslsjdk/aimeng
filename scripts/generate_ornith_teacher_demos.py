#!/usr/bin/env python3
"""Generate pending AIMENG teacher demonstrations using the official Ornith GGUF quant.

Requires a CUDA-enabled llama.cpp build with llama-server available on PATH.
Input JSONL rows: {task_id, topic, prompt, difficulty?, expected_verifier}
The server is started locally and stopped on exit; outputs are never auto-approved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M"
PROMPT_VERSION = "aimeng_ornith_teacher_gguf_v2"


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
    """Remove explicit reasoning-tag blocks; never store hidden reasoning traces."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"<analysis>.*?</analysis>", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"^\s*<final>\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s*</final>\s*$", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON response from {url}")
    return parsed


def wait_until_ready(base_url: str, process: subprocess.Popen[bytes], timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with code {process.returncode}; "
                "inspect the server log above for download, architecture, or GPU errors"
            )
        try:
            request_json(f"{base_url}/health", timeout=3)
            return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for llama-server at {base_url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="JSONL task prompts; one task per row")
    parser.add_argument("--output", required=True, help="New JSONL output file; existing files are refused")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model repo plus quant, e.g. repo:Q4_K_M")
    parser.add_argument("--llama-server", default="llama-server", help="Path/name of CUDA-enabled llama-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=99, help="Number of layers to offload; 99 means all possible")
    parser.add_argument("--startup-timeout", type=int, default=900, help="Seconds allowed for model download/load")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 means all tasks")
    args = parser.parse_args()

    task_path = Path(args.tasks).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not task_path.is_file():
        parser.error(f"tasks file not found: {task_path}")
    if task_path == output_path:
        parser.error("output must not overwrite the tasks file")
    if output_path.exists():
        parser.error(f"output already exists; refusing to overwrite: {output_path}")
    if (args.max_new_tokens < 1 or args.max_input_tokens < 16 or args.context_size < 256
            or args.gpu_layers < 0 or args.startup_timeout < 1 or args.limit < 0
            or not (1 <= args.port <= 65535)):
        parser.error("invalid token, context, port, GPU-layer, timeout, or limit setting")
    server = shutil.which(args.llama_server) or (args.llama_server if Path(args.llama_server).is_file() else None)
    if server is None:
        raise SystemExit(
            "llama-server was not found. Install/use a CUDA-enabled llama.cpp build first; "
            "this script intentionally does not fall back to the full BF16 Transformers checkpoint."
        )

    tasks = read_tasks(task_path)
    if args.limit:
        tasks = tasks[:args.limit]
    base_url = f"http://{args.host}:{args.port}"
    command = [
        server, "-hf", args.model, "--host", args.host, "--port", str(args.port),
        "--ctx-size", str(args.context_size), "--n-gpu-layers", str(args.gpu_layers),
        "--parallel", "1",
    ]
    print("Starting official quantized teacher with command:", " ".join(command), flush=True)
    print("Note: --n-gpu-layers requests GPU offload; llama.cpp may still leave unsupported layers on CPU.", flush=True)
    process = subprocess.Popen(command, stdout=None, stderr=subprocess.STDOUT)
    model_hash = hashlib.sha256(args.model.encode("utf-8")).hexdigest()
    count = 0
    try:
        wait_until_ready(base_url, process, args.startup_timeout)
        # Only create the output after the server has loaded successfully.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as out:
            for index, task in enumerate(tasks, 1):
                user_prompt = (
                    "Answer the user's task accurately and directly. Provide the useful final answer only; "
                    "do not include private chain-of-thought or hidden analysis. If the task is underspecified, "
                    "state the uncertainty. Do not claim tests, browsing, or verification you did not perform.\n\n"
                    + task["prompt"].strip()
                )
                result = request_json(
                    f"{base_url}/v1/chat/completions",
                    {
                        "model": args.model,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "temperature": 0,
                        "max_tokens": args.max_new_tokens,
                        "stream": False,
                    },
                    timeout=max(120, args.max_new_tokens * 2),
                )
                choices = result.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError(f"Task {task['task_id']} returned no completion choices")
                message = choices[0].get("message", {})
                answer = strip_reasoning(message.get("content", "") if isinstance(message, dict) else "")
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
                        "revision": "repository-selected-quantization",
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
                        "temperature": 0,
                        "quantization": "official-gguf-Q4_K_M",
                        "backend": "llama.cpp",
                        "requested_gpu_layers": args.gpu_layers,
                        "context_size": args.context_size,
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
            "teacher_quant": args.model,
            "next_step": "independently verify, deduplicate, assign leakage-safe splits, then run the student pipeline",
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
