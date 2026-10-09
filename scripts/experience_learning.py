#!/usr/bin/env python3
"""Persistent, verifier-gated strategy memory for a frozen GGUF reasoning model.

This learns which prompting strategy works for a task family. It does not train
or modify GGUF weights, and it never treats unknown/self-reported quality as truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "aimeng.experience.v1"
STRATEGIES = {
    "direct": (
        "Solve the user's task directly. Prioritize correctness and relevance. "
        "Do not add unnecessary explanation."
    ),
    "decompose": (
        "Break the task into small reasoning steps, solve each dependency, then "
        "check that the steps support the result. Return only the requested format."
    ),
    "verify": (
        "First derive a candidate answer, then independently check it against every "
        "explicit constraint and look for a likely mistake. Return the corrected final answer."
    ),
}
MIN_SAMPLES = 2


def prompt_fingerprint(prompt: str) -> str:
    normalized = " ".join(prompt.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_experiences(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid experience JSON at line {line_no}: {exc.msg}") from exc
            if not isinstance(row, dict) or row.get("schema_version") != SCHEMA:
                raise ValueError(f"unsupported experience record at line {line_no}")
            if row.get("quality_label") not in {"pass", "fail"}:
                raise ValueError(f"untrusted quality label at line {line_no}")
            if row.get("strategy") not in STRATEGIES:
                raise ValueError(f"unknown strategy at line {line_no}")
            rows.append(row)
    return rows


def choose_strategy(
    task_family: str,
    experiences: list[dict[str, Any]],
    *,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, Any]:
    """Pick the best independently verified strategy, or safe direct baseline."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    candidates: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        rows = [
            row for row in experiences
            if row.get("task_family") == task_family
            and row.get("strategy") == strategy
            and row.get("quality_label") in {"pass", "fail"}
        ]
        if len(rows) < min_samples:
            continue
        pass_rate = sum(row["quality_label"] == "pass" for row in rows) / len(rows)
        measured = [
            float(row["latency_ms"]) for row in rows
            if isinstance(row.get("latency_ms"), (int, float))
            and not isinstance(row.get("latency_ms"), bool)
            and math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0
            and row["quality_label"] == "pass"
        ]
        candidates.append({
            "strategy": strategy,
            "samples": len(rows),
            "pass_rate": pass_rate,
            "median_passing_latency_ms": statistics.median(measured) if measured else None,
        })
    if not candidates:
        return {
            "strategy": "direct",
            "reason": "insufficient_verified_history",
            "samples": 0,
        }
    # Quality is the primary objective; latency only breaks ties among strategies
    # with the same verified pass rate. Missing latency sorts after measured latency.
    candidates.sort(key=lambda item: (
        -item["pass_rate"],
        item["median_passing_latency_ms"] is None,
        item["median_passing_latency_ms"] if item["median_passing_latency_ms"] is not None else float("inf"),
        -item["samples"],
        item["strategy"],
    ))
    best = candidates[0]
    return {"strategy": best["strategy"], "reason": "best_verified_task_family_history", **best}


def build_prompt(prompt: str, strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {tuple(STRATEGIES)}")
    return f"[AIMENG strategy={strategy}]\n{STRATEGIES[strategy]}\n\nTask:\n{prompt}"


def append_experience(
    path: str | Path,
    *,
    task_id: str,
    task_family: str,
    strategy: str,
    quality_label: str,
    verifier: str,
    latency_ms: float | None = None,
    prompt: str = "",
) -> dict[str, Any]:
    """Persist only externally verified pass/fail feedback; unknown is rejected."""
    if quality_label not in {"pass", "fail"}:
        raise ValueError("only independently verified pass/fail outcomes can be persisted")
    if not verifier or verifier == "none":
        raise ValueError("a named independent verifier is required")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {tuple(STRATEGIES)}")
    if not task_id.strip() or not task_family.strip():
        raise ValueError("task_id and task_family must be non-empty")
    if latency_ms is not None and (
        isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float))
        or not math.isfinite(latency_ms) or latency_ms < 0
    ):
        raise ValueError("latency_ms must be null or a finite non-negative number")
    row = {
        "schema_version": SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "task_id": task_id,
        "task_family": task_family,
        "strategy": strategy,
        "quality_label": quality_label,
        "verifier": verifier,
        "latency_ms": latency_ms,
        "prompt_sha256": prompt_fingerprint(prompt) if prompt else None,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Validate existing records before appending so a corrupt file fails closed.
    load_experiences(destination)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", required=True, help="Verified experience JSONL")
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()
    try:
        rows = load_experiences(args.memory)
        decision = choose_strategy(args.task_family, rows)
        result = {**decision, "prompt": build_prompt(args.prompt, decision["strategy"]) if args.show_prompt else None}
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
