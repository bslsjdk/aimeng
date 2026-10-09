#!/usr/bin/env python3
"""Validate AIMENG telemetry JSONL before any training.

This checks record shape and data integrity, not whether measurements or labels are truthful.
Missing measurements must be null, never silently replaced with zero.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

QUALITY = {"pass", "fail", "unknown"}
STATUS = {"completed", "failed", "timeout", "oom", "cancelled"}
SPLITS = {"train", "validation", "test"}
BUDGET_RANK = {"small": 0, "medium": 1, "full": 2}

REQUIRED_PATHS = (
    "schema_version", "run_id", "task_id", "timestamp_utc",
    "model.name", "model.backend", "environment.device_class",
    "task.task_family", "task.dataset_split", "budget.budget_id",
    "budget.execution_mode", "result.status", "result.quality_label",
    "performance.total_latency_ms", "memory.measurement_method",
    "routing.controller_version", "provenance.config_sha256",
)


def get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_record(record: Any, line_no: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"line {line_no}: each JSONL row must be an object"]

    for path in REQUIRED_PATHS:
        value = get_path(record, path)
        if value is None or value == "":
            errors.append(f"line {line_no}: missing required field {path}")

    if record.get("schema_version") != "aimeng.telemetry.v1":
        errors.append(f"line {line_no}: schema_version must be aimeng.telemetry.v1")

    timestamp = record.get("timestamp_utc")
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                errors.append(f"line {line_no}: timestamp_utc must include timezone")
        except ValueError:
            errors.append(f"line {line_no}: timestamp_utc is not valid ISO-8601")

    result = record.get("result", {})
    if isinstance(result, dict):
        if result.get("quality_label") not in QUALITY:
            errors.append(f"line {line_no}: result.quality_label must be pass/fail/unknown")
        if result.get("status") not in STATUS:
            errors.append(f"line {line_no}: result.status has unsupported value")
        score = result.get("quality_score")
        if score is not None and (not is_number(score) or not 0 <= score <= 1):
            errors.append(f"line {line_no}: quality_score must be null or a number in [0, 1]")

    task = record.get("task", {})
    if isinstance(task, dict) and task.get("dataset_split") not in SPLITS:
        errors.append(f"line {line_no}: dataset_split must be train/validation/test")

    budget = record.get("budget", {})
    if isinstance(budget, dict):
        mode = budget.get("execution_mode")
        if mode is not None and mode not in {"normal", "simulation", "verified_skip"}:
            errors.append(f"line {line_no}: invalid budget.execution_mode")
        for key in ("n_ctx_requested", "n_ctx_actual", "n_batch_requested",
                    "n_batch_actual", "max_tokens", "threads", "concurrency"):
            value = budget.get(key)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                errors.append(f"line {line_no}: budget.{key} must be null or a positive integer")

    for path in ("performance.total_latency_ms", "performance.ttft_ms",
                 "performance.tokens_per_second", "memory.process_peak_rss_mib",
                 "memory.process_peak_pss_mib", "memory.backend_buffer_peak_mib",
                 "memory.kv_cache_estimated_mib", "environment.gpu_peak_allocated_mib",
                 "environment.gpu_peak_reserved_mib"):
        value = get_path(record, path)
        if value is not None and (not is_number(value) or value < 0):
            errors.append(f"line {line_no}: {path} must be null or a finite non-negative number")

    routing = record.get("routing", {})
    if isinstance(routing, dict):
        skipped = routing.get("verified_compute_skipped")
        if skipped is True and budget.get("execution_mode") != "verified_skip":
            errors.append(f"line {line_no}: verified compute skip requires execution_mode=verified_skip")
        if budget.get("execution_mode") == "simulation" and skipped is True:
            errors.append(f"line {line_no}: simulation cannot claim verified compute was skipped")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Path to telemetry JSONL file")
    parser.add_argument("--require-paired-budgets", action="store_true",
                        help="Require each pair_group_id to contain small, medium, and full budget labels")
    parser.add_argument("--check-task-split-leakage", action="store_true",
                        help="Reject a task_id appearing in more than one dataset split")
    args = parser.parse_args()
    path = Path(args.jsonl)
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    errors: list[str] = []
    seen_runs: set[str] = set()
    task_splits: dict[str, set[str]] = defaultdict(set)
    pairs: dict[str, set[str]] = defaultdict(set)
    count = 0

    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            count += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc.msg}")
                continue
            errors.extend(validate_record(record, line_no))
            if not isinstance(record, dict):
                continue
            run_id = record.get("run_id")
            if isinstance(run_id, str) and run_id:
                if run_id in seen_runs:
                    errors.append(f"line {line_no}: duplicate run_id {run_id!r}")
                seen_runs.add(run_id)
            task_id = record.get("task_id")
            split = get_path(record, "task.dataset_split")
            if isinstance(task_id, str) and isinstance(split, str):
                task_splits[task_id].add(split)
            pair_id = record.get("pair_group_id")
            budget_id = get_path(record, "budget.budget_id")
            if isinstance(pair_id, str) and pair_id and isinstance(budget_id, str):
                # Normalize only explicit conventional names; never infer from arbitrary IDs.
                label = next((name for name in BUDGET_RANK if name in budget_id.lower()), None)
                if label:
                    pairs[pair_id].add(label)

    if args.check_task_split_leakage:
        for task_id, splits in sorted(task_splits.items()):
            if len(splits) > 1:
                errors.append(f"task_id {task_id!r} leaks across splits: {sorted(splits)}")

    if args.require_paired_budgets:
        expected = set(BUDGET_RANK)
        for pair_id, budgets in sorted(pairs.items()):
            if budgets != expected:
                errors.append(f"pair_group_id {pair_id!r} has budgets {sorted(budgets)}, expected {sorted(expected)}")
        if not pairs:
            errors.append("no recognizable pair_group_id/budget_id pairs found")

    if errors:
        print(f"FAIL: {len(errors)} issue(s) in {count} record(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS: {count} telemetry record(s) validated. This is a format/integrity check only, not proof that measurements are real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
