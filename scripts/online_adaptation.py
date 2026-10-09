#!/usr/bin/env python3
"""Conservative task-local budget adaptation from independently verified outcomes.

This module selects the next action; it does not run inference, update model weights,
or persistently train a controller. Only whole-application PSS can establish the
Android runtime memory gate. Missing memory measurements never count as safe.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

BUDGETS = ("small", "medium", "full")


def decide_next_action(
    observation: dict[str, Any],
    *,
    hard_limit_mib: float = 4096.0,
    safety_margin_mib: float = 512.0,
) -> dict[str, Any]:
    """Return a deterministic, auditable next action for one task iteration."""
    current = observation.get("budget")
    quality = observation.get("quality_label", "unknown")
    status = observation.get("status", "completed")
    pss = observation.get("whole_app_pss_mib")
    pss_source = observation.get("memory_measurement_source")
    trusted_pss = pss_source == "android_whole_app"

    if current not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}")
    if quality not in {"pass", "fail", "unknown"}:
        raise ValueError("quality_label must be pass/fail/unknown")
    if status not in {"completed", "failed", "timeout", "oom", "cancelled"}:
        raise ValueError("unsupported status")
    if pss is not None and (
        isinstance(pss, bool) or not isinstance(pss, (int, float))
        or not math.isfinite(pss) or pss < 0
    ):
        raise ValueError("whole_app_pss_mib must be null or a finite non-negative number")
    if (
        not math.isfinite(hard_limit_mib) or not math.isfinite(safety_margin_mib)
        or hard_limit_mib <= 0 or safety_margin_mib < 0
        or safety_margin_mib >= hard_limit_mib
    ):
        raise ValueError("invalid memory limit or safety margin")

    base = {
        "schema_version": "aimeng.online_decision.v1",
        "current_budget": current,
        "quality_label": quality,
        "status": status,
        "whole_app_pss_mib": pss,
        "memory_measurement_source": pss_source,
        "persistent_update": False,
    }

    if trusted_pss and pss is not None and pss >= hard_limit_mib:
        return {**base, "action": "stop_and_flag_memory_violation",
                "next_budget": None, "reason": "whole_app_pss_at_or_above_hard_limit"}
    if status in {"oom", "cancelled", "failed", "timeout"}:
        return {**base, "action": "stop_and_record_failure",
                "next_budget": None, "reason": f"execution_status_{status}"}
    if quality == "pass":
        return {**base, "action": "stop_success",
                "next_budget": None, "reason": "independent_verifier_passed"}
    if quality == "unknown":
        return {**base, "action": "stop_for_reliable_feedback",
                "next_budget": None, "reason": "unknown_is_not_a_training_label"}
    if pss is None or not trusted_pss:
        return {**base, "action": "stop_for_memory_measurement",
                "next_budget": None,
                "reason": "missing_or_untrusted_whole_app_memory_measurement"}

    soft_limit = hard_limit_mib - safety_margin_mib
    if pss >= soft_limit:
        return {**base, "action": "stop_for_memory_headroom",
                "next_budget": None, "reason": "insufficient_memory_headroom"}

    next_index = BUDGETS.index(current) + 1
    if next_index >= len(BUDGETS):
        return {**base, "action": "stop_quality_failure",
                "next_budget": None, "reason": "highest_budget_failed_verification"}
    return {**base, "action": "retry_with_higher_budget",
            "next_budget": BUDGETS[next_index],
            "reason": "verified_quality_failure_with_measured_memory_headroom"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file with one observation")
    parser.add_argument("--output", help="Optional path for decision JSON; stdout by default")
    parser.add_argument("--hard-limit-mib", type=float, default=4096.0)
    parser.add_argument("--safety-margin-mib", type=float, default=512.0)
    args = parser.parse_args()
    try:
        observation = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(observation, dict):
            raise ValueError("input JSON must be an object")
        decision = decide_next_action(
            observation,
            hard_limit_mib=args.hard_limit_mib,
            safety_margin_mib=args.safety_margin_mib,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
