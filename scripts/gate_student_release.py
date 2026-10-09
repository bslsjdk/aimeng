"""Gate student promotion on held-out quality, regression, and target-device memory.

This script never trains a model. It writes an auditable decision and, only when
all gates pass, can atomically update a JSON pointer to the candidate release.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path


def load_metrics(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: metrics must be a JSON object")
    return value


def evaluate(baseline: dict, candidate: dict, min_gain: float, max_regression_drop: float, max_android_mib: float) -> dict:
    required = ("suite_hash", "heldout_success_rate", "regression_success_rate")
    errors = []
    for label, metrics in (("baseline", baseline), ("candidate", candidate)):
        for key in required:
            if key not in metrics:
                errors.append(f"{label} missing {key}")
        for key in ("heldout_success_rate", "regression_success_rate"):
            if key in metrics and (not isinstance(metrics[key], (int, float)) or not 0 <= metrics[key] <= 1):
                errors.append(f"{label}.{key} must be between 0 and 1")
    if errors:
        return {"promote": False, "reasons": errors}
    if baseline["suite_hash"] != candidate["suite_hash"]:
        errors.append("baseline and candidate must use the same frozen evaluation suite hash")
    gain = candidate["heldout_success_rate"] - baseline["heldout_success_rate"]
    regression_drop = baseline["regression_success_rate"] - candidate["regression_success_rate"]
    if gain < min_gain:
        errors.append(f"held-out gain {gain:.4f} is below required {min_gain:.4f}")
    if regression_drop > max_regression_drop:
        errors.append(f"regression suite dropped {regression_drop:.4f}, limit {max_regression_drop:.4f}")
    scope = candidate.get("memory_measurement_scope")
    peak = candidate.get("peak_memory_mib")
    if scope != "android_app" or candidate.get("memory_measured") is not True:
        errors.append("candidate lacks measured whole-Android-app peak memory")
    elif not isinstance(peak, (int, float)) or peak <= 0 or peak >= max_android_mib:
        errors.append(f"Android app peak memory must be >0 and below {max_android_mib:.0f} MiB")
    return {
        "promote": not errors,
        "reasons": errors,
        "metrics": {
            "heldout_gain": gain,
            "regression_drop": regression_drop,
            "candidate_android_peak_mib": peak,
            "android_memory_limit_mib": max_android_mib,
        },
        "suite_hash": candidate["suite_hash"],
        "candidate_version": candidate.get("model_version"),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--promote-pointer", help="Optional JSON pointer; updated only if every gate passes")
    parser.add_argument("--min-heldout-gain", type=float, default=0.01)
    parser.add_argument("--max-regression-drop", type=float, default=0.01)
    parser.add_argument("--max-android-mib", type=float, default=3800.0,
                        help="Must be below the hard 4096 MiB cap to retain safety margin")
    args = parser.parse_args()
    if args.min_heldout_gain < 0 or args.max_regression_drop < 0 or not 0 < args.max_android_mib < 4096:
        parser.error("invalid thresholds; Android cap must be strictly below 4096 MiB")
    baseline, candidate = load_metrics(Path(args.baseline)), load_metrics(Path(args.candidate))
    decision = evaluate(baseline, candidate, args.min_heldout_gain, args.max_regression_drop, args.max_android_mib)
    decision["baseline_metrics_file"] = str(Path(args.baseline))
    decision["candidate_metrics_file"] = str(Path(args.candidate))
    atomic_json(Path(args.decision), decision)
    if decision["promote"] and args.promote_pointer:
        atomic_json(Path(args.promote_pointer), {
            "schema_version": "aimeng.release_pointer.v1",
            "active_model_version": candidate.get("model_version"),
            "metrics_file": str(Path(args.candidate)),
            "suite_hash": candidate["suite_hash"],
        })
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
