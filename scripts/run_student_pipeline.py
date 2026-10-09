"""One-command preflight, baseline evaluation, SFT, and candidate evaluation pipeline."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], log_path: Path) -> None:
    print("\n=== " + name + " ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preflight -> baseline eval -> SFT -> candidate eval in one command."
    )
    parser.add_argument("--data", required=True, help="Verified candidate JSONL with train/validation/test splits")
    parser.add_argument("--model", required=True, help="Trainable Hugging Face causal LM ID or local checkpoint")
    parser.add_argument("--heldout", required=True, help="Frozen aimeng.eval_task.v1 JSONL")
    parser.add_argument("--regression", required=True, help="Frozen old-capability aimeng.eval_task.v1 JSONL")
    parser.add_argument("--output", required=True, help="New, empty/nonexistent run directory; existing paths are never overwritten")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    if args.epochs <= 0 or args.lr <= 0 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("epochs/lr must be positive and batch-size/grad-accum must be >= 1")
    if args.max_length < 16 or args.max_new_tokens < 1:
        parser.error("max-length must be >= 16 and max-new-tokens must be >= 1")

    source_data = Path(args.data).expanduser().resolve()
    source_heldout = Path(args.heldout).expanduser().resolve()
    source_regression = Path(args.regression).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (source_data, source_heldout, source_regression):
        if not path.is_file():
            parser.error(f"required input file does not exist: {path}")
    if output.exists():
        parser.error(f"output path already exists; refusing to overwrite/resume ambiguously: {output}")
    if output in (source_data, source_heldout, source_regression):
        parser.error("output must not equal an input file")

    output.mkdir(parents=True)
    snapshots = output / "input_snapshot"
    snapshots.mkdir()
    data = snapshots / "verified_candidates.jsonl"
    heldout = snapshots / "heldout.jsonl"
    regression = snapshots / "regression.jsonl"
    shutil.copy2(source_data, data)
    shutil.copy2(source_heldout, heldout)
    shutil.copy2(source_regression, regression)
    (output / "pipeline_config.json").write_text(json.dumps({
        "schema_version": "aimeng.pipeline_run.v1",
        "model": args.model,
        "source_files": {"data": str(source_data), "heldout": str(source_heldout), "regression": str(source_regression)},
        "snapshots": {"data": str(data), "heldout": str(heldout), "regression": str(regression)},
        "config": vars(args) | {"output": str(output)},
        "status": "preflight",
        "note": "The input snapshots are fixed for this run. Promotion is not automatic; Android whole-app peak RAM must be measured separately."
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logs = output / "logs"
    logs.mkdir()

    try:
        run_step("Validate SFT data structure", [
            sys.executable, str(ROOT / "scripts/validate_sft_candidates.py"), "--input", str(data)
        ], logs / "01_validate.log")
        trainer = [
            sys.executable, str(ROOT / "scripts/train_student_sft.py"),
            "--data", str(data), "--model", args.model, "--output", str(output / "student"),
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
            "--max-length", str(args.max_length), "--seed", str(args.seed), "--dry-run"
        ]
        if args.gradient_checkpointing:
            trainer.append("--gradient-checkpointing")
        run_step("Training preflight (no weights loaded)", trainer, logs / "02_preflight.log")
        common_eval = [
            "--heldout", str(heldout), "--regression", str(regression),
            "--max-new-tokens", str(args.max_new_tokens), "--device", args.device
        ]
        run_step("Baseline evaluation", [
            sys.executable, str(ROOT / "scripts/evaluate_student_sft.py"),
            "--model", args.model, "--model-version", "baseline", "--output", str(output / "baseline_metrics.json"),
            *common_eval
        ], logs / "03_baseline_eval.log")
        trainer[trainer.index("--dry-run"):] = []
        run_step("Student SFT training", trainer, logs / "04_train.log")
        run_step("Candidate evaluation", [
            sys.executable, str(ROOT / "scripts/evaluate_student_sft.py"),
            "--model", str(output / "student"), "--model-version", "candidate",
            "--output", str(output / "candidate_metrics.json"), *common_eval
        ], logs / "05_candidate_eval.log")
    except Exception as exc:
        config_path = output / "pipeline_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["status"] = "failed"
        config["failure"] = str(exc)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        print(f"Inputs, logs and any partial checkpoint are preserved at: {output}", file=sys.stderr)
        return 2

    config_path = output / "pipeline_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["status"] = "training_and_evaluation_complete"
    config["artifacts"] = {
        "baseline_metrics": str(output / "baseline_metrics.json"),
        "candidate_metrics": str(output / "candidate_metrics.json"),
        "checkpoint": str(output / "student"),
        "logs": str(logs),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nPIPELINE COMPLETE")
    print(f"Checkpoint: {output / 'student'}")
    print(f"Metrics: {output / 'baseline_metrics.json'} and {output / 'candidate_metrics.json'}")
    print("Promotion is intentionally not automatic. Measure whole-app Android peak RAM and run the release gate before replacing any active model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
