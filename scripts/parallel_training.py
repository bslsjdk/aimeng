#!/usr/bin/env python3
"""Run independent training jobs concurrently under an explicit memory budget.

This runner does NOT synchronize gradients or merge model weights. Every job must
own its output directory/checkpoint. Use a distributed-training framework when
workers are meant to update the same model.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class Job:
    job_id: str
    argv: tuple[str, ...]
    memory_mb: int
    output_dir: Path

def load_plan(path: Path) -> tuple[list[Job], int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read plan: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError("plan must contain a jobs array")
    budget, workers = payload.get("memory_budget_mb"), payload.get("max_workers", 1)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("memory_budget_mb must be a positive integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    jobs, ids, outputs = [], set(), set()
    for index, item in enumerate(payload["jobs"]):
        if not isinstance(item, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        job_id, argv, memory_mb, output_dir = (item.get(k) for k in ("id", "argv", "memory_mb", "output_dir"))
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError(f"jobs[{index}].id must be a non-empty string")
        if job_id in ids:
            raise ValueError(f"duplicate job id: {job_id}")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError(f"job {job_id}: argv must be a non-empty string array")
        if not isinstance(memory_mb, int) or isinstance(memory_mb, bool) or memory_mb <= 0:
            raise ValueError(f"job {job_id}: memory_mb must be a positive integer")
        if memory_mb > budget:
            raise ValueError(f"job {job_id}: estimated memory exceeds total budget")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError(f"job {job_id}: output_dir is required")
        normalized = str(Path(output_dir).resolve())
        if normalized in outputs:
            raise ValueError(f"jobs must not share output_dir: {output_dir}")
        ids.add(job_id); outputs.add(normalized)
        jobs.append(Job(job_id, tuple(argv), memory_mb, Path(output_dir)))
    if not jobs:
        raise ValueError("plan contains no jobs")
    return jobs, budget, workers

def run_plan(jobs: list[Job], memory_budget_mb: int, max_workers: int,
             *, dry_run: bool = False, fail_fast: bool = False) -> dict[str, Any]:
    """Run independent commands; declared memory is an estimate, not OS isolation."""
    pending, running, results = list(jobs), {}, []
    reserved_mb, stopped = 0, False
    if dry_run:
        for job in jobs:
            print(json.dumps({"event":"planned","id":job.job_id,"argv":list(job.argv),
                "memory_mb":job.memory_mb,"output_dir":str(job.output_dir)}, ensure_ascii=False))
        return {"status":"dry_run","results":[],"memory_budget_mb":memory_budget_mb}
    while pending or running:
        launched, finished_ids = False, []
        if not stopped:
            for job in list(pending):
                if len(running) >= max_workers:
                    break
                if reserved_mb + job.memory_mb > memory_budget_mb:
                    continue
                job.output_dir.mkdir(parents=True, exist_ok=True)
                log_path = job.output_dir / "training.log"
                handle = log_path.open("wb")
                try:
                    process = subprocess.Popen(list(job.argv), stdout=handle,
                        stderr=subprocess.STDOUT, shell=False, env=os.environ.copy())
                except OSError as exc:
                    handle.close()
                    results.append({"id":job.job_id,"returncode":None,"status":"launch_error","error":str(exc)})
                    pending.remove(job)
                    if fail_fast: stopped = True
                    continue
                running[job.job_id] = (job, process, handle, time.time())
                reserved_mb += job.memory_mb; pending.remove(job); launched = True
                print(json.dumps({"event":"started","id":job.job_id,"pid":process.pid,
                    "reserved_memory_mb":reserved_mb}, ensure_ascii=False))
        if not running:
            for job in pending:
                results.append({"id":job.job_id,"returncode":None,"status":"not_started",
                    "error":"scheduler stopped or budget unavailable"})
            pending.clear()
            break
        for job_id, (job, process, handle, started_at) in list(running.items()):
            code = process.poll()
            if code is None: continue
            handle.close()
            elapsed = round(time.time() - started_at, 3)
            status = "passed" if code == 0 else "failed"
            results.append({"id":job_id,"returncode":code,"status":status,"elapsed_seconds":elapsed,
                "log":str(job.output_dir / "training.log")})
            reserved_mb -= job.memory_mb; finished_ids.append(job_id)
            print(json.dumps({"event":"finished","id":job_id,"status":status,"returncode":code,
                "elapsed_seconds":elapsed,"reserved_memory_mb":reserved_mb}, ensure_ascii=False))
            if code != 0 and fail_fast: stopped = True
        for job_id in finished_ids: del running[job_id]
        if not launched and not finished_ids: time.sleep(0.1)
    failures = [r for r in results if r["status"] != "passed"]
    return {"status":"passed" if not failures else "failed","memory_budget_mb":memory_budget_mb,
        "max_workers":max_workers,"results":sorted(results,key=lambda item:item["id"])}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        jobs, budget, workers = load_plan(args.plan)
        report = run_plan(jobs, budget, workers, dry_run=args.dry_run, fail_fast=args.fail_fast)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return 0 if report["status"] in {"passed","dry_run"} else 1

if __name__ == "__main__":
    raise SystemExit(main())
