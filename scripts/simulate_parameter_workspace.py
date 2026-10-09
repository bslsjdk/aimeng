"""Trace-only simulator for a bounded dynamic parameter working set.

This simulates metadata and budget decisions only. It does not load model weights,
free operating-system pages, skip model computation, or measure real RAM.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WorkspaceError(RuntimeError):
    pass


@dataclass
class Unit:
    unit_id: str
    size_bytes: int
    dependencies: tuple[str, ...] = ()
    state: str = "unloaded"
    last_used: int = 0
    in_use: int = 0


@dataclass
class BoundedWorkspace:
    budget_bytes: int
    units: dict[str, Unit]
    clock: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: dict[str, Any]) -> "BoundedWorkspace":
        if not isinstance(plan, dict):
            raise WorkspaceError("plan must be an object")
        budget = plan.get("budget_bytes")
        raw_units = plan.get("units")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise WorkspaceError("budget_bytes must be a positive integer")
        if not isinstance(raw_units, list) or not raw_units:
            raise WorkspaceError("units must be a non-empty array")
        units: dict[str, Unit] = {}
        for item in raw_units:
            if not isinstance(item, dict):
                raise WorkspaceError("each unit must be an object")
            unit_id = item.get("unit_id")
            size = item.get("size_bytes")
            deps = item.get("dependencies", [])
            if not isinstance(unit_id, str) or not unit_id.strip():
                raise WorkspaceError("unit_id must be a non-empty string")
            if unit_id in units:
                raise WorkspaceError(f"duplicate unit_id: {unit_id}")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise WorkspaceError(f"{unit_id}.size_bytes must be a positive integer")
            if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
                raise WorkspaceError(f"{unit_id}.dependencies must be an array of strings")
            units[unit_id] = Unit(unit_id, size, tuple(deps))
        for unit in units.values():
            missing = set(unit.dependencies) - set(units)
            if missing:
                raise WorkspaceError(f"{unit.unit_id} has unknown dependencies: {sorted(missing)}")
            if unit.unit_id in unit.dependencies:
                raise WorkspaceError(f"{unit.unit_id} cannot depend on itself")
        workspace = cls(budget, units)
        for unit_id in units:
            workspace._dependency_order(unit_id)
        workspace._record("initialized", budget_bytes=budget, unit_count=len(units))
        return workspace

    @property
    def resident_bytes(self) -> int:
        return sum(u.size_bytes for u in self.units.values() if u.state in {"resident", "in_use", "evict_pending"})

    def _record(self, event: str, **details: Any) -> None:
        self.trace.append({
            "step": len(self.trace),
            "event": event,
            "resident_bytes": self.resident_bytes,
            **details,
        })

    def _dependency_order(self, unit_id: str) -> list[str]:
        if unit_id not in self.units:
            raise WorkspaceError(f"unknown unit: {unit_id}")
        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(current: str) -> None:
            if current in visiting:
                raise WorkspaceError(f"dependency cycle includes {current}")
            if current in visited:
                return
            visiting.add(current)
            for dependency in self.units[current].dependencies:
                visit(dependency)
            visiting.remove(current)
            visited.add(current)
            result.append(current)

        visit(unit_id)
        return result

    def _evict_one(self, protected: set[str]) -> bool:
        candidates = [
            unit for unit in self.units.values()
            if unit.state == "resident" and unit.in_use == 0 and unit.unit_id not in protected
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda unit: (unit.last_used, unit.unit_id))
        victim.state = "unloaded"
        self._record("evicted", unit_id=victim.unit_id, size_bytes=victim.size_bytes)
        return True

    def load(self, unit_id: str, fail_at: str | None = None) -> None:
        order = self._dependency_order(unit_id)
        protected = set(order)
        needed = sum(
            self.units[name].size_bytes for name in order
            if self.units[name].state == "unloaded"
        )
        if needed > self.budget_bytes:
            raise WorkspaceError(
                f"dependency-closed request needs {needed} additional bytes; "
                f"budget is {self.budget_bytes}"
            )
        while self.resident_bytes + needed > self.budget_bytes:
            if not self._evict_one(protected):
                raise WorkspaceError("cannot fit request without evicting in-use/protected units")
        newly_loaded: list[str] = []
        try:
            for name in order:
                unit = self.units[name]
                if unit.state in {"resident", "in_use"}:
                    continue
                unit.state = "loading"
                self._record("loading", unit_id=name, size_bytes=unit.size_bytes)
                if fail_at == name:
                    raise WorkspaceError(f"simulated load failure at {name}")
                unit.state = "resident"
                newly_loaded.append(name)
                self.clock += 1
                unit.last_used = self.clock
                self._record("loaded", unit_id=name, size_bytes=unit.size_bytes)
        except Exception:
            for name in reversed(newly_loaded):
                self.units[name].state = "unloaded"
            for unit in self.units.values():
                if unit.state == "loading":
                    unit.state = "unloaded"
            self._record("load_rollback", requested_unit=unit_id)
            raise
        self._record("load_complete", requested_unit=unit_id, dependency_order=order)

    def begin_use(self, unit_id: str) -> None:
        order = self._dependency_order(unit_id)
        if any(self.units[name].state not in {"resident", "in_use"} for name in order):
            raise WorkspaceError(f"{unit_id} and all dependencies must be resident before use")
        for name in order:
            unit = self.units[name]
            unit.in_use += 1
            unit.state = "in_use"
            self.clock += 1
            unit.last_used = self.clock
        self._record("begin_use", unit_id=unit_id, dependency_order=order)

    def end_use(self, unit_id: str) -> None:
        order = self._dependency_order(unit_id)
        if any(self.units[name].in_use <= 0 for name in order):
            raise WorkspaceError(f"unbalanced end_use for {unit_id}")
        for name in order:
            unit = self.units[name]
            unit.in_use -= 1
            if unit.in_use == 0:
                unit.state = "resident"
        self._record("end_use", unit_id=unit_id, dependency_order=order)

    def unload(self, unit_id: str) -> None:
        if unit_id not in self.units:
            raise WorkspaceError(f"unknown unit: {unit_id}")
        unit = self.units[unit_id]
        if unit.in_use:
            unit.state = "evict_pending"
            self._record("evict_deferred_in_use", unit_id=unit_id)
            return
        unit.state = "unloaded"
        self._record("unloaded", unit_id=unit_id)

    def run(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(actions, list):
            raise WorkspaceError("actions must be an array")
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise WorkspaceError(f"action {index} must be an object")
            op = action.get("op")
            unit_id = action.get("unit_id")
            if op == "load":
                self.load(unit_id, action.get("simulate_failure_at"))
            elif op == "begin_use":
                self.begin_use(unit_id)
            elif op == "end_use":
                self.end_use(unit_id)
            elif op == "unload":
                self.unload(unit_id)
            else:
                raise WorkspaceError(f"action {index} has unsupported op: {op}")
            if self.resident_bytes > self.budget_bytes:
                raise WorkspaceError("internal invariant violated: resident bytes exceed budget")
        return {
            "schema_version": "aimeng.parameter_workspace_simulation.v1",
            "simulation_only": True,
            "budget_bytes": self.budget_bytes,
            "final_resident_bytes": self.resident_bytes,
            "peak_resident_bytes": max((row["resident_bytes"] for row in self.trace), default=0),
            "unit_states": {
                key: {"state": unit.state, "size_bytes": unit.size_bytes, "in_use": unit.in_use}
                for key, unit in self.units.items()
            },
            "trace": self.trace,
            "limitations": [
                "No model weights were loaded or unloaded.",
                "No actual OS/device memory was measured or released.",
                "No model computation was skipped or executed.",
                "The byte budget models parameter-unit residency only, not activations, KV cache, staging buffers, or runtime overhead.",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Simulation plan JSON")
    parser.add_argument("--output", required=True, help="Simulation report JSON")
    args = parser.parse_args()
    try:
        plan = json.loads(Path(args.input).read_text(encoding="utf-8"))
        workspace = BoundedWorkspace.from_plan(plan)
        report = workspace.run(plan.get("actions", []))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, WorkspaceError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "simulation_only": True, "peak_resident_bytes": report["peak_resident_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
