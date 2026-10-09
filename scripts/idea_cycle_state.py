"""Crash-resilient state machine for the AIMENG idea/verification lifecycle.

This module manages durable task metadata only. It does not call a model, claim
that ideas are correct, train weights, or promote plastic modules.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = {
    "CREATED", "CONTEXT_READY", "EXPLORE", "HYPOTHESIS_READY", "CHECKING",
    "REVISING", "SOLVED_PENDING_REVIEW", "PLASTIC_TRIAL",
    "CANDIDATE_EVALUATION", "CONSOLIDATE", "DONE",
    "FAILED_RECOVERABLE", "ABORTED_BUDGET",
}
TERMINAL = {"DONE", "ABORTED_BUDGET"}
TRANSITIONS = {
    "CREATED": {"CONTEXT_READY", "FAILED_RECOVERABLE"},
    "CONTEXT_READY": {"EXPLORE", "FAILED_RECOVERABLE", "ABORTED_BUDGET"},
    "EXPLORE": {"HYPOTHESIS_READY", "ABORTED_BUDGET", "FAILED_RECOVERABLE"},
    "HYPOTHESIS_READY": {"CHECKING", "ABORTED_BUDGET", "FAILED_RECOVERABLE"},
    "CHECKING": {"REVISING", "SOLVED_PENDING_REVIEW", "ABORTED_BUDGET", "FAILED_RECOVERABLE"},
    "REVISING": {"HYPOTHESIS_READY", "SOLVED_PENDING_REVIEW", "ABORTED_BUDGET", "FAILED_RECOVERABLE"},
    "SOLVED_PENDING_REVIEW": {"PLASTIC_TRIAL", "CONSOLIDATE", "FAILED_RECOVERABLE", "ABORTED_BUDGET"},
    "PLASTIC_TRIAL": {"CANDIDATE_EVALUATION", "CONSOLIDATE", "FAILED_RECOVERABLE", "ABORTED_BUDGET"},
    "CANDIDATE_EVALUATION": {"CONSOLIDATE", "FAILED_RECOVERABLE", "ABORTED_BUDGET"},
    "CONSOLIDATE": {"DONE", "FAILED_RECOVERABLE"},
    "FAILED_RECOVERABLE": set(),
    "DONE": set(),
    "ABORTED_BUDGET": set(),
}


class StateError(ValueError):
    """Raised when a task state transition violates the lifecycle contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class IdeaCycleState:
    """A small, persistent state machine; each transition is atomically committed."""

    def __init__(self, path: str | Path, document: dict[str, Any]):
        self.path = Path(path)
        self.document = document

    @classmethod
    def create(cls, path: str | Path, task_id: str, budget: dict[str, Any] | None = None):
        if not isinstance(task_id, str) or not task_id.strip():
            raise StateError("task_id must be a non-empty string")
        target = Path(path)
        if target.exists():
            raise StateError(f"refusing to overwrite existing task state: {target}")
        doc = {
            "schema_version": "aimeng.task_state.v1",
            "task_id": task_id,
            "state": "CREATED",
            "resume_state": None,
            "budget": budget or {},
            "counters": {"ideas_created": 0, "verification_calls": 0, "plastic_trials": 0},
            "artifacts": [],
            "events": [{
                "at": _now(), "from": None, "to": "CREATED",
                "reason": "task_created", "metadata": {}
            }],
            "created_at": _now(),
            "updated_at": _now(),
        }
        instance = cls(target, doc)
        instance._commit()
        return instance

    @classmethod
    def load(cls, path: str | Path):
        target = Path(path)
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StateError(f"task state does not exist: {target}") from exc
        except json.JSONDecodeError as exc:
            raise StateError(f"task state JSON is corrupt: {target}") from exc
        if not isinstance(doc, dict) or doc.get("schema_version") != "aimeng.task_state.v1":
            raise StateError("unsupported or invalid task state schema")
        if not isinstance(doc.get("state"), str) or doc.get("state") not in STATES:
            raise StateError("unknown task state")
        if not isinstance(doc.get("events"), list) or not isinstance(doc.get("counters"), dict):
            raise StateError("task state is missing events or counters")
        return cls(target, doc)

    @property
    def state(self) -> str:
        return self.document["state"]

    def _commit(self) -> None:
        self.document["updated_at"] = _now()
        _atomic_json(self.path, self.document)

    def transition(self, target: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
        if not isinstance(target, str) or target not in STATES:
            raise StateError(f"unknown target state: {target}")
        if self.state in TERMINAL:
            raise StateError(f"terminal state {self.state} cannot transition")
        if self.state == "FAILED_RECOVERABLE":
            raise StateError("use resume() to recover from FAILED_RECOVERABLE")
        if target not in TRANSITIONS[self.state]:
            raise StateError(f"illegal transition: {self.state} -> {target}")
        if not isinstance(reason, str) or not reason.strip():
            raise StateError("transition reason must be non-empty")
        previous = self.state
        self.document["state"] = target
        self.document["events"].append({
            "at": _now(), "from": previous, "to": target,
            "reason": reason, "metadata": metadata or {}
        })
        self._commit()

    def record_artifact(self, artifact_ref: str, sha256: str, artifact_type: str) -> None:
        import re
        if not isinstance(artifact_ref, str) or not artifact_ref.strip():
            raise StateError("artifact_ref must be non-empty")
        if not isinstance(sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
            raise StateError("sha256 must contain exactly 64 hexadecimal characters")
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            raise StateError("artifact_type must be non-empty")
        self.document["artifacts"].append({
            "ref": artifact_ref, "sha256": sha256.lower(),
            "type": artifact_type, "recorded_at": _now()
        })
        self._commit()

    def increment(self, counter: str, amount: int = 1, maximum: int | None = None) -> int:
        if counter not in self.document["counters"]:
            raise StateError(f"unknown counter: {counter}")
        if not isinstance(amount, int) or amount < 0:
            raise StateError("counter increment must be a non-negative integer")
        current = self.document["counters"][counter]
        new_value = current + amount
        if maximum is not None and new_value > maximum:
            raise StateError(f"budget exceeded for {counter}: {new_value} > {maximum}")
        self.document["counters"][counter] = new_value
        self._commit()
        return new_value

    def fail_recoverably(self, reason: str, metadata: dict[str, Any] | None = None) -> None:
        if self.state in TERMINAL or self.state == "FAILED_RECOVERABLE":
            raise StateError(f"cannot mark {self.state} as recoverable failure")
        previous = self.state
        self.document["resume_state"] = previous
        self.document["state"] = "FAILED_RECOVERABLE"
        self.document["events"].append({
            "at": _now(), "from": previous, "to": "FAILED_RECOVERABLE",
            "reason": reason, "metadata": metadata or {}
        })
        self._commit()

    def resume(self, reason: str = "resume_after_recovery") -> str:
        if self.state != "FAILED_RECOVERABLE":
            raise StateError("resume() requires FAILED_RECOVERABLE state")
        target = self.document.get("resume_state")
        if not isinstance(target, str) or target not in STATES or target in TERMINAL or target == "FAILED_RECOVERABLE":
            raise StateError("invalid resume_state")
        self.document["state"] = target
        self.document["resume_state"] = None
        self.document["events"].append({
            "at": _now(), "from": "FAILED_RECOVERABLE", "to": target,
            "reason": reason, "metadata": {"recovered": True}
        })
        self._commit()
        return target

    def abort_for_budget(self, reason: str, metadata: dict[str, Any] | None = None) -> None:
        self.transition("ABORTED_BUDGET", reason, metadata)
