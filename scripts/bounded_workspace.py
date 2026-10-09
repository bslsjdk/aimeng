"""Fixed-capacity resource-slot scheduler.

Metadata only: this module does NOT load model weights, force OS page eviction,
or prove a process/whole-app RAM bound. A real backend adapter must implement
and measure those operations before claiming memory savings.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Slot:
    slot_id: int
    resource_id: str | None = None
    artifact_path: str | None = None
    size_bytes: int = 0
    pinned: bool = False
    last_used: int = 0
    metadata: dict[str, Any] | None = None


class BudgetExceeded(ValueError):
    """Raised when a resource cannot fit in the fixed workspace budget."""


class NoEvictableSlot(RuntimeError):
    """Raised when every slot is pinned and no free slot is available."""


class BoundedWorkspace:
    """A deterministic, byte-budgeted LRU pool with a fixed slot count."""

    def __init__(self, *, max_slots: int, max_bytes: int):
        if not isinstance(max_slots, int) or max_slots < 1:
            raise ValueError("max_slots must be a positive integer")
        if not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.max_slots = max_slots
        self.max_bytes = max_bytes
        self._clock = 0
        self._slots = [Slot(slot_id=i) for i in range(max_slots)]

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(self._slots)

    @property
    def used_bytes(self) -> int:
        return sum(s.size_bytes for s in self._slots if s.resource_id is not None)

    @property
    def active_resources(self) -> tuple[str, ...]:
        return tuple(s.resource_id for s in self._slots if s.resource_id is not None)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def touch(self, resource_id: str) -> Slot:
        for slot in self._slots:
            if slot.resource_id == resource_id:
                slot.last_used = self._tick()
                return slot
        raise KeyError(resource_id)

    def admit(self, resource_id: str, artifact_path: str, size_bytes: int, *,
              pinned: bool = False, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        if not isinstance(size_bytes, int) or size_bytes < 1:
            raise ValueError("size_bytes must be a positive integer")
        if size_bytes > self.max_bytes:
            raise BudgetExceeded("resource exceeds the entire workspace byte budget")

        for slot in self._slots:
            if slot.resource_id == resource_id:
                if slot.size_bytes != size_bytes or slot.artifact_path != str(Path(artifact_path)):
                    raise ValueError("resource_id already exists with different artifact metadata")
                slot.pinned = pinned
                slot.metadata = dict(metadata or {})
                slot.last_used = self._tick()
                return {"action": "reuse", "slot_id": slot.slot_id, "evicted": None}

        evicted = None
        while self.used_bytes + size_bytes > self.max_bytes:
            candidates = [s for s in self._slots if s.resource_id is not None and not s.pinned]
            if not candidates:
                raise NoEvictableSlot("byte budget is full and all resident slots are pinned")
            victim = min(candidates, key=lambda s: (s.last_used, s.slot_id))
            evicted = victim.resource_id
            self._clear(victim)

        slot = next((s for s in self._slots if s.resource_id is None), None)
        if slot is None:
            candidates = [s for s in self._slots if not s.pinned]
            if not candidates:
                raise NoEvictableSlot("all workspace slots are pinned")
            slot = min(candidates, key=lambda s: (s.last_used, s.slot_id))
            evicted = slot.resource_id
            self._clear(slot)

        slot.resource_id = resource_id
        slot.artifact_path = str(Path(artifact_path))
        slot.size_bytes = size_bytes
        slot.pinned = pinned
        slot.last_used = self._tick()
        slot.metadata = dict(metadata or {})
        return {"action": "admit", "slot_id": slot.slot_id, "evicted": evicted}

    def release(self, resource_id: str) -> bool:
        for slot in self._slots:
            if slot.resource_id == resource_id:
                self._clear(slot)
                return True
        return False

    @staticmethod
    def _clear(slot: Slot) -> None:
        slot.resource_id = None
        slot.artifact_path = None
        slot.size_bytes = 0
        slot.pinned = False
        slot.last_used = 0
        slot.metadata = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "aimeng.workspace.v1",
            "max_slots": self.max_slots,
            "max_bytes": self.max_bytes,
            "used_bytes": self.used_bytes,
            "slot_count": sum(s.resource_id is not None for s in self._slots),
            "resources": [
                {"slot_id": s.slot_id, "resource_id": s.resource_id,
                 "size_bytes": s.size_bytes, "pinned": s.pinned, "last_used": s.last_used}
                for s in self._slots if s.resource_id is not None
            ],
            "ram_bound_verified": False,
            "note": "Metadata scheduler only; backend residency and whole-app PSS are not measured.",
        }
