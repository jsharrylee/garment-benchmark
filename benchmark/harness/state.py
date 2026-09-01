from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .schemas import ORDERED_STAGES, TERMINAL_STAGES, Stage


class TransitionError(ValueError):
    pass


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


class RunState:
    """Atomically persisted job states; one job never skips a stage."""

    def __init__(self, path: Path):
        self.path = path
        self.payload = self._read()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "jobs": {}}
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        stored.setdefault("version", 1)
        stored.setdefault("jobs", {})
        return stored

    def stage(self, job_id: str) -> Stage | None:
        value = self.payload["jobs"].get(job_id, {}).get("stage")
        return Stage(value) if value else None

    def transition(self, job_id: str, target: Stage, action_id: str) -> bool:
        current = self.stage(job_id)
        if current == target:
            return False
        if current in TERMINAL_STAGES:
            raise TransitionError(f"terminal job {job_id} cannot transition from {current}")
        if current is None:
            if target != Stage.DISCOVER:
                raise TransitionError("new jobs must begin at DISCOVER")
        elif target not in TERMINAL_STAGES:
            expected = ORDERED_STAGES[ORDERED_STAGES.index(current) + 1]
            if target != expected:
                raise TransitionError(f"expected {expected}, got {target}")
        self.payload["jobs"][job_id] = {"stage": target.value, "last_action_id": action_id}
        atomic_json_write(self.path, self.payload)
        return True
