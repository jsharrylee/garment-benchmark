from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .schemas import ActionIntent, ActionReceipt, Stage
from .state import RunState


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Executor:
    """Records intent before work and receipt afterwards; completed actions are idempotent."""

    def __init__(self, root: Path):
        self.root = root
        self.state = RunState(root / "state" / "run_state.json")
        self.events = root / "state" / "events.jsonl"
        self.receipts: set[str] = set()
        if self.events.exists():
            self.receipts = {json.loads(line)["action_id"] for line in self.events.read_text(encoding="utf-8").splitlines() if json.loads(line)["type"] == "receipt"}

    def _event(self, event: dict) -> None:
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def run(
        self,
        job_id: str,
        stage: Stage,
        command: str,
        work: Callable[[], ActionReceipt],
        *,
        attempt: int = 1,
        repair_hypothesis: str | None = None,
        input_hashes: dict[str, str] | None = None,
        expected_outputs: list[str] | None = None,
    ) -> ActionReceipt | None:
        action_id = f"{job_id}:{stage.value}:{attempt}"
        if action_id in self.receipts:
            return None
        current = self.state.stage(job_id)
        if current is not None and current != stage:
            raise ValueError(f"job {job_id} is at {current}, not requested stage {stage}")
        intent = ActionIntent(
            action_id,
            job_id,
            stage,
            command,
            input_hashes or {},
            expected_outputs or [],
            attempt,
            now(),
            repair_hypothesis,
        )
        self._event({"type": "intent", **intent.as_dict()})
        # Persist the active first stage before dispatch so interruption can resume safely.
        if self.state.stage(job_id) is None:
            self.state.transition(job_id, Stage.DISCOVER, action_id)
        receipt = work()
        self._event({"type": "receipt", **receipt.as_dict(), "job_id": job_id, "stage": stage.value})
        self.receipts.add(action_id)
        self.state.transition(job_id, receipt.next_stage, action_id)
        return receipt
