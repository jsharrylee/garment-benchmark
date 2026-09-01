from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    DISCOVER = "DISCOVER"
    ACQUIRE = "ACQUIRE"
    VALIDATE_RAW = "VALIDATE_RAW"
    GROUP_VIEWS = "GROUP_VIEWS"
    PREPROCESS = "PREPROCESS"
    VALIDATE_INPUT = "VALIDATE_INPUT"
    PREPARE_MODEL = "PREPARE_MODEL"
    OFFICIAL_SMOKE = "OFFICIAL_SMOKE"
    EXTERNAL_INFERENCE = "EXTERNAL_INFERENCE"
    VALIDATE_OUTPUT = "VALIDATE_OUTPUT"
    VISUAL_REVIEW = "VISUAL_REVIEW"
    COMPARE = "COMPARE"
    ACCEPTED = "ACCEPTED"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
    FAILED_BUDGET = "FAILED_BUDGET"
    FAILED_VALIDATION = "FAILED_VALIDATION"


TERMINAL_STAGES = {Stage.ACCEPTED, Stage.BLOCKED_EXTERNAL, Stage.FAILED_BUDGET, Stage.FAILED_VALIDATION}
ORDERED_STAGES = tuple(stage for stage in Stage if stage not in TERMINAL_STAGES)


@dataclass(frozen=True)
class ActionIntent:
    action_id: str
    job_id: str
    stage: Stage
    command: str
    input_hashes: dict[str, str]
    expected_outputs: list[str]
    attempt: int
    started_at: str
    repair_hypothesis: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "stage": self.stage.value}


@dataclass(frozen=True)
class ActionReceipt:
    action_id: str
    exit_status: int
    ended_at: str
    stdout_log: str
    stderr_log: str
    produced_hashes: dict[str, str] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    peak_vram_mib: int | None = None
    next_stage: Stage = Stage.FAILED_VALIDATION

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "next_stage": self.next_stage.value}
