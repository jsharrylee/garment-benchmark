from __future__ import annotations

from .schemas import Stage


def next_failure_state(consecutive_non_improvements: int) -> Stage | None:
    """Two non-improving repairs stop a lane; otherwise a single repair is allowed."""
    return Stage.FAILED_VALIDATION if consecutive_non_improvements >= 2 else None
