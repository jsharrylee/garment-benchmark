from __future__ import annotations

import math
from typing import Iterable


def finite_nonempty(values: Iterable[float]) -> tuple[bool, str]:
    values = list(values)
    if not values:
        return False, "EMPTY_OUTPUT"
    if not all(math.isfinite(value) for value in values):
        return False, "NONFINITE_OUTPUT"
    return True, "OK"


def valid_references(references: Iterable[int], edge_count: int) -> tuple[bool, str]:
    if any(reference < 0 or reference >= edge_count for reference in references):
        return False, "INVALID_PATTERN_STRUCTURE"
    return True, "OK"


def coherent_views(view_ids: Iterable[str]) -> tuple[bool, str]:
    selected = list(view_ids)
    if len(selected) != 4 or len(set(selected)) != 4:
        return False, "NO_COHERENT_MULTIVIEW_GROUP"
    return True, "OK"


def nonempty_mask(foreground_pixels: int) -> tuple[bool, str]:
    return (True, "OK") if foreground_pixels > 0 else (False, "MASK_FAILURE")


def provenance_matches(input_hash: str, recorded_input_hash: str) -> tuple[bool, str]:
    return (True, "OK") if input_hash == recorded_input_hash else (False, "OUTPUT_NOT_BOUND_TO_INPUT")
