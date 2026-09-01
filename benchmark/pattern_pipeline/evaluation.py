from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ORTHOGONAL_VIEWS = ("front", "back", "left", "right")


def silhouette_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference).astype(bool)
    candidate = np.asarray(candidate).astype(bool)
    if reference.shape != candidate.shape:
        raise ValueError(f"mask shape mismatch: {reference.shape} != {candidate.shape}")
    union = np.logical_or(reference, candidate).sum()
    return 1.0 if union == 0 else float(np.logical_and(reference, candidate).sum() / union)


def compare_orthogonal_masks(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict:
    missing = [view for view in ORTHOGONAL_VIEWS if view not in reference or view not in candidate]
    if missing:
        return {"accepted": False, "failure": "MISSING_ORTHOGONAL_VIEW", "missing": missing}
    scores = {view: silhouette_iou(reference[view], candidate[view]) for view in ORTHOGONAL_VIEWS}
    return {"accepted": True, "per_view_iou": scores, "mean_iou": float(np.mean(list(scores.values())))}
