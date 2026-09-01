from __future__ import annotations

import numpy as np


def resample_closed_contour(points_xy: np.ndarray, count: int = 256) -> np.ndarray:
    points = np.asarray(points_xy, np.float64).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("a closed contour needs at least three points")
    closed = np.vstack((points, points[0]))
    delta = np.diff(closed, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
    if cumulative[-1] <= 1e-8:
        raise ValueError("degenerate closed contour")
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return np.stack([np.interp(samples, cumulative, closed[:, axis]) for axis in (0, 1)], axis=1).astype(np.float32)


def contour_from_probability(probability: np.ndarray, threshold: float = 0.5, count: int = 256) -> np.ndarray:
    import cv2

    mask = (np.asarray(probability) >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("predicted mask has no external contour")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    contour[:, 0] /= max(mask.shape[1] - 1, 1)
    contour[:, 1] /= max(mask.shape[0] - 1, 1)
    return resample_closed_contour(contour, count)


def symmetric_chamfer(first: np.ndarray, second: np.ndarray) -> float:
    distances = np.linalg.norm(np.asarray(first)[:, None] - np.asarray(second)[None], axis=-1)
    return float((distances.min(1).mean() + distances.min(0).mean()) / 2.0)


__all__ = ["contour_from_probability", "resample_closed_contour", "symmetric_chamfer"]
