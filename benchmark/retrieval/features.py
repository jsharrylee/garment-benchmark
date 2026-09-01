from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _foreground_mask(path: Path) -> np.ndarray:
    """Return a normalized foreground mask from alpha/mask or white-background RGB."""
    with Image.open(path) as image:
        if image.mode in {"1", "L", "I", "F"}:
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            return gray > 0.5
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
            if alpha.min() < alpha.max():
                return alpha > 0.5
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    # Render inputs use white backgrounds. A generous threshold keeps faded body context.
    return np.max(1.0 - rgb, axis=2) > 0.035


def _binned_profile(values: np.ndarray, bins: int = 8) -> list[float]:
    parts = np.array_split(values.astype(np.float32), bins)
    return [float(part.mean()) if part.size else 0.0 for part in parts]


def single_view_descriptor(path: Path) -> tuple[float, ...]:
    mask = _foreground_mask(path)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"empty foreground mask: {path}")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    occupancy = float(mask.mean())
    bbox_width = float((x1 - x0) / width)
    bbox_height = float((y1 - y0) / height)
    centroid_x = float((xs.mean() + 0.5) / width)
    centroid_y = float((ys.mean() + 0.5) / height)
    row_profile = _binned_profile(mask.mean(axis=1))
    col_profile = _binned_profile(mask.mean(axis=0))
    return tuple([occupancy, bbox_width, bbox_height, centroid_x, centroid_y, *row_profile, *col_profile])


def multiview_descriptor(paths: Iterable[Path]) -> tuple[float, ...]:
    ordered = tuple(sorted((Path(path) for path in paths), key=lambda value: value.name.lower()))
    if len(ordered) != 4:
        raise ValueError(f"exactly four views are required, found {len(ordered)}")
    if len({path.resolve() for path in ordered}) != 4:
        raise ValueError("four distinct view files are required")
    values: list[float] = []
    for path in ordered:
        values.extend(single_view_descriptor(path))
    return tuple(values)


def normalized_l1_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("descriptor shapes must be equal one-dimensional vectors")
    return float(1.0 / (1.0 + 4.0 * np.mean(np.abs(a - b))))
