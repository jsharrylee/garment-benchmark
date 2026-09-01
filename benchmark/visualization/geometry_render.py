from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _equal_limits(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max(float((maximum - minimum).max()) / 2, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def render_reweaver_geometry(npz_path: Path, output_path: Path, *, title: str) -> dict:
    archive = np.load(npz_path, allow_pickle=True)
    patches = np.asarray(archive["patch_points"], dtype=np.float64)
    curves = np.asarray(archive["curve_points"], dtype=np.float64)
    patch_valid = np.asarray(archive["patch_valid_prob"]) >= 0.5
    curve_valid = np.asarray(archive["curve_valid_prob"]) >= 0.5
    valid_patches = patches[patch_valid]
    valid_curves = curves[curve_valid]
    all_points = np.concatenate([valid_patches.reshape(-1, 3), valid_curves.reshape(-1, 3)], axis=0)
    figure = plt.figure(figsize=(12, 6))
    for slot, (azimuth, label) in enumerate(((35, "three-quarter"), (180, "opposite")), start=1):
        axis = figure.add_subplot(1, 2, slot, projection="3d")
        for index, patch in enumerate(valid_patches):
            axis.scatter(patch[:, 0], patch[:, 1], patch[:, 2], s=0.8, alpha=0.22, label=None)
        for curve in valid_curves:
            axis.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=0.65, alpha=0.75, color="black")
        axis.view_init(elev=16, azim=azimuth)
        _equal_limits(axis, all_points)
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    return {
        "valid_patch_count": int(patch_valid.sum()),
        "valid_curve_count": int(curve_valid.sum()),
        "point_count": int(len(all_points)),
        "output": str(output_path),
    }
