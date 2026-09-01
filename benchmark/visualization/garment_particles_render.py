from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.pattern_pipeline.garment_particles_convert import sample_generated_edge


def _sample_edge(start: np.ndarray, vector: np.ndarray, samples: int = 30) -> np.ndarray:
    return sample_generated_edge(start, vector, samples)


def render_garment_particles_pattern(npz_path: Path, output_path: Path, *, title: str) -> dict:
    archive = np.load(npz_path)
    edges = np.asarray(archive["edges"], dtype=np.float64)
    valid = np.asarray(archive["edge_valid_mask"], dtype=bool)
    shifts = np.asarray(archive["panel_shifts"], dtype=np.float64)
    stitch = np.asarray(archive["stitch_flags"], dtype=bool)
    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20")
    plotted = []
    closure_gaps = []
    for panel in np.flatnonzero(valid.any(axis=1)):
        cursor = np.zeros(2, dtype=np.float64)
        panel_edges = edges[panel, valid[panel]]
        panel_stitches = stitch[panel, valid[panel]]
        for edge_index, (edge, stitched) in enumerate(zip(panel_edges, panel_stitches, strict=True)):
            points = _sample_edge(cursor, edge) + shifts[panel]
            plotted.append(points)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=colors(panel % 20),
                linestyle="--" if stitched else "-",
                linewidth=1.5,
            )
            cursor += edge[:2]
        closure_gaps.append(float(np.linalg.norm(cursor)))
        centroid = np.concatenate(plotted[-len(panel_edges) :]).mean(axis=0)
        axis.text(centroid[0], centroid[1], str(panel), fontsize=7, ha="center")
    points = np.concatenate(plotted)
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    axis.set_title(title + "\n(dashed edges: predicted stitches)")
    axis.set_xlabel("packed pattern x")
    axis.set_ylabel("packed pattern y")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    return {
        "panel_count": int(valid.any(axis=1).sum()),
        "edge_count": int(valid.sum()),
        "closure_gap_mean": float(np.mean(closure_gaps)),
        "closure_gap_max": float(np.max(closure_gaps)),
        "bbox": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "output": str(output_path),
    }


def _equal_limits(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max(float((maximum - minimum).max()) / 2, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def render_garment_particles_geometry(npz_path: Path, output_path: Path, *, title: str) -> dict:
    archive = np.load(npz_path)
    particles = np.asarray(archive["particles"], dtype=np.float64)
    xyz = particles[:, 2:5]
    display = xyz[:, [0, 2, 1]]  # keep the dataset's y (body height) vertical
    boundary = particles[:, -1] >= 0.5
    figure = plt.figure(figsize=(12, 6))
    for slot, (azimuth, label) in enumerate(((35, "three-quarter"), (180, "opposite")), start=1):
        axis = figure.add_subplot(1, 2, slot, projection="3d")
        axis.scatter(display[~boundary, 0], display[~boundary, 1], display[~boundary, 2], s=0.7, alpha=0.28, color="#268bd2")
        axis.scatter(display[boundary, 0], display[boundary, 1], display[boundary, 2], s=1.4, alpha=0.75, color="#dc322f")
        axis.view_init(elev=16, azim=azimuth)
        _equal_limits(axis, display)
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("z")
        axis.set_zlabel("y (body height)")
    figure.suptitle(title + "\n(red: boundary particles)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    return {"particle_count": int(len(xyz)), "boundary_particle_count": int(boundary.sum()), "output": str(output_path)}
