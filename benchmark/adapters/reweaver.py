from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REWEAVER_COMMIT = "e5640769f89d9540d118a9cc605ebe2d7404bc02"
VIEW_ORDER = ("CAM000", "CAM001", "CAM002", "CAM003")
IMAGE_MEAN = np.array([0.9329, 0.9249, 0.92], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGE_STD = np.array([0.1885, 0.2093, 0.2226], dtype=np.float32).reshape(1, 3, 1, 1)


def order_panel_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedily order and orient an unordered panel-edge set into its lowest-gap cycle."""
    edges = np.asarray(edges)
    best_edges = None
    best_gaps = None
    best_cost = float("inf")
    for start in range(len(edges)):
        for start_flip in (False, True):
            current = edges[start][::-1] if start_flip else edges[start]
            ordered = [current]
            unused = set(range(len(edges))) - {start}
            while unused:
                choice = None
                for index in unused:
                    for flip in (False, True):
                        candidate = edges[index][::-1] if flip else edges[index]
                        distance = float(np.linalg.norm(ordered[-1][-1] - candidate[0]))
                        if choice is None or distance < choice[0]:
                            choice = (distance, index, candidate)
                assert choice is not None
                _, index, candidate = choice
                ordered.append(candidate)
                unused.remove(index)
            cycle = np.stack(ordered)
            gaps = np.linalg.norm(cycle[:, -1, :] - np.roll(cycle[:, 0, :], -1, axis=0), axis=1)
            cost = float(gaps.sum())
            if cost < best_cost:
                best_edges, best_gaps, best_cost = cycle, gaps, cost
    assert best_edges is not None and best_gaps is not None
    return best_edges, best_gaps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input_directory(path: Path) -> dict:
    from PIL import Image

    files = [path / f"{camera}.png" for camera in VIEW_ORDER]
    missing = [str(item) for item in files if not item.is_file()]
    if missing:
        return {"valid": False, "failure": "INPUT_ADAPTER", "missing": missing}
    dimensions = []
    hashes = []
    for item in files:
        with Image.open(item) as image:
            image.verify()
        with Image.open(item) as image:
            dimensions.append([image.width, image.height])
        hashes.append(sha256(item))
    valid = dimensions == [[518, 518]] * 4 and len(set(hashes)) == 4
    return {"valid": valid, "failure": None if valid else "INPUT_ADAPTER", "dimensions": dimensions, "sha256": hashes}


def summarize_output(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        return {"valid": False, "failure": "EMPTY_OUTPUT"}
    archive = np.load(path, allow_pickle=True)
    required = {
        "flatten_pred",
        "patch_curve_connectivity",
        "curve_points",
        "curve_valid_prob",
        "patch_points",
        "patch_valid_prob",
        "patch_points_scaled",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        return {"valid": False, "failure": "INVALID_PATTERN_STRUCTURE", "missing_keys": missing}
    numeric_keys = [key for key in required if key not in {"flatten_pred", "patch_points_scaled"}]
    arrays = {key: np.asarray(archive[key]) for key in numeric_keys}
    if any(array.size == 0 for array in arrays.values()):
        return {"valid": False, "failure": "EMPTY_OUTPUT"}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        return {"valid": False, "failure": "NONFINITE_OUTPUT"}
    scaled = archive["patch_points_scaled"]
    if scaled.size == 0:
        return {"valid": False, "failure": "EMPTY_OUTPUT"}
    for points in scaled.flat:
        points_array = np.asarray(points)
        if points_array.size == 0:
            return {"valid": False, "failure": "EMPTY_OUTPUT"}
        if not np.isfinite(points_array).all():
            return {"valid": False, "failure": "NONFINITE_OUTPUT"}
    panel_count = int(np.count_nonzero(arrays["patch_valid_prob"] >= 0.5))
    curve_count = int(np.count_nonzero(arrays["curve_valid_prob"] >= 0.5))
    connectivity = arrays["patch_curve_connectivity"].astype(bool)
    if (
        panel_count <= 0
        or curve_count <= 0
        or connectivity.ndim != 2
        or connectivity.shape != (len(arrays["patch_valid_prob"]), len(arrays["curve_valid_prob"]))
    ):
        return {"valid": False, "failure": "INVALID_PATTERN_STRUCTURE", "panel_count": panel_count, "curve_count": curve_count}
    flatten_pred = archive["flatten_pred"].item()
    if not isinstance(flatten_pred, dict) or len(flatten_pred) != panel_count:
        return {
            "valid": False,
            "failure": "INVALID_PATTERN_STRUCTURE",
            "panel_count": panel_count,
            "flatten_panel_count": len(flatten_pred) if isinstance(flatten_pred, dict) else None,
        }
    panel_edge_counts: list[int] = []
    boundary_closure_gaps: list[float] = []
    for panel in flatten_pred.values():
        edges = np.asarray(panel.get("edge_points"))
        if edges.ndim != 3 or edges.shape[0] == 0 or edges.shape[-1] != 2 or not np.isfinite(edges).all():
            return {"valid": False, "failure": "INVALID_PATTERN_STRUCTURE"}
        panel_edge_counts.append(int(edges.shape[0]))
        _, gaps = order_panel_edges(edges)
        boundary_closure_gaps.append(float(gaps.mean()))
    summary = {
        "valid": True,
        "failure": None,
        "keys": sorted(archive.files),
        "panel_count": panel_count,
        "curve_count": curve_count,
        "connectivity_shape": list(connectivity.shape),
        "connected_references": int(connectivity.sum()),
        "panel_edge_counts": panel_edge_counts,
        "mean_boundary_closure_gap": float(np.mean(boundary_closure_gaps)),
        "patch_bbox_min": np.min(arrays["patch_points"], axis=tuple(range(arrays["patch_points"].ndim - 1))).tolist(),
        "patch_bbox_max": np.max(arrays["patch_points"], axis=tuple(range(arrays["patch_points"].ndim - 1))).tolist(),
        "output_sha256": sha256(path),
    }
    return summary


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
