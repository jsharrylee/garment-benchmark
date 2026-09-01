from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .evaluation import compare_orthogonal_masks


CAMERA_TO_VIEW = {
    "CAM000": "front",
    "CAM001": "back",
    "CAM002": "left",
    "CAM003": "right",
}


def load_reference_masks(directory: Path) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for camera, view in CAMERA_TO_VIEW.items():
        path = directory / f"{camera}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            masks[view] = np.asarray(image.convert("L")) > 0
        if not masks[view].any():
            raise ValueError(f"empty reference mask: {path}")
    return masks


def _rasterize_projection(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rasterize a dense generated particle projection into the reference frame.

    The particle model predicts garment coordinates without the source camera
    extrinsics.  We therefore align only its aspect-preserving bounding box to
    the reference mask bbox.  This is a silhouette-shape proxy, not simulation.
    """
    height, width = reference.shape
    ys, xs = np.nonzero(reference)
    ref_left, ref_right = int(xs.min()), int(xs.max())
    ref_top, ref_bottom = int(ys.min()), int(ys.max())
    values = np.asarray(points, dtype=float)
    span = np.ptp(values, axis=0)
    if len(values) == 0 or np.any(span <= 1e-8) or not np.isfinite(values).all():
        return np.zeros_like(reference, dtype=bool)
    ref_width = max(1, ref_right - ref_left)
    ref_height = max(1, ref_bottom - ref_top)
    scale = min(ref_width / span[0], ref_height / span[1])
    normalized = (values - values.min(axis=0)) * scale
    offset_x = ref_left + (ref_width - span[0] * scale) * 0.5
    offset_y = ref_top + (ref_height - span[1] * scale) * 0.5
    pixels_x = np.clip(np.rint(normalized[:, 0] + offset_x), 0, width - 1).astype(int)
    # Model Y is up; image Y is down.
    pixels_y = np.clip(np.rint(ref_bottom - normalized[:, 1]), 0, height - 1).astype(int)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    radius = max(2, int(round(min(width, height) / 130)))
    for x, y in zip(pixels_x, pixels_y, strict=True):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    image = image.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return np.asarray(image) > 0


def particle_silhouette_proxy(prediction: Path, reference_masks: Mapping[str, np.ndarray]) -> dict:
    archive = np.load(prediction)
    particles = np.asarray(archive["particles"], dtype=float)[:, 2:5]
    projections = {
        "front": particles[:, [0, 1]],
        "back": particles[:, [0, 1]] * np.array([-1.0, 1.0]),
        "left": particles[:, [2, 1]],
        "right": particles[:, [2, 1]] * np.array([-1.0, 1.0]),
    }
    rendered = {
        view: _rasterize_projection(points, np.asarray(reference_masks[view]).astype(bool))
        for view, points in projections.items()
    }
    result = compare_orthogonal_masks(reference_masks, rendered)
    result.update(
        {
            "method": "aspect_aligned_generated_particle_projection",
            "is_simulation": False,
            "camera_extrinsics_used": False,
        }
    )
    return result


def final_validation(receipt: dict) -> dict:
    export = next(stage for stage in receipt["stages"] if stage["stage"] == "EXPORT")
    return export["validation"]


def candidate_rank(record: dict) -> tuple[float, ...]:
    validation = record["validation"]
    metrics = validation["metrics"]
    proxy = record.get("particle_silhouette_proxy", {})
    return (
        0.0 if validation["accepted"] else 1.0,
        float(metrics["error_count"]),
        -float(proxy.get("mean_iou", 0.0)),
        float(metrics["max_closure_gap_cm"]),
        float(metrics["mean_seam_length_mismatch"]),
    )


def select_generated_candidate(records: list[dict]) -> dict:
    if not records:
        raise ValueError("at least one generated candidate is required")
    selected = min(records, key=candidate_rank)
    return {
        "selected_candidate_id": selected["candidate_id"],
        "selected_rank": list(candidate_rank(selected)),
        "status": "STRUCTURALLY_VALID" if selected["validation"]["accepted"] else "DRAFT_REQUIRES_REPAIR",
        "manufacturing_ready": False,
        "method": "validation_guided_stochastic_regeneration",
        "variable_topology": True,
        "template_retrieval": False,
        "nearest_pattern_selection": False,
    }
