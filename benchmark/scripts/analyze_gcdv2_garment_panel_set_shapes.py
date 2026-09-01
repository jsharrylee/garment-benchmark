from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt

from benchmark.scripts.render_gcdv2_garment_panel_set_review import _predicted_points


def _cyclic_error(predicted: np.ndarray, target: np.ndarray, *, allow_reverse: bool) -> float | None:
    if len(predicted) != len(target):
        return None
    candidates = [predicted]
    if allow_reverse:
        candidates.append(predicted[::-1])
    return min(
        float(np.linalg.norm(np.roll(candidate, shift, axis=0) - target, axis=1).mean() * 1024.0)
        for candidate in candidates
        for shift in range(len(target))
    )


def _prediction_mask(panel: dict[str, Any], size: int) -> np.ndarray:
    boundary = []
    for edge_index in range(len(panel["predicted_vertices_uv"])):
        points = _predicted_points(panel, edge_index)
        boundary.extend((float(x) * size, float(y) * size) for x, y in points[:-1])
    image = Image.new("1", (size, size), 0)
    ImageDraw.Draw(image).polygon(boundary, fill=1)
    return np.asarray(image, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure shape-level metrics separately from ordered CAD graph metrics.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/test_predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/shape_metrics.json"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    index = {row["panel_uid"]: row for line in args.index.read_text(encoding="utf-8").splitlines() if line for row in [json.loads(line)]}
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    ious, chamfers, cyclic, cyclic_reverse = [], [], [], []
    by_category: dict[str, dict[str, list[float]]] = {}
    for garment in predictions:
        category = garment["target_category"]
        category_values = by_category.setdefault(category, {"iou": [], "chamfer": []})
        for panel in garment["panels"]:
            row = index[panel["panel_uid"]]
            with Image.open(row["panel_image_path"]) as image:
                truth = np.asarray(image.convert("L").resize((args.size, args.size), Image.Resampling.LANCZOS)) >= 128
            predicted_mask = _prediction_mask(panel, args.size)
            intersection = np.logical_and(truth, predicted_mask).sum()
            union = np.logical_or(truth, predicted_mask).sum()
            iou = float(intersection / max(union, 1))
            true_boundary = truth & ~binary_erosion(truth)
            predicted_boundary = predicted_mask & ~binary_erosion(predicted_mask)
            if true_boundary.any() and predicted_boundary.any():
                distance_to_truth = distance_transform_edt(~true_boundary)
                distance_to_prediction = distance_transform_edt(~predicted_boundary)
                chamfer = float((distance_to_truth[predicted_boundary].mean() + distance_to_prediction[true_boundary].mean()) * 0.5 * (1024.0 / args.size))
            else:
                chamfer = float(args.size)
            ious.append(iou)
            chamfers.append(chamfer)
            category_values["iou"].append(iou)
            category_values["chamfer"].append(chamfer)
            predicted_vertices = np.asarray(panel["predicted_vertices_uv"], np.float32)
            target_vertices = np.asarray(panel["target_vertices_uv"], np.float32)
            value = _cyclic_error(predicted_vertices, target_vertices, allow_reverse=False)
            if value is not None:
                cyclic.append(value)
            value = _cyclic_error(predicted_vertices, target_vertices, allow_reverse=True)
            if value is not None:
                cyclic_reverse.append(value)
    result: dict[str, Any] = {
        "status": "PASS",
        "panel_count": len(ious),
        "silhouette_iou_mean": float(np.mean(ious)),
        "silhouette_iou_median": float(np.median(ious)),
        "boundary_chamfer_px_at_1024_mean": float(np.mean(chamfers)),
        "boundary_chamfer_px_at_1024_median": float(np.median(chamfers)),
        "same_count_panel_count": len(cyclic),
        "cyclic_aligned_vertex_mae_px": float(np.mean(cyclic)),
        "cyclic_or_reversed_aligned_vertex_mae_px": float(np.mean(cyclic_reverse)),
        "by_category": {
            category: {
                "panel_count": len(values["iou"]),
                "silhouette_iou_mean": float(np.mean(values["iou"])),
                "boundary_chamfer_px_at_1024_mean": float(np.mean(values["chamfer"])),
            }
            for category, values in sorted(by_category.items())
        },
        "claim_boundary": "Shape metrics allow cyclic alignment or raster overlap and do not prove recovery of the source ordered CAD graph.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
