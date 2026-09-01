from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from scipy.optimize import linear_sum_assignment

from benchmark.gcdv2_exact.geometry import CURVE_COLORS, _render_label
from benchmark.gcdv2_exact.pattern_learning import (
    CATEGORIES,
    IMAGE_SIZE,
    MAXIMUM_EDGES,
    MAXIMUM_PANELS,
    PRIMITIVE_TYPES,
    PatternExample,
    build_pattern_parser_model,
    hungarian_matches,
    padded_pattern_batch,
    read_pattern_examples,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _tensor_batch(raw: Mapping[str, Any], device):
    import torch

    return {
        key: torch.from_numpy(value).to(device)
        for key, value in raw.items()
        if isinstance(value, np.ndarray)
    }


def _infer(model, examples: Sequence[PatternExample], *, batch_size: int, device):
    import torch

    results: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            current = examples[offset : offset + batch_size]
            raw = padded_pattern_batch(current)
            batch = _tensor_batch(raw, device)
            output = model(batch["spatial_features"])
            edge_matches, panel_matches = hungarian_matches(output, batch)
            for row, example in enumerate(current):
                edge_query, edge_target = edge_matches[row]
                panel_query, panel_target = panel_matches[row]
                results[example.sample_id] = {
                    "category_logits": output["category_logits"][row].float().cpu().numpy(),
                    "edge_presence": output["edge_presence_logits"][row]
                    .sigmoid()
                    .float()
                    .cpu()
                    .numpy(),
                    "edge_types": output["edge_type_logits"][row]
                    .float()
                    .cpu()
                    .numpy(),
                    "edge_geometry": output["edge_geometry"][row].float().cpu().numpy(),
                    "panel_presence": output["panel_presence_logits"][row]
                    .sigmoid()
                    .float()
                    .cpu()
                    .numpy(),
                    "panel_boxes": output["panel_boxes"][row].float().cpu().numpy(),
                    # Matches are retained solely for evaluation annotations.
                    # They never decide which prediction is rendered.
                    "edge_match_query": edge_query.cpu().numpy(),
                    "edge_match_target": edge_target.cpu().numpy(),
                    "panel_match_query": panel_query.cpu().numpy(),
                    "panel_match_target": panel_target.cpu().numpy(),
                }
    return results


def _binary_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    expected = targets.astype(bool)
    tp = int(np.sum(predicted & expected))
    fp = int(np.sum(predicted & ~expected))
    fn = int(np.sum(~predicted & expected))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_positive": int(np.sum(predicted)),
        "truth_positive": int(np.sum(expected)),
    }


def _calibrate(scores: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.linspace(0.05, 0.95, 181)
    rows = [_binary_metrics(scores, targets, float(value)) for value in candidates]
    best = max(rows, key=lambda row: (row["f1"], row["precision"], -abs(row["threshold"] - 0.5)))
    return float(best["threshold"]), {
        "selection_rule": "maximum F1 on validation queries only; ties prefer precision then proximity to 0.5",
        "candidate_min": 0.05,
        "candidate_max": 0.95,
        "candidate_step": 0.005,
        "selected": best,
        "at_default_0_5": _binary_metrics(scores, targets, 0.5),
    }


def _presence_arrays(
    examples: Sequence[PatternExample],
    inference: Mapping[str, Mapping[str, Any]],
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    scores, targets = [], []
    maximum = MAXIMUM_EDGES if kind == "edge" else MAXIMUM_PANELS
    for example in examples:
        result = inference[example.sample_id]
        current = np.zeros(maximum, dtype=bool)
        current[result[f"{kind}_match_query"]] = True
        scores.append(result[f"{kind}_presence"])
        targets.append(current)
    return np.concatenate(scores), np.concatenate(targets)


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box, *, fill, width=2, dash=12):
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    for start in range(x0, x1, 2 * dash):
        draw.line((start, y0, min(start + dash, x1), y0), fill=fill, width=width)
        draw.line((start, y1, min(start + dash, x1), y1), fill=fill, width=width)
    for start in range(y0, y1, 2 * dash):
        draw.line((x0, start, x0, min(start + dash, y1)), fill=fill, width=width)
        draw.line((x1, start, x1, min(start + dash, y1)), fill=fill, width=width)


def _detected_edges(inference: Mapping[str, Any], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for query, score in enumerate(inference["edge_presence"]):
        if float(score) < threshold:
            continue
        geometry = np.asarray(inference["edge_geometry"][query], dtype=float)
        primitive_index = int(np.argmax(inference["edge_types"][query]))
        rows.append(
            {
                "query_index": query,
                "presence_probability": float(score),
                "primitive_type": PRIMITIVE_TYPES[primitive_index],
                "packed_start_uv": geometry[0:2].tolist(),
                "packed_end_uv": geometry[2:4].tolist(),
                "rendered_length_canvas_fraction": float(geometry[4]),
                "direction_sin_cos_image_xy": geometry[5:7].tolist(),
            }
        )
    return rows


def _cluster_endpoints(
    edges: Sequence[Mapping[str, Any]], merge_radius_uv: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge independently predicted endpoints without consulting any GT.

    Connected components are used instead of greedy insertion so the result is
    invariant to query serialization.  Coordinates are confidence-weighted
    centroids; final IDs are sorted geometrically for deterministic output.
    """

    if not edges:
        return [], []
    points = []
    weights = []
    for edge in edges:
        points.extend((edge["packed_start_uv"], edge["packed_end_uv"]))
        weights.extend((edge["presence_probability"], edge["presence_probability"]))
    points_array = np.asarray(points, dtype=np.float64)
    parent = list(range(len(points_array)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    distances = np.linalg.norm(points_array[:, None] - points_array[None, :], axis=-1)
    for left, right in zip(*np.where(np.triu(distances <= merge_radius_uv, k=1))):
        union(int(left), int(right))
    groups: dict[int, list[int]] = {}
    for index in range(len(points_array)):
        groups.setdefault(find(index), []).append(index)
    clusters = []
    for members in groups.values():
        member_weights = np.asarray([weights[index] for index in members], dtype=np.float64)
        coordinate = np.average(points_array[members], axis=0, weights=member_weights)
        clusters.append(
            {
                "members": members,
                "packed_uv": coordinate.tolist(),
                "supporting_endpoint_count": len(members),
                "mean_presence_probability": float(member_weights.mean()),
            }
        )
    clusters.sort(key=lambda row: (round(row["packed_uv"][1], 8), round(row["packed_uv"][0], 8)))
    endpoint_to_id = {}
    predicted_points = []
    for point_id, cluster in enumerate(clusters):
        for endpoint in cluster.pop("members"):
            endpoint_to_id[endpoint] = point_id
        predicted_points.append({"point_id": point_id, **cluster})
    output_edges = []
    for edge_index, edge in enumerate(edges):
        output_edges.append(
            {
                **dict(edge),
                "start_point_id": endpoint_to_id[2 * edge_index],
                "end_point_id": endpoint_to_id[2 * edge_index + 1],
            }
        )
    return predicted_points, output_edges


def _predicted_panels(inference: Mapping[str, Any], threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "panel_query_id": index,
            "presence_probability": float(score),
            "packed_bbox_uv": np.asarray(bbox, dtype=float).tolist(),
        }
        for index, (score, bbox) in enumerate(
            zip(inference["panel_presence"], inference["panel_boxes"])
        )
        if float(score) >= threshold
    ]


def _assign_predicted_panels(
    edges: Sequence[Mapping[str, Any]], panels: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for edge in edges:
        midpoint = 0.5 * (
            np.asarray(edge["packed_start_uv"], dtype=float)
            + np.asarray(edge["packed_end_uv"], dtype=float)
        )
        candidates = []
        for panel in panels:
            x0, y0, x1, y1 = panel["packed_bbox_uv"]
            if x0 <= midpoint[0] <= x1 and y0 <= midpoint[1] <= y1:
                candidates.append(((x1 - x0) * (y1 - y0), panel["panel_query_id"]))
        panel_id = min(candidates)[1] if candidates else None
        output.append(
            {
                **dict(edge),
                "predicted_panel_query_id": panel_id,
                "panel_assignment_method": (
                    "smallest_predicted_bbox_containing_edge_midpoint" if panel_id is not None else "unassigned"
                ),
            }
        )
    return output


def _truth_graph(example: PatternExample):
    points = []
    keys: dict[tuple[float, float], int] = {}
    edges = []
    for edge_index, geometry in enumerate(example.edge_geometry):
        endpoint_ids = []
        for coordinate in (geometry[0:2], geometry[2:4]):
            key = (round(float(coordinate[0]), 7), round(float(coordinate[1]), 7))
            if key not in keys:
                keys[key] = len(points)
                points.append(np.asarray(coordinate, dtype=np.float64))
            endpoint_ids.append(keys[key])
        edges.append(
            {
                "start_point_id": endpoint_ids[0],
                "end_point_id": endpoint_ids[1],
                "primitive_type": PRIMITIVE_TYPES[int(example.edge_types[edge_index])],
            }
        )
    return np.asarray(points, dtype=np.float64), edges


def _point_graph_metrics(
    example: PatternExample,
    predicted_points: Sequence[Mapping[str, Any]],
    predicted_edges: Sequence[Mapping[str, Any]],
    *,
    tolerance_uv: float,
) -> dict[str, Any]:
    truth_points, truth_edges = _truth_graph(example)
    points = np.asarray([row["packed_uv"] for row in predicted_points], dtype=np.float64).reshape((-1, 2))
    if len(points) and len(truth_points):
        distances = np.linalg.norm(points[:, None] - truth_points[None, :], axis=-1)
        predicted_index, truth_index = linear_sum_assignment(distances)
        accepted = distances[predicted_index, truth_index] <= tolerance_uv
        accepted_predicted = predicted_index[accepted]
        accepted_truth = truth_index[accepted]
        mapping = {int(left): int(right) for left, right in zip(accepted_predicted, accepted_truth)}
        coordinate_errors = distances[accepted_predicted, accepted_truth]
    else:
        mapping = {}
        coordinate_errors = np.asarray([], dtype=float)
    point_tp = len(mapping)
    point_fp = len(points) - point_tp
    point_fn = len(truth_points) - point_tp
    precision = point_tp / max(point_tp + point_fp, 1)
    recall = point_tp / max(point_tp + point_fn, 1)
    point_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    truth_pairs = {
        tuple(sorted((int(edge["start_point_id"]), int(edge["end_point_id"]))))
        for edge in truth_edges
    }
    predicted_pairs = []
    correct_pairs = set()
    for edge in predicted_edges:
        left = mapping.get(int(edge["start_point_id"]))
        right = mapping.get(int(edge["end_point_id"]))
        pair = tuple(sorted((left, right))) if left is not None and right is not None else None
        predicted_pairs.append(pair)
        if pair in truth_pairs:
            correct_pairs.add(pair)
    graph_tp = len(correct_pairs)
    graph_fp = len(predicted_edges) - graph_tp
    graph_fn = len(truth_pairs) - graph_tp
    graph_precision = graph_tp / max(graph_tp + graph_fp, 1)
    graph_recall = graph_tp / max(graph_tp + graph_fn, 1)
    return {
        "point_match_tolerance_px": tolerance_uv * IMAGE_SIZE,
        "predicted_point_count": len(points),
        "truth_point_count": len(truth_points),
        "point_true_positive": point_tp,
        "point_false_positive": point_fp,
        "point_false_negative": point_fn,
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": point_f1,
        "matched_point_coordinate_mae_px": (
            float(coordinate_errors.mean() * IMAGE_SIZE) if len(coordinate_errors) else None
        ),
        "predicted_edge_count": len(predicted_edges),
        "truth_edge_count": len(truth_pairs),
        "graph_edge_true_positive": graph_tp,
        "graph_edge_false_positive": graph_fp,
        "graph_edge_false_negative": graph_fn,
        "graph_edge_precision": graph_precision,
        "graph_edge_recall": graph_recall,
        "graph_edge_f1": 2 * graph_precision * graph_recall / max(graph_precision + graph_recall, 1e-12),
    }


def _aggregate_graph_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    point_tp = sum(row["point_true_positive"] for row in rows)
    point_fp = sum(row["point_false_positive"] for row in rows)
    point_fn = sum(row["point_false_negative"] for row in rows)
    graph_tp = sum(row["graph_edge_true_positive"] for row in rows)
    graph_fp = sum(row["graph_edge_false_positive"] for row in rows)
    graph_fn = sum(row["graph_edge_false_negative"] for row in rows)
    point_precision = point_tp / max(point_tp + point_fp, 1)
    point_recall = point_tp / max(point_tp + point_fn, 1)
    graph_precision = graph_tp / max(graph_tp + graph_fp, 1)
    graph_recall = graph_tp / max(graph_tp + graph_fn, 1)
    coordinate = [row["matched_point_coordinate_mae_px"] for row in rows if row["matched_point_coordinate_mae_px"] is not None]
    return {
        "sample_count": len(rows),
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": 2 * point_precision * point_recall / max(point_precision + point_recall, 1e-12),
        "mean_per_sample_matched_point_coordinate_mae_px": float(np.mean(coordinate)) if coordinate else None,
        "graph_edge_precision": graph_precision,
        "graph_edge_recall": graph_recall,
        "graph_edge_f1": 2 * graph_precision * graph_recall / max(graph_precision + graph_recall, 1e-12),
        "point_true_positive": point_tp,
        "point_false_positive": point_fp,
        "point_false_negative": point_fn,
        "graph_edge_true_positive": graph_tp,
        "graph_edge_false_positive": graph_fp,
        "graph_edge_false_negative": graph_fn,
    }


def _calibrate_point_merge(
    examples: Sequence[PatternExample],
    inference: Mapping[str, Mapping[str, Any]],
    *,
    edge_threshold: float,
    point_tolerance_uv: float,
) -> tuple[float, dict[str, Any]]:
    candidates_px = (4, 8, 12, 16, 24, 32, 48, 64)
    trials = []
    for pixels in candidates_px:
        radius = pixels / IMAGE_SIZE
        rows = []
        for example in examples:
            detected = _detected_edges(inference[example.sample_id], edge_threshold)
            points, graph_edges = _cluster_endpoints(detected, radius)
            rows.append(
                _point_graph_metrics(
                    example, points, graph_edges, tolerance_uv=point_tolerance_uv
                )
            )
        aggregate = _aggregate_graph_metrics(rows)
        trials.append({"merge_radius_px": pixels, **aggregate})
    best = max(
        trials,
        key=lambda row: (
            row["point_f1"],
            row["graph_edge_f1"],
            -row["mean_per_sample_matched_point_coordinate_mae_px"],
            -row["merge_radius_px"],
        ),
    )
    return float(best["merge_radius_px"] / IMAGE_SIZE), {
        "selection_rule": "maximum validation point F1, then graph-edge F1, coordinate error, smaller radius",
        "point_match_tolerance_px_fixed_before_test": point_tolerance_uv * IMAGE_SIZE,
        "selected": best,
        "trials": trials,
    }


def _prediction_overlay(
    example: PatternExample,
    inference: Mapping[str, Any],
    *,
    edge_threshold: float,
    panel_threshold: float,
    point_merge_radius_uv: float,
    destination: Path,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = Image.open(example.pattern_path).convert("RGB")
    base = ImageEnhance.Brightness(base).enhance(0.62).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    panels = _predicted_panels(inference, panel_threshold)
    for panel in panels:
        index = panel["panel_query_id"]
        score = panel["presence_probability"]
        bbox = panel["packed_bbox_uv"]
        _dashed_rectangle(
            draw,
            np.asarray(bbox) * IMAGE_SIZE,
            fill=(238, 238, 238, 190),
            width=2,
        )
        x0, y0 = (np.asarray(bbox[:2]) * IMAGE_SIZE).tolist()
        draw.text(
            (x0 + 3, y0 + 3),
            f"Pq{index} {score:.2f}",
            font=_font(13, bold=True),
            fill=(255, 255, 255, 230),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
    detected = _detected_edges(inference, edge_threshold)
    predicted_points, detected = _cluster_endpoints(detected, point_merge_radius_uv)
    detected = _assign_predicted_panels(detected, panels)
    for edge in detected:
        query = edge["query_index"]
        score = edge["presence_probability"]
        primitive = edge["primitive_type"]
        color_hex = CURVE_COLORS[primitive]
        color = tuple(int(color_hex[index : index + 2], 16) for index in (1, 3, 5))
        start = tuple((np.asarray(edge["packed_start_uv"]) * IMAGE_SIZE).tolist())
        end = tuple((np.asarray(edge["packed_end_uv"]) * IMAGE_SIZE).tolist())
        # The Stage-1 head predicts endpoints and primitive class, not Bezier
        # controls/radius.  Non-line predictions are therefore shown as their
        # predicted chord, never as a fabricated curve.
        draw.line((start, end), fill=(*color, 235), width=5)
        radius = 5
        draw.ellipse(
            (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
            fill=(255, 255, 255, 245),
            outline=(0, 0, 0, 255),
            width=2,
        )
        draw.ellipse(
            (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
            fill=(255, 226, 89, 245),
            outline=(0, 0, 0, 255),
            width=2,
        )
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        draw.text(
            (midpoint[0] + 4, midpoint[1] - 8),
            f"q{query} {primitive.split('_')[0]} {score:.2f}",
            font=_font(11, bold=True),
            fill=(*color, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 230),
        )
    for point in predicted_points:
        x, y = (np.asarray(point["packed_uv"]) * IMAGE_SIZE).tolist()
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(36, 210, 255, 245), outline=(0, 0, 0, 255), width=2)
        draw.text(
            (x + 7, y + 5),
            f"p{point['point_id']}",
            font=_font(11, bold=True),
            fill=(210, 247, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 230),
        )
    output = Image.alpha_composite(base, layer).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination, quality=95)
    return destination, predicted_points, detected, panels


def _sample_evaluation(example: PatternExample, inference: Mapping[str, Any], threshold: float):
    matched_queries = np.asarray(inference["edge_match_query"], dtype=int)
    matched_targets = np.asarray(inference["edge_match_target"], dtype=int)
    prediction = inference["edge_geometry"][matched_queries]
    expected = example.edge_geometry[matched_targets]
    detected = inference["edge_presence"][matched_queries] >= threshold
    primitive_prediction = np.argmax(inference["edge_types"][matched_queries], axis=-1)
    primitive_expected = example.edge_types[matched_targets]
    cosine = np.clip(np.sum(prediction[:, 5:7] * expected[:, 5:7], axis=-1), -1.0, 1.0)
    return {
        "annotation_contract": "evaluation only; Hungarian matches do not affect rendered prediction count or geometry",
        "matched_target_count": int(len(matched_targets)),
        "matched_detected_count": int(np.sum(detected)),
        "matched_endpoint_mae_px": float(np.mean(np.abs(prediction[:, :4] - expected[:, :4])) * IMAGE_SIZE),
        "matched_length_mae_px": float(np.mean(np.abs(prediction[:, 4] - expected[:, 4])) * IMAGE_SIZE),
        "matched_direction_mean_angular_error_deg": float(np.degrees(np.arccos(cosine)).mean()),
        "matched_primitive_accuracy": float(np.mean(primitive_prediction == primitive_expected)),
    }


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    output = Image.new("RGB", size, "#111318")
    output.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return output


def _board(
    example: PatternExample,
    inference: Mapping[str, Any],
    truth_overlay: Path,
    prediction_overlay: Path,
    predicted_points: Sequence[Mapping[str, Any]],
    detected: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    *,
    edge_threshold: float,
    panel_threshold: float,
    destination: Path,
) -> Path:
    width, height = 2860, 1280
    board = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(board)
    predicted_category = CATEGORIES[int(np.argmax(inference["category_logits"]))]
    draw.text(
        (50, 26),
        f"Stage 1 image-only parser review · {example.category.upper()} · {example.sample_id}",
        font=_font(34, bold=True),
        fill="#17191d",
    )
    draw.text(
        (50, 78),
        f"category truth={example.category} / predicted={predicted_category}   |   edge threshold={edge_threshold:.3f}, panel threshold={panel_threshold:.3f} (validation-only calibration)",
        font=_font(21),
        fill="#3b4048",
    )
    titles = (
        "A. CLEAN PATTERN INPUT",
        "B. EXACT TRUTH OVERLAY",
        "C. INDEPENDENT MODEL OUTPUT",
    )
    images = (
        Image.open(example.pattern_path).convert("RGB"),
        Image.open(truth_overlay).convert("RGB"),
        Image.open(prediction_overlay).convert("RGB"),
    )
    cell_w, cell_h, gap = 880, 880, 50
    for index, (title, image) in enumerate(zip(titles, images)):
        x = 50 + index * (cell_w + gap)
        draw.rounded_rectangle((x, 128, x + cell_w, 1062), radius=18, fill="#ffffff", outline="#cac5bb", width=2)
        draw.text((x + 18, 145), title, font=_font(22, bold=True), fill="#16191d")
        board.paste(_fit(image, (832, 832)), (x + 24, 205))
    truth_edge_count = len(example.edge_geometry)
    truth_panel_count = len(example.panel_boxes)
    draw.text(
        (56, 1090),
        f"Truth sidecar: panels={truth_panel_count}, edges={truth_edge_count}.  Exact native curve geometry remains in labels.json.",
        font=_font(19),
        fill="#30343b",
    )
    draw.text(
        (980, 1090),
        f"Independent detections: panels={(inference['panel_presence'] >= panel_threshold).sum()}, points={len(predicted_points)}, edges={len(detected)}.  No GT count/top-k/matching used.",
        font=_font(19, bold=True),
        fill="#712b6f",
    )
    draw.text(
        (56, 1132),
        "Prediction colors: cyan=line · orange=quadratic · pink=cubic · green=arc. Blue p# = merged predicted point ID. Non-lines are endpoint chords because controls/radius are not Stage-1 outputs.",
        font=_font(17),
        fill="#30343b",
    )
    draw.text(
        (56, 1172),
        f"Evaluation-only match annotation: endpoint MAE={evaluation['matched_endpoint_mae_px']:.1f}px, length MAE={evaluation['matched_length_mae_px']:.1f}px, primitive acc={evaluation['matched_primitive_accuracy']:.3f}, direction error={evaluation['matched_direction_mean_angular_error_deg']:.1f}°.",
        font=_font(17),
        fill="#5a5f67",
    )
    draw.text(
        (56, 1210),
        "Important: packed UV is display geometry. Panel ownership is only an independent predicted-bbox containment heuristic; it is not a learned edge-panel relation.",
        font=_font(17, bold=True),
        fill="#9a3b2f",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, quality=94)
    return destination


def _contact_sheet(boards: Sequence[Path], destination: Path) -> Path:
    columns = 2
    thumb_w, thumb_h = 1420, 636
    rows = (len(boards) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "#ddd9d0")
    for index, path in enumerate(boards):
        image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render leakage-safe Stage-1 parser review boards.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/pattern_fpn_tokens.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_exact/pattern_set_parser_hungarian.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/pattern_parser_hungarian_review"))
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_pattern_parser_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    examples = read_pattern_examples(args.index, feature_path=args.features)
    assignments = checkpoint["split_assignments"]
    validation = tuple(item for item in examples if assignments[item.sample_id] == "validation")
    test = tuple(item for item in examples if assignments[item.sample_id] == "test")
    if not validation or not test:
        raise RuntimeError("checkpoint must contain non-empty validation and test splits")
    validation_inference = _infer(model, validation, batch_size=args.batch_size, device=device)
    edge_scores, edge_targets = _presence_arrays(validation, validation_inference, kind="edge")
    panel_scores, panel_targets = _presence_arrays(validation, validation_inference, kind="panel")
    edge_threshold, edge_calibration = _calibrate(edge_scores, edge_targets)
    panel_threshold, panel_calibration = _calibrate(panel_scores, panel_targets)
    point_match_tolerance_uv = 32.0 / IMAGE_SIZE
    point_merge_radius_uv, point_merge_calibration = _calibrate_point_merge(
        validation,
        validation_inference,
        edge_threshold=edge_threshold,
        point_tolerance_uv=point_match_tolerance_uv,
    )

    calibration = {
        "schema_version": "gcdv2-pattern-parser-presence-calibration-1.0",
        "status": "PASS",
        "leakage_contract": {
            "threshold_source": "validation split only",
            "test_labels_used_for_threshold": False,
            "ground_truth_count_used_at_inference": False,
            "top_k_from_ground_truth_count_used_at_inference": False,
            "hungarian_use": "validation presence-target construction and evaluation annotations only",
        },
        "checkpoint": str(args.checkpoint.as_posix()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "validation_sample_count": len(validation),
        "edge_presence": edge_calibration,
        "panel_presence": panel_calibration,
        "predicted_point_merge": point_merge_calibration,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output / "presence_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    test_inference = _infer(model, test, batch_size=args.batch_size, device=device)
    randomizer = random.Random(args.seed)
    selected = []
    for category, count in (("top", 4), ("skirt", 3), ("pants", 3)):
        pool = sorted((item for item in test if item.category == category), key=lambda item: item.sample_id)
        selected.extend(randomizer.sample(pool, count))
    randomizer.shuffle(selected)
    boards, sample_rows = [], []
    for index, example in enumerate(selected, start=1):
        inference = test_inference[example.sample_id]
        prefix = f"{index:02d}_{example.category}_{example.sample_id}"
        truth_path = args.output / f"{prefix}_truth_overlay.png"
        prediction_path = args.output / f"{prefix}_prediction_overlay.png"
        board_path = args.output / f"{prefix}_board.png"
        label = json.loads(example.label_path.read_text(encoding="utf-8"))
        _render_label(label, truth_path, overlay=True, size=IMAGE_SIZE)
        _, predicted_points, detected, predicted_panels = _prediction_overlay(
            example,
            inference,
            edge_threshold=edge_threshold,
            panel_threshold=panel_threshold,
            point_merge_radius_uv=point_merge_radius_uv,
            destination=prediction_path,
        )
        evaluation = _sample_evaluation(example, inference, edge_threshold)
        graph_evaluation = _point_graph_metrics(
            example,
            predicted_points,
            detected,
            tolerance_uv=point_match_tolerance_uv,
        )
        _board(
            example,
            inference,
            truth_path,
            prediction_path,
            predicted_points,
            detected,
            evaluation,
            edge_threshold=edge_threshold,
            panel_threshold=panel_threshold,
            destination=board_path,
        )
        boards.append(board_path)
        sample_rows.append(
            {
                "sample_id": example.sample_id,
                "category_truth": example.category,
                "category_predicted": CATEGORIES[int(np.argmax(inference["category_logits"]))],
                "clean_pattern": str(example.pattern_path.as_posix()),
                "truth_overlay": str(truth_path.as_posix()),
                "prediction_overlay": str(prediction_path.as_posix()),
                "board": str(board_path.as_posix()),
                "truth_panel_count": len(example.panel_boxes),
                "independent_predicted_panel_count": int(np.sum(inference["panel_presence"] >= panel_threshold)),
                "truth_edge_count": len(example.edge_geometry),
                "independent_predicted_edge_count": len(detected),
                "independent_predicted_point_count": len(predicted_points),
                "predicted_points": predicted_points,
                "predicted_panels": predicted_panels,
                "detected_edges": detected,
                "evaluation_only": evaluation,
                "point_graph_evaluation_only": graph_evaluation,
            }
        )
    contact_path = _contact_sheet(boards, args.output / "contact_sheet_10.png")

    test_edge_scores, test_edge_targets = _presence_arrays(test, test_inference, kind="edge")
    test_panel_scores, test_panel_targets = _presence_arrays(test, test_inference, kind="panel")
    full_test_graph_rows = []
    for example in test:
        detected = _detected_edges(test_inference[example.sample_id], edge_threshold)
        points, graph_edges = _cluster_endpoints(detected, point_merge_radius_uv)
        full_test_graph_rows.append(
            _point_graph_metrics(
                example,
                points,
                graph_edges,
                tolerance_uv=point_match_tolerance_uv,
            )
        )
    summary = {
        "schema_version": "gcdv2-pattern-parser-visual-review-1.0",
        "status": "PASS",
        "selection": {
            "method": "deterministic random category-stratified test selection",
            "seed": args.seed,
            "counts": {"top": 4, "skirt": 3, "pants": 3},
            "selection_used_model_quality": False,
        },
        "inference_contract": calibration["leakage_contract"],
        "thresholds": {
            "edge_presence": edge_threshold,
            "panel_presence": panel_threshold,
            "endpoint_merge_radius_uv": point_merge_radius_uv,
            "endpoint_merge_radius_px_at_1024": point_merge_radius_uv * IMAGE_SIZE,
            "point_evaluation_tolerance_px": point_match_tolerance_uv * IMAGE_SIZE,
        },
        "test_detection_metrics_evaluation_only": {
            "edge_presence": _binary_metrics(test_edge_scores, test_edge_targets, edge_threshold),
            "panel_presence": _binary_metrics(test_panel_scores, test_panel_targets, panel_threshold),
            "predicted_point_and_graph": _aggregate_graph_metrics(full_test_graph_rows),
        },
        "limitations": [
            "Stage 1 predicts independent edge endpoints; unique point IDs are deterministic distance-cluster postprocessing, not a learned incidence head.",
            "Edge-to-panel ownership is not learned. Review JSON uses predicted-bbox containment when possible and otherwise leaves the edge unassigned.",
            "Bezier controls and circular-arc radius/sweep are not outputs, so prediction overlays draw endpoint chords for all primitive classes.",
            "Packed UV is display geometry; the image-only model does not infer absolute centimetres.",
        ],
        "sample_count": len(sample_rows),
        "category_counts": dict(sorted(Counter(row["category_truth"] for row in sample_rows).items())),
        "samples": sample_rows,
        "contact_sheet": str(contact_path.as_posix()),
        "calibration": str(calibration_path.as_posix()),
    }
    summary_path = args.output / "review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "edge_threshold": edge_threshold,
                "panel_threshold": panel_threshold,
                "point_merge_radius_px": point_merge_radius_uv * IMAGE_SIZE,
                "contact_sheet": str(contact_path.as_posix()),
                "calibration": str(calibration_path.as_posix()),
                "summary": str(summary_path.as_posix()),
                "sample_ids": [row["sample_id"] for row in sample_rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
