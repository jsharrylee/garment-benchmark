from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONTOUR_POINTS = 256
SEGMENT_POINTS = 32
PRIMITIVES = ("line", "quadratic_bezier", "cubic_bezier", "circular_arc")


def intrinsic_contour_features(points: np.ndarray) -> np.ndarray:
    """Rigid/scale invariant cyclic features; absolute x/y never leave here."""
    points = np.asarray(points, np.float32)
    steps = np.roll(points, -1, axis=0) - points
    perimeter = max(float(np.linalg.norm(steps, axis=1).sum()), 1e-8)
    features = []
    for offset in (1, 2, 4, 8, 16):
        before = points - np.roll(points, offset, axis=0)
        after = np.roll(points, -offset, axis=0) - points
        before_length = np.linalg.norm(before, axis=1)
        after_length = np.linalg.norm(after, axis=1)
        denominator = np.maximum(before_length * after_length, 1e-8)
        cosine = np.clip((before * after).sum(1) / denominator, -1.0, 1.0)
        sine = (before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]) / denominator
        across = np.linalg.norm(np.roll(points, -offset, axis=0) - np.roll(points, offset, axis=0), axis=1)
        features.extend((before_length / perimeter, after_length / perimeter, sine, cosine, across / perimeter))
    return np.stack(features, axis=1).astype(np.float32)


def nearest_contour_indices(contour: np.ndarray, uv: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(uv, np.float32)
    return np.asarray([int(np.square(contour - point).sum(1).argmin()) for point in values], np.int64)


def circular_corner_target(indices: Sequence[int], size: int = CONTOUR_POINTS, sigma: float = 1.25) -> np.ndarray:
    axis = np.arange(size)
    target = np.zeros(size, np.float32)
    for index in indices:
        distance = np.minimum((axis - int(index)) % size, (int(index) - axis) % size)
        target = np.maximum(target, np.exp(-0.5 * np.square(distance / sigma)))
    return target


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    delta = np.diff(points, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in (0, 1)], axis=1).astype(np.float32)


def segment_between(contour: np.ndarray, start: int, end: int) -> np.ndarray:
    if end <= start:
        indices = np.concatenate((np.arange(start, len(contour)), np.arange(0, end + 1)))
    else:
        indices = np.arange(start, end + 1)
    values = contour[indices]
    if len(values) < 3:
        values = np.vstack((values[0], (values[0] + values[-1]) / 2.0, values[-1]))
    return _resample_polyline(values, SEGMENT_POINTS)


def intrinsic_segment_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return chord-frame sequence features and invariant regression targets."""
    points = np.asarray(points, np.float64)
    chord = points[-1] - points[0]
    chord_length = max(float(np.linalg.norm(chord)), 1e-8)
    x_axis = chord / chord_length
    y_axis = np.asarray([-x_axis[1], x_axis[0]])
    relative = points - points[0]
    local = np.stack((relative @ x_axis, relative @ y_axis), axis=1) / chord_length
    delta = np.gradient(local, axis=0)
    tangent_norm = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    tangent = delta / tangent_norm
    turn = np.arctan2(tangent[:, 1], tangent[:, 0])
    curvature = np.gradient(np.unwrap(turn))
    step = np.linalg.norm(np.diff(local, axis=0), axis=1)
    step = np.concatenate((step, step[-1:]))
    features = np.column_stack((local, tangent, np.sin(turn), np.cos(turn), curvature, step)).astype(np.float32)

    t = np.linspace(0.0, 1.0, len(local))
    a = np.column_stack((3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2))
    fixed = ((1 - t) ** 3)[:, None] * local[0] + (t**3)[:, None] * local[-1]
    controls = np.linalg.lstsq(a, local - fixed, rcond=None)[0]
    arc_over_chord = float(np.linalg.norm(np.diff(local, axis=0), axis=1).sum())
    target = np.asarray(
        [
            controls[0, 0], controls[0, 1], controls[1, 0], controls[1, 1],
            arc_over_chord,
            tangent[0, 0], tangent[0, 1], tangent[-1, 0], tangent[-1, 1],
        ],
        np.float32,
    )
    return features, target


def merged_primitive(curves: Sequence[Mapping[str, Any]]) -> str:
    kinds = [str(curve["primitive"]) for curve in curves]
    if len(kinds) == 1:
        return kinds[0]
    if set(kinds) == {"line"}:
        return "line"
    if set(kinds) == {"circular_arc"}:
        return "circular_arc"
    # Several smooth source pieces are represented by one visible cubic path.
    return "cubic_bezier"


def _visible_adjacency_primitives(graph: Mapping[str, Any]) -> dict[frozenset[int], str]:
    visible = [index for index, point in enumerate(graph["points"]) if point["visual_supervision_eligible"]]
    point_count = len(graph["points"])
    output = {}
    for local, start in enumerate(visible):
        end = visible[(local + 1) % len(visible)]
        curve_indices = []
        current = start
        while current != end:
            curve_indices.append(current)
            current = (current + 1) % point_count
            if len(curve_indices) > point_count:
                raise ValueError("visible adjacency failed to close")
        output[frozenset((start, end))] = merged_primitive([graph["curves"][value] for value in curve_indices])
    return output


def _cyclic_source_direction(source_indices: Sequence[int], visible_indices: Sequence[int]) -> int:
    """Return the source-boundary direction followed by contour-ordered vertices."""
    positions = {value: index for index, value in enumerate(visible_indices)}
    size = len(visible_indices)
    forward = sum((positions[b] - positions[a]) % size for a, b in zip(source_indices, source_indices[1:] + source_indices[:1]))
    reverse = sum((positions[a] - positions[b]) % size for a, b in zip(source_indices, source_indices[1:] + source_indices[:1]))
    return 1 if forward <= reverse else -1


def _merged_primitive_between(graph: Mapping[str, Any], start: int, end: int, direction: int) -> str:
    curves = []
    current = start
    point_count = len(graph["points"])
    while current != end:
        curve_index = current if direction > 0 else (current - 1) % point_count
        curves.append(graph["curves"][curve_index])
        current = (current + direction) % point_count
        if len(curves) > point_count:
            raise ValueError("source path failed to close")
    return merged_primitive(curves)


def build_intrinsic_arrays(
    panel_rows: Sequence[Mapping[str, Any]], *, contours_override: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    contours, corners, counts, split_codes = [], [], [], []
    segment_features, segment_targets, segment_primitives, segment_splits, segment_panels = [], [], [], [], []
    split_lookup = {"train": 0, "validation": 1, "test": 2}
    for panel_index, row in enumerate(panel_rows):
        if contours_override is None:
            with np.load(row["visual_truth_path"]) as visual:
                contour = visual["dense_contour_uv_f32"].astype(np.float32)
        else:
            contour = np.asarray(contours_override[panel_index], np.float32)
        graph = json.loads(Path(row["formal_graph_path"]).read_text(encoding="utf-8"))
        visible_points = [point for point in graph["points"] if point["visual_supervision_eligible"]]
        visible_source_indices = [int(point["point_id"][1:]) for point in visible_points]
        indices = nearest_contour_indices(contour, [point["uv"] for point in visible_points])
        order = np.argsort(indices)
        indices = indices[order]
        visible_source_indices = [visible_source_indices[index] for index in order]
        # Rare raster collisions cannot support two separate visible vertices.
        keep = np.concatenate(([True], np.diff(indices) > 0))
        indices = indices[keep]
        visible_source_indices = [value for value, accepted in zip(visible_source_indices, keep, strict=True) if accepted]
        contours.append(contour)
        corners.append(circular_corner_target(indices))
        counts.append(len(indices))
        split_code = split_lookup[row["split"]]
        split_codes.append(split_code)
        visible_graph_indices = [index for index, point in enumerate(graph["points"]) if point["visual_supervision_eligible"]]
        source_direction = _cyclic_source_direction(visible_source_indices, visible_graph_indices)
        for local_index, start in enumerate(indices):
            end = int(indices[(local_index + 1) % len(indices)])
            start_source = visible_source_indices[local_index]
            end_source = visible_source_indices[(local_index + 1) % len(indices)]
            segment = segment_between(contour, int(start), end)
            features, target = intrinsic_segment_features(segment)
            segment_features.append(features)
            segment_targets.append(target)
            primitive = _merged_primitive_between(graph, start_source, end_source, source_direction)
            segment_primitives.append(PRIMITIVES.index(primitive))
            segment_splits.append(split_code)
            segment_panels.append(panel_index)
    return {
        "contours": np.asarray(contours, np.float32),
        "corner_targets": np.asarray(corners, np.float16),
        "corner_counts": np.asarray(counts, np.int16),
        "panel_splits": np.asarray(split_codes, np.int8),
        "segment_features": np.asarray(segment_features, np.float16),
        "segment_targets": np.asarray(segment_targets, np.float32),
        "segment_primitives": np.asarray(segment_primitives, np.int8),
        "segment_splits": np.asarray(segment_splits, np.int8),
        "segment_panels": np.asarray(segment_panels, np.int32),
    }


def build_corner_model(feature_dim: int = 25, width: int = 96, heads: int = 4, layers: int = 2, maximum_count: int = 36):
    import torch

    class IntrinsicCornerNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = torch.nn.Linear(feature_dim, width)
            self.local = torch.nn.Sequential(
                torch.nn.Conv1d(width, width, 5, padding=2, padding_mode="circular"), torch.nn.GELU(),
                torch.nn.Conv1d(width, width, 5, padding=2, padding_mode="circular"), torch.nn.GELU(),
            )
            layer = torch.nn.TransformerEncoderLayer(width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.encoder = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.corner_head = torch.nn.Linear(width, 1)
            self.count_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, maximum_count + 1))

        def forward(self, features):
            hidden = self.project(features)
            hidden = hidden + self.local(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = self.encoder(hidden)
            return {"corner_logits": self.corner_head(hidden).squeeze(-1), "count_logits": self.count_head(hidden.mean(1))}

    return IntrinsicCornerNet()


def build_segment_model(feature_dim: int = 8, width: int = 96, heads: int = 4, layers: int = 2):
    import torch

    class IntrinsicSegmentNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = torch.nn.Linear(feature_dim, width)
            layer = torch.nn.TransformerEncoderLayer(width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.encoder = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.primitive = torch.nn.Linear(width, len(PRIMITIVES))
            self.geometry = torch.nn.Linear(width, 9)

        def forward(self, features):
            hidden = self.encoder(self.project(features)).mean(1)
            geometry = self.geometry(hidden)
            # Direction vectors are always normalized before loss/evaluation.
            geometry = torch.cat((geometry[:, :5], torch.nn.functional.normalize(geometry[:, 5:7], dim=-1), torch.nn.functional.normalize(geometry[:, 7:9], dim=-1)), 1)
            return {"primitive_logits": self.primitive(hidden), "geometry": geometry}

    return IntrinsicSegmentNet()


def select_cyclic_peaks(probability: np.ndarray, count: int, radius: int = 3) -> list[int]:
    values = np.asarray(probability).copy()
    selected = []
    for _ in range(max(0, int(count))):
        index = int(values.argmax())
        selected.append(index)
        for delta in range(-radius, radius + 1):
            values[(index + delta) % len(values)] = -np.inf
    return sorted(selected)


__all__ = [
    "CONTOUR_POINTS", "PRIMITIVES", "SEGMENT_POINTS", "build_corner_model", "build_intrinsic_arrays",
    "build_segment_model", "intrinsic_contour_features", "intrinsic_segment_features", "select_cyclic_peaks",
]
