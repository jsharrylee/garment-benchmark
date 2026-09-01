"""Spatial four-view prediction of drafting-style curve formula parameters.

This module is the local-detail successor to the global ResNet descriptor
baseline.  It deliberately predicts a small, auditable contract before any
full CAD decoder is attempted:

* named front/back neckline and armhole paths plus a sleeve-head path;
* endpoints in a canonical panel frame (landmark-like values);
* normalized chord and arc length; and
* two connected cubic Bezier segments in a chord-local coordinate frame.

The two cubics remain children of one semantic path.  In particular, an
armhole serialized as two source primitives is *not* presented as two
armholes.  Missing roles are masked and are also predicted by a separate
presence head; target presence is never supplied to the visual encoder.

The image model consumes frozen, multi-scale ResNet-50-FPN feature maps.  The
paper/model checkpoint is loaded only from an explicit local path: none of the
helpers in this file can trigger a network download.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .multiview_pattern_semantics import VIEW_NAMES, _split_lookup
from .schema import DraftingSemanticRecord, PanelAnnotation
from .semantic_paths import merge_predicted_semantic_paths


CURVE_QUERY_NAMES = (
    "front_neckline",
    "back_neckline",
    "front_armhole",
    "back_armhole",
    "sleeve_head",
)
LANDMARK_PARAMETER_NAMES = (
    "start_u_in_panel",
    "start_v_in_panel",
    "end_u_in_panel",
    "end_v_in_panel",
)
METRIC_PARAMETER_NAMES = ("chord_over_garment", "arc_over_garment")
CONTROL_PARAMETER_NAMES = (
    "knot_x_over_chord",
    "knot_y_over_chord",
    "segment_0_control_1_x",
    "segment_0_control_1_y",
    "segment_0_control_2_x",
    "segment_0_control_2_y",
    "segment_1_control_1_x",
    "segment_1_control_1_y",
    "segment_1_control_2_x",
    "segment_1_control_2_y",
)
CURVE_PARAMETER_NAMES = (
    *LANDMARK_PARAMETER_NAMES,
    *METRIC_PARAMETER_NAMES,
    *CONTROL_PARAMETER_NAMES,
)
CURVE_TRUTH_GENERATOR_FORMULA = "GENERATOR_FORMULA_CAPTURE"
CURVE_TRUTH_DENSE_APPROXIMATION = "DENSE_CURVE_TWO_CUBIC_APPROXIMATION"
LANDMARK_SLICE = slice(0, len(LANDMARK_PARAMETER_NAMES))
METRIC_SLICE = slice(LANDMARK_SLICE.stop, LANDMARK_SLICE.stop + len(METRIC_PARAMETER_NAMES))
CONTROL_SLICE = slice(METRIC_SLICE.stop, len(CURVE_PARAMETER_NAMES))


@dataclass(frozen=True)
class CurveFormulaTargets:
    values: np.ndarray
    role_mask: np.ndarray
    fit_rmse_over_chord: np.ndarray
    observation_count: np.ndarray
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class MultiviewCurveExample:
    sample_id: str
    split: str
    view_paths: tuple[str, ...]
    pattern_path: str
    curve_target: np.ndarray
    role_mask: np.ndarray
    fit_rmse_over_chord: np.ndarray
    target_provenance: tuple[str, ...]
    spatial_features: np.ndarray | None = None


@dataclass(frozen=True)
class CurveParameterStandardizer:
    means: tuple[tuple[float, ...], ...]
    standard_deviations: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(cls, examples: Sequence[MultiviewCurveExample]) -> "CurveParameterStandardizer":
        if not examples:
            raise ValueError("at least one example is required")
        values = np.stack([item.curve_target for item in examples])
        masks = np.stack([item.role_mask for item in examples])
        means = np.zeros(values.shape[1:], dtype=np.float32)
        deviations = np.ones(values.shape[1:], dtype=np.float32)
        for query in range(values.shape[1]):
            observed = values[masks[:, query], query]
            if not len(observed):
                continue
            means[query] = observed.mean(axis=0)
            deviations[query] = np.maximum(observed.std(axis=0), 1e-4)
        return cls(
            tuple(tuple(float(value) for value in row) for row in means),
            tuple(tuple(float(value) for value in row) for row in deviations),
        )

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.means, dtype=np.float32)) / np.asarray(
            self.standard_deviations, dtype=np.float32
        )

    def decode(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.standard_deviations, dtype=np.float32) + np.asarray(
            self.means, dtype=np.float32
        )


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Arc-length resample a polyline, retaining exact endpoints."""

    if len(points) < 2:
        raise ValueError("a curve requires at least two points")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-9))
    points = points[keep]
    cumulative = cumulative[keep]
    if len(points) < 2 or cumulative[-1] <= 1e-9:
        raise ValueError("a curve must have nonzero arc length")
    desired = np.linspace(0.0, cumulative[-1], count)
    return np.stack(
        [np.interp(desired, cumulative, points[:, axis]) for axis in range(2)], axis=1
    ).astype(np.float32)


def _bezier(points: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    one_minus = 1.0 - parameter
    basis = np.stack(
        (
            one_minus**3,
            3.0 * one_minus**2 * parameter,
            3.0 * one_minus * parameter**2,
            parameter**3,
        ),
        axis=1,
    )
    return basis @ points


def _fit_cubic_fixed_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares fit P1/P2 with P0/P3 fixed to the observations."""

    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(length)))
    parameter = cumulative / max(float(cumulative[-1]), 1e-9)
    one_minus = 1.0 - parameter
    first = 3.0 * one_minus**2 * parameter
    second = 3.0 * one_minus * parameter**2
    design = np.stack((first, second), axis=1)
    residual = points - one_minus[:, None] ** 3 * points[0] - parameter[:, None] ** 3 * points[-1]
    controls, _, _, _ = np.linalg.lstsq(design, residual, rcond=None)
    control_1, control_2 = controls
    fitted = _bezier(
        np.stack((points[0], control_1, control_2, points[-1])), parameter
    )
    rmse = float(np.sqrt(np.mean(np.sum((fitted - points) ** 2, axis=1))))
    return control_1.astype(np.float32), control_2.astype(np.float32), rmse


def fit_two_cubic_formula(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Encode one semantic path as two cubic formulas in its chord frame.

    The output excludes implicit P0=(0,0) and P3=(1,0).  It stores the shared
    knot followed by two control points for each segment.  Thus source
    primitives can be preserved during target construction without turning
    them into multiple semantic roles.
    """

    sampled = _resample_polyline(np.asarray(points, dtype=np.float32), 65)
    origin = sampled[0]
    chord_vector = sampled[-1] - origin
    chord = float(np.linalg.norm(chord_vector))
    if chord <= 1e-6:
        raise ValueError("curve chord is too short for a stable local frame")
    axis_x = chord_vector / chord
    axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float32)
    relative = sampled - origin
    local = np.stack((relative @ axis_x, relative @ axis_y), axis=1) / chord
    first = local[:33]
    second = local[32:]
    first_1, first_2, first_error = _fit_cubic_fixed_endpoints(first)
    second_1, second_2, second_error = _fit_cubic_fixed_endpoints(second)
    knot = local[32]
    controls = np.concatenate((knot, first_1, first_2, second_1, second_2)).astype(
        np.float32
    )
    return controls, float(np.sqrt((first_error**2 + second_error**2) * 0.5))


def sample_two_cubic_formula(parameters: np.ndarray, samples_per_segment: int = 17) -> np.ndarray:
    """Reconstruct normalized curve samples from the 10 control parameters."""

    values = np.asarray(parameters, dtype=np.float32)
    if values.shape != (len(CONTROL_PARAMETER_NAMES),):
        raise ValueError(f"expected {len(CONTROL_PARAMETER_NAMES)} control values")
    knot = values[0:2]
    segment_0 = np.stack((np.asarray((0.0, 0.0)), values[2:4], values[4:6], knot))
    segment_1 = np.stack((knot, values[6:8], values[8:10], np.asarray((1.0, 0.0))))
    parameter = np.linspace(0.0, 1.0, samples_per_segment, dtype=np.float32)
    first = _bezier(segment_0, parameter)
    second = _bezier(segment_1, parameter)
    return np.concatenate((first, second[1:]), axis=0).astype(np.float32)


def _join_dense_edges(edges: Sequence[np.ndarray]) -> np.ndarray:
    output: np.ndarray | None = None
    for raw in edges:
        points = np.asarray(raw, dtype=np.float32)
        if len(points) < 2:
            continue
        if output is None:
            output = points.copy()
            continue
        forward_distance = float(np.linalg.norm(output[-1] - points[0]))
        reverse_distance = float(np.linalg.norm(output[-1] - points[-1]))
        if reverse_distance < forward_distance:
            points = points[::-1]
        if np.linalg.norm(output[-1] - points[0]) < 1e-5:
            points = points[1:]
        output = np.concatenate((output, points), axis=0)
    return np.empty((0, 2), dtype=np.float32) if output is None else output


def _dense_panel_points(canonical_panel: Mapping[str, Any]) -> np.ndarray:
    return _join_dense_edges(
        [np.asarray(edge.get("points", ()), dtype=np.float32) for edge in canonical_panel.get("edges", ())]
    )


def _landmark(panel: PanelAnnotation, name: str) -> np.ndarray | None:
    for item in panel.landmarks:
        if item.name == name:
            return np.asarray(item.xy_cm, dtype=np.float32)
    return None


def _mirror_body_panel(
    panel: PanelAnnotation,
    panel_points: np.ndarray,
    path_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    names = {item.name: np.asarray(item.xy_cm, dtype=np.float32) for item in panel.landmarks}
    center_name = "FNP" if panel.role == "front_bodice" else "BNP"
    center = names.get(center_name)
    shoulder_neck = names.get("SNP")
    if center is not None and shoulder_neck is not None and shoulder_neck[0] < center[0]:
        panel_points = panel_points.copy()
        path_points = path_points.copy()
        panel_points[:, 0] *= -1.0
        path_points[:, 0] *= -1.0
        names = {key: np.asarray((-value[0], value[1]), dtype=np.float32) for key, value in names.items()}
    return panel_points, path_points, names


def _orient_path(
    query: str,
    points: np.ndarray,
    landmarks: Mapping[str, np.ndarray],
) -> np.ndarray:
    anchor_name = None
    if query == "front_neckline":
        anchor_name = "FNP"
    elif query == "back_neckline":
        anchor_name = "BNP"
    elif query in {"front_armhole", "back_armhole"}:
        anchor_name = "SP"
    anchor = landmarks.get(anchor_name) if anchor_name else None
    reverse = False
    if anchor is not None:
        reverse = np.linalg.norm(points[-1] - anchor) < np.linalg.norm(points[0] - anchor)
    else:
        reverse = bool(
            points[0, 0] > points[-1, 0]
            or (np.isclose(points[0, 0], points[-1, 0]) and points[0, 1] > points[-1, 1])
        )
    return points[::-1].copy() if reverse else points


def _query_for(panel_role: str, path_role: str) -> str | None:
    if path_role == "neckline" and panel_role == "front_bodice":
        return "front_neckline"
    if path_role == "neckline" and panel_role == "back_bodice":
        return "back_neckline"
    if path_role == "armhole" and panel_role == "front_bodice":
        return "front_armhole"
    if path_role == "armhole" and panel_role == "back_bodice":
        return "back_armhole"
    if path_role == "sleeve_head" and panel_role == "sleeve":
        return "sleeve_head"
    return None


def curve_formula_targets(
    record: DraftingSemanticRecord,
    canonical_pattern: Mapping[str, Any],
    *,
    generator_formula_truth: Mapping[str, Sequence[float]] | None = None,
) -> CurveFormulaTargets:
    """Derive named, formula-style curve targets from one paired CAD pattern.

    ``generator_formula_truth`` is an optional bridge for an instrumented
    GarmentCode/FreeSewing generator.  Its vectors must already obey
    ``CURVE_PARAMETER_NAMES`` and are marked as exact captured formula truth.
    All other targets are explicitly marked as dense-curve approximations;
    the least-squares control points must not be reported as original authoring
    controls.
    """

    canonical_panels = {str(panel["id"]): panel for panel in canonical_pattern.get("panels", ())}
    dense_panels = [_dense_panel_points(panel) for panel in canonical_panels.values()]
    garment_scale = max(
        (float(np.max(np.ptp(points, axis=0))) for points in dense_panels if len(points)),
        default=1.0,
    )
    garment_scale = max(garment_scale, 1e-6)
    observations: dict[str, list[tuple[np.ndarray, float]]] = {
        name: [] for name in CURVE_QUERY_NAMES
    }
    for panel in record.panels:
        canonical_panel = canonical_panels.get(panel.id)
        if canonical_panel is None:
            continue
        panel_points = _dense_panel_points(canonical_panel)
        if len(panel_points) < 2:
            continue
        dense_edges = {
            str(edge["id"]): np.asarray(edge.get("points", ()), dtype=np.float32)
            for edge in canonical_panel.get("edges", ())
        }
        paths = merge_predicted_semantic_paths(
            tuple(edge.role for edge in panel.edges),
            edge_ids=tuple(edge.id for edge in panel.edges),
        )
        for path in paths:
            query = _query_for(panel.role, path.role)
            if query is None:
                continue
            points = _join_dense_edges([dense_edges.get(identifier, np.empty((0, 2))) for identifier in path.edge_ids])
            if len(points) < 2:
                continue
            transformed_panel = panel_points.copy()
            landmarks = {item.name: np.asarray(item.xy_cm, dtype=np.float32) for item in panel.landmarks}
            if panel.role in {"front_bodice", "back_bodice"}:
                transformed_panel, points, landmarks = _mirror_body_panel(
                    panel, transformed_panel, points
                )
            points = _orient_path(query, points, landmarks)
            span = np.ptp(transformed_panel, axis=0)
            if float(np.max(span)) <= 1e-6:
                continue
            minimum = transformed_panel.min(axis=0)
            safe_span = np.maximum(span, 1e-6)
            start = (points[0] - minimum) / safe_span
            end = (points[-1] - minimum) / safe_span
            chord = float(np.linalg.norm(points[-1] - points[0]))
            arc = _polyline_length(points)
            if chord <= 1e-6 or arc <= 1e-6:
                continue
            try:
                controls, fit_rmse = fit_two_cubic_formula(points)
            except ValueError:
                continue
            value = np.concatenate(
                (
                    start,
                    end,
                    np.asarray((chord / garment_scale, arc / garment_scale), dtype=np.float32),
                    controls,
                )
            ).astype(np.float32)
            observations[query].append((value, fit_rmse))

    values = np.zeros((len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES)), dtype=np.float32)
    mask = np.zeros(len(CURVE_QUERY_NAMES), dtype=bool)
    fit_error = np.zeros(len(CURVE_QUERY_NAMES), dtype=np.float32)
    counts = np.zeros(len(CURVE_QUERY_NAMES), dtype=np.int64)
    provenance = ["ABSENT"] * len(CURVE_QUERY_NAMES)
    for index, query in enumerate(CURVE_QUERY_NAMES):
        if generator_formula_truth is not None and query in generator_formula_truth:
            captured = np.asarray(generator_formula_truth[query], dtype=np.float32)
            if captured.shape != (len(CURVE_PARAMETER_NAMES),):
                raise ValueError(
                    f"captured formula for {query} must have shape {(len(CURVE_PARAMETER_NAMES),)}"
                )
            values[index] = captured
            counts[index] = 1
            mask[index] = True
            provenance[index] = CURVE_TRUTH_GENERATOR_FORMULA
            continue
        current = observations[query]
        if not current:
            continue
        values[index] = np.mean([item[0] for item in current], axis=0)
        fit_error[index] = float(np.mean([item[1] for item in current]))
        counts[index] = len(current)
        mask[index] = True
        provenance[index] = CURVE_TRUTH_DENSE_APPROXIMATION
    return CurveFormulaTargets(values, mask, fit_error, counts, tuple(provenance))


def _read_records(path: Path) -> dict[str, DraftingSemanticRecord]:
    records: dict[str, DraftingSemanticRecord] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = DraftingSemanticRecord.from_dict(json.loads(line))
                records[record.sample_id] = record
    return records


def read_multiview_curve_examples(
    index_path: Path,
    split_path: Path,
    semantic_records_path: Path,
    spatial_features_path: Path | None = None,
    *,
    split_prefix: str | None = "garments_5000_0/default_body",
) -> tuple[MultiviewCurveExample, ...]:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    split = _split_lookup(split_path, split_prefix=split_prefix)
    records = _read_records(semantic_records_path)
    precomputed: dict[str, np.ndarray] = {}
    if spatial_features_path is not None:
        archive = np.load(spatial_features_path, allow_pickle=False)
        ids = [str(value) for value in archive["sample_ids"].tolist()]
        features = archive["features"]
        if features.ndim != 4 or features.shape[1] != len(VIEW_NAMES):
            raise ValueError(f"invalid spatial feature shape: {features.shape}")
        precomputed = dict(zip(ids, features))
    output = []
    for row in payload["records"]:
        sample_id = str(row["sample_id"])
        record = records.get(sample_id)
        if record is None or (precomputed and sample_id not in precomputed):
            continue
        pattern_path = Path(str(row["source_pattern"]))
        canonical = json.loads(pattern_path.read_text(encoding="utf-8"))
        target = curve_formula_targets(record, canonical)
        output.append(
            MultiviewCurveExample(
                sample_id=sample_id,
                split=split.get(sample_id, "auxiliary"),
                view_paths=tuple(str(value) for value in row["source_views"]),
                pattern_path=str(pattern_path),
                curve_target=target.values,
                role_mask=target.role_mask,
                fit_rmse_over_chord=target.fit_rmse_over_chord,
                target_provenance=target.provenance,
                spatial_features=(
                    np.asarray(precomputed[sample_id]) if precomputed else None
                ),
            )
        )
    return tuple(output)


def multiview_curve_batch(
    examples: Sequence[MultiviewCurveExample],
    standardizer: CurveParameterStandardizer,
) -> dict[str, Any]:
    raw = np.stack([item.curve_target for item in examples])
    output: dict[str, Any] = {
        "curve_targets": standardizer.encode(raw).astype(np.float32),
        "raw_curve_targets": raw.astype(np.float32),
        "role_mask": np.stack([item.role_mask for item in examples]),
        "presence_targets": np.stack([item.role_mask for item in examples]).astype(np.float32),
        "fit_rmse_over_chord": np.stack([item.fit_rmse_over_chord for item in examples]),
        "target_provenance": tuple(item.target_provenance for item in examples),
        "sample_ids": tuple(item.sample_id for item in examples),
        "view_paths": tuple(item.view_paths for item in examples),
    }
    if all(item.spatial_features is not None for item in examples):
        output["spatial_features"] = np.stack([item.spatial_features for item in examples])
    elif any(item.spatial_features is not None for item in examples):
        raise ValueError("a batch cannot mix image-backed and precomputed spatial examples")
    return output


def build_local_maskrcnn_fpn_backbone(weights_path: Path):
    """Load only the FPN backbone from a local Mask R-CNN v2 checkpoint."""

    import torch
    from torchvision.models import resnet50
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"local pretrained weights not found: {path}")
    body = resnet50(weights=None)
    backbone = _resnet_fpn_extractor(
        body, trainable_layers=0, norm_layer=torch.nn.BatchNorm2d
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"checkpoint does not contain a Mask R-CNN backbone: {path}")
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


def spatial_token_layout(grid_sizes: Sequence[int]) -> tuple[dict[str, float | int], ...]:
    output = []
    for level, grid in enumerate(grid_sizes):
        for row in range(int(grid)):
            for column in range(int(grid)):
                output.append(
                    {
                        "level": level,
                        "row": row,
                        "column": column,
                        "x": (column + 0.5) / grid,
                        "y": (row + 0.5) / grid,
                    }
                )
    return tuple(output)


def build_spatial_curve_model(config: Mapping[str, Any], *, backbone=None):
    """Build the role-query spatial curve model.

    ``backbone`` is optional so unit tests and ordinary training can consume
    precomputed FPN tokens.  Supplying images without one raises an error
    instead of silently downloading weights.
    """

    import torch
    import torch.nn.functional as functional

    grid_sizes = tuple(int(value) for value in config.get("pyramid_grid_sizes", (8, 4, 2, 1)))
    level_names = tuple(str(value) for value in config.get("pyramid_levels", ("0", "1", "2", "3")))
    if len(grid_sizes) != len(level_names):
        raise ValueError("pyramid_grid_sizes and pyramid_levels must have equal lengths")
    layout = spatial_token_layout(grid_sizes)
    tokens_per_view = len(layout)

    class FrozenPyramidExtractor(torch.nn.Module):
        def __init__(self, value) -> None:
            super().__init__()
            self.backbone = value
            self.register_buffer(
                "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            )
            self.input_size = int(config.get("image_size", 256))

        def train(self, mode: bool = True):
            super().train(False)
            self.backbone.eval()
            return self

        def forward(self, images):
            images = functional.interpolate(
                images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
            images = (images - self.mean) / self.std
            with torch.no_grad():
                pyramid = self.backbone(images)
            tokens = []
            for name, grid in zip(level_names, grid_sizes):
                if name not in pyramid:
                    raise KeyError(f"FPN output is missing configured level {name!r}")
                pooled = functional.adaptive_avg_pool2d(pyramid[name], (grid, grid))
                tokens.append(pooled.flatten(2).transpose(1, 2))
            return torch.cat(tokens, dim=1).float()

    class InspectableSpatialDecoderLayer(torch.nn.Module):
        def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
            super().__init__()
            self.query_norm = torch.nn.LayerNorm(width)
            self.self_attention = torch.nn.MultiheadAttention(
                width, heads, dropout=dropout, batch_first=True
            )
            self.memory_norm = torch.nn.LayerNorm(width)
            self.cross_attention = torch.nn.MultiheadAttention(
                width, heads, dropout=dropout, batch_first=True
            )
            self.ff_norm = torch.nn.LayerNorm(width)
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(width, feedforward),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(feedforward, width),
            )
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, queries, memory, padding, capture_attention: bool):
            normalized = self.query_norm(queries)
            attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
            queries = queries + self.dropout(attended)
            attended, weights = self.cross_attention(
                self.query_norm(queries),
                self.memory_norm(memory),
                self.memory_norm(memory),
                key_padding_mask=padding,
                need_weights=capture_attention,
                average_attn_weights=False,
            )
            queries = queries + self.dropout(attended)
            queries = queries + self.dropout(self.feedforward(self.ff_norm(queries)))
            return queries, weights if capture_attention else None

    class SpatialCurveParameterTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            heads = int(config["heads"])
            dropout = float(config["dropout"])
            feedforward = width * int(config.get("feedforward_multiplier", 4))
            feature_dim = int(config.get("spatial_feature_dim", 256))
            self.extractor = FrozenPyramidExtractor(backbone) if backbone is not None else None
            self.patch_projection = torch.nn.Linear(feature_dim, width)
            self.view_embedding = torch.nn.Parameter(
                torch.zeros(1, len(VIEW_NAMES), 1, width)
            )
            self.level_embedding = torch.nn.Parameter(
                torch.zeros(1, 1, tokens_per_view, width)
            )
            coordinates = torch.tensor(
                [[item["x"], item["y"], item["x"] ** 2, item["y"] ** 2] for item in layout],
                dtype=torch.float32,
            )
            self.register_buffer("patch_coordinates", coordinates)
            self.coordinate_projection = torch.nn.Linear(4, width)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_encoder = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=int(config.get("memory_layers", 1)),
                enable_nested_tensor=False,
            )
            self.role_queries = torch.nn.Parameter(
                torch.zeros(1, len(CURVE_QUERY_NAMES), width)
            )
            self.decoder_layers = torch.nn.ModuleList(
                InspectableSpatialDecoderLayer(width, heads, feedforward, dropout)
                for _ in range(int(config.get("decoder_layers", 3)))
            )

            def head(size: int):
                return torch.nn.Sequential(
                    torch.nn.LayerNorm(width),
                    torch.nn.Linear(width, width),
                    torch.nn.GELU(),
                    torch.nn.Linear(width, size),
                )

            self.landmark_head = head(len(LANDMARK_PARAMETER_NAMES))
            self.metric_head = head(len(METRIC_PARAMETER_NAMES))
            self.control_head = head(len(CONTROL_PARAMETER_NAMES))
            self.presence_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, 1)
            )
            torch.nn.init.normal_(self.view_embedding, std=0.02)
            torch.nn.init.normal_(self.level_embedding, std=0.02)
            torch.nn.init.normal_(self.role_queries, std=0.02)

        def train(self, mode: bool = True):
            super().train(mode)
            if self.extractor is not None:
                self.extractor.eval()
            return self

        def forward(
            self,
            images=None,
            *,
            spatial_features=None,
            view_valid=None,
            spatial_valid=None,
            capture_attention: bool = False,
        ):
            if (images is None) == (spatial_features is None):
                raise ValueError("supply exactly one of images or spatial_features")
            if images is not None:
                if self.extractor is None:
                    raise RuntimeError(
                        "image input requires an explicitly loaded local FPN backbone"
                    )
                batch, views = images.shape[:2]
                if views != len(VIEW_NAMES) or images.shape[2] != 3:
                    raise ValueError("images must have shape [batch, 4, 3, height, width]")
                extracted = self.extractor(images.reshape(batch * views, *images.shape[2:]))
                spatial_features = extracted.reshape(batch, views, tokens_per_view, -1)
            batch, views, patches, _ = spatial_features.shape
            if views != len(VIEW_NAMES) or patches != tokens_per_view:
                raise ValueError(
                    f"spatial_features must have shape [batch, 4, {tokens_per_view}, channels]"
                )
            if view_valid is None:
                view_valid = torch.ones((batch, views), dtype=torch.bool, device=spatial_features.device)
            if spatial_valid is None:
                spatial_valid = view_valid[:, :, None].expand(-1, -1, patches)
            else:
                spatial_valid = spatial_valid & view_valid[:, :, None]
            hidden = self.patch_projection(spatial_features)
            coordinate = self.coordinate_projection(self.patch_coordinates)[None, None]
            hidden = hidden + self.view_embedding + self.level_embedding + coordinate
            hidden = hidden.reshape(batch, views * patches, -1)
            padding = ~spatial_valid.reshape(batch, views * patches)
            hidden = self.memory_encoder(hidden, src_key_padding_mask=padding)
            queries = self.role_queries.expand(batch, -1, -1)
            attention = []
            for layer in self.decoder_layers:
                queries, weights = layer(queries, hidden, padding, capture_attention)
                if capture_attention:
                    attention.append(weights.reshape(batch, weights.shape[1], len(CURVE_QUERY_NAMES), views, patches))
            landmark = self.landmark_head(queries)
            metric = self.metric_head(queries)
            control = self.control_head(queries)
            return {
                "curve_prediction": torch.cat((landmark, metric, control), dim=-1),
                "landmark_prediction": landmark,
                "metric_prediction": metric,
                "control_prediction": control,
                "presence_logits": self.presence_head(queries).squeeze(-1),
                "spatial_attention": attention,
            }

    model = SpatialCurveParameterTransformer()
    model.spatial_layout = layout
    model.tokens_per_view = tokens_per_view
    return model


def spatial_attention_maps(attention, grid_sizes: Sequence[int]):
    """Split [B,H,Q,V,T] attention into inspectable maps per FPN level."""

    output = []
    offset = 0
    for grid in grid_sizes:
        count = int(grid) ** 2
        current = attention[..., offset : offset + count]
        output.append(current.reshape(*current.shape[:-1], int(grid), int(grid)))
        offset += count
    if offset != attention.shape[-1]:
        raise ValueError("attention token count does not match grid sizes")
    return tuple(output)


def curve_reconstruction_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    *,
    samples_per_segment: int = 33,
) -> dict[str, Any]:
    """Evaluate reconstructed curves in chord-normalized local coordinates."""

    predicted = np.asarray(predicted, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    mask = np.asarray(role_mask, dtype=bool)
    expected_shape = (len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES))
    if predicted.shape != expected.shape or predicted.shape[1:] != expected_shape:
        raise ValueError("predicted and expected must have shape [N, query, parameter]")
    if mask.shape != predicted.shape[:2]:
        raise ValueError("role_mask must have shape [N, query]")
    per_query: dict[str, dict[str, float | int]] = {}
    all_rmse: list[float] = []
    all_chamfer: list[float] = []
    all_hausdorff: list[float] = []
    for query_index, query in enumerate(CURVE_QUERY_NAMES):
        rmses: list[float] = []
        chamfers: list[float] = []
        hausdorffs: list[float] = []
        landmark_errors: list[float] = []
        metric_errors: list[float] = []
        for row in np.flatnonzero(mask[:, query_index]):
            left = sample_two_cubic_formula(
                predicted[row, query_index, CONTROL_SLICE], samples_per_segment
            )
            right = sample_two_cubic_formula(
                expected[row, query_index, CONTROL_SLICE], samples_per_segment
            )
            distances = np.linalg.norm(left[:, None] - right[None, :], axis=-1)
            rmses.append(float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1)))))
            chamfers.append(
                float(0.5 * (distances.min(axis=0).mean() + distances.min(axis=1).mean()))
            )
            hausdorffs.append(
                float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
            )
            landmark_errors.append(
                float(
                    np.sqrt(
                        np.mean(
                            (predicted[row, query_index, LANDMARK_SLICE]
                             - expected[row, query_index, LANDMARK_SLICE]) ** 2
                        )
                    )
                )
            )
            metric_errors.append(
                float(
                    np.mean(
                        np.abs(
                            predicted[row, query_index, METRIC_SLICE]
                            - expected[row, query_index, METRIC_SLICE]
                        )
                    )
                )
            )
        if not rmses:
            per_query[query] = {"support": 0}
            continue
        per_query[query] = {
            "support": len(rmses),
            "pointwise_rmse_over_chord": float(np.mean(rmses)),
            "symmetric_chamfer_over_chord": float(np.mean(chamfers)),
            "hausdorff_over_chord": float(np.mean(hausdorffs)),
            "panel_landmark_rmse": float(np.mean(landmark_errors)),
            "normalized_length_mae": float(np.mean(metric_errors)),
        }
        all_rmse.extend(rmses)
        all_chamfer.extend(chamfers)
        all_hausdorff.extend(hausdorffs)
    return {
        "macro_pointwise_rmse_over_chord": float(np.mean(all_rmse)) if all_rmse else None,
        "macro_symmetric_chamfer_over_chord": float(np.mean(all_chamfer)) if all_chamfer else None,
        "macro_hausdorff_over_chord": float(np.mean(all_hausdorff)) if all_hausdorff else None,
        "per_query": per_query,
    }


def sample_two_cubic_formula_torch(control, samples_per_segment: int = 17):
    import torch

    parameter = torch.linspace(0.0, 1.0, samples_per_segment, device=control.device, dtype=control.dtype)
    one_minus = 1.0 - parameter
    basis = torch.stack(
        (
            one_minus**3,
            3.0 * one_minus**2 * parameter,
            3.0 * one_minus * parameter**2,
            parameter**3,
        ),
        dim=-1,
    )
    knot = control[..., 0:2]
    zeros = torch.zeros_like(knot)
    end = torch.cat((torch.ones_like(knot[..., :1]), torch.zeros_like(knot[..., :1])), dim=-1)
    first_points = torch.stack((zeros, control[..., 2:4], control[..., 4:6], knot), dim=-2)
    second_points = torch.stack((knot, control[..., 6:8], control[..., 8:10], end), dim=-2)
    first = torch.einsum("...pc,sp->...sc", first_points, basis)
    second = torch.einsum("...pc,sp->...sc", second_points, basis)
    return torch.cat((first, second[..., 1:, :]), dim=-2)


def curve_formula_loss(
    output: Mapping[str, Any],
    targets,
    role_mask,
    standardizer: CurveParameterStandardizer,
    *,
    parameter_weight: float = 1.0,
    sampled_curve_weight: float = 0.35,
    presence_weight: float = 0.3,
):
    """Dimension-balanced masked parameter, curve-shape, and presence loss."""

    import torch
    import torch.nn.functional as functional

    prediction = output["curve_prediction"]
    mask = role_mask.bool()
    error = functional.smooth_l1_loss(prediction, targets, reduction="none")
    expanded = mask[..., None].expand_as(error)
    per_dimension = (error * expanded).sum(dim=0) / expanded.sum(dim=0).clamp_min(1)
    supported = expanded.sum(dim=0) > 0
    parameter_loss = per_dimension[supported].mean() if supported.any() else error.sum() * 0.0

    means = torch.as_tensor(standardizer.means, device=prediction.device, dtype=prediction.dtype)
    deviations = torch.as_tensor(
        standardizer.standard_deviations, device=prediction.device, dtype=prediction.dtype
    )
    raw_prediction = prediction * deviations + means
    raw_target = targets * deviations + means
    predicted_curve = sample_two_cubic_formula_torch(raw_prediction[..., CONTROL_SLICE])
    expected_curve = sample_two_cubic_formula_torch(raw_target[..., CONTROL_SLICE])
    shape_error = functional.smooth_l1_loss(predicted_curve, expected_curve, reduction="none").mean(dim=(-1, -2))
    curve_loss = (shape_error * mask).sum() / mask.sum().clamp_min(1)
    presence_loss = functional.binary_cross_entropy_with_logits(
        output["presence_logits"], mask.to(prediction.dtype)
    )
    total = (
        float(parameter_weight) * parameter_loss
        + float(sampled_curve_weight) * curve_loss
        + float(presence_weight) * presence_loss
    )
    return {
        "loss": total,
        "parameter_loss": parameter_loss,
        "sampled_curve_loss": curve_loss,
        "presence_loss": presence_loss,
    }


__all__ = [
    "CONTROL_PARAMETER_NAMES",
    "CONTROL_SLICE",
    "CURVE_PARAMETER_NAMES",
    "CURVE_QUERY_NAMES",
    "CURVE_TRUTH_DENSE_APPROXIMATION",
    "CURVE_TRUTH_GENERATOR_FORMULA",
    "CurveFormulaTargets",
    "CurveParameterStandardizer",
    "LANDMARK_PARAMETER_NAMES",
    "LANDMARK_SLICE",
    "METRIC_PARAMETER_NAMES",
    "METRIC_SLICE",
    "MultiviewCurveExample",
    "build_local_maskrcnn_fpn_backbone",
    "build_spatial_curve_model",
    "curve_formula_loss",
    "curve_formula_targets",
    "curve_reconstruction_metrics",
    "fit_two_cubic_formula",
    "multiview_curve_batch",
    "read_multiview_curve_examples",
    "sample_two_cubic_formula",
    "sample_two_cubic_formula_torch",
    "spatial_attention_maps",
    "spatial_token_layout",
]
