"""Four-view regression of continuous, source-derived 2D pattern elements.

The target remains deliberately bounded: normalized panel extents and
semantic-path measurements, not panel vertices, spline control points, or a
stitch graph.  Missing garment elements are masked instead of trained as zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .multigarment_learning import GARMENT_ROLES, InspectableEncoderLayer
from .multiview_pattern_semantics import (
    PANEL_COUNT_NAMES,
    SEMANTIC_COUNT_NAMES,
    VIEW_NAMES,
    read_multiview_pattern_examples,
)
from .schema import DraftingSemanticRecord
from .semantic_paths import merge_predicted_semantic_paths


GEOMETRY_PANEL_ROLES = (
    "front_bodice", "back_bodice", "front_skirt", "back_skirt",
    "front_pants", "back_pants", "sleeve",
)
GEOMETRY_PATH_ROLES = (
    "neckline", "shoulder", "armhole", "center_front", "center_back",
    "side_seam", "waistline", "hemline", "sleeve_head", "sleeve_underarm",
    "sleeve_hem", "inseam", "outseam", "crotch_curve",
)
CURVEDNESS_ROLES = {"neckline", "armhole", "sleeve_head", "crotch_curve"}
PANEL_GEOMETRY_COMPONENTS = ("mean_major_extent", "mean_minor_extent", "mean_polygon_area")
PATH_GEOMETRY_COMPONENTS = ("mean_length", "mean_chord", "mean_primitive_curvedness")
PRESENCE_TARGET_NAMES = (
    *(f"panel:{role}" for role in GEOMETRY_PANEL_ROLES),
    *(f"path:{role}" for role in GEOMETRY_PATH_ROLES),
    "seam:sleeve_head_to_armhole",
)
GEOMETRY_TARGET_NAMES = (
    *(
        f"panel:{role}:{component}"
        for role in GEOMETRY_PANEL_ROLES
        for component in PANEL_GEOMETRY_COMPONENTS
    ),
    *(
        f"path:{role}:{component}"
        for role in GEOMETRY_PATH_ROLES
        for component in PATH_GEOMETRY_COMPONENTS
    ),
    "seam:sleeve_head_to_armhole_ratio",
)


@dataclass(frozen=True)
class MultiviewGeometryExample:
    sample_id: str
    split: str
    category_target: int
    view_features: np.ndarray
    geometry_target: np.ndarray
    geometry_mask: np.ndarray
    presence_target: np.ndarray
    view_paths: tuple[str, ...]
    pattern_path: str


@dataclass(frozen=True)
class MaskedTargetStandardizer:
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    @classmethod
    def fit(cls, examples: Sequence[MultiviewGeometryExample]) -> "MaskedTargetStandardizer":
        values = np.stack([item.geometry_target for item in examples])
        masks = np.stack([item.geometry_mask for item in examples])
        means = np.zeros(values.shape[1], dtype=np.float32)
        deviations = np.ones(values.shape[1], dtype=np.float32)
        for index in range(values.shape[1]):
            observed = values[masks[:, index], index]
            if len(observed):
                means[index] = float(observed.mean())
                deviations[index] = float(max(observed.std(), 1e-4))
        return cls(tuple(float(value) for value in means), tuple(float(value) for value in deviations))

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.means, dtype=np.float32)) / np.asarray(
            self.standard_deviations, dtype=np.float32
        )

    def decode(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.standard_deviations, dtype=np.float32) + np.asarray(
            self.means, dtype=np.float32
        )


def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _dense_panel_points(panel: Mapping[str, Any]) -> np.ndarray:
    points: list[list[float]] = []
    for edge in panel.get("edges", []):
        current = [[float(value[0]), float(value[1])] for value in edge.get("points", [])]
        if points and current and np.allclose(points[-1], current[0]):
            current = current[1:]
        points.extend(current)
    return np.asarray(points, dtype=np.float32)


def _polyline_measurements(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(points[-1] - points[0]))
    return length, chord


def geometry_targets(
    record: DraftingSemanticRecord,
    canonical_pattern: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized geometry values, an applicability mask, and presence."""

    canonical_panels = {
        str(panel["id"]): panel for panel in canonical_pattern.get("panels", [])
    }
    panel_spans: list[tuple[float, float]] = []
    for panel in record.panels:
        dense = _dense_panel_points(canonical_panels.get(panel.id, {}))
        if len(dense):
            span = np.ptp(dense, axis=0)
            panel_spans.append((float(span[0]), float(span[1])))
    garment_scale = max((max(width, height) for width, height in panel_spans), default=1.0)
    garment_scale = max(garment_scale, 1e-6)

    panel_values: dict[str, list[tuple[float, float, float]]] = {
        role: [] for role in GEOMETRY_PANEL_ROLES
    }
    path_values: dict[str, list[tuple[float, float, float]]] = {
        role: [] for role in GEOMETRY_PATH_ROLES
    }
    total_armhole = 0.0
    total_sleeve_head = 0.0
    for panel in record.panels:
        canonical_panel = canonical_panels.get(panel.id, {})
        dense = _dense_panel_points(canonical_panel)
        dense_edges = {
            str(edge["id"]): np.asarray(edge.get("points", []), dtype=np.float32)
            for edge in canonical_panel.get("edges", [])
        }
        if panel.role in panel_values and len(dense):
            span = np.ptp(dense, axis=0)
            major = float(max(span[0], span[1]) / garment_scale)
            minor = float(min(span[0], span[1]) / garment_scale)
            panel_values[panel.role].append(
                (
                    major,
                    minor,
                    float(_polygon_area(dense) / (garment_scale * garment_scale)),
                )
            )
        roles = tuple(edge.role for edge in panel.edges)
        lengths = tuple(float(edge.length_cm) for edge in panel.edges)
        paths = merge_predicted_semantic_paths(
            roles,
            edge_ids=tuple(edge.id for edge in panel.edges),
            edge_lengths_cm=lengths,
        )
        for path in paths:
            if path.role not in path_values:
                continue
            path_edge_points = [dense_edges.get(edge_id) for edge_id in path.edge_ids]
            if any(points is None or len(points) < 2 for points in path_edge_points):
                continue
            length = 0.0
            primitive_chords = 0.0
            for points in path_edge_points:
                primitive_length, primitive_chord = _polyline_measurements(points)
                length += primitive_length
                primitive_chords += primitive_chord
            chord = float(np.linalg.norm(path_edge_points[-1][-1] - path_edge_points[0][0]))
            curvedness = max(length - primitive_chords, 0.0) / max(length, 1e-6)
            path_values[path.role].append(
                (
                    length / garment_scale,
                    chord / garment_scale,
                    curvedness if path.role in CURVEDNESS_ROLES else float("nan"),
                )
            )
            if path.role == "armhole":
                total_armhole += length
            elif path.role == "sleeve_head":
                total_sleeve_head += length

    values: list[float] = []
    mask: list[bool] = []
    presence: list[float] = []
    for role in GEOMETRY_PANEL_ROLES:
        observed = panel_values[role]
        present = bool(observed)
        mean = np.mean(observed, axis=0) if present else np.zeros(3, dtype=np.float32)
        values.extend(float(value) for value in mean)
        mask.extend((present,) * len(PANEL_GEOMETRY_COMPONENTS))
        presence.append(float(present))
    for role in GEOMETRY_PATH_ROLES:
        observed = path_values[role]
        present = bool(observed)
        if present:
            array = np.asarray(observed, dtype=np.float32)
            valid_components = np.any(np.isfinite(array), axis=0)
            mean = np.asarray(
                [np.nanmean(array[:, index]) if valid_components[index] else 0.0 for index in range(3)],
                dtype=np.float32,
            )
        else:
            mean = np.zeros(3, dtype=np.float32)
            valid_components = np.zeros(3, dtype=bool)
        values.extend(float(value) for value in mean)
        mask.extend(bool(value) for value in valid_components)
        presence.append(float(present))
    seam_present = total_armhole > 1e-6 and total_sleeve_head > 1e-6
    values.append(total_sleeve_head / total_armhole if seam_present else 0.0)
    mask.append(seam_present)
    presence.append(float(seam_present))
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(mask, dtype=bool),
        np.asarray(presence, dtype=np.float32),
    )


def _read_semantic_records(path: Path) -> dict[str, DraftingSemanticRecord]:
    output: dict[str, DraftingSemanticRecord] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = DraftingSemanticRecord.from_dict(json.loads(line))
                output[record.sample_id] = record
    return output


def read_multiview_geometry_examples(
    index_path: Path,
    split_path: Path,
    semantic_records_path: Path,
    precomputed_features_path: Path,
) -> tuple[MultiviewGeometryExample, ...]:
    base = read_multiview_pattern_examples(
        index_path, split_path, semantic_records_path, precomputed_features_path
    )
    records = _read_semantic_records(semantic_records_path)
    output = []
    for item in base:
        canonical = json.loads(Path(item.pattern_path).read_text(encoding="utf-8"))
        values, mask, presence = geometry_targets(records[item.sample_id], canonical)
        output.append(
            MultiviewGeometryExample(
                sample_id=item.sample_id,
                split=item.split,
                category_target=item.category_target,
                view_features=item.view_features,
                geometry_target=values,
                geometry_mask=mask,
                presence_target=presence,
                view_paths=item.view_paths,
                pattern_path=item.pattern_path,
            )
        )
    return tuple(output)


def multiview_geometry_batch(
    examples: Sequence[MultiviewGeometryExample],
    standardizer: MaskedTargetStandardizer,
) -> dict[str, Any]:
    raw = np.stack([item.geometry_target for item in examples])
    return {
        "view_features": np.stack([item.view_features for item in examples]),
        "category_targets": np.asarray([item.category_target for item in examples], dtype=np.int64),
        "geometry_targets": standardizer.encode(raw).astype(np.float32),
        "raw_geometry_targets": raw,
        "geometry_mask": np.stack([item.geometry_mask for item in examples]),
        "presence_targets": np.stack([item.presence_target for item in examples]),
        "sample_ids": tuple(item.sample_id for item in examples),
    }


def build_multiview_geometry_model(config: Mapping[str, Any]):
    import torch

    class InspectableRoleDecoderLayer(torch.nn.Module):
        def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(width)
            self.self_attention = torch.nn.MultiheadAttention(
                width, heads, dropout=dropout, batch_first=True
            )
            self.norm2 = torch.nn.LayerNorm(width)
            self.cross_attention = torch.nn.MultiheadAttention(
                width, heads, dropout=dropout, batch_first=True
            )
            self.norm3 = torch.nn.LayerNorm(width)
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(width, feedforward),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(feedforward, width),
            )
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, queries, memory, memory_padding, capture_attention: bool):
            normalized = self.norm1(queries)
            attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
            queries = queries + self.dropout(attended)
            attended, weights = self.cross_attention(
                self.norm2(queries),
                memory,
                memory,
                key_padding_mask=memory_padding,
                need_weights=capture_attention,
                average_attn_weights=False,
            )
            queries = queries + self.dropout(attended)
            queries = queries + self.dropout(self.feedforward(self.norm3(queries)))
            return queries, weights if capture_attention else None

    class MultiviewElementGeometryTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            heads = int(config["heads"])
            feedforward = width * int(config["feedforward_multiplier"])
            dropout = float(config["dropout"])
            self.view_projection = torch.nn.Linear(int(config["view_feature_dim"]), width)
            self.view_embedding = torch.nn.Parameter(torch.zeros(1, len(VIEW_NAMES), width))
            self.class_token = torch.nn.Parameter(torch.zeros(1, 1, width))
            self.layers = torch.nn.ModuleList(
                InspectableEncoderLayer.build(width, heads, feedforward, dropout)
                for _ in range(int(config["layers"]))
            )
            self.role_queries = torch.nn.Parameter(
                torch.zeros(1, len(PRESENCE_TARGET_NAMES), width)
            )
            self.decoder_layers = torch.nn.ModuleList(
                InspectableRoleDecoderLayer(width, heads, feedforward, dropout)
                for _ in range(int(config.get("decoder_layers", 2)))
            )
            self.category_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, len(GARMENT_ROLES))
            )
            self.geometry_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, 3),
            )
            self.presence_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, 1)
            )
            torch.nn.init.normal_(self.view_embedding, std=0.02)
            torch.nn.init.normal_(self.class_token, std=0.02)
            torch.nn.init.normal_(self.role_queries, std=0.02)

        def forward(self, view_features, *, view_valid=None, capture_attention: bool = False):
            batch = view_features.shape[0]
            hidden = self.view_projection(view_features) + self.view_embedding
            hidden = torch.cat((self.class_token.expand(batch, -1, -1), hidden), dim=1)
            if view_valid is None:
                view_valid = torch.ones(
                    (batch, len(VIEW_NAMES)), dtype=torch.bool, device=view_features.device
                )
            padding = torch.cat(
                (
                    torch.zeros((batch, 1), dtype=torch.bool, device=view_features.device),
                    ~view_valid,
                ),
                dim=1,
            )
            attention = []
            for layer in self.layers:
                hidden, weights = layer(hidden, padding, capture_attention)
                if capture_attention:
                    attention.append(weights)
            pooled = hidden[:, 0]
            queries = self.role_queries.expand(batch, -1, -1)
            role_attention = []
            for layer in self.decoder_layers:
                queries, weights = layer(queries, hidden[:, 1:], ~view_valid, capture_attention)
                if capture_attention:
                    role_attention.append(weights)
            per_role_geometry = self.geometry_head(queries)
            # Every panel/path role query emits three measurements.  The
            # final seam query emits only its first scalar ratio.
            geometry_prediction = torch.cat(
                (per_role_geometry[:, :-1].reshape(batch, -1), per_role_geometry[:, -1, :1]),
                dim=1,
            )
            return {
                "category_logits": self.category_head(pooled),
                "geometry_prediction": geometry_prediction,
                "presence_logits": self.presence_head(queries).squeeze(-1),
                "attention": attention,
                "role_attention": role_attention,
            }

    return MultiviewElementGeometryTransformer()


__all__ = [
    "GEOMETRY_TARGET_NAMES",
    "GEOMETRY_PANEL_ROLES",
    "GEOMETRY_PATH_ROLES",
    "PANEL_GEOMETRY_COMPONENTS",
    "PATH_GEOMETRY_COMPONENTS",
    "PRESENCE_TARGET_NAMES",
    "MaskedTargetStandardizer",
    "MultiviewGeometryExample",
    "build_multiview_geometry_model",
    "geometry_targets",
    "multiview_geometry_batch",
    "read_multiview_geometry_examples",
]
