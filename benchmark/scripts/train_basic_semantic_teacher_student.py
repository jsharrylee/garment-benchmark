"""Train the bounded common-garment 2D teacher -> four-view student pilot.

The script intentionally keeps two truth domains visible:

* formula-generated common blocks are ``PROVISIONAL_EXPERT_REVIEW``; and
* GarmentCode records are exact to that generator, then receive a train-only
  query-isotropic residual-distribution alignment before being mixed with the
  common blocks.  Exact path endpoints remain untouched.  That alignment is
  a provisional transfer, not expert truth.

At student inference only frozen four-view features and a category id are
accepted.  Source image paths and pixels are never copied to the outputs.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.basic_blocks import (
    BasicBlock,
    build_basic_block,
    generate_corpus,
    load_corpus_json,
)
from benchmark.drafting_semantics.basic_semantic_targets import (
    BasicSemanticTarget,
    common_basic_category,
    filter_common_basic_records,
    resolved_common_basic_edge_roles,
    semantic_target_from_basic_block,
    semantic_target_from_drafting_record,
    stack_semantic_targets,
)
from benchmark.drafting_semantics.dataset import CURVATURE_TYPES, read_records
from benchmark.drafting_semantics.lower_body_semantics import extract_lower_body_semantics
from benchmark.drafting_semantics.multigarment_learning import (
    EDGE_FEATURE_DIM,
    GARMENT_ROLES,
    MULTIGARMENT_EDGE_ROLES,
    MULTIGARMENT_PANEL_ROLES,
    MultiGarmentExample,
    MultiPanelExample,
    padded_garment_batch,
    read_gcd_multigarment_examples,
)
from benchmark.drafting_semantics.multiview_curve_parameters import (
    CONTROL_SLICE,
    CURVE_QUERY_NAMES,
    CURVE_TRUTH_DENSE_APPROXIMATION,
    CurveFormulaTargets,
    curve_formula_targets,
    sample_two_cubic_formula,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    CATEGORY_NAMES,
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INVENTORY,
    SEMANTIC_QUERY_SCHEMA_VERSION,
    build_four_view_semantic_student,
    build_vector_graph_teacher,
    detached_teacher_forward,
    freeze_semantic_teacher,
    semantic_distillation_loss,
    semantic_token_reconstruction_loss,
)


SCHEMA_VERSION = "basic-semantic-teacher-student-result/v1"
CALIBRATION_STATUS = "PROVISIONAL_TRAIN_ONLY_DISTRIBUTION_ALIGNMENT"
EDIT_CALIBRATION_SCHEMA_VERSION = "semantic-edit-validation-gate/v1"
# `include_stitch_features=False` guarantees local feature 14 is zero for
# boundary rows.  This semantic-teacher graph reserves it as an explicit
# non-boundary construction/reference-line marker without changing the shared
# 19D tensor shape used by existing GCD features.
CONSTRUCTION_LINE_FEATURE_INDEX = 14
GRAPH_REFERENCE_NAMES = {
    "tshirt": frozenset({"BL", "WL", "HL"}),
    "pants": frozenset({"WL", "HL", "CL", "KL", "GRAIN"}),
    "skirt": frozenset({"WL", "HL", "GRAIN"}),
}


@dataclass(frozen=True)
class SemanticTrainingExample:
    sample_id: str
    category: str
    graph: MultiGarmentExample
    target: BasicSemanticTarget
    global_features: np.ndarray | None = None
    spatial_features: np.ndarray | None = None

    def validate(self) -> None:
        if self.sample_id != self.graph.sample_id or self.sample_id != self.target.sample_id:
            raise ValueError("graph and semantic target sample IDs must match")
        if self.category != self.target.category:
            raise ValueError("example category does not match semantic target")
        if self.global_features is not None:
            if self.global_features.ndim != 2 or self.global_features.shape[0] != 4:
                raise ValueError("global features must have shape [4, channels]")
        if self.spatial_features is not None:
            if self.spatial_features.ndim != 3 or self.spatial_features.shape[0] != 4:
                raise ValueError("spatial features must have shape [4, patches, channels]")


def deterministic_category_split(
    sample_ids: Sequence[str],
    categories: Sequence[str],
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    """Create a repeatable, category-stratified sample-ID split.

    This is deliberately not a recipe-family split.  Each ID appears in one
    partition and category membership is preserved as closely as integer
    counts permit.
    """

    if len(sample_ids) != len(categories):
        raise ValueError("sample_ids and categories must have equal length")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique before splitting")
    names = ("train", "validation", "test")
    values = {name: float(fractions[name]) for name in names}
    if any(value < 0.0 for value in values.values()) or not math.isclose(
        sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("split fractions must be non-negative and sum to one")
    grouped: dict[str, list[str]] = {}
    for sample_id, category in zip(sample_ids, categories, strict=True):
        if category not in CATEGORY_NAMES:
            raise ValueError(f"unsupported category: {category!r}")
        grouped.setdefault(category, []).append(str(sample_id))

    result: dict[str, str] = {}
    for category, identifiers in sorted(grouped.items()):
        identifiers.sort(
            key=lambda value: hashlib.sha256(
                f"{seed}:{category}:{value}".encode("utf-8")
            ).digest()
        )
        count = len(identifiers)
        if count >= 3 and all(values[name] > 0 for name in names):
            validation_count = max(1, int(round(count * values["validation"])))
            test_count = max(1, int(round(count * values["test"])))
            if validation_count + test_count >= count:
                overflow = validation_count + test_count - (count - 1)
                if test_count >= validation_count:
                    test_count = max(1, test_count - overflow)
                else:
                    validation_count = max(1, validation_count - overflow)
            train_count = count - validation_count - test_count
        else:
            train_count = int(round(count * values["train"]))
            validation_count = int(round(count * values["validation"]))
            test_count = count - train_count - validation_count
        boundaries = (train_count, train_count + validation_count)
        for index, sample_id in enumerate(identifiers):
            split = "train" if index < boundaries[0] else (
                "validation" if index < boundaries[1] else "test"
            )
            result[sample_id] = split
    return result


def _robust_center_scale(values: np.ndarray, minimum_scale: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    center = float(np.median(finite))
    low, high = np.quantile(finite, (0.25, 0.75))
    scale = float((high - low) / 1.349)
    if scale < minimum_scale:
        scale = float(np.std(finite))
    return center, max(scale, minimum_scale)


@dataclass(frozen=True)
class TrainOnlyCoordinateCalibrator:
    """Train-only query-isotropic residual-distribution alignment.

    Per-channel medians may differ, but every fitted coordinate within one
    category/query shares a single robust scalar.  This preserves a query's
    geometry better than independently stretching each axis.  Path endpoint
    channels 0:4 and all construction-line endpoints are never calibrated
    because authoritative named landmarks and exact source topology own them.
    """

    source_center: np.ndarray
    source_scale: np.ndarray
    reference_center: np.ndarray
    reference_scale: np.ndarray
    fitted_mask: np.ndarray
    source_support: np.ndarray
    reference_support: np.ndarray
    minimum_scale: float
    clamp_standard_deviations: float
    status: str = CALIBRATION_STATUS

    @classmethod
    def fit(
        cls,
        source_train: Sequence[BasicSemanticTarget],
        reference_train: Sequence[BasicSemanticTarget],
        *,
        minimum_support: int = 8,
        minimum_scale: float = 1e-3,
        clamp_standard_deviations: float = 4.0,
    ) -> "TrainOnlyCoordinateCalibrator":
        if not source_train or not reference_train:
            raise ValueError("calibration requires source and reference training targets")
        coordinate_shape = (
            len(CATEGORY_NAMES), len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM
        )
        query_shape = (len(CATEGORY_NAMES), len(SEMANTIC_QUERY_INVENTORY))
        source_center = np.zeros(coordinate_shape, dtype=np.float32)
        source_scale = np.ones(query_shape, dtype=np.float32)
        reference_center = np.zeros(coordinate_shape, dtype=np.float32)
        reference_scale = np.ones(query_shape, dtype=np.float32)
        fitted = np.zeros(coordinate_shape, dtype=np.bool_)
        source_support = np.zeros(coordinate_shape, dtype=np.int32)
        reference_support = np.zeros(coordinate_shape, dtype=np.int32)
        for category_id in range(len(CATEGORY_NAMES)):
            source = [item for item in source_train if item.category_id == category_id]
            reference = [item for item in reference_train if item.category_id == category_id]
            for query_index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
                source_residuals: list[np.ndarray] = []
                reference_residuals: list[np.ndarray] = []
                for channel in range(MAX_COORDINATE_DIM):
                    source_values = np.asarray(
                        [
                            item.coordinates[query_index, channel]
                            for item in source
                            if item.coordinate_mask[query_index, channel]
                        ],
                        dtype=np.float64,
                    )
                    reference_values = np.asarray(
                        [
                            item.coordinates[query_index, channel]
                            for item in reference
                            if item.coordinate_mask[query_index, channel]
                        ],
                        dtype=np.float64,
                    )
                    source_support[category_id, query_index, channel] = len(source_values)
                    reference_support[category_id, query_index, channel] = len(reference_values)
                    if len(source_values) < minimum_support or len(reference_values) < minimum_support:
                        continue
                    # Exact garment-frame endpoints are not distribution
                    # transferred; doing so can break shared seams/landmarks.
                    if (query.kind == "path" and channel < 4) or query.kind == "reference_line":
                        continue
                    source_channel_center = float(np.median(source_values))
                    reference_channel_center = float(np.median(reference_values))
                    source_center[category_id, query_index, channel] = source_channel_center
                    reference_center[category_id, query_index, channel] = reference_channel_center
                    source_residuals.append(source_values - source_channel_center)
                    reference_residuals.append(reference_values - reference_channel_center)
                    fitted[category_id, query_index, channel] = True
                if source_residuals:
                    _, source_scale[category_id, query_index] = _robust_center_scale(
                        np.concatenate(source_residuals), minimum_scale
                    )
                    _, reference_scale[category_id, query_index] = _robust_center_scale(
                        np.concatenate(reference_residuals), minimum_scale
                    )
        return cls(
            source_center,
            source_scale,
            reference_center,
            reference_scale,
            fitted,
            source_support,
            reference_support,
            float(minimum_scale),
            float(clamp_standard_deviations),
        )

    def transform(self, target: BasicSemanticTarget) -> BasicSemanticTarget:
        coordinates = target.coordinates.copy()
        category_id = target.category_id
        active = target.coordinate_mask & self.fitted_mask[category_id]
        if active.any():
            standardized = (
                coordinates - self.source_center[category_id]
            ) / self.source_scale[category_id, :, None]
            standardized = np.clip(
                standardized,
                -self.clamp_standard_deviations,
                self.clamp_standard_deviations,
            )
            aligned = (
                standardized * self.reference_scale[category_id, :, None]
                + self.reference_center[category_id]
            )
            coordinates[active] = aligned[active]
        output = replace(
            target,
            coordinates=coordinates,
            provenance_status=f"{target.provenance_status}+{self.status}",
        )
        output.validate()
        return output

    def to_dict(self) -> dict[str, Any]:
        fitted = np.argwhere(self.fitted_mask)
        entries = []
        for category_id, query_index, channel in fitted:
            query = SEMANTIC_QUERY_INVENTORY[int(query_index)]
            entries.append(
                {
                    "category": CATEGORY_NAMES[int(category_id)],
                    "query": query.key,
                    "coordinate": query.coordinate_names[int(channel)],
                    "source_center": float(self.source_center[category_id, query_index, channel]),
                    "query_isotropic_source_scale": float(self.source_scale[category_id, query_index]),
                    "reference_center": float(self.reference_center[category_id, query_index, channel]),
                    "query_isotropic_reference_scale": float(self.reference_scale[category_id, query_index]),
                    "source_support": int(self.source_support[category_id, query_index, channel]),
                    "reference_support": int(self.reference_support[category_id, query_index, channel]),
                }
            )
        return {
            "status": self.status,
            "fit_partition": "train_only",
            "method": "train_only_query_isotropic_residual_distribution_alignment",
            "minimum_scale": self.minimum_scale,
            "clamp_standard_deviations": self.clamp_standard_deviations,
            "fitted_coordinate_count": len(entries),
            "entries": entries,
            "claim_boundary": (
                "This transfers train-partition residual distributions with one robust "
                "scale per category/query; path and construction-line endpoint channels remain exact and "
                "uncalibrated. It is not CAD truth, expert validation, or sample-specific "
                "geometric correction."
            ),
        }


def _sampled_path_points(block: BasicBlock, panel_id: str) -> dict[str, np.ndarray]:
    document = block.to_pattern_document(curve_samples=16)
    panel = next(item for item in document.panels if item.id == panel_id)
    return {edge.id: np.asarray(edge.points, dtype=np.float32) for edge in panel.edges}


def _construction_line_feature(
    points_cm: Sequence[Sequence[float]],
    *,
    panel_center: np.ndarray,
    panel_scale: float,
    garment_scale: float,
) -> tuple[np.ndarray, float]:
    """Encode one non-boundary line in the same 19D geometric frame."""

    if len(points_cm) != 2:
        raise ValueError("construction line requires exactly two endpoints")
    ordered = sorted(
        (np.asarray(point, dtype=np.float32) for point in points_cm),
        key=lambda point: (float(point[0]), float(point[1])),
    )
    start = (ordered[0] - panel_center) / panel_scale
    end = (ordered[1] - panel_center) / panel_scale
    delta = end - start
    angle = math.atan2(float(delta[1]), float(delta[0]))
    length = float(np.linalg.norm(ordered[1] - ordered[0]))
    curvature = np.zeros(len(CURVATURE_TYPES), dtype=np.float32)
    curvature[CURVATURE_TYPES.index("line")] = 1.0
    feature = np.asarray(
        [
            start[0], start[1], end[0], end[1], delta[0], delta[1],
            length / panel_scale, math.sin(angle), math.cos(angle),
            *curvature,
            0.0,
            1.0,  # explicit non-boundary construction-line token
            0.0,
            0.0,
            length / garment_scale,
            panel_scale / garment_scale,
        ],
        dtype=np.float32,
    )
    if len(feature) != EDGE_FEATURE_DIM or feature[CONSTRUCTION_LINE_FEATURE_INDEX] != 1.0:
        raise AssertionError("construction-line feature contract drifted")
    return feature, length


def basic_block_to_graph(block: BasicBlock) -> MultiGarmentExample:
    """Encode a provisional block with the existing 19D source-agnostic graph contract."""

    block.validate()
    sampled = {panel.id: _sampled_path_points(block, panel.id) for panel in block.panels}
    # Match `dataset.edge_features` exactly: the frame is built from source
    # boundary vertices/endpoints, never dense curve samples.  Dense samples
    # are used only for the physical arc-length feature below.
    panel_vertices: dict[str, np.ndarray] = {}
    panel_scales: dict[str, float] = {}
    for panel in block.panels:
        names: list[str] = []
        seen: set[str] = set()
        paths = {path.name: path for path in panel.paths}
        for path_name in panel.boundary_order:
            for name in paths[path_name].landmark_sequence:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        vertices = np.asarray([panel.landmark(name).xy_cm for name in names], dtype=np.float32)
        panel_vertices[panel.id] = vertices
        panel_scales[panel.id] = float(max(np.ptp(vertices, axis=0).max(), 1e-6))
    garment_scale = max(panel_scales.values(), default=1.0)
    graph_panels: list[MultiPanelExample] = []
    for panel in block.panels:
        paths = {path.name: path for path in panel.paths}
        vertices = panel_vertices[panel.id]
        center = vertices.mean(axis=0)
        panel_scale = panel_scales[panel.id]
        feature_rows = []
        targets = []
        lengths = []
        edge_ids = []
        primitives: list[tuple[Any, str, str, str, float, str]] = []
        for path_name in panel.boundary_order:
            path = paths[path_name]
            if path.geometry_kind == "cubic_bezier":
                points = sampled[panel.id][path_name]
                primitives.append(
                    (
                        path,
                        path.landmark_sequence[0],
                        path.landmark_sequence[-1],
                        path_name,
                        float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                        "cubic",
                    )
                )
            else:
                # A semantic polyline remains one path target, but the vector
                # teacher sees each authored primitive.  Otherwise SIDE_HIP,
                # SIDE_KNEE and equivalent drafting breakpoints disappear
                # entirely from its input graph.
                for segment_index, (start_name, end_name) in enumerate(
                    zip(path.landmark_sequence, path.landmark_sequence[1:])
                ):
                    start_point = np.asarray(
                        panel.landmark(start_name).xy_cm, dtype=np.float32
                    )
                    end_point = np.asarray(
                        panel.landmark(end_name).xy_cm, dtype=np.float32
                    )
                    primitives.append(
                        (
                            path,
                            start_name,
                            end_name,
                            f"{path_name}#{segment_index}",
                            float(np.linalg.norm(end_point - start_point)),
                            "line",
                        )
                    )
        for index, (path, start_name, end_name, edge_id, length, curvature_kind) in enumerate(primitives):
            start_cm = np.asarray(
                panel.landmark(start_name).xy_cm, dtype=np.float32
            )
            end_cm = np.asarray(
                panel.landmark(end_name).xy_cm, dtype=np.float32
            )
            start = (start_cm - center) / panel_scale
            end = (end_cm - center) / panel_scale
            delta = end - start
            angle = math.atan2(float(delta[1]), float(delta[0]))
            curvature = np.zeros(len(CURVATURE_TYPES), dtype=np.float32)
            curvature[CURVATURE_TYPES.index(curvature_kind)] = 1.0
            phase = 2.0 * math.pi * index / max(len(primitives), 1)
            local = np.asarray(
                [
                    start[0], start[1], end[0], end[1], delta[0], delta[1],
                    length / panel_scale, math.sin(angle), math.cos(angle),
                    *curvature, 0.0, 0.0, math.sin(phase), math.cos(phase),
                    length / garment_scale, panel_scale / garment_scale,
                ],
                dtype=np.float32,
            )
            if len(local) != EDGE_FEATURE_DIM:
                raise AssertionError("basic-block graph feature contract drifted")
            feature_rows.append(local)
            targets.append(
                MULTIGARMENT_EDGE_ROLES.index(path.role)
                if path.role in MULTIGARMENT_EDGE_ROLES
                else -100
            )
            lengths.append(length)
            edge_ids.append(edge_id)
        # Only query-aligned construction lines are graph tokens.  Sleeve
        # grain/bicep helpers have no shared target and are absent in real GCD;
        # including them would leak the provisional source domain.
        for line in panel.reference_lines:
            if line.name not in GRAPH_REFERENCE_NAMES[block.category]:
                continue
            feature, length = _construction_line_feature(
                (line.start_cm, line.end_cm),
                panel_center=center,
                panel_scale=panel_scale,
                garment_scale=garment_scale,
            )
            feature_rows.append(feature)
            targets.append(-100)
            lengths.append(length)
            edge_ids.append(f"@reference:{line.name}")
        graph_panels.append(
            MultiPanelExample(
                panel_id=panel.id,
                panel_target=(
                    MULTIGARMENT_PANEL_ROLES.index(panel.role)
                    if panel.role in MULTIGARMENT_PANEL_ROLES
                    else MULTIGARMENT_PANEL_ROLES.index("other")
                ),
                features=np.stack(feature_rows),
                edge_targets=np.asarray(targets, dtype=np.int64),
                edge_lengths_cm=np.asarray(lengths, dtype=np.float32),
                edge_ids=tuple(edge_ids),
                panel_scale_cm=panel_scale,
            )
        )
    garment_category = "top" if block.category == "tshirt" else block.category
    return MultiGarmentExample(
        sample_id=block.sample_id,
        split="provisional",
        source="provisional_common_basic_block",
        garment_target=GARMENT_ROLES.index(garment_category),
        panels=tuple(graph_panels),
    )


def graph_padding_audit(
    examples: Sequence[SemanticTrainingExample],
    *,
    maximum_panels: int,
    maximum_edges: int,
) -> dict[str, Any]:
    """Fail before optimization if graph padding would drop any token."""

    observed_panels = max((len(item.graph.panels) for item in examples), default=0)
    observed_edges = max(
        (
            len(panel.features)
            for item in examples
            for panel in item.graph.panels
        ),
        default=0,
    )
    oversized_panels = sorted(
        (item.sample_id, len(item.graph.panels))
        for item in examples
        if len(item.graph.panels) > maximum_panels
    )
    oversized_edges = sorted(
        (item.sample_id, panel.panel_id, len(panel.features))
        for item in examples
        for panel in item.graph.panels
        if len(panel.features) > maximum_edges
    )
    if oversized_panels or oversized_edges:
        raise ValueError(
            "graph padding would truncate semantic inputs: "
            f"panels={oversized_panels[:3]} edges={oversized_edges[:3]}"
        )
    return {
        "example_count": int(len(examples)),
        "panel_count": int(sum(len(item.graph.panels) for item in examples)),
        "configured_maximum_panels": int(maximum_panels),
        "configured_maximum_edges": int(maximum_edges),
        "observed_maximum_panels": int(observed_panels),
        "observed_maximum_edges": int(observed_edges),
        "status": "PASS_NO_TRUNCATION",
    }


def _batch_graph_targets(
    examples: Sequence[SemanticTrainingExample],
    *,
    maximum_panels: int,
    maximum_edges: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = padded_garment_batch(
        [item.graph for item in examples],
        maximum_panels=maximum_panels,
        maximum_edges=maximum_edges,
    )
    targets = stack_semantic_targets([item.target for item in examples])
    return graph, targets


def supervised_teacher_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Query + primitive-role supervision for one teacher batch."""

    import torch
    import torch.nn.functional as functional

    device = output["presence_logits"].device
    query_mask = torch.as_tensor(targets["query_mask"], device=device, dtype=torch.bool)
    presence = torch.as_tensor(targets["presence_targets"], device=device, dtype=torch.float32)
    coordinates = torch.as_tensor(
        targets["coordinate_targets"], device=device, dtype=torch.float32
    )
    coordinate_mask = torch.as_tensor(
        targets["coordinate_mask"], device=device, dtype=torch.bool
    )
    reconstruction = semantic_token_reconstruction_loss(
        output,
        presence_targets=presence,
        coordinate_targets=coordinates,
        coordinate_mask=coordinate_mask,
        query_mask=query_mask,
    )
    presence_loss = reconstruction["presence_loss"]
    coordinate_loss = reconstruction["coordinate_loss"]

    def role_loss(logits: Any, raw_targets: np.ndarray) -> Any:
        labels = torch.as_tensor(raw_targets, device=device, dtype=torch.long)
        valid = labels != -100
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return functional.cross_entropy(logits[valid], labels[valid])

    panel_role_loss = role_loss(output["panel_role_logits"], graph["panel_targets"])
    edge_role_loss = role_loss(output["edge_role_logits"], graph["edge_targets"])
    total = (
        float(weights["presence"]) * presence_loss
        + float(weights["coordinate"]) * coordinate_loss
        + float(weights["panel_role"]) * panel_role_loss
        + float(weights["edge_role"]) * edge_role_loss
    )
    return {
        "loss": total,
        "token_reconstruction_loss": reconstruction["loss"],
        "presence_loss": presence_loss,
        "coordinate_loss": coordinate_loss,
        "panel_role_loss": panel_role_loss,
        "edge_role_loss": edge_role_loss,
    }


def _forward_teacher(model: Any, graph: Mapping[str, Any], targets: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return model(
        torch.as_tensor(graph["features"], device=device, dtype=torch.float32),
        edge_valid=torch.as_tensor(graph["edge_valid"], device=device, dtype=torch.bool),
        panel_valid=torch.as_tensor(graph["panel_valid"], device=device, dtype=torch.bool),
        category_ids=torch.as_tensor(targets["category_ids"], device=device, dtype=torch.long),
    )


def _visual_tensors(
    examples: Sequence[SemanticTrainingExample],
    *,
    mode: str,
    device: Any,
    missing_view: int | None = None,
) -> tuple[Any | None, Any | None, Any]:
    import torch

    global_features = None
    spatial_features = None
    if mode in {"global", "global+spatial"}:
        global_features = torch.as_tensor(
            np.stack([item.global_features for item in examples]),
            device=device,
            dtype=torch.float32,
        )
    if mode in {"spatial", "global+spatial"}:
        spatial_features = torch.as_tensor(
            np.stack([item.spatial_features for item in examples]),
            device=device,
            dtype=torch.float32,
        )
    view_valid = torch.ones((len(examples), 4), device=device, dtype=torch.bool)
    if missing_view is not None:
        view_valid[:, missing_view] = False
        if global_features is not None:
            global_features[:, missing_view] = 0
        if spatial_features is not None:
            spatial_features[:, missing_view] = 0
    return global_features, spatial_features, view_valid


def student_training_step(
    student: Any,
    teacher: Any,
    examples: Sequence[SemanticTrainingExample],
    *,
    mode: str,
    maximum_panels: int,
    maximum_edges: int,
    device: Any,
    weights: Mapping[str, float],
    missing_view: int | None = None,
) -> dict[str, Any]:
    """One differentiable student step; useful for smoke tests and the CLI."""

    import torch

    graph, targets = _batch_graph_targets(
        examples, maximum_panels=maximum_panels, maximum_edges=maximum_edges
    )
    category_ids = torch.as_tensor(targets["category_ids"], device=device, dtype=torch.long)
    teacher_output = detached_teacher_forward(
        teacher,
        torch.as_tensor(graph["features"], device=device, dtype=torch.float32),
        edge_valid=torch.as_tensor(graph["edge_valid"], device=device, dtype=torch.bool),
        panel_valid=torch.as_tensor(graph["panel_valid"], device=device, dtype=torch.bool),
        category_ids=category_ids,
    )
    global_features, spatial_features, view_valid = _visual_tensors(
        examples, mode=mode, device=device, missing_view=missing_view
    )
    student_output = student(
        category_ids=category_ids,
        global_features=global_features,
        spatial_features=spatial_features,
        view_valid=view_valid,
    )
    return semantic_distillation_loss(
        student_output,
        teacher_output,
        presence_targets=torch.as_tensor(
            targets["presence_targets"], device=device, dtype=torch.float32
        ),
        coordinate_targets=torch.as_tensor(
            np.nan_to_num(targets["coordinate_targets"]), device=device, dtype=torch.float32
        ),
        coordinate_mask=torch.as_tensor(
            targets["coordinate_mask"], device=device, dtype=torch.bool
        ),
        query_mask=torch.as_tensor(targets["query_mask"], device=device, dtype=torch.bool),
        weights=weights,
    )


def _binary_f1(probabilities: np.ndarray, targets: np.ndarray) -> float:
    predicted = probabilities >= 0.5
    expected = targets >= 0.5
    tp = int(np.sum(predicted & expected))
    fp = int(np.sum(predicted & ~expected))
    fn = int(np.sum(~predicted & expected))
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def semantic_metrics(
    presence_probabilities: np.ndarray,
    coordinate_predictions: np.ndarray,
    targets: Sequence[BasicSemanticTarget],
) -> dict[str, Any]:
    """Report presence macro-F1 and normalized geometry errors by category/kind."""

    stacked = stack_semantic_targets(targets)
    expected_presence = stacked["presence_targets"]
    query_mask = stacked["query_mask"]
    expected_coordinates = stacked["coordinate_targets"]
    coordinate_mask = stacked["coordinate_mask"]
    category_ids = stacked["category_ids"]
    per_category: dict[str, Any] = {}
    all_f1: list[float] = []
    all_errors: list[float] = []
    for category_id, category in enumerate(CATEGORY_NAMES):
        rows = category_ids == category_id
        f1_values = []
        query_values: dict[str, Any] = {}
        kind_errors: dict[str, list[float]] = {
            "panel": [],
            "path": [],
            "landmark": [],
            "reference_line": [],
        }
        landmark_l2: list[float] = []
        reference_line_endpoint_l2: list[float] = []
        for query_index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
            if query.category != category:
                continue
            applicable = rows & query_mask[:, query_index]
            if not applicable.any():
                continue
            f1 = _binary_f1(
                presence_probabilities[applicable, query_index],
                expected_presence[applicable, query_index],
            )
            positive_support = int(
                expected_presence[applicable, query_index].sum()
            )
            negative_support = int(applicable.sum()) - positive_support
            f1_values.append(f1)
            active = rows[:, None] & coordinate_mask[:, query_index, :]
            error = np.abs(
                coordinate_predictions[:, query_index, :] - expected_coordinates[:, query_index, :]
            )
            values = error[active]
            query_mae = float(values.mean()) if len(values) else None
            if len(values):
                kind_errors[query.kind].extend(float(value) for value in values)
                all_errors.extend(float(value) for value in values)
            if query.kind == "landmark":
                xy_active = rows & coordinate_mask[:, query_index, 0] & coordinate_mask[:, query_index, 1]
                if xy_active.any():
                    delta = (
                        coordinate_predictions[xy_active, query_index, :2]
                        - expected_coordinates[xy_active, query_index, :2]
                    )
                    landmark_l2.extend(np.linalg.norm(delta, axis=1).astype(float).tolist())
            elif query.kind == "reference_line":
                line_active = rows & coordinate_mask[:, query_index, :4].all(axis=1)
                if line_active.any():
                    delta = (
                        coordinate_predictions[line_active, query_index, :4]
                        - expected_coordinates[line_active, query_index, :4]
                    ).reshape(-1, 2, 2)
                    reference_line_endpoint_l2.extend(
                        np.linalg.norm(delta, axis=2).mean(axis=1).astype(float).tolist()
                    )
            query_values[query.key] = {
                "applicable_support": int(applicable.sum()),
                "positive_support": positive_support,
                "negative_support": negative_support,
                "presence_f1": f1,
                "presence_metric_status": (
                    "BOTH_CLASSES_OBSERVED"
                    if positive_support and negative_support
                    else (
                        "POSITIVE_ONLY_NO_ABSENCE_EVIDENCE"
                        if positive_support
                        else "NEGATIVE_ONLY_NO_PRESENCE_EVIDENCE"
                    )
                ),
                "coordinate_channel_support": int(active.sum()),
                "coordinate_normalized_mae": query_mae,
            }
        all_f1.extend(f1_values)
        per_category[category] = {
            "sample_count": int(rows.sum()),
            "presence_macro_f1": float(np.mean(f1_values)) if f1_values else None,
            "coordinate_normalized_mae": {
                kind: float(np.mean(values)) if values else None
                for kind, values in kind_errors.items()
            },
            "landmark_uv_l2_mean": float(np.mean(landmark_l2)) if landmark_l2 else None,
            "reference_line_endpoint_uv_l2_mean": (
                float(np.mean(reference_line_endpoint_l2))
                if reference_line_endpoint_l2
                else None
            ),
            "queries": query_values,
        }
    return {
        "sample_count": len(targets),
        "presence_macro_f1": float(np.mean(all_f1)) if all_f1 else None,
        "coordinate_normalized_mae": float(np.mean(all_errors)) if all_errors else None,
        "per_category": per_category,
    }


@dataclass(frozen=True)
class CategoryMeanBaseline:
    presence: np.ndarray
    coordinates: np.ndarray

    @classmethod
    def fit(cls, targets: Sequence[BasicSemanticTarget]) -> "CategoryMeanBaseline":
        shape_presence = (len(CATEGORY_NAMES), len(SEMANTIC_QUERY_INVENTORY))
        shape_coordinates = (*shape_presence, MAX_COORDINATE_DIM)
        presence = np.zeros(shape_presence, dtype=np.float32)
        coordinates = np.zeros(shape_coordinates, dtype=np.float32)
        for category_id in range(len(CATEGORY_NAMES)):
            selected = [item for item in targets if item.category_id == category_id]
            for query_index in range(len(SEMANTIC_QUERY_INVENTORY)):
                applicable = [item for item in selected if item.query_applicability[query_index]]
                if applicable:
                    presence[category_id, query_index] = float(
                        np.mean([item.presence[query_index] for item in applicable])
                    )
                for channel in range(MAX_COORDINATE_DIM):
                    values = [
                        item.coordinates[query_index, channel]
                        for item in selected
                        if item.coordinate_mask[query_index, channel]
                    ]
                    if values:
                        coordinates[category_id, query_index, channel] = float(np.mean(values))
        return cls(presence, coordinates)

    def predict(self, targets: Sequence[BasicSemanticTarget]) -> tuple[np.ndarray, np.ndarray]:
        category_ids = np.asarray([item.category_id for item in targets], dtype=np.int64)
        return self.presence[category_ids], self.coordinates[category_ids]


def _role_metrics(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray], names: Sequence[str]) -> dict[str, Any]:
    predicted = np.concatenate([value.reshape(-1) for value in predictions])
    expected = np.concatenate([value.reshape(-1) for value in targets])
    valid = expected != -100
    predicted, expected = predicted[valid], expected[valid]
    per_role = {}
    values = []
    for index, name in enumerate(names):
        support = int(np.sum(expected == index))
        if not support:
            continue
        f1 = _binary_f1((predicted == index).astype(np.float32), (expected == index).astype(np.float32))
        per_role[name] = {"support": support, "f1": f1}
        values.append(f1)
    return {
        "support": int(len(expected)),
        "accuracy": float(np.mean(predicted == expected)) if len(expected) else None,
        "macro_f1": float(np.mean(values)) if values else None,
        "per_role": per_role,
    }


def _evaluate_teacher(
    model: Any,
    examples: Sequence[SemanticTrainingExample],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probabilities, coordinates = [], []
    panel_predictions, panel_targets = [], []
    edge_predictions, edge_targets = [], []
    batch_size = int(config["teacher"]["batch_size"])
    padding = config["padding"]
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            current = examples[start : start + batch_size]
            graph, targets = _batch_graph_targets(
                current,
                maximum_panels=int(padding["maximum_panels"]),
                maximum_edges=int(padding["maximum_edges"]),
            )
            output = _forward_teacher(model, graph, targets, device)
            probabilities.append(output["presence_logits"].sigmoid().cpu().numpy())
            coordinates.append(output["coordinates"].cpu().numpy())
            panel_predictions.append(output["panel_role_logits"].argmax(-1).cpu().numpy())
            panel_targets.append(graph["panel_targets"])
            edge_predictions.append(output["edge_role_logits"].argmax(-1).cpu().numpy())
            edge_targets.append(graph["edge_targets"])
    probability = np.concatenate(probabilities)
    coordinate = np.concatenate(coordinates)
    metrics = semantic_metrics(probability, coordinate, [item.target for item in examples])
    metrics["panel_roles"] = _role_metrics(panel_predictions, panel_targets, MULTIGARMENT_PANEL_ROLES)
    metrics["edge_roles"] = _role_metrics(edge_predictions, edge_targets, MULTIGARMENT_EDGE_ROLES)
    return metrics, probability, coordinate


def _evaluate_student(
    model: Any,
    examples: Sequence[SemanticTrainingExample],
    config: Mapping[str, Any],
    device: Any,
    *,
    missing_view: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probabilities, coordinates = [], []
    batch_size = int(config["student"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            current = examples[start : start + batch_size]
            targets = stack_semantic_targets([item.target for item in current])
            category_ids = torch.as_tensor(targets["category_ids"], device=device, dtype=torch.long)
            global_features, spatial_features, view_valid = _visual_tensors(
                current,
                mode=str(config["visual_feature_mode"]),
                device=device,
                missing_view=missing_view,
            )
            output = model(
                category_ids=category_ids,
                global_features=global_features,
                spatial_features=spatial_features,
                view_valid=view_valid,
            )
            probabilities.append(output["presence_logits"].sigmoid().cpu().numpy())
            coordinates.append(output["coordinates"].cpu().numpy())
    probability = np.concatenate(probabilities)
    coordinate = np.concatenate(coordinates)
    return (
        semantic_metrics(probability, coordinate, [item.target for item in examples]),
        probability,
        coordinate,
    )


def _selection_score(metrics: Mapping[str, Any]) -> float:
    f1 = metrics.get("presence_macro_f1")
    mae = metrics.get("coordinate_normalized_mae")
    return float(0.0 if f1 is None else f1) - float(1.0 if mae is None else mae)


def _clone_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_teacher(
    train: list[SemanticTrainingExample],
    validation: Sequence[SemanticTrainingExample],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    model = build_vector_graph_teacher(config["model"]).to(device)
    section = config["teacher"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(int(config["seed"]) + 11)
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    padding = config["padding"]
    started = time.perf_counter()
    for epoch in range(1, int(section["epochs"]) + 1):
        generator.shuffle(train)
        model.train()
        losses = []
        for start in range(0, len(train), int(section["batch_size"])):
            current = train[start : start + int(section["batch_size"])]
            graph, targets = _batch_graph_targets(
                current,
                maximum_panels=int(padding["maximum_panels"]),
                maximum_edges=int(padding["maximum_edges"]),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = _forward_teacher(model, graph, targets, device)
                loss = supervised_teacher_loss(
                    output, targets, graph, weights=section["loss_weights"]
                )
            scaler.scale(loss["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss["loss"].detach().cpu()))
        metrics, _, _ = _evaluate_teacher(model, validation, config, device)
        score = _selection_score(metrics)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_presence_macro_f1": metrics["presence_macro_f1"],
            "validation_coordinate_normalized_mae": metrics["coordinate_normalized_mae"],
            "selection_score": score,
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"stage": "teacher", **row}), flush=True)
        if score > best_score + 1e-6:
            best_score, best_epoch, best_state = score, epoch, _clone_state(model)
        if epoch - best_epoch >= int(section["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("teacher training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "model_state": best_state,
    }


def _train_student(
    teacher: Any,
    train: list[SemanticTrainingExample],
    validation: Sequence[SemanticTrainingExample],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    freeze_semantic_teacher(teacher)
    student = build_four_view_semantic_student(config["model"]).to(device)
    section = config["student"]
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(int(config["seed"]) + 29)
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(section["epochs"]) + 1):
        generator.shuffle(train)
        student.train()
        losses = []
        for start in range(0, len(train), int(section["batch_size"])):
            current = train[start : start + int(section["batch_size"])]
            missing_view = None
            if generator.random() < float(section["view_dropout_probability"]):
                missing_view = int(generator.integers(4))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = student_training_step(
                    student,
                    teacher,
                    current,
                    mode=str(config["visual_feature_mode"]),
                    maximum_panels=int(config["padding"]["maximum_panels"]),
                    maximum_edges=int(config["padding"]["maximum_edges"]),
                    device=device,
                    weights=section["loss_weights"],
                    missing_view=missing_view,
                )
            scaler.scale(loss["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss["loss"].detach().cpu()))
        metrics, _, _ = _evaluate_student(student, validation, config, device)
        score = _selection_score(metrics)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_presence_macro_f1": metrics["presence_macro_f1"],
            "validation_coordinate_normalized_mae": metrics["coordinate_normalized_mae"],
            "selection_score": score,
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"stage": "student", **row}), flush=True)
        if score > best_score + 1e-6:
            best_score, best_epoch, best_state = score, epoch, _clone_state(student)
        if epoch - best_epoch >= int(section["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("student training produced no checkpoint")
    student.load_state_dict(best_state)
    return student, {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "model_state": best_state,
    }


def _load_feature_archive(path: Path, expected_ndim: int) -> tuple[tuple[str, ...], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        sample_ids = tuple(str(value) for value in archive["sample_ids"].tolist())
        # Do not cast here: the spatial archive is intentionally float16 and
        # can exceed 1 GB.  A float32 copy is made only for the active batch.
        features = archive["features"]
    if features.ndim != expected_ndim or features.shape[1] != 4:
        raise ValueError(f"invalid feature archive shape: {features.shape}")
    if len(sample_ids) != len(features) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("feature archive sample IDs are missing or duplicated")
    return sample_ids, features


def overlay_dense_upper_curve_targets(
    target: BasicSemanticTarget,
    record: Any,
    canonical_pattern: Mapping[str, Any],
) -> BasicSemanticTarget:
    """Overlay five paired dense-canonical upper curves onto query channels.

    The fitted two-cubic values are not authoring controls.  They are an exact
    deterministic target *of the benchmark approximation* and remain labelled
    ``DENSE_CURVE_TWO_CUBIC_APPROXIMATION``.  This supplies the depth/tangent
    channels that ``DraftingSemanticRecord`` itself cannot recover.
    """

    if target.category != "tshirt":
        return target
    formula = curve_formula_targets(record, canonical_pattern)
    return _apply_dense_curve_formula_overlay(target, formula)


def _apply_dense_curve_formula_overlay(
    target: BasicSemanticTarget,
    formula: CurveFormulaTargets,
) -> BasicSemanticTarget:
    """Fill arc/depth/tangent channels without replacing garment-frame UV.

    ``curve_formula_targets`` stores endpoints in a panel frame, whereas the
    shared semantic target stores endpoints in a garment frame.  The original
    exact semantic endpoints must therefore remain untouched.  Arc/depth are
    garment-scale normalized and tangent angles are dimensionless, so only
    channels 4:8 are compatible with the shared contract.
    """

    original_endpoints = target.coordinates[:, :4].copy()
    coordinates = target.coordinates.copy()
    coordinate_mask = target.coordinate_mask.copy()
    applicability = target.query_applicability.copy()
    presence = target.presence.copy()
    evidence = list(target.evidence)
    query_lookup = {
        query.name: index
        for index, query in enumerate(SEMANTIC_QUERY_INVENTORY)
        if query.category == "tshirt" and query.kind == "path"
    }
    for formula_index, name in enumerate(CURVE_QUERY_NAMES):
        if not bool(formula.role_mask[formula_index]) or name not in query_lookup:
            continue
        raw = np.asarray(formula.values[formula_index], dtype=np.float32)
        local = sample_two_cubic_formula(raw[CONTROL_SLICE], samples_per_segment=33)
        signed_depth_local = float(local[np.argmax(np.abs(local[:, 1])), 1])
        first_control = raw[CONTROL_SLICE][2:4]
        final_control = raw[CONTROL_SLICE][8:10]
        start_tangent = math.atan2(float(first_control[1]), float(first_control[0])) / math.pi
        end_tangent = math.atan2(
            float(-final_control[1]), float(1.0 - final_control[0])
        ) / math.pi
        query_index = query_lookup[name]
        coordinates[query_index, 4:8] = np.asarray(
            [
                raw[5],
                signed_depth_local * float(raw[4]),
                start_tangent,
                end_tangent,
            ],
            dtype=np.float32,
        )
        coordinate_mask[query_index, 4:8] = True
        applicability[query_index] = True
        presence[query_index] = 1.0
        evidence[query_index] = (
            f"{CURVE_TRUTH_DENSE_APPROXIMATION}:paired_canonical_pattern:"
            f"channels_4_to_7_only;garment_frame_endpoints_preserved;"
            f"fit_rmse_over_chord={float(formula.fit_rmse_over_chord[formula_index]):.8f}"
        )
    if not np.array_equal(coordinates[:, :4], original_endpoints, equal_nan=True):
        raise AssertionError("dense curve overlay must preserve garment-frame endpoint channels")
    output = replace(
        target,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        query_applicability=applicability,
        presence=presence,
        evidence=tuple(evidence),
        provenance_status=(
            f"{target.provenance_status}+{CURVE_TRUTH_DENSE_APPROXIMATION}"
        ),
    )
    output.validate()
    return output


def _limit_by_category(
    examples: Sequence[SemanticTrainingExample], count: int | None, *, seed: int
) -> tuple[SemanticTrainingExample, ...]:
    if count is None:
        return tuple(examples)
    output = []
    for category in CATEGORY_NAMES:
        values = [item for item in examples if item.category == category]
        values.sort(
            key=lambda item: hashlib.sha256(
                f"limit:{seed}:{item.sample_id}".encode("utf-8")
            ).digest()
        )
        output.extend(values[:count])
    return tuple(output)


def _build_basic_examples(
    config: Mapping[str, Any], corpus_path: Path | None
) -> tuple[SemanticTrainingExample, ...]:
    if corpus_path is not None:
        corpus = load_corpus_json(corpus_path)
    else:
        corpus = generate_corpus(
            int(config["basic_block_variations_per_category"]),
            seed=int(config["seed"]),
        )
    output = []
    for block in corpus.records:
        target = semantic_target_from_basic_block(block)
        example = SemanticTrainingExample(
            block.sample_id,
            block.category,
            basic_block_to_graph(block),
            target,
        )
        example.validate()
        output.append(example)
    return tuple(output)


def _gcd_training_reference_lines(
    record: Any,
) -> dict[str, tuple[tuple[str, tuple[tuple[float, float], tuple[float, float]]], ...]]:
    category = common_basic_category(record)
    output: dict[str, list[tuple[str, tuple[tuple[float, float], tuple[float, float]]]]] = {}
    if category == "tshirt":
        for panel in record.panels:
            for line in panel.reference_lines:
                if line.name.upper() not in {"BL", "WL", "HL"} or not line.training_eligible:
                    continue
                output.setdefault(panel.id, []).append(
                    (line.name.upper(), tuple(tuple(float(x) for x in point) for point in line.points_cm))
                )
    elif category in {"pants", "skirt"}:
        lower = extract_lower_body_semantics(record)
        wanted = {"WL", "HL", "GRAIN"}
        if category == "pants":
            wanted.update({"CL", "KNEE_LINE"})
        for panel in lower.panels:
            for line in panel.reference_lines:
                if line.name not in wanted or not line.available or line.points_cm is None:
                    continue
                if not line.training_eligible and line.name != "KNEE_LINE":
                    continue
                output.setdefault(panel.panel_id, []).append(
                    (line.name, tuple(tuple(float(x) for x in point) for point in line.points_cm))
                )
    return {
        panel_id: tuple(sorted(values, key=lambda item: item[0]))
        for panel_id, values in output.items()
    }


def _append_gcd_reference_line_tokens(
    graph: MultiGarmentExample, record: Any
) -> MultiGarmentExample:
    """Append exact/eligible construction lines without giving them edge labels."""

    references = _gcd_training_reference_lines(record)
    raw_panels = {panel.id: panel for panel in record.panels}
    garment_scale = max((panel.panel_scale_cm for panel in graph.panels), default=1.0)
    panels = []
    for panel in graph.panels:
        values = references.get(panel.panel_id, ())
        if not values:
            panels.append(panel)
            continue
        raw = raw_panels[panel.panel_id]
        center = np.asarray(raw.vertices_cm, dtype=np.float32).mean(axis=0)
        features = [panel.features]
        lengths = [panel.edge_lengths_cm]
        ids = list(panel.edge_ids)
        for name, points in values:
            feature, length = _construction_line_feature(
                points,
                panel_center=center,
                panel_scale=panel.panel_scale_cm,
                garment_scale=garment_scale,
            )
            features.append(feature[None, :])
            lengths.append(np.asarray([length], dtype=np.float32))
            ids.append(f"@reference:{name}")
        count = len(values)
        panels.append(
            replace(
                panel,
                features=np.concatenate(features, axis=0),
                edge_targets=np.concatenate(
                    (panel.edge_targets, np.full(count, -100, dtype=np.int64))
                ),
                edge_lengths_cm=np.concatenate(lengths),
                edge_ids=tuple(ids),
            )
        )
    return replace(graph, panels=tuple(panels))


def _build_gcd_examples(
    *,
    records_path: Path,
    index_path: Path,
    global_features_path: Path,
    spatial_features_path: Path,
    mode: str,
) -> tuple[SemanticTrainingExample, ...]:
    records = filter_common_basic_records(read_records(records_path))
    record_by_id = {record.sample_id: record for record in records}
    graph_by_id = {}
    for graph in read_gcd_multigarment_examples(records_path):
        if graph.sample_id not in record_by_id:
            continue
        record = record_by_id[graph.sample_id]
        roles = resolved_common_basic_edge_roles(record)
        panels = []
        for panel in graph.panels:
            targets = np.asarray(
                [
                    -100
                    if roles.get(edge_id) is None
                    else MULTIGARMENT_EDGE_ROLES.index(str(roles[edge_id]))
                    for edge_id in panel.edge_ids
                ],
                dtype=np.int64,
            )
            panels.append(replace(panel, edge_targets=targets))
        resolved_graph = replace(graph, panels=tuple(panels))
        graph_by_id[graph.sample_id] = _append_gcd_reference_line_tokens(
            resolved_graph, record
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_rows = {str(row["sample_id"]): row for row in index_payload["records"]}
    indexed_ids = set(indexed_rows)

    global_lookup: dict[str, int] = {}
    global_values = None
    spatial_lookup: dict[str, int] = {}
    spatial_values = None
    if mode in {"global", "global+spatial"}:
        ids, global_values = _load_feature_archive(global_features_path, 3)
        global_lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    if mode in {"spatial", "global+spatial"}:
        ids, spatial_values = _load_feature_archive(spatial_features_path, 4)
        spatial_lookup = {sample_id: index for index, sample_id in enumerate(ids)}

    output = []
    for sample_id, record in sorted(record_by_id.items()):
        if sample_id not in indexed_ids or sample_id not in graph_by_id:
            continue
        if global_values is not None and sample_id not in global_lookup:
            continue
        if spatial_values is not None and sample_id not in spatial_lookup:
            continue
        target = semantic_target_from_drafting_record(record, require_common_basic=True)
        canonical_path = Path(str(indexed_rows[sample_id]["source_pattern"]))
        if not canonical_path.is_file():
            raise FileNotFoundError(
                f"paired canonical pattern is required for dense curve overlay: {sample_id}"
            )
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        target = overlay_dense_upper_curve_targets(target, record, canonical)
        example = SemanticTrainingExample(
            sample_id=sample_id,
            category=target.category,
            graph=graph_by_id[sample_id],
            target=target,
            global_features=(
                None if global_values is None else global_values[global_lookup[sample_id]]
            ),
            spatial_features=(
                None if spatial_values is None else spatial_values[spatial_lookup[sample_id]]
            ),
        )
        example.validate()
        output.append(example)
    if not output:
        raise ValueError("no strict common GCD records have paired requested visual features")
    return tuple(output)


def _partition(
    examples: Sequence[SemanticTrainingExample], split: Mapping[str, str]
) -> dict[str, list[SemanticTrainingExample]]:
    return {
        name: [item for item in examples if split[item.sample_id] == name]
        for name in ("train", "validation", "test")
    }


def _count_categories(examples: Sequence[SemanticTrainingExample]) -> dict[str, int]:
    return {category: sum(item.category == category for item in examples) for category in CATEGORY_NAMES}


def deterministic_category_oversample(
    examples: Sequence[SemanticTrainingExample], *, seed: int
) -> list[SemanticTrainingExample]:
    """Balance only a training partition; validation/test stay natural."""

    grouped = {
        category: [item for item in examples if item.category == category]
        for category in CATEGORY_NAMES
    }
    if any(not values for values in grouped.values()):
        raise ValueError("category-balanced training requires every category")
    maximum = max(len(values) for values in grouped.values())
    output: list[SemanticTrainingExample] = []
    for category, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"oversample:{seed}:{category}:{item.sample_id}".encode("utf-8")
            ).digest(),
        )
        output.extend(ordered[index % len(ordered)] for index in range(maximum))
    return output


def _prediction_rows(
    examples: Sequence[SemanticTrainingExample],
    probabilities: np.ndarray,
    coordinates: np.ndarray,
    coordinate_confidence_calibration: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    reliability = (
        coordinate_confidence_calibration.get("per_query", {})
        if coordinate_confidence_calibration is not None
        else {}
    )
    rows = []
    for row, example in enumerate(examples):
        values = []
        for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
            if query.category != example.category:
                continue
            active = example.target.coordinate_mask[index]
            presence_probability = float(probabilities[row, index])
            values.append(
                {
                    "query": query.key,
                    "presence_probability": presence_probability,
                    "coordinate_confidence": min(
                        presence_probability, float(reliability.get(query.key, 0.0))
                    ),
                    # Predictions follow the static query schema and are never
                    # hidden using a ground-truth mask unavailable at inference.
                    "predicted_coordinates": {
                        name: float(coordinates[row, index, channel])
                        for channel, name in enumerate(query.coordinate_names)
                    },
                    "query_supervised_in_ground_truth": bool(
                        example.target.query_applicability[index]
                    ),
                    "target_present": (
                        bool(example.target.presence[index] > 0.5)
                        if example.target.query_applicability[index]
                        else None
                    ),
                    "target_coordinates": {
                        name: (
                            float(example.target.coordinates[index, channel])
                            if active[channel]
                            else None
                        )
                        for channel, name in enumerate(query.coordinate_names)
                    },
                    "coordinate_supervision_mask": {
                        name: bool(active[channel])
                        for channel, name in enumerate(query.coordinate_names)
                    },
                }
            )
        rows.append({"sample_id": example.sample_id, "category": example.category, "queries": values})
    return rows


def coordinate_confidence_from_validation(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build a fail-closed same-domain reliability hook from validation MAE."""

    per_query: dict[str, float] = {}
    for query in SEMANTIC_QUERY_INVENTORY:
        row = (
            metrics.get("per_category", {})
            .get(query.category, {})
            .get("queries", {})
            .get(query.key)
        )
        if not row or int(row.get("coordinate_channel_support", 0)) <= 0:
            per_query[query.key] = 0.0
            continue
        mae = row.get("coordinate_normalized_mae")
        presence_f1 = row.get("presence_f1")
        if mae is None or presence_f1 is None:
            per_query[query.key] = 0.0
            continue
        per_query[query.key] = float(
            np.clip(math.exp(-float(mae) / 0.15) * float(presence_f1), 0.0, 1.0)
        )
    return {
        "schema_version": "semantic-coordinate-confidence/v1",
        "method": "validation_per_query_reliability",
        "per_query": per_query,
        "fallback": "FAIL_CLOSED",
        "claim_boundary": (
            "Heuristic same-generator validation reliability: exp(-normalized_MAE/0.15) "
            "times presence F1. It is not calibrated probability, cross-source evidence, "
            "or CAD validity. Queries without coordinate support receive zero."
        ),
    }


def semantic_edit_calibration_from_validation(
    metrics: Mapping[str, Any],
    examples: Sequence[SemanticTrainingExample],
    coordinate_confidence_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose student residuals only where they beat the default anchor.

    The comparison uses only the already-frozen validation partition.  It is
    deliberately query-level rather than sample-level, so inference cannot
    inspect the unknown target pattern.  Anchor-preferred queries receive a
    retention weight used by semantic projection to reject collateral edits.
    """

    reliability = coordinate_confidence_calibration.get("per_query", {})
    if not isinstance(reliability, Mapping):
        raise ValueError("coordinate confidence calibration requires per_query mapping")
    anchors = {
        category: semantic_target_from_basic_block(build_basic_block(category))
        for category in CATEGORY_NAMES
    }
    rows: dict[str, dict[str, Any]] = {}
    minimum_support = 8
    for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
        anchor = anchors[query.category]
        absolute_errors: list[float] = []
        for example in examples:
            if example.category != query.category:
                continue
            mask = example.target.coordinate_mask[index] & anchor.coordinate_mask[index]
            if bool(mask.any()):
                absolute_errors.extend(
                    np.abs(
                        example.target.coordinates[index]
                        - anchor.coordinates[index]
                    )[mask].tolist()
                )
        anchor_mae = (
            float(np.mean(absolute_errors)) if absolute_errors else None
        )
        metric_row = (
            metrics.get("per_category", {})
            .get(query.category, {})
            .get("queries", {})
            .get(query.key)
        )
        student_mae = (
            float(metric_row["coordinate_normalized_mae"])
            if metric_row is not None
            and metric_row.get("coordinate_normalized_mae") is not None
            else None
        )
        support = len(absolute_errors)
        editable_kind = query.kind in {"landmark", "path"}
        allow_student_edit = bool(
            editable_kind
            and support >= minimum_support
            and anchor_mae is not None
            and student_mae is not None
            and student_mae < anchor_mae
        )
        validation_advantage = (
            None
            if anchor_mae is None or student_mae is None
            else anchor_mae - student_mae
        )
        query_reliability = float(reliability.get(query.key, 0.0))
        if (
            support >= minimum_support
            and anchor_mae is not None
            and student_mae is not None
            and student_mae > anchor_mae
        ):
            relative_anchor_advantage = (student_mae - anchor_mae) / max(
                student_mae, 1e-8
            )
            anchor_retention_weight = float(
                np.clip(query_reliability * relative_anchor_advantage, 0.0, 1.0)
            )
        else:
            anchor_retention_weight = 0.0
        rows[query.key] = {
            "kind": query.kind,
            "coordinate_channel_support": support,
            "minimum_support": minimum_support,
            "anchor_validation_mae": anchor_mae,
            "student_validation_mae": student_mae,
            "student_validation_advantage": validation_advantage,
            "allow_student_edit": allow_student_edit,
            "anchor_retention_weight": anchor_retention_weight,
        }
    return {
        "schema_version": EDIT_CALIBRATION_SCHEMA_VERSION,
        "method": "student_vs_default_anchor_validation_mae",
        "query_keys": [query.key for query in SEMANTIC_QUERY_INVENTORY],
        "per_query": rows,
        "fallback": "FAIL_CLOSED",
        "validation_partition_only": True,
        "test_ground_truth_used": False,
        "claim_boundary": (
            "Same-generator validation selection between a visual student and a "
            "PROVISIONAL_EXPERT_REVIEW default anchor. It is not cross-source, "
            "family-disjoint, industrial CAD, or expert-approved evidence."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train common-block vector teacher and image-only four-view student."
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("benchmark/configs/basic_semantic_teacher_student.json"),
    )
    parser.add_argument(
        "--records", type=Path,
        default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"),
    )
    parser.add_argument(
        "--index", type=Path,
        default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"),
    )
    parser.add_argument(
        "--global-features", type=Path,
        default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"),
    )
    parser.add_argument(
        "--spatial-features", type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/resnet50_fpn_tokens.npz"),
    )
    parser.add_argument("--basic-block-corpus", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/drafting_semantics/basic_semantic_teacher_student"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/drafting_semantics/basic_semantic_teacher_student"),
    )
    parser.add_argument(
        "--visual-feature-mode", "--feature-mode", dest="feature_mode",
        choices=("global", "spatial", "global+spatial"),
    )
    parser.add_argument("--epochs", type=int, help="Override both teacher and student epochs.")
    parser.add_argument("--max-samples-per-category", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "basic-semantic-teacher-student/v1":
        raise ValueError("unsupported training config schema")
    if args.feature_mode:
        config["visual_feature_mode"] = args.feature_mode
    if args.epochs is not None:
        config["teacher"]["epochs"] = args.epochs
        config["student"]["epochs"] = args.epochs
    limit = args.max_samples_per_category
    if args.smoke:
        config["basic_block_variations_per_category"] = 12
        config["teacher"]["epochs"] = 1 if args.epochs is None else args.epochs
        config["student"]["epochs"] = 1 if args.epochs is None else args.epochs
        config["teacher"]["batch_size"] = 8
        config["student"]["batch_size"] = 8
        limit = 12 if limit is None else min(limit, 12)
        if args.feature_mode is None:
            # The full experiment defaults to spatial FPN.  Smoke deliberately
            # uses the 34 MB global archive so it does not materialize the
            # optional ~1.5 GB FPN feature file.
            config["visual_feature_mode"] = "global"
    mode = str(config["visual_feature_mode"])
    if mode not in {"global", "spatial", "global+spatial"}:
        raise ValueError(f"unsupported visual feature mode: {mode}")
    config["model"]["edge_feature_dim"] = EDGE_FEATURE_DIM
    config["model"]["panel_role_count"] = len(MULTIGARMENT_PANEL_ROLES)
    config["model"]["edge_role_count"] = len(MULTIGARMENT_EDGE_ROLES)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    basic_examples = _build_basic_examples(config, args.basic_block_corpus)
    gcd_examples = _build_gcd_examples(
        records_path=args.records,
        index_path=args.index,
        global_features_path=args.global_features,
        spatial_features_path=args.spatial_features,
        mode=mode,
    )
    padding_limits = {
        "maximum_panels": int(config["padding"]["maximum_panels"]),
        "maximum_edges": int(config["padding"]["maximum_edges"]),
    }
    padding_audit = {
        "provisional_basic_blocks": graph_padding_audit(
            basic_examples, **padding_limits
        ),
        "strict_common_gcdv2": graph_padding_audit(
            gcd_examples, **padding_limits
        ),
        "combined": graph_padding_audit(
            (*basic_examples, *gcd_examples), **padding_limits
        ),
    }
    gcd_examples = _limit_by_category(gcd_examples, limit, seed=seed)
    basic_split = deterministic_category_split(
        [item.sample_id for item in basic_examples],
        [item.category for item in basic_examples],
        seed=seed,
        fractions=config["split_fractions"],
    )
    gcd_split = deterministic_category_split(
        [item.sample_id for item in gcd_examples],
        [item.category for item in gcd_examples],
        seed=seed + 1,
        fractions=config["split_fractions"],
    )
    basic_partitions = _partition(basic_examples, basic_split)
    gcd_partitions = _partition(gcd_examples, gcd_split)

    calibration_config = config["coordinate_calibration"]
    calibrator = TrainOnlyCoordinateCalibrator.fit(
        [item.target for item in gcd_partitions["train"]],
        [item.target for item in basic_partitions["train"]],
        minimum_support=int(calibration_config["minimum_support"]),
        minimum_scale=float(calibration_config["minimum_scale"]),
        clamp_standard_deviations=float(calibration_config["clamp_standard_deviations"]),
    )
    gcd_partitions = {
        name: [replace(item, target=calibrator.transform(item.target)) for item in values]
        for name, values in gcd_partitions.items()
    }
    balanced_gcd_train = deterministic_category_oversample(
        gcd_partitions["train"], seed=seed + 43
    )
    teacher_partitions = {
        name: [
            *basic_partitions[name],
            *(balanced_gcd_train if name == "train" else gcd_partitions[name]),
        ]
        for name in ("train", "validation", "test")
    }
    if any(not teacher_partitions[name] for name in teacher_partitions):
        raise ValueError("teacher split contains an empty partition")
    if any(not gcd_partitions[name] for name in gcd_partitions):
        raise ValueError("visual GCD split contains an empty partition")

    # Feature dimensions are inferred from the actually selected archive and
    # validated by the student.  The unselected FPN archive was never opened.
    first_visual = gcd_examples[0]
    if first_visual.global_features is not None:
        config["model"]["global_feature_dim"] = int(first_visual.global_features.shape[-1])
    if first_visual.spatial_features is not None:
        config["model"]["spatial_feature_dim"] = int(first_visual.spatial_features.shape[-1])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    teacher, teacher_run = _train_teacher(
        teacher_partitions["train"], teacher_partitions["validation"], config, device
    )
    teacher_validation, _, _ = _evaluate_teacher(
        teacher, teacher_partitions["validation"], config, device
    )
    teacher_test, _, _ = _evaluate_teacher(
        teacher, teacher_partitions["test"], config, device
    )
    teacher_basic_test, _, _ = _evaluate_teacher(
        teacher, basic_partitions["test"], config, device
    )
    teacher_gcd_test, _, _ = _evaluate_teacher(
        teacher, gcd_partitions["test"], config, device
    )
    student, student_run = _train_student(
        teacher, balanced_gcd_train, gcd_partitions["validation"], config, device
    )
    student_validation, _, _ = _evaluate_student(
        student, gcd_partitions["validation"], config, device
    )
    coordinate_confidence_calibration = coordinate_confidence_from_validation(
        student_validation
    )
    semantic_edit_calibration = semantic_edit_calibration_from_validation(
        student_validation,
        gcd_partitions["validation"],
        coordinate_confidence_calibration,
    )
    student_test, test_probabilities, test_coordinates = _evaluate_student(
        student, gcd_partitions["test"], config, device
    )
    baseline = CategoryMeanBaseline.fit([item.target for item in gcd_partitions["train"]])
    baseline_presence, baseline_coordinates = baseline.predict(
        [item.target for item in gcd_partitions["test"]]
    )
    baseline_test = semantic_metrics(
        baseline_presence,
        baseline_coordinates,
        [item.target for item in gcd_partitions["test"]],
    )
    leave_one_view_out = {
        name: _evaluate_student(student, gcd_partitions["test"], config, device, missing_view=index)[0]
        for index, name in enumerate(("front", "back", "left", "right"))
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "vector_graph_teacher",
            "model_state": teacher_run.pop("model_state"),
            "model_config": config["model"],
            "query_keys": [query.key for query in SEMANTIC_QUERY_INVENTORY],
            "semantic_query_schema_version": SEMANTIC_QUERY_SCHEMA_VERSION,
            "calibration": calibrator.to_dict(),
            "provenance_status": "MIXED_SOURCE_WITH_PROVISIONAL_EXPERT_REVIEW_REFERENCE",
        },
        args.checkpoint_dir / "teacher.pt",
    )
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "four_view_student",
            "model_state": student_run.pop("model_state"),
            "model_config": config["model"],
            "visual_feature_mode": mode,
            "query_keys": [query.key for query in SEMANTIC_QUERY_INVENTORY],
            "semantic_query_schema_version": SEMANTIC_QUERY_SCHEMA_VERSION,
            "inference_contract": "four_view_features_plus_category_only_no_pattern_graph",
            "coordinate_confidence_calibration": coordinate_confidence_calibration,
            "semantic_edit_calibration": semantic_edit_calibration,
        },
        args.checkpoint_dir / "student.pt",
    )
    predictions_path = args.output_dir / "test_predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "semantic_query_schema_version": SEMANTIC_QUERY_SCHEMA_VERSION,
                "contains_source_images": False,
                "contains_source_paths": False,
                "coordinate_confidence_calibration": coordinate_confidence_calibration,
                "semantic_edit_calibration": semantic_edit_calibration,
                "rows": _prediction_rows(
                    gcd_partitions["test"],
                    test_probabilities,
                    test_coordinates,
                    coordinate_confidence_calibration,
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantic_query_schema_version": SEMANTIC_QUERY_SCHEMA_VERSION,
        "status": (
            "PASS_PIPELINE_SMOKE_ONLY_NO_PERFORMANCE_CLAIM"
            if args.smoke
            else "COMPLETE_BOUNDED_SAME_GENERATOR_TEACHER_STUDENT_EXPERIMENT"
        ),
        "run_mode": "smoke" if args.smoke else "full",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "visual_feature_mode": mode,
        "data": {
            "basic_block_provenance": "PROVISIONAL_EXPERT_REVIEW",
            "gcd_source": "GarmentCodeData_v2_strict_common_basic_top_pants_skirt",
            "split_contract": "deterministic_category_stratified_sample_id_unseen_not_family_disjoint",
            "graph_padding_audit": padding_audit,
            "basic_split_counts": {
                name: {"total": len(values), "per_category": _count_categories(values)}
                for name, values in basic_partitions.items()
            },
            "gcd_split_counts": {
                name: {"total": len(values), "per_category": _count_categories(values)}
                for name, values in gcd_partitions.items()
            },
            "gcd_effective_balanced_train_counts": {
                "total": len(balanced_gcd_train),
                "per_category": _count_categories(balanced_gcd_train),
                "method": "deterministic_category_oversample_train_only",
                "validation_and_test_remain_natural": True,
            },
        },
        "coordinate_calibration": calibrator.to_dict(),
        "teacher": {
            **teacher_run,
            "validation": teacher_validation,
            "test_mixed_provisional_and_gcd": teacher_test,
            "test_provisional_basic_blocks": teacher_basic_test,
            "test_calibrated_gcd_same_generator_unseen_ids": teacher_gcd_test,
        },
        "student": {
            **student_run,
            "validation": student_validation,
            "test_same_generator_unseen_ids": student_test,
            "category_mean_train_only_baseline": baseline_test,
            "leave_one_view_out": leave_one_view_out,
            "coordinate_confidence_calibration": coordinate_confidence_calibration,
            "semantic_edit_calibration": semantic_edit_calibration,
        },
        "artifacts": {
            "teacher_checkpoint": str((args.checkpoint_dir / "teacher.pt").as_posix()),
            "student_checkpoint": str((args.checkpoint_dir / "student.pt").as_posix()),
            "numeric_predictions": str(predictions_path.as_posix()),
            "source_images_copied": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": str(metrics_path),
                "teacher_test": {
                    "presence_macro_f1": teacher_test["presence_macro_f1"],
                    "coordinate_normalized_mae": teacher_test["coordinate_normalized_mae"],
                    "panel_role_macro_f1": teacher_test["panel_roles"]["macro_f1"],
                    "edge_role_macro_f1": teacher_test["edge_roles"]["macro_f1"],
                },
                "student_test": {
                    "presence_macro_f1": student_test["presence_macro_f1"],
                    "coordinate_normalized_mae": student_test["coordinate_normalized_mae"],
                },
                "category_mean_baseline": {
                    "presence_macro_f1": baseline_test["presence_macro_f1"],
                    "coordinate_normalized_mae": baseline_test["coordinate_normalized_mae"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
