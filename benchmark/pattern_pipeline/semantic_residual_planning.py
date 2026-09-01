"""Turn semantic teacher/student predictions into safe basic-block edits.

The four-view student does not emit CAD.  It emits the same fixed semantic
query table as the 2D-pattern teacher.  This module is the deliberately small
adapter between that table and :mod:`semantic_editing`::

    predicted semantic coordinates - anchor semantic coordinates
        -> bounded named landmark/path residuals
        -> topology-preserving edit of an already valid basic block

There is no topology creation here.  A query must be present in both the
anchor target and the prediction, have sufficient confidence, and be exposed
by the anchor's ``semantic_landmarks``/``semantic_paths`` annotations.  This
is especially important for slits and closures: a visual presence prediction
alone can never create one.

The eight path channels are interpreted according to
``semantic_teacher_student.PATH_COORDINATES``.  Endpoint coordinates control
the chord scale and normal translation.  Signed depth, relative arc length,
and endpoint tangent spread jointly provide a conservative normal scale.  A
``PathResidual`` cannot reproduce arbitrary Bezier controls, so this is an
auditable bounded approximation rather than a claim of exact CAD recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import numpy as np

from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    PATH_COORDINATES,
    SEMANTIC_QUERY_INVENTORY,
)

from .schema import PatternDocument
from .semantic_editing import (
    LandmarkResidual,
    PathResidual,
    SemanticResidualPlan,
    semantic_annotation_entries,
)


if PATH_COORDINATES != (
    "start_u",
    "start_v",
    "end_u",
    "end_v",
    "arc_length_norm",
    "signed_depth_norm",
    "start_tangent_angle_norm",
    "end_tangent_angle_norm",
):
    raise RuntimeError("semantic residual planner is out of sync with PATH_COORDINATES")


@dataclass(frozen=True)
class ResidualPlanningConfig:
    """Safety bounds for converting normalized predictions to centimetres."""

    presence_threshold: float = 0.50
    confidence_threshold: float = 0.55
    max_landmark_displacement_cm: float = 5.0
    landmark_influence_radius_cm: float = 8.0
    min_chord_scale: float = 0.78
    max_chord_scale: float = 1.22
    min_normal_scale: float = 0.60
    max_normal_scale: float = 1.55
    max_normal_offset_cm: float = 3.0
    minimum_path_chord_cm: float = 0.25
    minimum_depth_cm: float = 0.20
    sleeve_ease_ratio_min: float = 0.98
    sleeve_ease_ratio_max: float = 1.08
    identity_tolerance: float = 1e-5

    def validate(self) -> None:
        finite = (
            self.presence_threshold,
            self.confidence_threshold,
            self.max_landmark_displacement_cm,
            self.landmark_influence_radius_cm,
            self.min_chord_scale,
            self.max_chord_scale,
            self.min_normal_scale,
            self.max_normal_scale,
            self.max_normal_offset_cm,
            self.minimum_path_chord_cm,
            self.minimum_depth_cm,
            self.sleeve_ease_ratio_min,
            self.sleeve_ease_ratio_max,
            self.identity_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("residual planning configuration must be finite")
        if not 0.0 <= self.presence_threshold <= 1.0:
            raise ValueError("presence_threshold must be in [0, 1]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_landmark_displacement_cm <= 0.0:
            raise ValueError("max_landmark_displacement_cm must be positive")
        if self.landmark_influence_radius_cm <= 0.0:
            raise ValueError("landmark_influence_radius_cm must be positive")
        if not 0.5 <= self.min_chord_scale <= self.max_chord_scale <= 1.5:
            raise ValueError("chord bounds must lie inside PathResidual [0.5, 1.5]")
        if not 0.25 <= self.min_normal_scale <= self.max_normal_scale <= 2.0:
            raise ValueError("normal bounds must lie inside PathResidual [0.25, 2.0]")
        if self.max_normal_offset_cm < 0.0:
            raise ValueError("max_normal_offset_cm must be non-negative")
        if self.minimum_path_chord_cm <= 0.0 or self.minimum_depth_cm <= 0.0:
            raise ValueError("minimum path dimensions must be positive")
        if not 0.0 < self.sleeve_ease_ratio_min <= self.sleeve_ease_ratio_max:
            raise ValueError("sleeve ease ratio bounds are invalid")
        if self.identity_tolerance < 0.0:
            raise ValueError("identity_tolerance must be non-negative")


@dataclass(frozen=True)
class _PathEstimate:
    residual: PathResidual
    anchor_arc: float
    predicted_arc: float
    confidence: float


def _as_document(anchor: Any) -> PatternDocument:
    if isinstance(anchor, PatternDocument):
        return anchor
    converter = getattr(anchor, "to_pattern_document", None)
    if callable(converter):
        document = converter()
        if isinstance(document, PatternDocument):
            return document
    raise TypeError("anchor must be a PatternDocument or expose to_pattern_document()")


def _as_vector(values: Any, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    if output.ndim == 2 and output.shape[0] == 1:
        output = output[0]
    expected = (len(SEMANTIC_QUERY_INVENTORY),)
    if output.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, received {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must contain finite probabilities")
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1], not logits")
    return output


def _as_coordinates(values: Any, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    expected = (len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM)
    if output.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, received {output.shape}")
    # NaN is the explicit representation for an absent/unavailable target.
    if np.any(np.isinf(output)):
        raise ValueError(f"{name} must not contain infinite coordinates")
    return output


def _document_scales(document: PatternDocument) -> tuple[float, float, float]:
    points = [
        point
        for panel in document.panels
        for edge in panel.edges
        for point in edge.points
    ]
    if not points:
        raise ValueError("anchor PatternDocument has no geometry")
    values = np.asarray(points, dtype=np.float64)
    width = max(float(np.ptp(values[:, 0])), 1.0)
    height = max(float(np.ptp(values[:, 1])), 1.0)
    return width, height, max(width, height)


def _source_y_axis_down(document: PatternDocument) -> bool:
    raw = document.annotations.get("semantic_coordinate_frame", {})
    if not isinstance(raw, Mapping):
        raise ValueError("semantic_coordinate_frame must be a mapping when present")
    value = raw.get("source_y_axis_down", False)
    if not isinstance(value, bool):
        raise ValueError("semantic_coordinate_frame.source_y_axis_down must be boolean")
    return value


def _declared_anchor_presence(document: PatternDocument, kind: str, name: str) -> bool:
    # A reviewed role-specific dart specialization can be derived from the
    # legacy front/back entries even when the legacy explicit map says the new
    # exact query name is absent.
    if semantic_annotation_entries(document, kind, name):
        return True
    values = document.annotations.get("semantic_query_presence")
    if values is None:
        return True
    if not isinstance(values, Mapping):
        raise ValueError("semantic_query_presence must be a mapping when present")
    return bool(values.get(name, False))


def _query_confidence(
    index: int,
    anchor_presence: np.ndarray,
    predicted_presence: np.ndarray,
    predicted_confidence: np.ndarray,
    config: ResidualPlanningConfig,
) -> float | None:
    if anchor_presence[index] < config.presence_threshold:
        return None
    if predicted_presence[index] < config.presence_threshold:
        return None
    confidence = min(
        float(anchor_presence[index]),
        float(predicted_presence[index]),
        float(predicted_confidence[index]),
    )
    return confidence if confidence >= config.confidence_threshold else None


def _coordinates_request_change(
    anchor: np.ndarray,
    predicted: np.ndarray,
    channel_count: int,
    tolerance: float,
) -> bool:
    anchor_values = anchor[:channel_count]
    predicted_values = predicted[:channel_count]
    if not np.all(np.isfinite(anchor_values)) or not np.all(np.isfinite(predicted_values)):
        return False
    return bool(np.max(np.abs(predicted_values - anchor_values)) > tolerance)


def _bounded_vector(dx: float, dy: float, limit: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= limit or length <= 1e-12:
        return dx, dy
    scale = limit / length
    return dx * scale, dy * scale


def _angle_distance(first: float, second: float) -> float:
    """Shortest distance for angles normalized to one half-turn in [-1, 1]."""

    return abs((float(first) - float(second) + 1.0) % 2.0 - 1.0)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if numerator <= 1e-8 or denominator <= 1e-8:
        return None
    return numerator / denominator


def _path_estimate(
    anchor: np.ndarray,
    predicted: np.ndarray,
    confidence: float,
    width_cm: float,
    height_cm: float,
    scale_cm: float,
    source_y_axis_down: bool,
    config: ResidualPlanningConfig,
) -> _PathEstimate | None:
    if not np.all(np.isfinite(anchor[: len(PATH_COORDINATES)])):
        return None
    if not np.all(np.isfinite(predicted[: len(PATH_COORDINATES)])):
        return None

    scale = np.asarray((width_cm, height_cm), dtype=np.float64)
    anchor_start = anchor[0:2] * scale
    anchor_end = anchor[2:4] * scale
    predicted_start = predicted[0:2] * scale
    predicted_end = predicted[2:4] * scale
    anchor_chord = anchor_end - anchor_start
    predicted_chord = predicted_end - predicted_start
    anchor_chord_length = float(np.linalg.norm(anchor_chord))
    predicted_chord_length = float(np.linalg.norm(predicted_chord))
    if anchor_chord_length < config.minimum_path_chord_cm:
        return None

    raw_chord_scale = predicted_chord_length / anchor_chord_length
    chord_scale = float(
        np.clip(raw_chord_scale, config.min_chord_scale, config.max_chord_scale)
    )
    tangent = anchor_chord / anchor_chord_length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    midpoint_delta = 0.5 * (
        predicted_start + predicted_end - anchor_start - anchor_end
    )
    normal_offset = float(midpoint_delta @ normal)

    anchor_depth_cm = float(anchor[5]) * scale_cm
    predicted_depth_cm = float(predicted[5]) * scale_cm
    shape_factors: list[tuple[float, float]] = []
    if abs(anchor_depth_cm) >= config.minimum_depth_cm:
        # A sign flip cannot be represented safely by a positive normal scale.
        # Retain magnitude information but heavily down-weight it.
        same_side = anchor_depth_cm * predicted_depth_cm >= 0.0
        depth_factor = max(abs(predicted_depth_cm) / abs(anchor_depth_cm), 0.05)
        shape_factors.append((depth_factor, 0.50 if same_side else 0.15))
    elif abs(predicted_depth_cm) >= config.minimum_depth_cm:
        # A straight anchor cannot acquire curvature through normal scaling.
        # A small bounded normal translation is the only topology-safe proxy.
        normal_offset += predicted_depth_cm - anchor_depth_cm

    anchor_arc = float(anchor[4])
    predicted_arc = float(predicted[4])
    arc_ratio = _safe_ratio(predicted_arc, anchor_arc)
    if arc_ratio is not None:
        shape_factors.append((arc_ratio / max(raw_chord_scale, 1e-8), 0.35))

    anchor_turn = _angle_distance(anchor[7], anchor[6])
    predicted_turn = _angle_distance(predicted[7], predicted[6])
    # Tangent spread is intentionally a small correction.  The residual
    # editor cannot rotate individual Bezier handles.
    tangent_factor = 1.0 + 0.45 * (predicted_turn - anchor_turn)
    shape_factors.append((max(tangent_factor, 0.05), 0.15))

    weight_sum = sum(weight for _, weight in shape_factors)
    if weight_sum:
        # Geometric averaging treats reciprocal widen/narrow changes evenly.
        log_scale = sum(
            weight * math.log(max(value, 1e-6))
            for value, weight in shape_factors
        ) / weight_sum
        normal_scale = math.exp(log_scale)
    else:
        normal_scale = 1.0
    normal_scale = float(
        np.clip(normal_scale, config.min_normal_scale, config.max_normal_scale)
    )
    normal_offset = float(
        np.clip(
            normal_offset,
            -config.max_normal_offset_cm,
            config.max_normal_offset_cm,
        )
    )
    # Canonical targets use y-up (bottom=0, top=1).  Reflection into a
    # y-down drafting document reverses the chord-local left normal.
    if source_y_axis_down:
        normal_offset = -normal_offset
    return _PathEstimate(
        residual=PathResidual(
            chord_scale=chord_scale,
            normal_scale=normal_scale,
            normal_offset_cm=normal_offset,
            confidence=confidence,
        ),
        anchor_arc=anchor_arc,
        predicted_arc=predicted_arc,
        confidence=confidence,
    )


def _is_identity_path(residual: PathResidual, tolerance: float) -> bool:
    return (
        abs(residual.chord_scale - 1.0) <= tolerance
        and abs(residual.normal_scale - 1.0) <= tolerance
        and abs(residual.normal_offset_cm) <= tolerance
    )


def _correct_tshirt_sleeve_compatibility(
    estimates: Mapping[str, _PathEstimate],
    config: ResidualPlanningConfig,
) -> PathResidual | None:
    required = ("front_armhole", "back_armhole", "sleeve_head")
    if any(name not in estimates for name in required):
        return None
    front, back, sleeve = (estimates[name] for name in required)
    predicted_armholes = front.predicted_arc + back.predicted_arc
    if predicted_armholes <= 1e-8 or sleeve.predicted_arc <= 1e-8:
        return None
    raw_ease = sleeve.predicted_arc / predicted_armholes
    bounded_ease = float(
        np.clip(
            raw_ease,
            config.sleeve_ease_ratio_min,
            config.sleeve_ease_ratio_max,
        )
    )
    correction = bounded_ease / raw_ease
    if abs(correction - 1.0) <= config.identity_tolerance:
        return sleeve.residual

    # Keep sleeve-head endpoints (bicep width) controlled by their predicted
    # chord and spend the compatibility correction on cap depth/curvature.
    # This is only an approximation to arc-length matching, bounded so a bad
    # visual prediction cannot invert or explode the sleeve cap.
    corrected_normal = float(
        np.clip(
            sleeve.residual.normal_scale * correction,
            config.min_normal_scale,
            config.max_normal_scale,
        )
    )
    return replace(
        sleeve.residual,
        normal_scale=corrected_normal,
        confidence=min(front.confidence, back.confidence, sleeve.confidence),
    )


def build_semantic_residual_plan(
    category: str,
    anchor_query_coordinates: Any,
    anchor_query_presence: Any,
    predicted_query_coordinates: Any,
    predicted_query_presence: Any,
    predicted_query_confidence: Any,
    anchor: Any,
    *,
    config: ResidualPlanningConfig | None = None,
) -> SemanticResidualPlan:
    """Build a bounded residual plan from a shared semantic query table.

    Arrays may be unbatched or have a leading singleton batch dimension.  They
    must span the full ``SEMANTIC_QUERY_INVENTORY``; absent coordinate rows may
    be NaN.  Presence and confidence inputs are probabilities, not logits.

    ``anchor`` may be either a provisional ``BasicBlock`` or an already
    converted ``PatternDocument``.  Only exact query names exposed in its
    semantic annotations are emitted into the plan.
    """

    if category not in {query.category for query in SEMANTIC_QUERY_INVENTORY}:
        raise ValueError(f"unsupported semantic residual category: {category!r}")
    resolved = config or ResidualPlanningConfig()
    resolved.validate()
    document = _as_document(anchor)
    anchor_coordinates = _as_coordinates(
        anchor_query_coordinates, "anchor_query_coordinates"
    )
    predicted_coordinates = _as_coordinates(
        predicted_query_coordinates, "predicted_query_coordinates"
    )
    anchor_presence_values = _as_vector(anchor_query_presence, "anchor_query_presence")
    predicted_presence_values = _as_vector(
        predicted_query_presence, "predicted_query_presence"
    )
    confidence_values = _as_vector(
        predicted_query_confidence, "predicted_query_confidence"
    )
    width_cm, height_cm, scale_cm = _document_scales(document)
    source_y_down = _source_y_axis_down(document)
    landmarks: dict[str, LandmarkResidual] = {}
    path_estimates: dict[str, _PathEstimate] = {}
    gated_queries: dict[str, str] = {}
    for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
        # Panels and construction/reference lines are diagnostic prediction
        # targets, not editable boundary residuals.  Skip them before looking
        # up PatternDocument edit annotations so a line prediction can never
        # be misinterpreted as a path or landmark operation.
        if query.category != category or query.kind not in {"landmark", "path"}:
            continue
        confidence = _query_confidence(
            index,
            anchor_presence_values,
            predicted_presence_values,
            confidence_values,
            resolved,
        )
        if confidence is None or not _declared_anchor_presence(
            document, query.kind, query.name
        ):
            continue
        anchor_values = anchor_coordinates[index]
        predicted_values = predicted_coordinates[index]
        if query.kind == "landmark":
            entries = semantic_annotation_entries(document, "landmark", query.name)
            if not entries:
                continue
            instance_count = len(entries)
            if instance_count != 1:
                if _coordinates_request_change(
                    anchor_values,
                    predicted_values,
                    len(query.coordinate_names),
                    resolved.identity_tolerance,
                ):
                    gated_queries[f"landmark:{query.name}"] = (
                        f"unsupported_one_to_many_query:{instance_count}_instances"
                    )
                continue
            if not np.all(np.isfinite(anchor_values[:2])) or not np.all(
                np.isfinite(predicted_values[:2])
            ):
                continue
            dx = float(predicted_values[0] - anchor_values[0]) * width_cm
            dy = float(predicted_values[1] - anchor_values[1]) * height_cm
            if source_y_down:
                dy = -dy
            dx, dy = _bounded_vector(
                dx, dy, resolved.max_landmark_displacement_cm
            )
            if math.hypot(dx, dy) <= resolved.identity_tolerance:
                continue
            landmarks[query.name] = LandmarkResidual(
                dx_cm=dx,
                dy_cm=dy,
                influence_radius_cm=resolved.landmark_influence_radius_cm,
                confidence=confidence,
            )
        elif query.kind == "path":
            entries = semantic_annotation_entries(document, "path", query.name)
            if not entries:
                continue
            instance_count = len(entries)
            if instance_count != 1:
                if _coordinates_request_change(
                    anchor_values,
                    predicted_values,
                    len(query.coordinate_names),
                    resolved.identity_tolerance,
                ):
                    gated_queries[f"path:{query.name}"] = (
                        f"unsupported_one_to_many_query:{instance_count}_instances"
                    )
                continue
            estimate = _path_estimate(
                anchor_values,
                predicted_values,
                confidence,
                width_cm,
                height_cm,
                scale_cm,
                source_y_down,
                resolved,
            )
            if estimate is not None:
                path_estimates[query.name] = estimate

    paths = {
        name: estimate.residual
        for name, estimate in path_estimates.items()
        if not _is_identity_path(estimate.residual, resolved.identity_tolerance)
    }
    if category == "tshirt":
        corrected_sleeve = _correct_tshirt_sleeve_compatibility(
            path_estimates, resolved
        )
        if corrected_sleeve is not None:
            if _is_identity_path(corrected_sleeve, resolved.identity_tolerance):
                paths.pop("sleeve_head", None)
            else:
                paths["sleeve_head"] = corrected_sleeve

    plan = SemanticResidualPlan(
        category=category,
        landmark_residuals=landmarks,
        path_residuals=paths,
        gated_queries=gated_queries,
        source="four_view_semantic_student_bounded_anchor_residual",
    )
    plan.validate()
    return plan


def plan_semantic_residuals(*args: Any, **kwargs: Any) -> SemanticResidualPlan:
    """Alias retained for pipeline code that uses verb-first naming."""

    return build_semantic_residual_plan(*args, **kwargs)


__all__ = [
    "ResidualPlanningConfig",
    "build_semantic_residual_plan",
    "plan_semantic_residuals",
]
