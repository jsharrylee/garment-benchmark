"""Typed truth schema for construction-time tracing of a basic T-shirt.

The older drafting-semantic schema describes a finished pattern.  This module
is deliberately concerned with the *construction trace*: named points,
curves, helper geometry, and mutations such as dart insertion are connected to
the operation that produced them.  Source-specific production annotations
(for example FreeSewing notches) remain distinguishable through their
``domain``, evidence, and provenance fields.

Role strings are intentionally open-ended.  The constants below are useful
canonical spellings, not enumerations enforced by :meth:`validate`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Point2D = tuple[float, float]
JsonObject = dict[str, Any]


CANONICAL_TSHIRT_POINTS = ("FNP", "BNP", "SNP", "SP", "BP")
CANONICAL_PANEL_ROLES = ("front", "back", "sleeve", "neckband")
CANONICAL_EDGE_ROLES = (
    "neckline",
    "shoulder",
    "armhole",
    "side_seam",
    "center_front",
    "center_back",
    "hemline",
    "sleeve_head",
    "sleeve_underarm",
    "sleeve_hem",
    "neckband_attachment",
    "dart_leg",
)
CANONICAL_CURVE_KINDS = ("line", "bezier", "quadratic_bezier", "cubic_bezier", "arc")
CANONICAL_SPLITS = ("train", "validation", "test", "unseen")
DRAFTING_FORMULA_ROLES = ("neckline", "armhole", "sleeve_head")
DRAFTING_FORMULA_SCALARS = (
    "width_cm",
    "height_cm",
    "depth_cm",
    "chord_cm",
    "arc_length_cm",
)


def _required_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")


def _finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")


def _xy(value: Sequence[Any], label: str) -> Point2D:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two coordinates")
    x, y = value
    _finite_number(x, f"{label}[0]")
    _finite_number(y, f"{label}[1]")
    return float(x), float(y)


def _validate_xy(value: Any, label: str) -> None:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a two-coordinate sequence")
    _xy(value, label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _items(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _strings(value: Iterable[Any], label: str) -> tuple[str, ...]:
    result = tuple(value)
    for index, item in enumerate(result):
        _required_text(item, f"{label}[{index}]")
    return result


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    return _strings(_items(value, label), label)


def _json_object(value: Any, label: str) -> JsonObject:
    return dict(_mapping(value, label))


def _annotation_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": value.get("domain", "garmentcode_runtime"),
        "evidence": value.get("evidence", "observed_runtime"),
        "provenance": _json_object(value.get("provenance", {}), "provenance"),
        "training_eligible": bool(value.get("training_eligible", True)),
        "confidence": float(value.get("confidence", 1.0)),
    }


def _validate_annotation(
    *,
    domain: str,
    evidence: str,
    provenance: Mapping[str, Any],
    training_eligible: bool,
    confidence: float,
    label: str,
) -> None:
    _required_text(domain, f"{label}.domain")
    _required_text(evidence, f"{label}.evidence")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label}.provenance must be an object")
    if not isinstance(training_eligible, bool):
        raise ValueError(f"{label}.training_eligible must be boolean")
    _finite_number(confidence, f"{label}.confidence")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"{label}.confidence must be between 0 and 1")


def _validate_measurements(values: Mapping[str, float], label: str) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be an object")
    for name, value in values.items():
        _required_text(name, f"{label} key")
        _finite_number(value, f"{label}.{name}")


@dataclass(frozen=True)
class TracedPoint:
    """A point at the instant a drafting recipe creates it."""

    id: str
    panel_id: str
    xy_cm: Point2D
    formula: str
    canonical_name: str | None = None
    source_name: str | None = None
    measurement_inputs: dict[str, float] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    operation_id: str | None = None
    domain: str = "garmentcode_runtime"
    evidence: str = "observed_runtime"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "point.id")
        _required_text(self.panel_id, f"point {self.id}.panel_id")
        _validate_xy(self.xy_cm, f"point {self.id}.xy_cm")
        _required_text(self.formula, f"point {self.id}.formula")
        if self.canonical_name is None and self.source_name is None:
            raise ValueError(f"point {self.id} requires canonical_name or source_name")
        if self.canonical_name is not None:
            _required_text(self.canonical_name, f"point {self.id}.canonical_name")
        if self.source_name is not None:
            _required_text(self.source_name, f"point {self.id}.source_name")
        _validate_measurements(self.measurement_inputs, f"point {self.id}.measurement_inputs")
        _strings(self.dependencies, f"point {self.id}.dependencies")
        if self.operation_id is not None:
            _required_text(self.operation_id, f"point {self.id}.operation_id")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"point {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TracedPoint":
        value = _mapping(value, "point")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            xy_cm=_xy(_items(value["xy_cm"], "point.xy_cm"), "point.xy_cm"),
            formula=str(value["formula"]),
            canonical_name=None if value.get("canonical_name") is None else str(value["canonical_name"]),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            measurement_inputs={
                str(name): float(raw)
                for name, raw in _mapping(value.get("measurement_inputs", {}), "point.measurement_inputs").items()
            },
            dependencies=_string_tuple(value.get("dependencies", ()), "point.dependencies"),
            operation_id=None if value.get("operation_id") is None else str(value["operation_id"]),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class CurveGeometry:
    """Lossless numerical geometry for a line, Bezier, arc, or extension."""

    kind: str
    start_cm: Point2D
    end_cm: Point2D
    control_points_cm: tuple[Point2D, ...] = ()
    center_cm: Point2D | None = None
    radius_cm: float | None = None
    start_angle_degrees: float | None = None
    end_angle_degrees: float | None = None
    clockwise: bool | None = None
    parameters: JsonObject = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(self.kind, "geometry.kind")
        _validate_xy(self.start_cm, "geometry.start_cm")
        _validate_xy(self.end_cm, "geometry.end_cm")
        for index, point in enumerate(self.control_points_cm):
            _validate_xy(point, f"geometry.control_points_cm[{index}]")
        if self.center_cm is not None:
            _validate_xy(self.center_cm, "geometry.center_cm")
        if self.radius_cm is not None:
            _finite_number(self.radius_cm, "geometry.radius_cm")
            if self.radius_cm <= 0.0:
                raise ValueError("geometry.radius_cm must be positive")
        for name, raw in (
            ("start_angle_degrees", self.start_angle_degrees),
            ("end_angle_degrees", self.end_angle_degrees),
        ):
            if raw is not None:
                _finite_number(raw, f"geometry.{name}")
        if (self.start_angle_degrees is None) != (self.end_angle_degrees is None):
            raise ValueError("arc angles must be supplied together")
        if self.clockwise is not None and not isinstance(self.clockwise, bool):
            raise ValueError("geometry.clockwise must be boolean or null")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("geometry.parameters must be an object")

        normalized = self.kind.strip().lower().replace("-", "_")
        if "bezier" in normalized and not self.control_points_cm:
            raise ValueError(f"{self.kind} geometry requires control_points_cm")
        if normalized == "quadratic_bezier" and len(self.control_points_cm) != 1:
            raise ValueError("quadratic_bezier geometry requires one control point")
        if normalized == "cubic_bezier" and len(self.control_points_cm) != 2:
            raise ValueError("cubic_bezier geometry requires two control points")
        if normalized == "arc":
            center_radius = self.center_cm is not None and self.radius_cm is not None
            through_points = bool(self.control_points_cm)
            if not center_radius and not through_points and not self.parameters:
                raise ValueError("arc geometry requires center/radius, through points, or exact source parameters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CurveGeometry":
        value = _mapping(value, "geometry")
        controls = tuple(
            _xy(_items(item, f"geometry.control_points_cm[{index}]"), f"geometry.control_points_cm[{index}]")
            for index, item in enumerate(_items(value.get("control_points_cm", ()), "geometry.control_points_cm"))
        )
        center = value.get("center_cm")
        return cls(
            kind=str(value["kind"]),
            start_cm=_xy(_items(value["start_cm"], "geometry.start_cm"), "geometry.start_cm"),
            end_cm=_xy(_items(value["end_cm"], "geometry.end_cm"), "geometry.end_cm"),
            control_points_cm=controls,
            center_cm=None if center is None else _xy(_items(center, "geometry.center_cm"), "geometry.center_cm"),
            radius_cm=None if value.get("radius_cm") is None else float(value["radius_cm"]),
            start_angle_degrees=(
                None if value.get("start_angle_degrees") is None else float(value["start_angle_degrees"])
            ),
            end_angle_degrees=(
                None if value.get("end_angle_degrees") is None else float(value["end_angle_degrees"])
            ),
            clockwise=value.get("clockwise"),
            parameters=_json_object(value.get("parameters", {}), "geometry.parameters"),
        )


@dataclass(frozen=True)
class TracedEdge:
    """A boundary or internal curve with exact geometry and dependencies."""

    id: str
    panel_id: str
    start_point_id: str
    end_point_id: str
    semantic_role: str
    geometry: CurveGeometry
    source_name: str | None = None
    formula: str | None = None
    dependencies: tuple[str, ...] = ()
    operation_id: str | None = None
    domain: str = "garmentcode_runtime"
    evidence: str = "observed_runtime"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    @property
    def role(self) -> str:
        return self.semantic_role

    def validate(self) -> None:
        _required_text(self.id, "edge.id")
        _required_text(self.panel_id, f"edge {self.id}.panel_id")
        _required_text(self.start_point_id, f"edge {self.id}.start_point_id")
        _required_text(self.end_point_id, f"edge {self.id}.end_point_id")
        _required_text(self.semantic_role, f"edge {self.id}.semantic_role")
        if self.source_name is not None:
            _required_text(self.source_name, f"edge {self.id}.source_name")
        if self.formula is not None:
            _required_text(self.formula, f"edge {self.id}.formula")
        _strings(self.dependencies, f"edge {self.id}.dependencies")
        if self.operation_id is not None:
            _required_text(self.operation_id, f"edge {self.id}.operation_id")
        if not isinstance(self.geometry, CurveGeometry):
            raise ValueError(f"edge {self.id}.geometry must be CurveGeometry")
        self.geometry.validate()
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"edge {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TracedEdge":
        value = _mapping(value, "edge")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            start_point_id=str(value["start_point_id"]),
            end_point_id=str(value["end_point_id"]),
            semantic_role=str(value.get("semantic_role", value.get("role", ""))),
            geometry=CurveGeometry.from_dict(_mapping(value["geometry"], "edge.geometry")),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            formula=None if value.get("formula") is None else str(value["formula"]),
            dependencies=_string_tuple(value.get("dependencies", ()), "edge.dependencies"),
            operation_id=None if value.get("operation_id") is None else str(value["operation_id"]),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class TracedPanel:
    """One T-shirt pattern piece and the trace-created objects it owns."""

    id: str
    semantic_role: str
    points: tuple[TracedPoint, ...]
    edges: tuple[TracedEdge, ...]
    source_name: str | None = None
    operation_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    @property
    def role(self) -> str:
        return self.semantic_role

    def validate(self) -> None:
        _required_text(self.id, "panel.id")
        _required_text(self.semantic_role, f"panel {self.id}.semantic_role")
        if self.source_name is not None:
            _required_text(self.source_name, f"panel {self.id}.source_name")
        if self.operation_id is not None:
            _required_text(self.operation_id, f"panel {self.id}.operation_id")
        if not isinstance(self.metadata, Mapping):
            raise ValueError(f"panel {self.id}.metadata must be an object")
        if not self.points:
            raise ValueError(f"panel {self.id} requires at least one point")
        if not self.edges:
            raise ValueError(f"panel {self.id} requires at least one edge")

        point_ids: set[str] = set()
        for point in self.points:
            if not isinstance(point, TracedPoint):
                raise ValueError(f"panel {self.id} contains an invalid point")
            point.validate()
            if point.id in point_ids:
                raise ValueError(f"duplicate point id in panel {self.id}: {point.id}")
            point_ids.add(point.id)
            if point.panel_id != self.id:
                raise ValueError(f"point {point.id} panel mismatch: {point.panel_id} != {self.id}")

        edge_ids: set[str] = set()
        point_by_id = {point.id: point for point in self.points}
        for edge in self.edges:
            if not isinstance(edge, TracedEdge):
                raise ValueError(f"panel {self.id} contains an invalid edge")
            edge.validate()
            if edge.id in edge_ids:
                raise ValueError(f"duplicate edge id in panel {self.id}: {edge.id}")
            edge_ids.add(edge.id)
            if edge.panel_id != self.id:
                raise ValueError(f"edge {edge.id} panel mismatch: {edge.panel_id} != {self.id}")
            if edge.start_point_id not in point_by_id:
                raise ValueError(f"edge {edge.id} references missing start point: {edge.start_point_id}")
            if edge.end_point_id not in point_by_id:
                raise ValueError(f"edge {edge.id} references missing end point: {edge.end_point_id}")
            start = point_by_id[edge.start_point_id].xy_cm
            end = point_by_id[edge.end_point_id].xy_cm
            if not _points_close(edge.geometry.start_cm, start):
                raise ValueError(f"edge {edge.id} start geometry does not match point {edge.start_point_id}")
            if not _points_close(edge.geometry.end_cm, end):
                raise ValueError(f"edge {edge.id} end geometry does not match point {edge.end_point_id}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TracedPanel":
        value = _mapping(value, "panel")
        return cls(
            id=str(value["id"]),
            semantic_role=str(value.get("semantic_role", value.get("role", ""))),
            points=tuple(
                TracedPoint.from_dict(_mapping(item, "panel point"))
                for item in _items(value.get("points", ()), "panel.points")
            ),
            edges=tuple(
                TracedEdge.from_dict(_mapping(item, "panel edge"))
                for item in _items(value.get("edges", ()), "panel.edges")
            ),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            operation_id=None if value.get("operation_id") is None else str(value["operation_id"]),
            metadata=_json_object(value.get("metadata", {}), "panel.metadata"),
        )


@dataclass(frozen=True)
class TracedReferenceLine:
    """A named construction/reference line, including non-boundary helpers."""

    id: str
    panel_id: str
    geometry: CurveGeometry
    formula: str
    canonical_name: str | None = None
    source_name: str | None = None
    measurement_inputs: dict[str, float] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    operation_id: str | None = None
    auxiliary: bool = True
    domain: str = "garmentcode_runtime"
    evidence: str = "observed_runtime"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "reference_line.id")
        _required_text(self.panel_id, f"reference_line {self.id}.panel_id")
        if self.canonical_name is None and self.source_name is None:
            raise ValueError(f"reference_line {self.id} requires canonical_name or source_name")
        if self.canonical_name is not None:
            _required_text(self.canonical_name, f"reference_line {self.id}.canonical_name")
        if self.source_name is not None:
            _required_text(self.source_name, f"reference_line {self.id}.source_name")
        _required_text(self.formula, f"reference_line {self.id}.formula")
        if not isinstance(self.geometry, CurveGeometry):
            raise ValueError(f"reference_line {self.id}.geometry must be CurveGeometry")
        self.geometry.validate()
        _validate_measurements(self.measurement_inputs, f"reference_line {self.id}.measurement_inputs")
        _strings(self.dependencies, f"reference_line {self.id}.dependencies")
        if self.operation_id is not None:
            _required_text(self.operation_id, f"reference_line {self.id}.operation_id")
        if not isinstance(self.auxiliary, bool):
            raise ValueError(f"reference_line {self.id}.auxiliary must be boolean")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"reference_line {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TracedReferenceLine":
        value = _mapping(value, "reference_line")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            geometry=CurveGeometry.from_dict(_mapping(value["geometry"], "reference_line.geometry")),
            formula=str(value["formula"]),
            canonical_name=None if value.get("canonical_name") is None else str(value["canonical_name"]),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            measurement_inputs={
                str(name): float(raw)
                for name, raw in _mapping(
                    value.get("measurement_inputs", {}), "reference_line.measurement_inputs"
                ).items()
            },
            dependencies=_string_tuple(value.get("dependencies", ()), "reference_line.dependencies"),
            operation_id=None if value.get("operation_id") is None else str(value["operation_id"]),
            auxiliary=bool(value.get("auxiliary", True)),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class GeometrySnapshot:
    """Panel-local geometry before or after a mutating construction step."""

    stage: str
    points_cm: dict[str, Point2D] = field(default_factory=dict)
    edges: dict[str, CurveGeometry] = field(default_factory=dict)
    operation_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(self.stage, "geometry_snapshot.stage")
        for point_id, point in self.points_cm.items():
            _required_text(point_id, "geometry_snapshot point id")
            _validate_xy(point, f"geometry_snapshot.points_cm.{point_id}")
        for edge_id, geometry in self.edges.items():
            _required_text(edge_id, "geometry_snapshot edge id")
            if not isinstance(geometry, CurveGeometry):
                raise ValueError(f"geometry_snapshot edge {edge_id} must be CurveGeometry")
            geometry.validate()
        if self.operation_id is not None:
            _required_text(self.operation_id, "geometry_snapshot.operation_id")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("geometry_snapshot.metadata must be an object")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeometrySnapshot":
        value = _mapping(value, "geometry_snapshot")
        return cls(
            stage=str(value["stage"]),
            points_cm={
                str(point_id): _xy(_items(raw, f"geometry_snapshot.points_cm.{point_id}"), f"geometry_snapshot.points_cm.{point_id}")
                for point_id, raw in _mapping(value.get("points_cm", {}), "geometry_snapshot.points_cm").items()
            },
            edges={
                str(edge_id): CurveGeometry.from_dict(_mapping(raw, f"geometry_snapshot.edges.{edge_id}"))
                for edge_id, raw in _mapping(value.get("edges", {}), "geometry_snapshot.edges").items()
            },
            operation_id=None if value.get("operation_id") is None else str(value["operation_id"]),
            metadata=_json_object(value.get("metadata", {}), "geometry_snapshot.metadata"),
        )


@dataclass(frozen=True)
class DartTrace:
    """Dart applicability plus insertion/rotation/closure snapshots."""

    id: str
    panel_id: str
    kind: str
    applicable: bool
    applicability_reason: str
    states: tuple[GeometrySnapshot, ...] = ()
    apex_point_id: str | None = None
    leg_edge_ids: tuple[str, ...] = ()
    affected_edge_ids: tuple[str, ...] = ()
    intake_cm: float | None = None
    depth_cm: float | None = None
    operation_ids: tuple[str, ...] = ()
    domain: str = "garmentcode_runtime"
    evidence: str = "observed_runtime"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    @property
    def applicability(self) -> bool:
        return self.applicable

    def validate(self) -> None:
        _required_text(self.id, "dart.id")
        _required_text(self.panel_id, f"dart {self.id}.panel_id")
        _required_text(self.kind, f"dart {self.id}.kind")
        if not isinstance(self.applicable, bool):
            raise ValueError(f"dart {self.id}.applicable must be boolean")
        _required_text(self.applicability_reason, f"dart {self.id}.applicability_reason")
        if self.apex_point_id is not None:
            _required_text(self.apex_point_id, f"dart {self.id}.apex_point_id")
        _strings(self.leg_edge_ids, f"dart {self.id}.leg_edge_ids")
        _strings(self.affected_edge_ids, f"dart {self.id}.affected_edge_ids")
        _strings(self.operation_ids, f"dart {self.id}.operation_ids")
        for name, raw in (("intake_cm", self.intake_cm), ("depth_cm", self.depth_cm)):
            if raw is not None:
                _finite_number(raw, f"dart {self.id}.{name}")
                if raw < 0.0:
                    raise ValueError(f"dart {self.id}.{name} must be non-negative")
        if self.applicable:
            if self.apex_point_id is None:
                raise ValueError(f"applicable dart {self.id} requires apex_point_id")
            if len(self.leg_edge_ids) != 2:
                raise ValueError(f"applicable dart {self.id} requires exactly two leg_edge_ids")
        stages: set[str] = set()
        for state in self.states:
            if not isinstance(state, GeometrySnapshot):
                raise ValueError(f"dart {self.id} contains an invalid geometry snapshot")
            state.validate()
            if state.stage in stages:
                raise ValueError(f"dart {self.id} has duplicate snapshot stage: {state.stage}")
            stages.add(state.stage)
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"dart {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DartTrace":
        value = _mapping(value, "dart")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            kind=str(value["kind"]),
            applicable=bool(value["applicable"]),
            applicability_reason=str(value["applicability_reason"]),
            states=tuple(
                GeometrySnapshot.from_dict(_mapping(item, "dart state"))
                for item in _items(value.get("states", ()), "dart.states")
            ),
            apex_point_id=None if value.get("apex_point_id") is None else str(value["apex_point_id"]),
            leg_edge_ids=_string_tuple(value.get("leg_edge_ids", ()), "dart.leg_edge_ids"),
            affected_edge_ids=_string_tuple(value.get("affected_edge_ids", ()), "dart.affected_edge_ids"),
            intake_cm=None if value.get("intake_cm") is None else float(value["intake_cm"]),
            depth_cm=None if value.get("depth_cm") is None else float(value["depth_cm"]),
            operation_ids=_string_tuple(value.get("operation_ids", ()), "dart.operation_ids"),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class ConstructionOperation:
    """One node in the construction DAG."""

    id: str
    order: int
    operation: str
    dependencies: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: JsonObject = field(default_factory=dict)
    source_reference: str | None = None
    pre_geometry: JsonObject = field(default_factory=dict)
    post_geometry: JsonObject = field(default_factory=dict)
    is_helper: bool = False
    status: str = "completed"
    domain: str = "garmentcode_runtime"
    evidence: str = "observed_runtime"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "operation.id")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError(f"operation {self.id}.order must be a non-negative integer")
        _required_text(self.operation, f"operation {self.id}.operation")
        _strings(self.dependencies, f"operation {self.id}.dependencies")
        _strings(self.inputs, f"operation {self.id}.inputs")
        _strings(self.outputs, f"operation {self.id}.outputs")
        if not isinstance(self.parameters, Mapping):
            raise ValueError(f"operation {self.id}.parameters must be an object")
        if self.source_reference is not None:
            _required_text(self.source_reference, f"operation {self.id}.source_reference")
        if not isinstance(self.pre_geometry, Mapping):
            raise ValueError(f"operation {self.id}.pre_geometry must be an object")
        if not isinstance(self.post_geometry, Mapping):
            raise ValueError(f"operation {self.id}.post_geometry must be an object")
        if not isinstance(self.is_helper, bool):
            raise ValueError(f"operation {self.id}.is_helper must be boolean")
        _required_text(self.status, f"operation {self.id}.status")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"operation {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstructionOperation":
        value = _mapping(value, "operation")
        return cls(
            id=str(value["id"]),
            order=int(value["order"]),
            operation=str(value["operation"]),
            dependencies=_string_tuple(value.get("dependencies", ()), "operation.dependencies"),
            inputs=_string_tuple(value.get("inputs", ()), "operation.inputs"),
            outputs=_string_tuple(value.get("outputs", ()), "operation.outputs"),
            parameters=_json_object(value.get("parameters", {}), "operation.parameters"),
            source_reference=None if value.get("source_reference") is None else str(value["source_reference"]),
            pre_geometry=_json_object(value.get("pre_geometry", {}), "operation.pre_geometry"),
            post_geometry=_json_object(value.get("post_geometry", {}), "operation.post_geometry"),
            is_helper=bool(value.get("is_helper", False)),
            status=str(value.get("status", "completed")),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class NamedPath:
    """A source-domain named path, distinct from inferred edge semantics."""

    id: str
    panel_id: str
    source_name: str
    semantic_role: str
    edge_ids: tuple[str, ...]
    closed: bool = False
    domain: str = "freesewing"
    evidence: str = "observed_source"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "named_path.id")
        _required_text(self.panel_id, f"named_path {self.id}.panel_id")
        _required_text(self.source_name, f"named_path {self.id}.source_name")
        _required_text(self.semantic_role, f"named_path {self.id}.semantic_role")
        if not self.edge_ids:
            raise ValueError(f"named_path {self.id} requires edge_ids")
        _strings(self.edge_ids, f"named_path {self.id}.edge_ids")
        if not isinstance(self.closed, bool):
            raise ValueError(f"named_path {self.id}.closed must be boolean")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"named_path {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NamedPath":
        value = _mapping(value, "named_path")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            source_name=str(value["source_name"]),
            semantic_role=str(value.get("semantic_role", value.get("role", ""))),
            edge_ids=_string_tuple(value.get("edge_ids", ()), "named_path.edge_ids"),
            closed=bool(value.get("closed", False)),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class Notch:
    id: str
    panel_id: str
    edge_id: str
    semantic_role: str
    position_fraction: float | None = None
    distance_cm: float | None = None
    xy_cm: Point2D | None = None
    source_name: str | None = None
    domain: str = "freesewing"
    evidence: str = "observed_source"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "notch.id")
        _required_text(self.panel_id, f"notch {self.id}.panel_id")
        _required_text(self.edge_id, f"notch {self.id}.edge_id")
        _required_text(self.semantic_role, f"notch {self.id}.semantic_role")
        if self.source_name is not None:
            _required_text(self.source_name, f"notch {self.id}.source_name")
        if self.position_fraction is None and self.distance_cm is None and self.xy_cm is None:
            raise ValueError(f"notch {self.id} requires an exact edge position or coordinate")
        if self.position_fraction is not None:
            _finite_number(self.position_fraction, f"notch {self.id}.position_fraction")
            if not 0.0 <= self.position_fraction <= 1.0:
                raise ValueError(f"notch {self.id}.position_fraction must be between 0 and 1")
        if self.distance_cm is not None:
            _finite_number(self.distance_cm, f"notch {self.id}.distance_cm")
            if self.distance_cm < 0.0:
                raise ValueError(f"notch {self.id}.distance_cm must be non-negative")
        if self.xy_cm is not None:
            _validate_xy(self.xy_cm, f"notch {self.id}.xy_cm")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"notch {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Notch":
        value = _mapping(value, "notch")
        point = value.get("xy_cm")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            edge_id=str(value["edge_id"]),
            semantic_role=str(value.get("semantic_role", value.get("role", ""))),
            position_fraction=(
                None if value.get("position_fraction") is None else float(value["position_fraction"])
            ),
            distance_cm=None if value.get("distance_cm") is None else float(value["distance_cm"]),
            xy_cm=None if point is None else _xy(_items(point, "notch.xy_cm"), "notch.xy_cm"),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class Grainline:
    id: str
    panel_id: str
    start_cm: Point2D
    end_cm: Point2D
    semantic_role: str = "grainline"
    start_point_id: str | None = None
    end_point_id: str | None = None
    source_name: str | None = None
    domain: str = "freesewing"
    evidence: str = "observed_source"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "grainline.id")
        _required_text(self.panel_id, f"grainline {self.id}.panel_id")
        _required_text(self.semantic_role, f"grainline {self.id}.semantic_role")
        _validate_xy(self.start_cm, f"grainline {self.id}.start_cm")
        _validate_xy(self.end_cm, f"grainline {self.id}.end_cm")
        if _points_close(self.start_cm, self.end_cm):
            raise ValueError(f"grainline {self.id} must have non-zero length")
        for name, raw in (("start_point_id", self.start_point_id), ("end_point_id", self.end_point_id)):
            if raw is not None:
                _required_text(raw, f"grainline {self.id}.{name}")
        if self.source_name is not None:
            _required_text(self.source_name, f"grainline {self.id}.source_name")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"grainline {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Grainline":
        value = _mapping(value, "grainline")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            start_cm=_xy(_items(value["start_cm"], "grainline.start_cm"), "grainline.start_cm"),
            end_cm=_xy(_items(value["end_cm"], "grainline.end_cm"), "grainline.end_cm"),
            semantic_role=str(value.get("semantic_role", value.get("role", "grainline"))),
            start_point_id=None if value.get("start_point_id") is None else str(value["start_point_id"]),
            end_point_id=None if value.get("end_point_id") is None else str(value["end_point_id"]),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class SeamAllowance:
    id: str
    panel_id: str
    edge_ids: tuple[str, ...]
    width_cm: float | None = None
    width_by_edge_cm: dict[str, float] = field(default_factory=dict)
    semantic_role: str = "seam_allowance"
    source_name: str | None = None
    offset_path: CurveGeometry | None = None
    domain: str = "freesewing"
    evidence: str = "observed_source"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "seam_allowance.id")
        _required_text(self.panel_id, f"seam_allowance {self.id}.panel_id")
        _required_text(self.semantic_role, f"seam_allowance {self.id}.semantic_role")
        if not self.edge_ids:
            raise ValueError(f"seam_allowance {self.id} requires edge_ids")
        _strings(self.edge_ids, f"seam_allowance {self.id}.edge_ids")
        if self.width_cm is None and not self.width_by_edge_cm:
            raise ValueError(f"seam_allowance {self.id} requires width_cm or width_by_edge_cm")
        if self.width_cm is not None:
            _finite_number(self.width_cm, f"seam_allowance {self.id}.width_cm")
            if self.width_cm < 0.0:
                raise ValueError(f"seam_allowance {self.id}.width_cm must be non-negative")
        _validate_measurements(self.width_by_edge_cm, f"seam_allowance {self.id}.width_by_edge_cm")
        for edge_id, width in self.width_by_edge_cm.items():
            if edge_id not in self.edge_ids:
                raise ValueError(f"seam_allowance {self.id} has width for unreferenced edge: {edge_id}")
            if width < 0.0:
                raise ValueError(f"seam_allowance {self.id} width for {edge_id} must be non-negative")
        if self.source_name is not None:
            _required_text(self.source_name, f"seam_allowance {self.id}.source_name")
        if self.offset_path is not None:
            if not isinstance(self.offset_path, CurveGeometry):
                raise ValueError(f"seam_allowance {self.id}.offset_path must be CurveGeometry")
            self.offset_path.validate()
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"seam_allowance {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeamAllowance":
        value = _mapping(value, "seam_allowance")
        offset = value.get("offset_path")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            edge_ids=_string_tuple(value.get("edge_ids", ()), "seam_allowance.edge_ids"),
            width_cm=None if value.get("width_cm") is None else float(value["width_cm"]),
            width_by_edge_cm={
                str(edge_id): float(raw)
                for edge_id, raw in _mapping(
                    value.get("width_by_edge_cm", {}), "seam_allowance.width_by_edge_cm"
                ).items()
            },
            semantic_role=str(value.get("semantic_role", value.get("role", "seam_allowance"))),
            source_name=None if value.get("source_name") is None else str(value["source_name"]),
            offset_path=(
                None if offset is None else CurveGeometry.from_dict(_mapping(offset, "seam_allowance.offset_path"))
            ),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class DraftingFormulaSegment:
    """One primitive represented in a whole-path, chord-normalized frame.

    Two control slots are always serialized so a learner can batch line,
    quadratic, and cubic primitives without guessing from array length.  The
    accompanying mask is authoritative: an arc midpoint is deliberately not
    mislabeled as a Bezier control point.
    """

    edge_id: str
    geometry_kind: str
    normalized_start: Point2D
    normalized_end: Point2D
    normalized_bezier_controls: tuple[Point2D, Point2D] = ((0.0, 0.0), (0.0, 0.0))
    bezier_control_mask: tuple[bool, bool] = (False, False)
    source_formula: str | None = None
    operation_id: str | None = None
    source_parameters: JsonObject = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(self.edge_id, "drafting_formula_segment.edge_id")
        _required_text(self.geometry_kind, f"drafting_formula_segment {self.edge_id}.geometry_kind")
        _validate_xy(self.normalized_start, f"drafting_formula_segment {self.edge_id}.normalized_start")
        _validate_xy(self.normalized_end, f"drafting_formula_segment {self.edge_id}.normalized_end")
        if len(self.normalized_bezier_controls) != 2 or len(self.bezier_control_mask) != 2:
            raise ValueError(f"drafting_formula_segment {self.edge_id} requires exactly two control slots/masks")
        for index, point in enumerate(self.normalized_bezier_controls):
            _validate_xy(point, f"drafting_formula_segment {self.edge_id}.normalized_bezier_controls[{index}]")
        if any(not isinstance(value, bool) for value in self.bezier_control_mask):
            raise ValueError(f"drafting_formula_segment {self.edge_id}.bezier_control_mask must be boolean")
        if self.source_formula is not None:
            _required_text(self.source_formula, f"drafting_formula_segment {self.edge_id}.source_formula")
        if self.operation_id is not None:
            _required_text(self.operation_id, f"drafting_formula_segment {self.edge_id}.operation_id")
        if not isinstance(self.source_parameters, Mapping):
            raise ValueError(f"drafting_formula_segment {self.edge_id}.source_parameters must be an object")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DraftingFormulaSegment":
        value = _mapping(value, "drafting_formula_segment")
        controls = tuple(
            _xy(_items(item, f"drafting_formula_segment.normalized_bezier_controls[{index}]"),
                f"drafting_formula_segment.normalized_bezier_controls[{index}]")
            for index, item in enumerate(
                _items(value.get("normalized_bezier_controls", ((0.0, 0.0), (0.0, 0.0))),
                       "drafting_formula_segment.normalized_bezier_controls")
            )
        )
        if len(controls) != 2:
            raise ValueError("drafting_formula_segment.normalized_bezier_controls requires two entries")
        masks = tuple(bool(item) for item in _items(
            value.get("bezier_control_mask", (False, False)), "drafting_formula_segment.bezier_control_mask"
        ))
        if len(masks) != 2:
            raise ValueError("drafting_formula_segment.bezier_control_mask requires two entries")
        return cls(
            edge_id=str(value["edge_id"]),
            geometry_kind=str(value["geometry_kind"]),
            normalized_start=_xy(_items(value["normalized_start"], "drafting_formula_segment.normalized_start"),
                                 "drafting_formula_segment.normalized_start"),
            normalized_end=_xy(_items(value["normalized_end"], "drafting_formula_segment.normalized_end"),
                               "drafting_formula_segment.normalized_end"),
            normalized_bezier_controls=(controls[0], controls[1]),
            bezier_control_mask=(masks[0], masks[1]),
            source_formula=None if value.get("source_formula") is None else str(value["source_formula"]),
            operation_id=None if value.get("operation_id") is None else str(value["operation_id"]),
            source_parameters=_json_object(value.get("source_parameters", {}),
                                           "drafting_formula_segment.source_parameters"),
        )


@dataclass(frozen=True)
class DraftingFormulaTarget:
    """Supervision for a named drafting curve, with explicit availability masks."""

    id: str
    panel_id: str
    panel_role: str
    semantic_role: str
    edge_ids: tuple[str, ...]
    endpoint_point_ids: tuple[str, str]
    endpoint_names: tuple[str | None, str | None]
    endpoint_name_mask: tuple[bool, bool]
    scalar_values: dict[str, float]
    scalar_mask: dict[str, bool]
    semantic_values: dict[str, float]
    semantic_mask: dict[str, bool]
    endpoint_tangents_unit: tuple[Point2D, Point2D]
    endpoint_tangent_mask: tuple[bool, bool]
    segments: tuple[DraftingFormulaSegment, ...]
    source_formula_parameters: dict[str, float] = field(default_factory=dict)
    source_parameter_mask: dict[str, bool] = field(default_factory=dict)
    operation_ids: tuple[str, ...] = ()
    domain: str = "garmentcode_runtime"
    evidence: str = "creation_event_binding"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "drafting_formula_target.id")
        _required_text(self.panel_id, f"drafting_formula_target {self.id}.panel_id")
        _required_text(self.panel_role, f"drafting_formula_target {self.id}.panel_role")
        _required_text(self.semantic_role, f"drafting_formula_target {self.id}.semantic_role")
        if self.semantic_role not in DRAFTING_FORMULA_ROLES:
            raise ValueError(f"unsupported drafting formula role: {self.semantic_role}")
        if not self.edge_ids or not self.segments:
            raise ValueError(f"drafting_formula_target {self.id} requires edges and segments")
        _strings(self.edge_ids, f"drafting_formula_target {self.id}.edge_ids")
        if len(self.endpoint_point_ids) != 2:
            raise ValueError(f"drafting_formula_target {self.id} requires two endpoint point ids")
        _strings(self.endpoint_point_ids, f"drafting_formula_target {self.id}.endpoint_point_ids")
        if len(self.endpoint_names) != 2 or len(self.endpoint_name_mask) != 2:
            raise ValueError(f"drafting_formula_target {self.id} requires two endpoint name slots/masks")
        if any(not isinstance(value, bool) for value in self.endpoint_name_mask):
            raise ValueError(f"drafting_formula_target {self.id}.endpoint_name_mask must be boolean")
        for index, (name, mask) in enumerate(zip(self.endpoint_names, self.endpoint_name_mask)):
            if mask and name is None:
                raise ValueError(f"drafting_formula_target {self.id} endpoint {index} is masked present but unnamed")
            if name is not None:
                _required_text(name, f"drafting_formula_target {self.id}.endpoint_names[{index}]")
        if set(self.scalar_values) != set(DRAFTING_FORMULA_SCALARS):
            raise ValueError(f"drafting_formula_target {self.id}.scalar_values must contain canonical scalar slots")
        if set(self.scalar_mask) != set(DRAFTING_FORMULA_SCALARS):
            raise ValueError(f"drafting_formula_target {self.id}.scalar_mask must contain canonical scalar slots")
        _validate_measurements(self.scalar_values, f"drafting_formula_target {self.id}.scalar_values")
        if any(not isinstance(value, bool) for value in self.scalar_mask.values()):
            raise ValueError(f"drafting_formula_target {self.id}.scalar_mask must be boolean")
        _validate_measurements(self.semantic_values, f"drafting_formula_target {self.id}.semantic_values")
        if set(self.semantic_mask) != set(self.semantic_values):
            raise ValueError(f"drafting_formula_target {self.id}.semantic_mask must match semantic_values")
        if any(not isinstance(value, bool) for value in self.semantic_mask.values()):
            raise ValueError(f"drafting_formula_target {self.id}.semantic_mask must be boolean")
        if len(self.endpoint_tangents_unit) != 2 or len(self.endpoint_tangent_mask) != 2:
            raise ValueError(f"drafting_formula_target {self.id} requires two tangent slots/masks")
        for index, tangent in enumerate(self.endpoint_tangents_unit):
            _validate_xy(tangent, f"drafting_formula_target {self.id}.endpoint_tangents_unit[{index}]")
        if any(not isinstance(value, bool) for value in self.endpoint_tangent_mask):
            raise ValueError(f"drafting_formula_target {self.id}.endpoint_tangent_mask must be boolean")
        for segment in self.segments:
            if not isinstance(segment, DraftingFormulaSegment):
                raise ValueError(f"drafting_formula_target {self.id} contains an invalid segment")
            segment.validate()
        if tuple(segment.edge_id for segment in self.segments) != self.edge_ids:
            raise ValueError(f"drafting_formula_target {self.id} segment order must match edge_ids")
        _validate_measurements(self.source_formula_parameters,
                               f"drafting_formula_target {self.id}.source_formula_parameters")
        if set(self.source_parameter_mask) != set(self.source_formula_parameters):
            raise ValueError(f"drafting_formula_target {self.id}.source_parameter_mask must match parameters")
        if any(not isinstance(value, bool) for value in self.source_parameter_mask.values()):
            raise ValueError(f"drafting_formula_target {self.id}.source_parameter_mask must be boolean")
        _strings(self.operation_ids, f"drafting_formula_target {self.id}.operation_ids")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"drafting_formula_target {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DraftingFormulaTarget":
        value = _mapping(value, "drafting_formula_target")
        endpoint_names_raw = _items(value["endpoint_names"], "drafting_formula_target.endpoint_names")
        endpoint_names = tuple(None if item is None else str(item) for item in endpoint_names_raw)
        endpoint_ids = _string_tuple(value["endpoint_point_ids"], "drafting_formula_target.endpoint_point_ids")
        endpoint_name_mask = tuple(bool(item) for item in _items(
            value["endpoint_name_mask"], "drafting_formula_target.endpoint_name_mask"
        ))
        tangents = tuple(
            _xy(_items(item, f"drafting_formula_target.endpoint_tangents_unit[{index}]"),
                f"drafting_formula_target.endpoint_tangents_unit[{index}]")
            for index, item in enumerate(_items(
                value["endpoint_tangents_unit"], "drafting_formula_target.endpoint_tangents_unit"
            ))
        )
        tangent_mask = tuple(bool(item) for item in _items(
            value["endpoint_tangent_mask"], "drafting_formula_target.endpoint_tangent_mask"
        ))
        if not all(len(items) == 2 for items in (endpoint_names, endpoint_ids, endpoint_name_mask, tangents, tangent_mask)):
            raise ValueError("drafting_formula_target endpoint fields require two entries")
        return cls(
            id=str(value["id"]),
            panel_id=str(value["panel_id"]),
            panel_role=str(value["panel_role"]),
            semantic_role=str(value["semantic_role"]),
            edge_ids=_string_tuple(value["edge_ids"], "drafting_formula_target.edge_ids"),
            endpoint_point_ids=(endpoint_ids[0], endpoint_ids[1]),
            endpoint_names=(endpoint_names[0], endpoint_names[1]),
            endpoint_name_mask=(endpoint_name_mask[0], endpoint_name_mask[1]),
            scalar_values={str(name): float(raw) for name, raw in _mapping(
                value["scalar_values"], "drafting_formula_target.scalar_values"
            ).items()},
            scalar_mask={str(name): bool(raw) for name, raw in _mapping(
                value["scalar_mask"], "drafting_formula_target.scalar_mask"
            ).items()},
            semantic_values={str(name): float(raw) for name, raw in _mapping(
                value.get("semantic_values", {}), "drafting_formula_target.semantic_values"
            ).items()},
            semantic_mask={str(name): bool(raw) for name, raw in _mapping(
                value.get("semantic_mask", {}), "drafting_formula_target.semantic_mask"
            ).items()},
            endpoint_tangents_unit=(tangents[0], tangents[1]),
            endpoint_tangent_mask=(tangent_mask[0], tangent_mask[1]),
            segments=tuple(DraftingFormulaSegment.from_dict(_mapping(item, "drafting_formula_segment"))
                           for item in _items(value["segments"], "drafting_formula_target.segments")),
            source_formula_parameters={str(name): float(raw) for name, raw in _mapping(
                value.get("source_formula_parameters", {}),
                "drafting_formula_target.source_formula_parameters",
            ).items()},
            source_parameter_mask={str(name): bool(raw) for name, raw in _mapping(
                value.get("source_parameter_mask", {}), "drafting_formula_target.source_parameter_mask"
            ).items()},
            operation_ids=_string_tuple(value.get("operation_ids", ()), "drafting_formula_target.operation_ids"),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class DraftingSeamRelation:
    """Aggregate sleeve-head/armhole compatibility with provenance and masks."""

    id: str
    sleeve_head_target_ids: tuple[str, ...]
    armhole_target_ids: tuple[str, ...]
    values: dict[str, float]
    value_mask: dict[str, bool]
    operation_ids: tuple[str, ...] = ()
    domain: str = "garmentcode_runtime"
    evidence: str = "creation_event_formula_and_live_geometry"
    provenance: JsonObject = field(default_factory=dict)
    training_eligible: bool = True
    confidence: float = 1.0

    def validate(self) -> None:
        _required_text(self.id, "drafting_seam_relation.id")
        if not self.sleeve_head_target_ids or not self.armhole_target_ids:
            raise ValueError(f"drafting_seam_relation {self.id} requires both target groups")
        _strings(self.sleeve_head_target_ids, f"drafting_seam_relation {self.id}.sleeve_head_target_ids")
        _strings(self.armhole_target_ids, f"drafting_seam_relation {self.id}.armhole_target_ids")
        required = {"sleeve_head_length_cm", "armhole_length_cm", "ease_difference_cm", "ease_ratio"}
        if set(self.values) != required or set(self.value_mask) != required:
            raise ValueError(f"drafting_seam_relation {self.id} requires canonical value slots/masks")
        _validate_measurements(self.values, f"drafting_seam_relation {self.id}.values")
        if any(not isinstance(value, bool) for value in self.value_mask.values()):
            raise ValueError(f"drafting_seam_relation {self.id}.value_mask must be boolean")
        _strings(self.operation_ids, f"drafting_seam_relation {self.id}.operation_ids")
        _validate_annotation(
            domain=self.domain,
            evidence=self.evidence,
            provenance=self.provenance,
            training_eligible=self.training_eligible,
            confidence=self.confidence,
            label=f"drafting_seam_relation {self.id}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DraftingSeamRelation":
        value = _mapping(value, "drafting_seam_relation")
        return cls(
            id=str(value["id"]),
            sleeve_head_target_ids=_string_tuple(value["sleeve_head_target_ids"],
                                                  "drafting_seam_relation.sleeve_head_target_ids"),
            armhole_target_ids=_string_tuple(value["armhole_target_ids"],
                                             "drafting_seam_relation.armhole_target_ids"),
            values={str(name): float(raw) for name, raw in _mapping(
                value["values"], "drafting_seam_relation.values"
            ).items()},
            value_mask={str(name): bool(raw) for name, raw in _mapping(
                value["value_mask"], "drafting_seam_relation.value_mask"
            ).items()},
            operation_ids=_string_tuple(value.get("operation_ids", ()), "drafting_seam_relation.operation_ids"),
            **_annotation_fields(value),
        )


@dataclass(frozen=True)
class TShirtTraceRecord:
    """Complete, training-ready trace for one body/design realization."""

    sample_id: str
    split: str
    source: JsonObject
    body: dict[str, float]
    design: JsonObject
    provenance: JsonObject
    panels: tuple[TracedPanel, ...]
    operations: tuple[ConstructionOperation, ...]
    recipe_id: str = "basic_tshirt"
    garment_type: str = "basic_tshirt"
    reference_lines: tuple[TracedReferenceLine, ...] = ()
    darts: tuple[DartTrace, ...] = ()
    named_paths: tuple[NamedPath, ...] = ()
    notches: tuple[Notch, ...] = ()
    grainlines: tuple[Grainline, ...] = ()
    seam_allowances: tuple[SeamAllowance, ...] = ()
    drafting_formula_targets: tuple[DraftingFormulaTarget, ...] = ()
    drafting_seam_relations: tuple[DraftingSeamRelation, ...] = ()
    metadata: JsonObject = field(default_factory=dict)
    schema_version: str = "tshirt-construction-trace-1.1"

    def validate(self) -> None:
        _required_text(self.sample_id, "sample_id")
        _required_text(self.split, "split")
        _required_text(self.recipe_id, "recipe_id")
        _required_text(self.garment_type, "garment_type")
        _required_text(self.schema_version, "schema_version")
        if not isinstance(self.source, Mapping) or not self.source:
            raise ValueError("source must be a non-empty object")
        _validate_measurements(self.body, "body")
        if not self.body:
            raise ValueError("body must contain at least one measurement")
        for name, value in (("design", self.design), ("provenance", self.provenance), ("metadata", self.metadata)):
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
        if not self.panels:
            raise ValueError("panels must contain at least one T-shirt panel")
        if not self.operations:
            raise ValueError("operations must contain the construction DAG")

        panel_ids: set[str] = set()
        point_to_panel: dict[str, str] = {}
        edge_to_panel: dict[str, str] = {}
        operation_references: list[tuple[str, str]] = []
        for panel in self.panels:
            if not isinstance(panel, TracedPanel):
                raise ValueError("panels contains an invalid panel")
            panel.validate()
            if panel.id in panel_ids:
                raise ValueError(f"duplicate panel id: {panel.id}")
            panel_ids.add(panel.id)
            if panel.operation_id is not None:
                operation_references.append((f"panel {panel.id}", panel.operation_id))
            for point in panel.points:
                if point.id in point_to_panel:
                    raise ValueError(f"duplicate global point id: {point.id}")
                point_to_panel[point.id] = panel.id
                if point.operation_id is not None:
                    operation_references.append((f"point {point.id}", point.operation_id))
            for edge in panel.edges:
                if edge.id in edge_to_panel:
                    raise ValueError(f"duplicate global edge id: {edge.id}")
                edge_to_panel[edge.id] = panel.id
                if edge.operation_id is not None:
                    operation_references.append((f"edge {edge.id}", edge.operation_id))

        operation_ids = self._validate_operation_dag()

        reference_ids: set[str] = set()
        for line in self.reference_lines:
            if not isinstance(line, TracedReferenceLine):
                raise ValueError("reference_lines contains an invalid reference line")
            line.validate()
            if line.id in reference_ids:
                raise ValueError(f"duplicate reference line id: {line.id}")
            reference_ids.add(line.id)
            _require_panel(line.panel_id, panel_ids, f"reference_line {line.id}")
            if line.operation_id is not None:
                operation_references.append((f"reference_line {line.id}", line.operation_id))

        dart_ids: set[str] = set()
        for dart in self.darts:
            if not isinstance(dart, DartTrace):
                raise ValueError("darts contains an invalid dart")
            dart.validate()
            if dart.id in dart_ids:
                raise ValueError(f"duplicate dart id: {dart.id}")
            dart_ids.add(dart.id)
            _require_panel(dart.panel_id, panel_ids, f"dart {dart.id}")
            if dart.apex_point_id is not None:
                _require_point_on_panel(dart.apex_point_id, dart.panel_id, point_to_panel, f"dart {dart.id}")
            for edge_id in (*dart.leg_edge_ids, *dart.affected_edge_ids):
                _require_edge_on_panel(edge_id, dart.panel_id, edge_to_panel, f"dart {dart.id}")
            for operation_id in dart.operation_ids:
                operation_references.append((f"dart {dart.id}", operation_id))
            for state in dart.states:
                if state.operation_id is not None:
                    operation_references.append((f"dart {dart.id} snapshot {state.stage}", state.operation_id))

        named_path_ids: set[str] = set()
        for path in self.named_paths:
            if not isinstance(path, NamedPath):
                raise ValueError("named_paths contains an invalid named path")
            path.validate()
            if path.id in named_path_ids:
                raise ValueError(f"duplicate named path id: {path.id}")
            named_path_ids.add(path.id)
            _require_panel(path.panel_id, panel_ids, f"named_path {path.id}")
            for edge_id in path.edge_ids:
                _require_edge_on_panel(edge_id, path.panel_id, edge_to_panel, f"named_path {path.id}")

        notch_ids: set[str] = set()
        for notch in self.notches:
            if not isinstance(notch, Notch):
                raise ValueError("notches contains an invalid notch")
            notch.validate()
            if notch.id in notch_ids:
                raise ValueError(f"duplicate notch id: {notch.id}")
            notch_ids.add(notch.id)
            _require_panel(notch.panel_id, panel_ids, f"notch {notch.id}")
            _require_edge_on_panel(notch.edge_id, notch.panel_id, edge_to_panel, f"notch {notch.id}")

        grainline_ids: set[str] = set()
        for grainline in self.grainlines:
            if not isinstance(grainline, Grainline):
                raise ValueError("grainlines contains an invalid grainline")
            grainline.validate()
            if grainline.id in grainline_ids:
                raise ValueError(f"duplicate grainline id: {grainline.id}")
            grainline_ids.add(grainline.id)
            _require_panel(grainline.panel_id, panel_ids, f"grainline {grainline.id}")
            for point_id in (grainline.start_point_id, grainline.end_point_id):
                if point_id is not None:
                    _require_point_on_panel(point_id, grainline.panel_id, point_to_panel, f"grainline {grainline.id}")

        allowance_ids: set[str] = set()
        for allowance in self.seam_allowances:
            if not isinstance(allowance, SeamAllowance):
                raise ValueError("seam_allowances contains an invalid seam allowance")
            allowance.validate()
            if allowance.id in allowance_ids:
                raise ValueError(f"duplicate seam allowance id: {allowance.id}")
            allowance_ids.add(allowance.id)
            _require_panel(allowance.panel_id, panel_ids, f"seam_allowance {allowance.id}")
            for edge_id in allowance.edge_ids:
                _require_edge_on_panel(edge_id, allowance.panel_id, edge_to_panel, f"seam_allowance {allowance.id}")

        formula_target_ids: set[str] = set()
        for target in self.drafting_formula_targets:
            if not isinstance(target, DraftingFormulaTarget):
                raise ValueError("drafting_formula_targets contains an invalid target")
            target.validate()
            if target.id in formula_target_ids:
                raise ValueError(f"duplicate drafting formula target id: {target.id}")
            formula_target_ids.add(target.id)
            _require_panel(target.panel_id, panel_ids, f"drafting_formula_target {target.id}")
            for point_id in target.endpoint_point_ids:
                _require_point_on_panel(point_id, target.panel_id, point_to_panel,
                                        f"drafting_formula_target {target.id}")
            for edge_id in target.edge_ids:
                _require_edge_on_panel(edge_id, target.panel_id, edge_to_panel,
                                       f"drafting_formula_target {target.id}")
            for operation_id in target.operation_ids:
                operation_references.append((f"drafting_formula_target {target.id}", operation_id))

        seam_relation_ids: set[str] = set()
        for relation in self.drafting_seam_relations:
            if not isinstance(relation, DraftingSeamRelation):
                raise ValueError("drafting_seam_relations contains an invalid relation")
            relation.validate()
            if relation.id in seam_relation_ids:
                raise ValueError(f"duplicate drafting seam relation id: {relation.id}")
            seam_relation_ids.add(relation.id)
            for target_id in (*relation.sleeve_head_target_ids, *relation.armhole_target_ids):
                if target_id not in formula_target_ids:
                    raise ValueError(f"drafting_seam_relation {relation.id} references missing target: {target_id}")
            for operation_id in relation.operation_ids:
                operation_references.append((f"drafting_seam_relation {relation.id}", operation_id))

        for owner, operation_id in operation_references:
            if operation_id not in operation_ids:
                raise ValueError(f"{owner} references missing operation: {operation_id}")

        try:
            json.dumps(self.to_dict(), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"record contains a non-JSON value: {error}") from error

    def _validate_operation_dag(self) -> set[str]:
        operations: dict[str, ConstructionOperation] = {}
        orders: set[int] = set()
        for operation in self.operations:
            if not isinstance(operation, ConstructionOperation):
                raise ValueError("operations contains an invalid construction operation")
            operation.validate()
            if operation.id in operations:
                raise ValueError(f"duplicate operation id: {operation.id}")
            if operation.order in orders:
                raise ValueError(f"duplicate operation order: {operation.order}")
            operations[operation.id] = operation
            orders.add(operation.order)

        for operation in operations.values():
            for dependency in operation.dependencies:
                if dependency not in operations:
                    raise ValueError(f"operation {operation.id} references missing dependency: {dependency}")

        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(operation_id: str) -> None:
            marker = state.get(operation_id, 0)
            if marker == 2:
                return
            if marker == 1:
                start = stack.index(operation_id)
                cycle = " -> ".join((*stack[start:], operation_id))
                raise ValueError(f"construction operation DAG contains a cycle: {cycle}")
            state[operation_id] = 1
            stack.append(operation_id)
            for dependency in operations[operation_id].dependencies:
                visit(dependency)
            stack.pop()
            state[operation_id] = 2

        for operation_id in operations:
            visit(operation_id)
        return set(operations)

    def topological_operations(self) -> tuple[ConstructionOperation, ...]:
        """Return a stable topological order, using trace order as a tie-break."""

        self._validate_operation_dag()
        operations = {operation.id: operation for operation in self.operations}
        dependents: dict[str, list[str]] = {operation_id: [] for operation_id in operations}
        indegree = {operation_id: 0 for operation_id in operations}
        for operation in operations.values():
            indegree[operation.id] = len(operation.dependencies)
            for dependency in operation.dependencies:
                dependents[dependency].append(operation.id)
        ready = sorted(
            (operations[operation_id] for operation_id, degree in indegree.items() if degree == 0),
            key=lambda item: (item.order, item.id),
        )
        result: list[ConstructionOperation] = []
        while ready:
            operation = ready.pop(0)
            result.append(operation)
            for dependent_id in dependents[operation.id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    ready.append(operations[dependent_id])
                    ready.sort(key=lambda item: (item.order, item.id))
        return tuple(result)

    def to_dict(self) -> JsonObject:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    write = write_json

    @classmethod
    def read_json(cls, path: str | Path) -> "TShirtTraceRecord":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("T-shirt trace JSON root must be an object")
        return cls.from_dict(value)

    read = read_json

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TShirtTraceRecord":
        value = _mapping(value, "T-shirt trace record")
        record = cls(
            sample_id=str(value["sample_id"]),
            split=str(value["split"]),
            source=_json_object(value["source"], "source"),
            body={str(name): float(raw) for name, raw in _mapping(value["body"], "body").items()},
            design=_json_object(value["design"], "design"),
            provenance=_json_object(value["provenance"], "provenance"),
            panels=tuple(
                TracedPanel.from_dict(_mapping(item, "panel"))
                for item in _items(value.get("panels", ()), "panels")
            ),
            operations=tuple(
                ConstructionOperation.from_dict(_mapping(item, "operation"))
                for item in _items(value.get("operations", ()), "operations")
            ),
            recipe_id=str(value.get("recipe_id", "basic_tshirt")),
            garment_type=str(value.get("garment_type", "basic_tshirt")),
            reference_lines=tuple(
                TracedReferenceLine.from_dict(_mapping(item, "reference_line"))
                for item in _items(value.get("reference_lines", ()), "reference_lines")
            ),
            darts=tuple(
                DartTrace.from_dict(_mapping(item, "dart"))
                for item in _items(value.get("darts", ()), "darts")
            ),
            named_paths=tuple(
                NamedPath.from_dict(_mapping(item, "named_path"))
                for item in _items(value.get("named_paths", ()), "named_paths")
            ),
            notches=tuple(
                Notch.from_dict(_mapping(item, "notch"))
                for item in _items(value.get("notches", ()), "notches")
            ),
            grainlines=tuple(
                Grainline.from_dict(_mapping(item, "grainline"))
                for item in _items(value.get("grainlines", ()), "grainlines")
            ),
            seam_allowances=tuple(
                SeamAllowance.from_dict(_mapping(item, "seam_allowance"))
                for item in _items(value.get("seam_allowances", ()), "seam_allowances")
            ),
            drafting_formula_targets=tuple(
                DraftingFormulaTarget.from_dict(_mapping(item, "drafting_formula_target"))
                for item in _items(value.get("drafting_formula_targets", ()), "drafting_formula_targets")
            ),
            drafting_seam_relations=tuple(
                DraftingSeamRelation.from_dict(_mapping(item, "drafting_seam_relation"))
                for item in _items(value.get("drafting_seam_relations", ()), "drafting_seam_relations")
            ),
            metadata=_json_object(value.get("metadata", {}), "metadata"),
            schema_version=str(value.get("schema_version", "tshirt-construction-trace-1.1")),
        )
        record.validate()
        return record


def _points_close(first: Point2D, second: Point2D, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(first[0]), float(second[0]), abs_tol=tolerance, rel_tol=tolerance) and math.isclose(
        float(first[1]), float(second[1]), abs_tol=tolerance, rel_tol=tolerance
    )


def _require_panel(panel_id: str, panel_ids: set[str], owner: str) -> None:
    if panel_id not in panel_ids:
        raise ValueError(f"{owner} references missing panel: {panel_id}")


def _require_point_on_panel(point_id: str, panel_id: str, point_to_panel: Mapping[str, str], owner: str) -> None:
    actual_panel = point_to_panel.get(point_id)
    if actual_panel is None:
        raise ValueError(f"{owner} references missing point: {point_id}")
    if actual_panel != panel_id:
        raise ValueError(f"{owner} references point {point_id} from panel {actual_panel}")


def _require_edge_on_panel(edge_id: str, panel_id: str, edge_to_panel: Mapping[str, str], owner: str) -> None:
    actual_panel = edge_to_panel.get(edge_id)
    if actual_panel is None:
        raise ValueError(f"{owner} references missing edge: {edge_id}")
    if actual_panel != panel_id:
        raise ValueError(f"{owner} references edge {edge_id} from panel {actual_panel}")


def read_tshirt_trace(path: str | Path) -> TShirtTraceRecord:
    return TShirtTraceRecord.read_json(path)


# Short aliases make the public API pleasant without giving up explicit class
# names in serialized records and documentation.
TracePoint = TracedPoint
TraceEdge = TracedEdge
TracePanel = TracedPanel
ReferenceLine = TracedReferenceLine
ConstructionStep = ConstructionOperation


__all__ = [
    "CANONICAL_TSHIRT_POINTS",
    "CANONICAL_PANEL_ROLES",
    "CANONICAL_EDGE_ROLES",
    "CANONICAL_CURVE_KINDS",
    "CANONICAL_SPLITS",
    "TracedPoint",
    "TracePoint",
    "CurveGeometry",
    "TracedEdge",
    "TraceEdge",
    "TracedPanel",
    "TracePanel",
    "TracedReferenceLine",
    "ReferenceLine",
    "GeometrySnapshot",
    "DartTrace",
    "ConstructionOperation",
    "DRAFTING_FORMULA_ROLES",
    "DRAFTING_FORMULA_SCALARS",
    "DraftingFormulaSegment",
    "DraftingFormulaTarget",
    "DraftingSeamRelation",
    "ConstructionStep",
    "NamedPath",
    "Notch",
    "Grainline",
    "SeamAllowance",
    "TShirtTraceRecord",
    "read_tshirt_trace",
]
