"""Provisional, expert-reviewable basic garment blocks.

This module intentionally does not claim to encode an industrial drafting
method.  It provides a small, deterministic synthetic domain for experiments
that need stable panel, path, landmark, reference-line, dart, and formula
names.  Every serialized record carries ``PROVISIONAL_EXPERT_REVIEW`` and is
invalid if it claims industrial truth or completed expert validation.

Coordinates are centimetres in a panel-local frame.  ``x`` increases from a
centre/fold line towards the side seam and ``y`` increases down the body.
Long semantic paths may contain several named landmarks (for example waist,
hip, knee, and hem) without pretending that every drafting breakpoint is a
separate semantic edge.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE_STATUS = "PROVISIONAL_EXPERT_REVIEW"
SCHEMA_VERSION = "basic-garment-blocks/v3"
SUPPORTED_CATEGORIES = ("tshirt", "pants", "skirt")
Point2D = tuple[float, float]


@dataclass(frozen=True)
class NumericBound:
    low: float
    high: float
    default: float
    unit: str = "cm"

    def validate(self, value: float, label: str) -> None:
        _finite(value, label)
        if not self.low <= float(value) <= self.high:
            raise ValueError(f"{label}={value} is outside [{self.low}, {self.high}] {self.unit}")


MEASUREMENT_BOUNDS: dict[str, dict[str, NumericBound]] = {
    "tshirt": {
        "bust_cm": NumericBound(80.0, 112.0, 92.0),
        "waist_cm": NumericBound(62.0, 102.0, 74.0),
        "hip_cm": NumericBound(84.0, 120.0, 98.0),
        "neck_circumference_cm": NumericBound(32.0, 43.0, 37.0),
        "shoulder_length_cm": NumericBound(11.0, 15.5, 13.0),
        "back_waist_length_cm": NumericBound(36.0, 46.0, 40.5),
        "bicep_circumference_cm": NumericBound(25.0, 39.0, 31.0),
        "bust_point_separation_cm": NumericBound(16.0, 22.0, 18.5),
        "shoulder_to_bust_cm": NumericBound(23.0, 31.0, 26.0),
    },
    "pants": {
        "waist_cm": NumericBound(62.0, 102.0, 74.0),
        "hip_cm": NumericBound(84.0, 120.0, 98.0),
        "outseam_cm": NumericBound(96.0, 112.0, 103.0),
        "inseam_cm": NumericBound(68.0, 88.0, 78.0),
    },
    "skirt": {
        "waist_cm": NumericBound(62.0, 102.0, 74.0),
        "hip_cm": NumericBound(84.0, 120.0, 98.0),
    },
}


DESIGN_BOUNDS: dict[str, dict[str, NumericBound]] = {
    "tshirt": {
        "chest_ease_cm": NumericBound(6.0, 16.0, 10.0),
        "waist_ease_cm": NumericBound(5.0, 16.0, 10.0),
        "hip_ease_cm": NumericBound(5.0, 16.0, 9.0),
        "body_length_cm": NumericBound(58.0, 72.0, 64.0),
        "neck_width_cm": NumericBound(6.8, 9.0, 7.6),
        "front_neck_depth_cm": NumericBound(6.5, 10.5, 8.2),
        "back_neck_depth_cm": NumericBound(1.8, 3.8, 2.5),
        "shoulder_drop_cm": NumericBound(1.5, 3.5, 2.4),
        "armhole_depth_cm": NumericBound(18.5, 25.5, 21.5),
        "sleeve_length_cm": NumericBound(16.0, 28.0, 21.0),
        "bicep_ease_cm": NumericBound(4.0, 10.0, 6.0),
        "sleeve_hem_reduction_cm": NumericBound(1.0, 5.0, 2.5),
    },
    "pants": {
        "waist_ease_cm": NumericBound(1.0, 5.0, 2.5),
        "hip_ease_cm": NumericBound(3.0, 8.0, 5.0),
        "hip_depth_cm": NumericBound(18.0, 24.0, 20.5),
        "knee_ease_cm": NumericBound(3.0, 9.0, 6.0),
        "knee_circumference_cm": NumericBound(38.0, 52.0, 44.0),
        "hem_circumference_cm": NumericBound(36.0, 48.0, 42.0),
        "front_dart_intake_cm": NumericBound(1.0, 2.2, 1.5),
        "front_dart_length_cm": NumericBound(7.0, 11.0, 9.0),
        "back_dart_intake_cm": NumericBound(2.0, 4.0, 3.0),
        "back_dart_length_cm": NumericBound(11.0, 16.0, 13.5),
        "front_crotch_extension_ratio": NumericBound(0.035, 0.060, 0.045, "ratio"),
        "back_crotch_extension_ratio": NumericBound(0.080, 0.125, 0.100, "ratio"),
        "back_waist_raise_cm": NumericBound(1.0, 3.0, 2.0),
    },
    "skirt": {
        "waist_ease_cm": NumericBound(0.5, 3.0, 1.5),
        "hip_ease_cm": NumericBound(2.0, 6.0, 4.0),
        "hip_depth_cm": NumericBound(17.0, 23.0, 20.0),
        "length_cm": NumericBound(52.0, 72.0, 60.0),
        "hem_flare_each_half_cm": NumericBound(0.0, 5.0, 1.0),
        "front_dart_intake_cm": NumericBound(1.2, 2.6, 1.8),
        "front_dart_length_cm": NumericBound(7.0, 11.0, 9.0),
        "back_dart_intake_cm": NumericBound(2.0, 4.0, 3.0),
        "back_dart_length_cm": NumericBound(11.0, 16.0, 13.5),
        "side_waist_drop_cm": NumericBound(0.5, 1.8, 1.0),
        "vent_length_cm": NumericBound(14.0, 24.0, 19.0),
    },
}


@dataclass(frozen=True)
class Formula:
    expression: str
    inputs: tuple[str, ...]

    def validate(self, label: str) -> None:
        _text(self.expression, f"{label}.expression")
        if not self.inputs:
            raise ValueError(f"{label}.inputs must not be empty")
        for index, name in enumerate(self.inputs):
            _text(name, f"{label}.inputs[{index}]")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Formula":
        return cls(str(raw["expression"]), tuple(str(value) for value in raw["inputs"]))


@dataclass(frozen=True)
class Landmark:
    name: str
    xy_cm: Point2D
    role: str
    formula: Formula

    def validate(self, label: str) -> None:
        _text(self.name, f"{label}.name")
        _text(self.role, f"{label}.role")
        _point(self.xy_cm, f"{label}.xy_cm")
        self.formula.validate(f"{label}.formula")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Landmark":
        return cls(
            name=str(raw["name"]),
            xy_cm=_point_tuple(raw["xy_cm"]),
            role=str(raw["role"]),
            formula=Formula.from_dict(raw["formula"]),
        )


@dataclass(frozen=True)
class VectorPath:
    name: str
    role: str
    landmark_sequence: tuple[str, ...]
    geometry_kind: str
    formula: Formula
    control_points_cm: tuple[Point2D, ...] = ()
    boundary: bool = True

    def validate(self, label: str, landmarks: set[str]) -> None:
        _text(self.name, f"{label}.name")
        _text(self.role, f"{label}.role")
        if self.geometry_kind not in {"line", "polyline", "cubic_bezier"}:
            raise ValueError(f"{label}.geometry_kind is unsupported: {self.geometry_kind}")
        if len(self.landmark_sequence) < 2:
            raise ValueError(f"{label}.landmark_sequence requires at least two names")
        missing = set(self.landmark_sequence) - landmarks
        if missing:
            raise ValueError(f"{label} references missing landmarks: {sorted(missing)}")
        if self.geometry_kind == "line" and len(self.landmark_sequence) != 2:
            raise ValueError(f"{label} line requires exactly two landmarks")
        if self.geometry_kind == "cubic_bezier":
            if len(self.landmark_sequence) != 2 or len(self.control_points_cm) != 2:
                raise ValueError(f"{label} cubic_bezier requires two endpoints and two controls")
        elif self.control_points_cm:
            raise ValueError(f"{label} controls are only valid for cubic_bezier")
        for index, point in enumerate(self.control_points_cm):
            _point(point, f"{label}.control_points_cm[{index}]")
        self.formula.validate(f"{label}.formula")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VectorPath":
        return cls(
            name=str(raw["name"]),
            role=str(raw["role"]),
            landmark_sequence=tuple(str(value) for value in raw["landmark_sequence"]),
            geometry_kind=str(raw["geometry_kind"]),
            formula=Formula.from_dict(raw["formula"]),
            control_points_cm=tuple(_point_tuple(value) for value in raw.get("control_points_cm", ())),
            boundary=bool(raw.get("boundary", True)),
        )


@dataclass(frozen=True)
class Dart:
    name: str
    apex_landmark: str
    inner_leg_path: str
    outer_leg_path: str
    intake_cm: float
    length_cm: float
    formula: Formula

    def validate(self, label: str, landmarks: set[str], paths: set[str]) -> None:
        _text(self.name, f"{label}.name")
        if self.apex_landmark not in landmarks:
            raise ValueError(f"{label} references missing apex {self.apex_landmark}")
        if self.inner_leg_path not in paths or self.outer_leg_path not in paths:
            raise ValueError(f"{label} references missing dart leg")
        _finite(self.intake_cm, f"{label}.intake_cm")
        _finite(self.length_cm, f"{label}.length_cm")
        if not 0.5 <= self.intake_cm <= 5.0:
            raise ValueError(f"{label}.intake_cm is implausible for this provisional corpus")
        if not 5.0 <= self.length_cm <= 18.0:
            raise ValueError(f"{label}.length_cm is implausible for this provisional corpus")
        self.formula.validate(f"{label}.formula")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Dart":
        return cls(
            name=str(raw["name"]),
            apex_landmark=str(raw["apex_landmark"]),
            inner_leg_path=str(raw["inner_leg_path"]),
            outer_leg_path=str(raw["outer_leg_path"]),
            intake_cm=float(raw["intake_cm"]),
            length_cm=float(raw["length_cm"]),
            formula=Formula.from_dict(raw["formula"]),
        )


@dataclass(frozen=True)
class ReferenceLine:
    name: str
    role: str
    start_cm: Point2D
    end_cm: Point2D
    formula: Formula

    def validate(self, label: str) -> None:
        _text(self.name, f"{label}.name")
        _text(self.role, f"{label}.role")
        _point(self.start_cm, f"{label}.start_cm")
        _point(self.end_cm, f"{label}.end_cm")
        if _distance(self.start_cm, self.end_cm) <= 1e-6:
            raise ValueError(f"{label} has zero length")
        self.formula.validate(f"{label}.formula")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReferenceLine":
        return cls(
            name=str(raw["name"]),
            role=str(raw["role"]),
            start_cm=_point_tuple(raw["start_cm"]),
            end_cm=_point_tuple(raw["end_cm"]),
            formula=Formula.from_dict(raw["formula"]),
        )


@dataclass(frozen=True)
class Symmetry:
    kind: str
    axis: str | None
    cut_quantity: int
    cut_on_fold: bool

    def validate(self, label: str) -> None:
        if self.kind not in {"cut_on_fold", "mirrored_pair", "full_piece"}:
            raise ValueError(f"{label}.kind is unsupported: {self.kind}")
        if self.cut_quantity < 1:
            raise ValueError(f"{label}.cut_quantity must be positive")
        if self.cut_on_fold and self.axis is None:
            raise ValueError(f"{label}.axis is required for a fold")
        if self.axis is not None:
            _text(self.axis, f"{label}.axis")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Symmetry":
        return cls(
            kind=str(raw["kind"]),
            axis=None if raw.get("axis") is None else str(raw["axis"]),
            cut_quantity=int(raw["cut_quantity"]),
            cut_on_fold=bool(raw["cut_on_fold"]),
        )


@dataclass(frozen=True)
class Panel:
    id: str
    role: str
    landmarks: tuple[Landmark, ...]
    paths: tuple[VectorPath, ...]
    boundary_order: tuple[str, ...]
    reference_lines: tuple[ReferenceLine, ...]
    darts: tuple[Dart, ...]
    symmetry: Symmetry
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _text(self.id, "panel.id")
        _text(self.role, f"panel {self.id}.role")
        landmark_names = _unique((item.name for item in self.landmarks), f"panel {self.id} landmark")
        path_names = _unique((item.name for item in self.paths), f"panel {self.id} path")
        if not landmark_names or not path_names:
            raise ValueError(f"panel {self.id} requires landmarks and paths")
        for index, landmark in enumerate(self.landmarks):
            landmark.validate(f"panel {self.id}.landmarks[{index}]")
        by_path = {item.name: item for item in self.paths}
        for index, path in enumerate(self.paths):
            path.validate(f"panel {self.id}.paths[{index}]", landmark_names)
        if set(self.boundary_order) != {item.name for item in self.paths if item.boundary}:
            raise ValueError(f"panel {self.id}.boundary_order must list every boundary path exactly once")
        if len(self.boundary_order) != len(set(self.boundary_order)):
            raise ValueError(f"panel {self.id}.boundary_order contains duplicates")
        for index, current_name in enumerate(self.boundary_order):
            current = by_path[current_name]
            following = by_path[self.boundary_order[(index + 1) % len(self.boundary_order)]]
            if current.landmark_sequence[-1] != following.landmark_sequence[0]:
                raise ValueError(
                    f"panel {self.id} boundary is open between {current.name} and {following.name}"
                )
        reference_names = _unique(
            (item.name for item in self.reference_lines), f"panel {self.id} reference line"
        )
        for index, line in enumerate(self.reference_lines):
            line.validate(f"panel {self.id}.reference_lines[{index}]")
        if len(reference_names) != len(self.reference_lines):
            raise AssertionError("unreachable duplicate reference line")
        dart_names = _unique((item.name for item in self.darts), f"panel {self.id} dart")
        for index, dart in enumerate(self.darts):
            dart.validate(f"panel {self.id}.darts[{index}]", landmark_names, path_names)
        if len(dart_names) != len(self.darts):
            raise AssertionError("unreachable duplicate dart")
        self.symmetry.validate(f"panel {self.id}.symmetry")
        if self.symmetry.cut_on_fold:
            fold = by_path.get(self.symmetry.axis or "")
            if fold is None:
                raise ValueError(f"panel {self.id} fold axis does not name a path")
            points = {item.name: item.xy_cm for item in self.landmarks}
            if any(abs(points[name][0]) > 1e-6 for name in fold.landmark_sequence):
                raise ValueError(f"panel {self.id} fold path must lie on x=0")
        if not isinstance(self.metadata, Mapping):
            raise ValueError(f"panel {self.id}.metadata must be an object")

    def landmark(self, name: str) -> Landmark:
        return next(item for item in self.landmarks if item.name == name)

    def path(self, name: str) -> VectorPath:
        return next(item for item in self.paths if item.name == name)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Panel":
        return cls(
            id=str(raw["id"]),
            role=str(raw["role"]),
            landmarks=tuple(Landmark.from_dict(item) for item in raw["landmarks"]),
            paths=tuple(VectorPath.from_dict(item) for item in raw["paths"]),
            boundary_order=tuple(str(value) for value in raw["boundary_order"]),
            reference_lines=tuple(ReferenceLine.from_dict(item) for item in raw.get("reference_lines", ())),
            darts=tuple(Dart.from_dict(item) for item in raw.get("darts", ())),
            symmetry=Symmetry.from_dict(raw["symmetry"]),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class SeamRelation:
    id: str
    role: str
    path_refs: tuple[str, ...]
    relation: str
    maximum_ease_ratio: float

    def validate(self, available: set[str]) -> None:
        _text(self.id, "seam.id")
        _text(self.role, f"seam {self.id}.role")
        _text(self.relation, f"seam {self.id}.relation")
        if len(self.path_refs) < 2:
            raise ValueError(f"seam {self.id} requires at least two paths")
        missing = set(self.path_refs) - available
        if missing:
            raise ValueError(f"seam {self.id} references missing paths: {sorted(missing)}")
        if not 0.0 <= self.maximum_ease_ratio <= 0.20:
            raise ValueError(f"seam {self.id}.maximum_ease_ratio is outside provisional bounds")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SeamRelation":
        return cls(
            id=str(raw["id"]),
            role=str(raw["role"]),
            path_refs=tuple(str(value) for value in raw["path_refs"]),
            relation=str(raw["relation"]),
            maximum_ease_ratio=float(raw["maximum_ease_ratio"]),
        )


@dataclass(frozen=True)
class Provenance:
    status: str = PROVENANCE_STATUS
    method: str = "bounded_correlated_parametric_basic_block_v2"
    expert_review: str = "PENDING"
    truth_scope: str = "PROVISIONAL_SYNTHETIC_TARGET"
    industrial_pattern_claim: bool = False
    notes: str = (
        "Formula ranges are deliberately conservative hypotheses for local experiments; "
        "a pattern expert must approve or revise them before gold evaluation."
    )

    def validate(self) -> None:
        if self.status != PROVENANCE_STATUS:
            raise ValueError(f"provenance.status must remain {PROVENANCE_STATUS}")
        if self.expert_review != "PENDING":
            raise ValueError("this generator cannot assert completed expert review")
        if self.industrial_pattern_claim:
            raise ValueError("provisional basic blocks cannot claim industrial pattern truth")
        _text(self.method, "provenance.method")
        _text(self.truth_scope, "provenance.truth_scope")
        _text(self.notes, "provenance.notes")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Provenance":
        return cls(
            status=str(raw["status"]),
            method=str(raw["method"]),
            expert_review=str(raw["expert_review"]),
            truth_scope=str(raw["truth_scope"]),
            industrial_pattern_claim=bool(raw["industrial_pattern_claim"]),
            notes=str(raw["notes"]),
        )


@dataclass(frozen=True)
class BasicBlock:
    sample_id: str
    category: str
    measurements: dict[str, float]
    design: dict[str, float]
    panels: tuple[Panel, ...]
    seams: tuple[SeamRelation, ...]
    provenance: Provenance = field(default_factory=Provenance)
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _text(self.sample_id, "sample_id")
        if self.category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"unsupported category: {self.category}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        _validate_values(self.measurements, MEASUREMENT_BOUNDS[self.category], "measurements")
        _validate_values(self.design, DESIGN_BOUNDS[self.category], "design")
        panel_ids = _unique((panel.id for panel in self.panels), "panel")
        for panel in self.panels:
            panel.validate()
        available = {
            f"{panel.id}:{path.name}"
            for panel in self.panels
            for path in panel.paths
        }
        seam_ids = _unique((seam.id for seam in self.seams), "seam")
        for seam in self.seams:
            seam.validate(available)
            if seam.relation in {
                "equal",
                "equal_after_neckline_review",
                "ease_checked_pair",
                "self_seam_equal",
                "sleeve_cap_ease_allowed",
            }:
                lengths = []
                for path_ref in seam.path_refs:
                    panel_id, path_name = path_ref.split(":", 1)
                    panel = self.panel(panel_id)
                    path = next(item for item in panel.paths if item.name == path_name)
                    points = _vector_path_points(panel, path, curve_samples=64)
                    lengths.append(sum(math.dist(a, b) for a, b in zip(points, points[1:])))
                shortest = min(lengths)
                if shortest <= 0.0:
                    raise ValueError(f"seam {seam.id} has a zero-length path")
                ease_ratio = (max(lengths) - shortest) / shortest
                if ease_ratio > seam.maximum_ease_ratio + 1e-6:
                    raise ValueError(
                        f"seam {seam.id} length mismatch {ease_ratio:.4f} exceeds "
                        f"{seam.maximum_ease_ratio:.4f}"
                    )
        if len(seam_ids) != len(self.seams):
            raise AssertionError("unreachable duplicate seam")
        self.provenance.validate()
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        _validate_category_contract(self, panel_ids)

    def panel(self, panel_id: str) -> Panel:
        return next(panel for panel in self.panels if panel.id == panel_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_pattern_document(self, *, curve_samples: int = 16):
        """Convert this block to a residual-editor-compatible PatternDocument.

        ``PatternDocument.annotations`` uses the exact query names from
        :mod:`semantic_teacher_student`.  Source-local names are retained in a
        documented adapter instead of silently renaming geometry.  The
        converter includes only boundary geometry and author-generated darts;
        it never invents zipper, notch, seam-allowance, or production-mark
        evidence.
        """

        if curve_samples < 4:
            raise ValueError("curve_samples must be at least 4")
        self.validate()
        from benchmark.pattern_pipeline.schema import (
            Edge as DocumentEdge,
            Panel as DocumentPanel,
            PatternDocument,
            Stitch,
            StitchSide,
        )

        document_panels: list[DocumentPanel] = []
        boundary_points: dict[tuple[str, str], tuple[Point2D, ...]] = {}
        for panel in self.panels:
            by_path = {path.name: path for path in panel.paths}
            edges = []
            for path_name in panel.boundary_order:
                path = by_path[path_name]
                points = _vector_path_points(panel, path, curve_samples=curve_samples)
                boundary_points[(panel.id, path.name)] = points
                edges.append(DocumentEdge(id=path.name, points=points, confidence=1.0))
            document_panels.append(DocumentPanel(id=panel.id, edges=tuple(edges), confidence=1.0))

        document_stitches = []
        for seam in self.seams:
            if len(seam.path_refs) != 2 or seam.relation == "same_panel_continuation":
                continue
            first_panel, first_path = seam.path_refs[0].split(":", 1)
            second_panel, second_path = seam.path_refs[1].split(":", 1)
            if (first_panel, first_path) not in boundary_points or (second_panel, second_path) not in boundary_points:
                continue
            document_stitches.append(
                Stitch(
                    id=seam.id,
                    side_a=StitchSide(first_panel, first_path, False),
                    side_b=StitchSide(second_panel, second_path, True),
                    confidence=1.0,
                )
            )

        annotations = _semantic_annotations(self, boundary_points)
        return PatternDocument(
            pattern_id=self.sample_id,
            generator="provisional_expert_review_basic_blocks",
            panels=tuple(document_panels),
            stitches=tuple(document_stitches),
            provenance=asdict(self.provenance),
            annotations=annotations,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BasicBlock":
        record = cls(
            sample_id=str(raw["sample_id"]),
            category=str(raw["category"]),
            measurements={str(name): float(value) for name, value in raw["measurements"].items()},
            design={str(name): float(value) for name, value in raw["design"].items()},
            panels=tuple(Panel.from_dict(item) for item in raw["panels"]),
            seams=tuple(SeamRelation.from_dict(item) for item in raw.get("seams", ())),
            provenance=Provenance.from_dict(raw["provenance"]),
            schema_version=str(raw["schema_version"]),
            metadata=dict(raw.get("metadata", {})),
        )
        record.validate()
        return record

    @classmethod
    def from_json(cls, payload: str) -> "BasicBlock":
        raw = json.loads(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("basic block JSON must be an object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class BasicBlockCorpus:
    records: tuple[BasicBlock, ...]
    seed: int
    provenance_status: str = PROVENANCE_STATUS
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.provenance_status != PROVENANCE_STATUS:
            raise ValueError(f"corpus provenance must be {PROVENANCE_STATUS}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"corpus schema must be {SCHEMA_VERSION}")
        _unique((record.sample_id for record in self.records), "sample")
        for record in self.records:
            record.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "provenance_status": self.provenance_status,
            "seed": self.seed,
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BasicBlockCorpus":
        records = tuple(BasicBlock.from_dict(item) for item in raw["records"])
        if int(raw.get("record_count", len(records))) != len(records):
            raise ValueError("corpus record_count does not match records")
        corpus = cls(
            records=records,
            seed=int(raw["seed"]),
            provenance_status=str(raw["provenance_status"]),
            schema_version=str(raw["schema_version"]),
        )
        corpus.validate()
        return corpus


def build_basic_block(
    category: str,
    *,
    measurements: Mapping[str, float] | None = None,
    design: Mapping[str, float] | None = None,
    sample_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BasicBlock:
    """Build one deterministic provisional block from bounded inputs."""

    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"unsupported category: {category}")
    measured = _with_defaults(MEASUREMENT_BOUNDS[category], measurements)
    styled = _with_defaults(DESIGN_BOUNDS[category], design)
    _validate_values(measured, MEASUREMENT_BOUNDS[category], "measurements")
    _validate_values(styled, DESIGN_BOUNDS[category], "design")
    builders = {"tshirt": _build_tshirt, "pants": _build_pants, "skirt": _build_skirt}
    panels, seams = builders[category](measured, styled)
    block = BasicBlock(
        sample_id=sample_id or f"{category}_basic_default",
        category=category,
        measurements=measured,
        design=styled,
        panels=panels,
        seams=seams,
        metadata={
            "coordinate_convention": "panel-local centimetres; x outward; y downward",
            "variation_policy": "bounded, topology-stable, no category mutation",
            **dict(metadata or {}),
        },
    )
    block.validate()
    return block


def generate_variations(category: str, count: int, *, seed: int = 0) -> tuple[BasicBlock, ...]:
    """Generate a repeatable, bounded set without changing category topology."""

    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"unsupported category: {category}")
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(_stable_seed(seed, category))
    records: list[BasicBlock] = []
    for index in range(count):
        measurements = _sample_values(MEASUREMENT_BOUNDS[category], rng)
        design = _sample_values(DESIGN_BOUNDS[category], rng)
        # Independent uniform body measurements produce combinations that are
        # legal per field but are not useful as common-garment blocks (for
        # example a 102 cm waist with an 84 cm hip).  Keep the declared bounds
        # while sampling conservative, explicitly documented proportions.
        if category == "tshirt":
            bust = _round(rng.uniform(82.0, 110.0))
            waist_low = max(MEASUREMENT_BOUNDS[category]["waist_cm"].low, bust - 28.0)
            waist_high = min(MEASUREMENT_BOUNDS[category]["waist_cm"].high, bust + 4.0)
            waist = _round(rng.uniform(waist_low, waist_high))
            hip_low = max(MEASUREMENT_BOUNDS[category]["hip_cm"].low, bust - 10.0, waist + 4.0)
            hip_high = min(MEASUREMENT_BOUNDS[category]["hip_cm"].high, bust + 16.0)
            measurements.update(bust_cm=bust, waist_cm=waist, hip_cm=_round(rng.uniform(hip_low, hip_high)))
        elif category in {"pants", "skirt"}:
            hip = _round(rng.uniform(86.0, 118.0))
            waist_low = max(MEASUREMENT_BOUNDS[category]["waist_cm"].low, hip - 34.0)
            waist_high = min(MEASUREMENT_BOUNDS[category]["waist_cm"].high, hip - 12.0)
            measurements.update(hip_cm=hip, waist_cm=_round(rng.uniform(waist_low, waist_high)))
        if category == "pants":
            # Keep the crotch/rise relationship coherent instead of sampling
            # two unrelated leg lengths.
            outseam = _round(rng.uniform(98.0, 110.0))
            rise = _round(rng.uniform(24.0, 29.0))
            measurements["outseam_cm"] = outseam
            measurements["inseam_cm"] = _round(outseam - rise)
        record = build_basic_block(
            category,
            measurements=measurements,
            design=design,
            sample_id=f"basic_{category}_{seed}_{index:05d}",
            metadata={"generator_seed": seed, "variation_index": index},
        )
        records.append(record)
    return tuple(records)


def generate_corpus(count_per_category: int | Mapping[str, int], *, seed: int = 0) -> BasicBlockCorpus:
    if isinstance(count_per_category, bool):
        raise ValueError("count_per_category must be an integer or mapping")
    counts = (
        {category: int(count_per_category) for category in SUPPORTED_CATEGORIES}
        if isinstance(count_per_category, int)
        else {category: int(count_per_category.get(category, 0)) for category in SUPPORTED_CATEGORIES}
    )
    if any(value < 0 for value in counts.values()):
        raise ValueError("category counts must be non-negative")
    records = tuple(
        record
        for category in SUPPORTED_CATEGORIES
        for record in generate_variations(category, counts[category], seed=seed)
    )
    corpus = BasicBlockCorpus(records=records, seed=seed)
    corpus.validate()
    return corpus


def write_corpus_json(corpus: BasicBlockCorpus, path: str | Path) -> Path:
    """Serialize a corpus; the caller chooses a tracked or ignored location."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(corpus.to_json() + "\n", encoding="utf-8")
    return destination


def load_corpus_json(path: str | Path) -> BasicBlockCorpus:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("corpus JSON must be an object")
    return BasicBlockCorpus.from_dict(raw)


def _build_tshirt(
    m: Mapping[str, float], d: Mapping[str, float]
) -> tuple[tuple[Panel, ...], tuple[SeamRelation, ...]]:
    shoulder_run = math.sqrt(m["shoulder_length_cm"] ** 2 - d["shoulder_drop_cm"] ** 2)
    sp_x = d["neck_width_cm"] + shoulder_run
    bust_x = (m["bust_cm"] + d["chest_ease_cm"]) / 4.0
    waist_x = (m["waist_cm"] + d["waist_ease_cm"]) / 4.0
    hem_x = (m["hip_cm"] + d["hip_ease_cm"]) / 4.0
    wl_y = m["back_waist_length_cm"]
    hl_y = min(d["body_length_cm"] - 2.0, wl_y + 18.0)

    def body_panel(panel_id: str, front: bool) -> Panel:
        center_name = "FNP" if front else "BNP"
        center_role = "front_neck_point" if front else "back_neck_point"
        depth_key = "front_neck_depth_cm" if front else "back_neck_depth_cm"
        center_path = "center_front" if front else "center_back"
        landmarks = (
            _lm(center_name, 0.0, d[depth_key], center_role, depth_key, (depth_key,)),
            _lm("SNP", d["neck_width_cm"], 0.0, "side_neck_point", "neck_width_cm", ("neck_width_cm",)),
            _lm(
                "SP",
                sp_x,
                d["shoulder_drop_cm"],
                "shoulder_point",
                "neck_width_cm + sqrt(shoulder_length_cm^2 - shoulder_drop_cm^2)",
                ("neck_width_cm", "shoulder_length_cm", "shoulder_drop_cm"),
            ),
            _lm(
                "UNDERARM",
                bust_x,
                d["armhole_depth_cm"],
                "armhole_side_point",
                "(bust_cm + chest_ease_cm) / 4, armhole_depth_cm",
                ("bust_cm", "chest_ease_cm", "armhole_depth_cm"),
            ),
            _lm(
                "WAIST_SIDE",
                waist_x,
                wl_y,
                "waist_side_point",
                "(waist_cm + waist_ease_cm) / 4, back_waist_length_cm",
                ("waist_cm", "waist_ease_cm", "back_waist_length_cm"),
            ),
            _lm(
                "SIDE_HEM",
                hem_x,
                d["body_length_cm"],
                "side_hem_point",
                "(hip_cm + hip_ease_cm) / 4, body_length_cm",
                ("hip_cm", "hip_ease_cm", "body_length_cm"),
            ),
            _lm("CF_HEM" if front else "CB_HEM", 0.0, d["body_length_cm"], "center_hem_point", "0, body_length_cm", ("body_length_cm",)),
            _lm(
                "BP",
                m["bust_point_separation_cm"] / 2.0,
                m["shoulder_to_bust_cm"],
                "bust_point_reference",
                "bust_point_separation_cm / 2, shoulder_to_bust_cm",
                ("bust_point_separation_cm", "shoulder_to_bust_cm"),
            ),
        )
        hem_center = "CF_HEM" if front else "CB_HEM"
        neckline_controls = (
            _p(d["neck_width_cm"] * 0.42, 0.0),
            _p(0.0, d[depth_key] * 0.48),
        )
        armhole_controls = (
            _p(bust_x - (0.7 if front else 0.4), d["armhole_depth_cm"] * 0.67),
            _p(sp_x + (2.0 if front else 1.4), d["shoulder_drop_cm"] + (5.8 if front else 4.8)),
        )
        paths = (
            _path(center_path, center_path, (center_name, hem_center), "line", "x = 0 fold line", ("body_length_cm",)),
            _path("hemline", "hemline", (hem_center, "SIDE_HEM"), "line", "quarter hip plus ease", ("hip_cm", "hip_ease_cm")),
            _path(
                "side_seam",
                "side_seam",
                ("SIDE_HEM", "WAIST_SIDE", "UNDERARM"),
                "polyline",
                "semantic side seam through HL/WL/BL levels",
                ("hip_cm", "waist_cm", "bust_cm", "body_length_cm", "back_waist_length_cm"),
            ),
            _path(
                "armhole",
                "armhole",
                ("UNDERARM", "SP"),
                "cubic_bezier",
                "bounded armhole from UNDERARM to SP",
                ("bust_cm", "chest_ease_cm", "armhole_depth_cm", "shoulder_drop_cm"),
                armhole_controls,
            ),
            _path("shoulder", "shoulder", ("SP", "SNP"), "line", "shoulder_length_cm at shoulder_drop_cm", ("shoulder_length_cm", "shoulder_drop_cm")),
            _path(
                "neckline",
                "neckline",
                ("SNP", center_name),
                "cubic_bezier",
                f"bounded {depth_key} neckline",
                ("neck_width_cm", depth_key),
                neckline_controls,
            ),
        )
        max_x = max(bust_x, waist_x, hem_x)
        references = (
            _ref("BL", "bust_line", 0.0, d["armhole_depth_cm"], max_x, d["armhole_depth_cm"], "y = armhole_depth_cm", ("armhole_depth_cm",)),
            _ref("WL", "waist_line", 0.0, wl_y, max_x, wl_y, "y = back_waist_length_cm", ("back_waist_length_cm",)),
            _ref("HL", "hip_line", 0.0, hl_y, max_x, hl_y, "y = min(body_length_cm - 2, WL + 18)", ("body_length_cm", "back_waist_length_cm")),
        )
        return Panel(
            id=panel_id,
            role="front_bodice" if front else "back_bodice",
            landmarks=landmarks,
            paths=paths,
            boundary_order=(center_path, "hemline", "side_seam", "armhole", "shoulder", "neckline"),
            reference_lines=references,
            darts=(),
            symmetry=Symmetry("cut_on_fold", center_path, 1, True),
            metadata={"dart_policy": "dartless knit basic", "half_pattern": True},
        )

    front = body_panel("front", True)
    back = body_panel("back", False)

    def path_length(panel: Panel, path_name: str) -> float:
        path = next(item for item in panel.paths if item.name == path_name)
        points = _vector_path_points(panel, path, curve_samples=64)
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    front_armhole_length = path_length(front, "armhole")
    back_armhole_length = path_length(back, "armhole")

    # A one-piece sleeve's flat bicep width is half its finished
    # circumference; ``sleeve_half`` is one half of that flat pattern.  The
    # former /2 construction doubled the sleeve width and made many cap seams
    # geometrically impossible to match to the armhole.
    sleeve_half = (m["bicep_circumference_cm"] + d["bicep_ease_cm"]) / 4.0

    def cap_controls(cap_height: float, front_half: bool) -> tuple[Point2D, Point2D]:
        if front_half:
            return (_p(-sleeve_half * 0.90, cap_height * 0.63), _p(-sleeve_half * 0.42, cap_height * 0.04))
        return (_p(sleeve_half * 0.45, cap_height * 0.02), _p(sleeve_half * 0.92, cap_height * 0.58))

    def cap_length(cap_height: float, front_half: bool) -> float:
        if front_half:
            endpoints = (_p(-sleeve_half, cap_height), _p(0.0, 0.0))
        else:
            endpoints = (_p(0.0, 0.0), _p(sleeve_half, cap_height))
        c1, c2 = cap_controls(cap_height, front_half)
        points: list[Point2D] = []
        for index in range(65):
            t = index / 64.0
            u = 1.0 - t
            points.append(
                _p(
                    u**3 * endpoints[0][0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t**3 * endpoints[1][0],
                    u**3 * endpoints[0][1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t**3 * endpoints[1][1],
                )
            )
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    # Solve one shared cap height against the two armholes.  A small 1% cap
    # ease is deliberate; it is recorded rather than hidden in the geometry.
    target_mean = (front_armhole_length + back_armhole_length) * 0.505
    low, high = 3.0, 36.0
    for _ in range(48):
        mid = (low + high) / 2.0
        mean_length = (cap_length(mid, True) + cap_length(mid, False)) / 2.0
        if mean_length < target_mean:
            low = mid
        else:
            high = mid
    cap_height = (low + high) / 2.0
    sleeve_hem_half = sleeve_half - d["sleeve_hem_reduction_cm"]
    if sleeve_hem_half <= 2.0:
        raise ValueError("sleeve hem reduction leaves an implausible sleeve opening")
    sleeve_bottom = cap_height + d["sleeve_length_cm"]
    sleeve = Panel(
        id="sleeve",
        role="sleeve",
        landmarks=(
            _lm("CAP_TOP", 0.0, 0.0, "sleeve_cap_top", "origin", ("armhole_depth_cm",)),
            _lm("BACK_UNDERARM", sleeve_half, cap_height, "back_underarm_point", "(bicep_circumference_cm + bicep_ease_cm) / 4; cap height solved from armhole lengths", ("bicep_circumference_cm", "bicep_ease_cm", "armhole_depth_cm")),
            _lm("BACK_HEM", sleeve_hem_half, sleeve_bottom, "back_sleeve_hem", "sleeve half width - reduction, cap height + sleeve length", ("bicep_circumference_cm", "bicep_ease_cm", "sleeve_hem_reduction_cm", "sleeve_length_cm")),
            _lm("FRONT_HEM", -sleeve_hem_half, sleeve_bottom, "front_sleeve_hem", "negative mirrored sleeve hem", ("bicep_circumference_cm", "bicep_ease_cm", "sleeve_hem_reduction_cm", "sleeve_length_cm")),
            _lm("FRONT_UNDERARM", -sleeve_half, cap_height, "front_underarm_point", "negative mirrored quarter bicep circumference; cap height solved from armhole lengths", ("bicep_circumference_cm", "bicep_ease_cm", "armhole_depth_cm")),
        ),
        paths=(
            _path("sleeve_head_back", "sleeve_head", ("CAP_TOP", "BACK_UNDERARM"), "cubic_bezier", "armhole-length-fitted back sleeve-cap half", ("armhole_depth_cm", "bicep_circumference_cm", "bicep_ease_cm"), cap_controls(cap_height, False)),
            _path("sleeve_underarm_back", "sleeve_underarm", ("BACK_UNDERARM", "BACK_HEM"), "line", "back underarm to sleeve hem", ("sleeve_length_cm", "sleeve_hem_reduction_cm")),
            _path("sleeve_hem", "sleeve_hem", ("BACK_HEM", "FRONT_HEM"), "line", "twice sleeve hem half width", ("bicep_circumference_cm", "bicep_ease_cm", "sleeve_hem_reduction_cm")),
            _path("sleeve_underarm_front", "sleeve_underarm", ("FRONT_HEM", "FRONT_UNDERARM"), "line", "front sleeve hem to underarm", ("sleeve_length_cm", "sleeve_hem_reduction_cm")),
            _path("sleeve_head_front", "sleeve_head", ("FRONT_UNDERARM", "CAP_TOP"), "cubic_bezier", "armhole-length-fitted front sleeve-cap half", ("armhole_depth_cm", "bicep_circumference_cm", "bicep_ease_cm"), cap_controls(cap_height, True)),
        ),
        boundary_order=("sleeve_head_back", "sleeve_underarm_back", "sleeve_hem", "sleeve_underarm_front", "sleeve_head_front"),
        reference_lines=(
            _ref("SLEEVE_GRAIN", "grainline", 0.0, 0.0, 0.0, sleeve_bottom, "x = 0 sleeve grain", ("sleeve_length_cm", "armhole_depth_cm")),
            _ref("BICEP_LINE", "bicep_line", -sleeve_half, cap_height, sleeve_half, cap_height, "y = cap height", ("armhole_depth_cm", "bicep_circumference_cm", "bicep_ease_cm")),
        ),
        darts=(),
        symmetry=Symmetry("mirrored_pair", None, 2, False),
        metadata={
            "left_right_pair": True,
            "front_back_cap_halves_are_not_interchangeable": True,
            "cap_fit_policy": "shared height fitted to front/back armholes with 1 percent mean cap ease",
            "front_armhole_length_cm": _round(front_armhole_length),
            "back_armhole_length_cm": _round(back_armhole_length),
            "front_cap_length_cm": _round(cap_length(cap_height, True)),
            "back_cap_length_cm": _round(cap_length(cap_height, False)),
        },
    )
    seams = (
        SeamRelation("shoulder", "shoulder", ("front:shoulder", "back:shoulder"), "equal_after_neckline_review", 0.02),
        SeamRelation("side", "side_seam", ("front:side_seam", "back:side_seam"), "equal", 0.03),
        SeamRelation("front_armhole", "armhole", ("front:armhole", "sleeve:sleeve_head_front"), "sleeve_cap_ease_allowed", 0.12),
        SeamRelation("back_armhole", "armhole", ("back:armhole", "sleeve:sleeve_head_back"), "sleeve_cap_ease_allowed", 0.12),
        SeamRelation("sleeve_underarm", "sleeve_underarm", ("sleeve:sleeve_underarm_front", "sleeve:sleeve_underarm_back"), "self_seam_equal", 0.03),
    )
    return (front, back, sleeve), seams


def _build_pants(
    m: Mapping[str, float], d: Mapping[str, float]
) -> tuple[tuple[Panel, ...], tuple[SeamRelation, ...]]:
    rise = m["outseam_cm"] - m["inseam_cm"]
    if not 20.0 <= rise <= 34.0:
        raise ValueError("pants outseam_cm - inseam_cm must be between 20 and 34 cm")
    knee_y = rise + m["inseam_cm"] * 0.52

    def leg_panel(panel_id: str, back: bool) -> Panel:
        prefix = "CB" if back else "CF"
        dart_intake_key = "back_dart_intake_cm" if back else "front_dart_intake_cm"
        dart_length_key = "back_dart_length_cm" if back else "front_dart_length_cm"
        crotch_ratio_key = "back_crotch_extension_ratio" if back else "front_crotch_extension_ratio"
        waist_raise = -d["back_waist_raise_cm"] if back else 0.0
        hip_quarter = (m["hip_cm"] + d["hip_ease_cm"]) / 4.0 + (1.0 if back else -0.4)
        waist_quarter = (m["waist_cm"] + d["waist_ease_cm"]) / 4.0
        side_waist_x = waist_quarter + d[dart_intake_key] + (0.6 if back else 0.0)
        # Each front/back panel carries half of one finished leg
        # circumference.  The former /4 value was then used as the *full*
        # panel width, halving both knee and hem circumferences.
        knee_panel_width = (d["knee_circumference_cm"] + d["knee_ease_cm"]) / 2.0
        hem_panel_width = d["hem_circumference_cm"] / 2.0
        leg_center_x = hip_quarter * 0.43
        side_knee_x = leg_center_x + knee_panel_width / 2.0
        inseam_knee_x = leg_center_x - knee_panel_width / 2.0
        side_hem_x = leg_center_x + hem_panel_width / 2.0
        inseam_hem_x = leg_center_x - hem_panel_width / 2.0
        crotch_extension = m["hip_cm"] * d[crotch_ratio_key]
        dart_center = side_waist_x * (0.46 if back else 0.55)
        dart_inner = dart_center - d[dart_intake_key] / 2.0
        dart_outer = dart_center + d[dart_intake_key] / 2.0
        center_waist = f"{prefix}_WAIST"
        center_hip = f"{prefix}_HIP"
        landmarks = (
            _lm(center_waist, 0.0, waist_raise, "center_waist_point", f"0, {'-back_waist_raise_cm' if back else '0'}", (("back_waist_raise_cm",) if back else ("waist_cm",))),
            _lm("WAIST_DART_INNER", dart_inner, _lerp(waist_raise, 0.0, dart_inner / side_waist_x), "waist_dart_leg", "dart centre - intake / 2 on waist", (dart_intake_key, "waist_cm", "waist_ease_cm")),
            _lm("WAIST_DART_APEX", dart_center, d[dart_length_key], "waist_dart_apex", "dart centre, dart length", (dart_length_key, dart_intake_key)),
            _lm("WAIST_DART_OUTER", dart_outer, _lerp(waist_raise, 0.0, dart_outer / side_waist_x), "waist_dart_leg", "dart centre + intake / 2 on waist", (dart_intake_key, "waist_cm", "waist_ease_cm")),
            _lm("SIDE_WAIST", side_waist_x, 0.0, "side_waist_point", "quarter waist + dart intake + back allowance", ("waist_cm", "waist_ease_cm", dart_intake_key)),
            _lm("SIDE_HIP", hip_quarter, d["hip_depth_cm"], "side_hip_point", "quarter hip plus ease and front/back balance", ("hip_cm", "hip_ease_cm", "hip_depth_cm")),
            _lm("SIDE_KNEE", side_knee_x, knee_y, "side_knee_point", "leg centre + quarter knee width", ("knee_circumference_cm", "knee_ease_cm", "outseam_cm", "inseam_cm")),
            _lm("SIDE_HEM", side_hem_x, m["outseam_cm"], "side_hem_point", "leg centre + quarter hem width", ("hem_circumference_cm", "outseam_cm")),
            _lm("INSEAM_HEM", inseam_hem_x, m["outseam_cm"], "inseam_hem_point", "leg centre - quarter hem width", ("hem_circumference_cm", "outseam_cm")),
            _lm("INSEAM_KNEE", inseam_knee_x, knee_y, "inseam_knee_point", "leg centre - quarter knee width", ("knee_circumference_cm", "knee_ease_cm", "outseam_cm", "inseam_cm")),
            _lm("CROTCH_POINT", -crotch_extension, rise, "crotch_extension_point", f"-hip_cm * {crotch_ratio_key}", ("hip_cm", crotch_ratio_key)),
            _lm(center_hip, 0.0, d["hip_depth_cm"], "center_hip_point", "x = 0 at hip depth", ("hip_depth_cm",)),
        )
        center_role = "center_back" if back else "center_front"
        paths = (
            _path("waist_inner", "waistline", (center_waist, "WAIST_DART_INNER"), "line", "waist before dart", ("waist_cm", "waist_ease_cm", dart_intake_key)),
            _path("dart_leg_inner", "dart_leg", ("WAIST_DART_INNER", "WAIST_DART_APEX"), "line", "waist dart inner leg", (dart_intake_key, dart_length_key)),
            _path("dart_leg_outer", "dart_leg", ("WAIST_DART_APEX", "WAIST_DART_OUTER"), "line", "waist dart outer leg", (dart_intake_key, dart_length_key)),
            _path("waist_outer", "waistline", ("WAIST_DART_OUTER", "SIDE_WAIST"), "line", "waist after dart", ("waist_cm", "waist_ease_cm", dart_intake_key)),
            _path("outseam", "outseam", ("SIDE_WAIST", "SIDE_HIP", "SIDE_KNEE", "SIDE_HEM"), "polyline", "one semantic outseam through WL/HL/KL/hem", ("hip_depth_cm", "outseam_cm", "inseam_cm", "knee_circumference_cm", "hem_circumference_cm")),
            _path("hemline", "hemline", ("SIDE_HEM", "INSEAM_HEM"), "line", "quarter hem circumference", ("hem_circumference_cm",)),
            _path("inseam", "inseam", ("INSEAM_HEM", "INSEAM_KNEE", "CROTCH_POINT"), "polyline", "one semantic inseam through hem/KL/crotch", ("outseam_cm", "inseam_cm", "knee_circumference_cm", "hem_circumference_cm")),
            _path("crotch_curve", "crotch_curve", ("CROTCH_POINT", center_hip), "cubic_bezier", "bounded front/back crotch extension to centre hip", ("hip_cm", crotch_ratio_key, "hip_depth_cm"), (_p(-crotch_extension * 0.45, rise), _p(0.0, d["hip_depth_cm"] + rise * 0.18))),
            _path(center_role, center_role, (center_hip, center_waist), "line", "centre grain-aligned rise above hip", ("hip_depth_cm", "back_waist_raise_cm") if back else ("hip_depth_cm", "waist_cm")),
        )
        dart = Dart(
            name="waist_dart",
            apex_landmark="WAIST_DART_APEX",
            inner_leg_path="dart_leg_inner",
            outer_leg_path="dart_leg_outer",
            intake_cm=d[dart_intake_key],
            length_cm=d[dart_length_key],
            formula=Formula("intake and length are bounded design parameters", (dart_intake_key, dart_length_key)),
        )
        min_x = -crotch_extension
        max_x = hip_quarter
        references = (
            _ref("WL", "waist_line", 0.0, waist_raise, side_waist_x, 0.0, "centre waist to side waist", ("waist_cm", "waist_ease_cm", "back_waist_raise_cm") if back else ("waist_cm", "waist_ease_cm")),
            _ref("HL", "hip_line", min_x, d["hip_depth_cm"], max_x, d["hip_depth_cm"], "y = hip_depth_cm", ("hip_depth_cm", "hip_cm", "hip_ease_cm")),
            _ref("CL", "crotch_line", min_x, rise, max_x, rise, "y = outseam_cm - inseam_cm", ("outseam_cm", "inseam_cm")),
            _ref("KL", "knee_line", inseam_knee_x, knee_y, side_knee_x, knee_y, "crotch depth + 0.52 * inseam", ("outseam_cm", "inseam_cm")),
            _ref("GRAIN", "grainline", leg_center_x, d["hip_depth_cm"], leg_center_x, m["outseam_cm"], "vertical leg centre", ("hip_cm", "hip_ease_cm", "outseam_cm")),
        )
        return Panel(
            id=panel_id,
            role="back_pants" if back else "front_pants",
            landmarks=landmarks,
            paths=paths,
            boundary_order=("waist_inner", "dart_leg_inner", "dart_leg_outer", "waist_outer", "outseam", "hemline", "inseam", "crotch_curve", center_role),
            reference_lines=references,
            darts=(dart,),
            symmetry=Symmetry("mirrored_pair", None, 2, False),
            metadata={"left_right_pair": True, "fit": "straight_leg"},
        )

    front = leg_panel("front_pants", False)
    back = leg_panel("back_pants", True)
    seams = (
        SeamRelation("side", "outseam", ("front_pants:outseam", "back_pants:outseam"), "ease_checked_pair", 0.06),
        SeamRelation("inside_leg", "inseam", ("front_pants:inseam", "back_pants:inseam"), "ease_checked_pair", 0.06),
        SeamRelation("front_rise", "crotch_curve", ("front_pants:crotch_curve", "front_pants:center_front"), "same_panel_continuation", 0.0),
        SeamRelation("back_rise", "crotch_curve", ("back_pants:crotch_curve", "back_pants:center_back"), "same_panel_continuation", 0.0),
    )
    return (front, back), seams


def _build_skirt(
    m: Mapping[str, float], d: Mapping[str, float]
) -> tuple[tuple[Panel, ...], tuple[SeamRelation, ...]]:
    def skirt_panel(panel_id: str, back: bool) -> Panel:
        prefix = "CB" if back else "CF"
        dart_intake_key = "back_dart_intake_cm" if back else "front_dart_intake_cm"
        dart_length_key = "back_dart_length_cm" if back else "front_dart_length_cm"
        hip_x = (m["hip_cm"] + d["hip_ease_cm"]) / 4.0 + (0.4 if back else -0.4)
        waist_x = (m["waist_cm"] + d["waist_ease_cm"]) / 4.0 + d[dart_intake_key]
        hem_x = hip_x + d["hem_flare_each_half_cm"]
        dart_center = waist_x * (0.43 if back else 0.56)
        inner_x = dart_center - d[dart_intake_key] / 2.0
        outer_x = dart_center + d[dart_intake_key] / 2.0
        center_waist = f"{prefix}_WAIST"
        center_hip = f"{prefix}_HIP"
        center_hem = f"{prefix}_HEM"
        center_path = "center_back" if back else "center_front"
        landmarks: list[Landmark] = [
            _lm(center_waist, 0.0, 0.0, "center_waist_point", "origin", ("waist_cm",)),
            _lm(center_hip, 0.0, d["hip_depth_cm"], "center_hip_point", "x = 0, y = hip depth", ("hip_depth_cm",)),
            _lm("WAIST_DART_INNER", inner_x, d["side_waist_drop_cm"] * inner_x / waist_x, "waist_dart_leg", "dart centre - intake / 2 on sloped waist", (dart_intake_key, "side_waist_drop_cm", "waist_cm")),
            _lm("WAIST_DART_APEX", dart_center, d[dart_length_key], "waist_dart_apex", "dart centre, dart length", (dart_intake_key, dart_length_key)),
            _lm("WAIST_DART_OUTER", outer_x, d["side_waist_drop_cm"] * outer_x / waist_x, "waist_dart_leg", "dart centre + intake / 2 on sloped waist", (dart_intake_key, "side_waist_drop_cm", "waist_cm")),
            _lm("SIDE_WAIST", waist_x, d["side_waist_drop_cm"], "side_waist_point", "quarter waist + dart intake, side waist drop", ("waist_cm", "waist_ease_cm", dart_intake_key, "side_waist_drop_cm")),
            _lm("SIDE_HIP", hip_x, d["hip_depth_cm"], "side_hip_point", "quarter hip plus ease and front/back balance", ("hip_cm", "hip_ease_cm", "hip_depth_cm")),
            _lm("SIDE_HEM", hem_x, d["length_cm"], "side_hem_point", "hip width plus bounded flare", ("hip_cm", "hip_ease_cm", "hem_flare_each_half_cm", "length_cm")),
            _lm(center_hem, 0.0, d["length_cm"], "center_hem_point", "x = 0, skirt length", ("length_cm",)),
        ]
        if back:
            landmarks.extend(
                (
                    _lm("SLIT_END", 0.0, d["length_cm"] - d["vent_length_cm"], "slit_end", "skirt length - vent length", ("length_cm", "vent_length_cm")),
                )
            )
        paths: list[VectorPath] = [
            _path("waist_inner", "waistline", (center_waist, "WAIST_DART_INNER"), "line", "waist before dart", ("waist_cm", "waist_ease_cm", dart_intake_key)),
            _path("dart_leg_inner", "dart_leg", ("WAIST_DART_INNER", "WAIST_DART_APEX"), "line", "waist dart inner leg", (dart_intake_key, dart_length_key)),
            _path("dart_leg_outer", "dart_leg", ("WAIST_DART_APEX", "WAIST_DART_OUTER"), "line", "waist dart outer leg", (dart_intake_key, dart_length_key)),
            _path("waist_outer", "waistline", ("WAIST_DART_OUTER", "SIDE_WAIST"), "line", "waist after dart", ("waist_cm", "waist_ease_cm", dart_intake_key)),
            _path("side_seam", "side_seam", ("SIDE_WAIST", "SIDE_HIP", "SIDE_HEM"), "polyline", "one semantic side seam through WL/HL/hem", ("waist_cm", "hip_cm", "hip_depth_cm", "length_cm", "hem_flare_each_half_cm")),
            _path("hemline", "hemline", ("SIDE_HEM", center_hem), "line", "bounded straight or slightly flared hem", ("hip_cm", "hip_ease_cm", "hem_flare_each_half_cm")),
        ]
        if back:
            paths.extend(
                (
                    _path("slit", "slit", (center_hem, "SLIT_END"), "line", "centre-back vent segment", ("length_cm", "vent_length_cm")),
                    _path(center_path, center_path, ("SLIT_END", center_hip, center_waist), "polyline", "centre-back seam through HL above vent; closure type not asserted", ("length_cm", "vent_length_cm", "hip_depth_cm")),
                )
            )
        else:
            paths.append(_path(center_path, center_path, (center_hem, center_hip, center_waist), "polyline", "x = 0 fold line through HL", ("length_cm", "hip_depth_cm")))
        dart = Dart(
            name="waist_dart",
            apex_landmark="WAIST_DART_APEX",
            inner_leg_path="dart_leg_inner",
            outer_leg_path="dart_leg_outer",
            intake_cm=d[dart_intake_key],
            length_cm=d[dart_length_key],
            formula=Formula("intake and length are bounded design parameters", (dart_intake_key, dart_length_key)),
        )
        references = (
            _ref("WL", "waist_line", 0.0, 0.0, waist_x, d["side_waist_drop_cm"], "centre waist to dropped side waist", ("waist_cm", "waist_ease_cm", "side_waist_drop_cm", dart_intake_key)),
            _ref("HL", "hip_line", 0.0, d["hip_depth_cm"], hip_x, d["hip_depth_cm"], "y = hip_depth_cm", ("hip_depth_cm", "hip_cm", "hip_ease_cm")),
            _ref("GRAIN", "grainline", hip_x * 0.5, d["hip_depth_cm"], hip_x * 0.5, d["length_cm"] - 4.0, "vertical grain", ("hip_depth_cm", "length_cm")),
        )
        return Panel(
            id=panel_id,
            role="back_skirt" if back else "front_skirt",
            landmarks=tuple(landmarks),
            paths=tuple(paths),
            boundary_order=(
                ("waist_inner", "dart_leg_inner", "dart_leg_outer", "waist_outer", "side_seam", "hemline", "slit", center_path)
                if back
                else ("waist_inner", "dart_leg_inner", "dart_leg_outer", "waist_outer", "side_seam", "hemline", center_path)
            ),
            reference_lines=references,
            darts=(dart,),
            symmetry=(
                Symmetry("mirrored_pair", None, 2, False)
                if back
                else Symmetry("cut_on_fold", "center_front", 1, True)
            ),
            metadata={
                "silhouette": "pencil" if d["hem_flare_each_half_cm"] <= 1.5 else "straight_slight_flare",
                "front_back_role_cue": "slit" if back else "fold",
                "closure": "NOT_ASSERTED",
            },
        )

    front = skirt_panel("front_skirt", False)
    back = skirt_panel("back_skirt", True)
    seams = (
        SeamRelation("side", "side_seam", ("front_skirt:side_seam", "back_skirt:side_seam"), "ease_checked_pair", 0.05),
        SeamRelation("back_center", "center_back", ("back_skirt:center_back", "back_skirt:slit"), "same_panel_continuation", 0.0),
    )
    return (front, back), seams


def _validate_category_contract(block: BasicBlock, panel_ids: set[str]) -> None:
    required_panels = {
        "tshirt": {"front", "back", "sleeve"},
        "pants": {"front_pants", "back_pants"},
        "skirt": {"front_skirt", "back_skirt"},
    }[block.category]
    if panel_ids != required_panels:
        raise ValueError(f"{block.category} requires panels {sorted(required_panels)}")
    if block.category == "tshirt":
        front = block.panel("front")
        back = block.panel("back")
        if front.landmark("FNP").xy_cm[1] <= back.landmark("BNP").xy_cm[1]:
            raise ValueError("basic T-shirt front neckline must be deeper than back neckline")
        for panel in (front, back):
            if panel.landmark("SP").xy_cm[0] <= panel.landmark("SNP").xy_cm[0]:
                raise ValueError("T-shirt SP must lie outward from SNP")
    elif block.category == "pants":
        for panel_id in required_panels:
            panel = block.panel(panel_id)
            if {line.name for line in panel.reference_lines} < {"WL", "HL", "CL", "KL"}:
                raise ValueError(f"{panel_id} lacks pants construction levels")
            if panel.path("outseam").landmark_sequence != (
                "SIDE_WAIST", "SIDE_HIP", "SIDE_KNEE", "SIDE_HEM"
            ):
                raise ValueError("pants outseam must remain one semantic path through named levels")
    elif block.category == "skirt":
        for panel_id, center_hip in (("front_skirt", "CF_HIP"), ("back_skirt", "CB_HIP")):
            panel = block.panel(panel_id)
            if center_hip not in {item.name for item in panel.landmarks}:
                raise ValueError(f"{panel_id} lacks required centre-hip landmark {center_hip}")
            if center_hip not in panel.path(
                "center_front" if panel_id == "front_skirt" else "center_back"
            ).landmark_sequence:
                raise ValueError(f"{panel_id} centre path must pass through {center_hip}")
        back = block.panel("back_skirt")
        required = {"SLIT_END"}
        if required - {item.name for item in back.landmarks}:
            raise ValueError("back skirt must retain a named slit endpoint")


def _with_defaults(
    bounds: Mapping[str, NumericBound], supplied: Mapping[str, float] | None
) -> dict[str, float]:
    values = {name: spec.default for name, spec in bounds.items()}
    if supplied:
        unknown = set(supplied) - set(bounds)
        if unknown:
            raise ValueError(f"unknown values: {sorted(unknown)}")
        values.update({name: float(value) for name, value in supplied.items()})
    return {name: _round(value) for name, value in values.items()}


def _validate_values(
    values: Mapping[str, float], bounds: Mapping[str, NumericBound], label: str
) -> None:
    if set(values) != set(bounds):
        missing = set(bounds) - set(values)
        extra = set(values) - set(bounds)
        raise ValueError(f"{label} keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
    for name, spec in bounds.items():
        spec.validate(float(values[name]), f"{label}.{name}")


def _sample_values(bounds: Mapping[str, NumericBound], rng: random.Random) -> dict[str, float]:
    return {name: _round(rng.uniform(spec.low, spec.high)) for name, spec in bounds.items()}


def _stable_seed(seed: int, category: str) -> int:
    digest = hashlib.sha256(f"{seed}:{category}:{SCHEMA_VERSION}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _lm(name: str, x: float, y: float, role: str, expression: str, inputs: Sequence[str]) -> Landmark:
    return Landmark(name, _p(x, y), role, Formula(expression, tuple(inputs)))


def _path(
    name: str,
    role: str,
    landmarks: Sequence[str],
    kind: str,
    expression: str,
    inputs: Sequence[str],
    controls: Sequence[Point2D] = (),
    *,
    boundary: bool = True,
) -> VectorPath:
    return VectorPath(
        name=name,
        role=role,
        landmark_sequence=tuple(landmarks),
        geometry_kind=kind,
        formula=Formula(expression, tuple(inputs)),
        control_points_cm=tuple(_p(*point) for point in controls),
        boundary=boundary,
    )


def _ref(
    name: str,
    role: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expression: str,
    inputs: Sequence[str],
) -> ReferenceLine:
    return ReferenceLine(name, role, _p(x1, y1), _p(x2, y2), Formula(expression, tuple(inputs)))


def _p(x: float, y: float) -> Point2D:
    return _round(x), _round(y)


def _round(value: float) -> float:
    return round(float(value), 6)


def _lerp(first: float, second: float, fraction: float) -> float:
    return first + (second - first) * fraction


def _distance(first: Point2D, second: Point2D) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _vector_path_points(panel: Panel, path: VectorPath, *, curve_samples: int) -> tuple[Point2D, ...]:
    by_name = {landmark.name: landmark.xy_cm for landmark in panel.landmarks}
    anchors = tuple(by_name[name] for name in path.landmark_sequence)
    if path.geometry_kind in {"line", "polyline"}:
        return anchors
    start, end = anchors
    first_control, second_control = path.control_points_cm
    points = []
    for index in range(curve_samples + 1):
        t = index / curve_samples
        u = 1.0 - t
        points.append(
            _p(
                u**3 * start[0]
                + 3.0 * u**2 * t * first_control[0]
                + 3.0 * u * t**2 * second_control[0]
                + t**3 * end[0],
                u**3 * start[1]
                + 3.0 * u**2 * t * first_control[1]
                + 3.0 * u * t**2 * second_control[1]
                + t**3 * end[1],
            )
        )
    return tuple(points)


def _semantic_annotations(
    block: BasicBlock,
    boundary_points: Mapping[tuple[str, str], tuple[Point2D, ...]],
) -> dict[str, Any]:
    """Build exact teacher/student query annotations for one anchor."""

    unknown_queries: set[str] = set()
    if block.category == "tshirt":
        panel_queries: dict[str, tuple[str, ...]] = {
            "front_bodice": ("front",),
            "back_bodice": ("back",),
            "sleeve": ("sleeve",),
        }
        landmark_queries: dict[str, tuple[tuple[str, str], ...]] = {
            "FNP": (("front", "FNP"),),
            "BNP": (("back", "BNP"),),
            "SNP_front": (("front", "SNP"),),
            "SNP_back": (("back", "SNP"),),
            "SP_front": (("front", "SP"),),
            "SP_back": (("back", "SP"),),
            "front_underarm": (("front", "UNDERARM"),),
            "back_underarm": (("back", "UNDERARM"),),
            "sleeve_cap_apex": (("sleeve", "CAP_TOP"),),
        }
        path_queries: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
            "front_neckline": (("front", ("neckline",)),),
            "back_neckline": (("back", ("neckline",)),),
            "front_shoulder": (("front", ("shoulder",)),),
            "back_shoulder": (("back", ("shoulder",)),),
            "front_armhole": (("front", ("armhole",)),),
            "back_armhole": (("back", ("armhole",)),),
            "front_side_seam": (("front", ("side_seam",)),),
            "back_side_seam": (("back", ("side_seam",)),),
            "front_hemline": (("front", ("hemline",)),),
            "back_hemline": (("back", ("hemline",)),),
            # One continuous cap path: front underarm -> cap apex -> back
            # underarm.  The order matters to topology-preserving residual
            # editing because the two dense edge polylines share CAP_TOP.
            "sleeve_head": (("sleeve", ("sleeve_head_front", "sleeve_head_back")),),
            # Front/back underarm seams are two distinct paths, not one
            # disjoint polyline.  Keep them as separate instances of the same
            # semantic query so an edit is applied independently to each.
            "sleeve_underarm": (
                ("sleeve", ("sleeve_underarm_back",)),
                ("sleeve", ("sleeve_underarm_front",)),
            ),
            "sleeve_hem": (("sleeve", ("sleeve_hem",)),),
        }
        reference_queries: dict[str, tuple[str, str]] = {
            "front_BL": ("front", "BL"),
            "back_BL": ("back", "BL"),
            "front_WL": ("front", "WL"),
            "back_WL": ("back", "WL"),
            "front_HL": ("front", "HL"),
            "back_HL": ("back", "HL"),
        }
    elif block.category == "pants":
        panel_queries = {"front_pants": ("front_pants",), "back_pants": ("back_pants",)}
        landmark_queries = {
            "CF_waist": (("front_pants", "CF_WAIST"),),
            "CB_waist": (("back_pants", "CB_WAIST"),),
            "front_side_waist": (("front_pants", "SIDE_WAIST"),),
            "back_side_waist": (("back_pants", "SIDE_WAIST"),),
            "front_side_hip": (("front_pants", "SIDE_HIP"),),
            "back_side_hip": (("back_pants", "SIDE_HIP"),),
            "front_center_hip": (("front_pants", "CF_HIP"),),
            "back_center_hip": (("back_pants", "CB_HIP"),),
            "front_crotch_point": (("front_pants", "CROTCH_POINT"),),
            "back_crotch_point": (("back_pants", "CROTCH_POINT"),),
            "front_knee_in": (("front_pants", "INSEAM_KNEE"),),
            "front_knee_out": (("front_pants", "SIDE_KNEE"),),
            "back_knee_in": (("back_pants", "INSEAM_KNEE"),),
            "back_knee_out": (("back_pants", "SIDE_KNEE"),),
            "front_hem_in": (("front_pants", "INSEAM_HEM"),),
            "front_hem_out": (("front_pants", "SIDE_HEM"),),
            "back_hem_in": (("back_pants", "INSEAM_HEM"),),
            "back_hem_out": (("back_pants", "SIDE_HEM"),),
            "front_dart_apex": (("front_pants", "WAIST_DART_APEX"),),
            "back_dart_apex": (("back_pants", "WAIST_DART_APEX"),),
            "front_dart_leg_left": (("front_pants", "WAIST_DART_INNER"),),
            "front_dart_leg_right": (("front_pants", "WAIST_DART_OUTER"),),
            "back_dart_leg_left": (("back_pants", "WAIST_DART_INNER"),),
            "back_dart_leg_right": (("back_pants", "WAIST_DART_OUTER"),),
        }
        path_queries = {
            "front_waistline": (("front_pants", ("waist_inner", "waist_outer")),),
            "back_waistline": (("back_pants", ("waist_inner", "waist_outer")),),
            "side_seam": (
                ("front_pants", ("outseam",)),
                ("back_pants", ("outseam",)),
            ),
            "inseam": (
                ("front_pants", ("inseam",)),
                ("back_pants", ("inseam",)),
            ),
            "front_crotch_curve": (("front_pants", ("crotch_curve",)),),
            "back_crotch_curve": (("back_pants", ("crotch_curve",)),),
            "hemline": (
                ("front_pants", ("hemline",)),
                ("back_pants", ("hemline",)),
            ),
            "front_dart_leg": (("front_pants", ("dart_leg_inner", "dart_leg_outer")),),
            "back_dart_leg": (("back_pants", ("dart_leg_inner", "dart_leg_outer")),),
        }
        reference_queries = {
            "front_WL": ("front_pants", "WL"),
            "back_WL": ("back_pants", "WL"),
            "front_HL": ("front_pants", "HL"),
            "back_HL": ("back_pants", "HL"),
            "front_KL": ("front_pants", "KL"),
            "back_KL": ("back_pants", "KL"),
            "front_CL": ("front_pants", "CL"),
            "back_CL": ("back_pants", "CL"),
            "front_GRAIN": ("front_pants", "GRAIN"),
            "back_GRAIN": ("back_pants", "GRAIN"),
        }
        # Deprecated combined dart queries cannot identify a physical front
        # or back instance and therefore must never broadcast one target to
        # both panels.
        unknown_queries.update({"dart_leg", "dart_apex"})
    else:
        panel_queries = {"skirt_panel": ("front_skirt", "back_skirt")}
        landmark_queries = {
            "front_center_waist": (("front_skirt", "CF_WAIST"),),
            "back_center_waist": (("back_skirt", "CB_WAIST"),),
            "front_side_waist": (("front_skirt", "SIDE_WAIST"),),
            "back_side_waist": (("back_skirt", "SIDE_WAIST"),),
            "front_side_hip": (("front_skirt", "SIDE_HIP"),),
            "back_side_hip": (("back_skirt", "SIDE_HIP"),),
            "front_center_hip": (("front_skirt", "CF_HIP"),),
            "back_center_hip": (("back_skirt", "CB_HIP"),),
            "front_hem_center": (("front_skirt", "CF_HEM"),),
            "back_hem_center": (("back_skirt", "CB_HEM"),),
            "front_hem_side": (("front_skirt", "SIDE_HEM"),),
            "back_hem_side": (("back_skirt", "SIDE_HEM"),),
            "front_dart_apex": (("front_skirt", "WAIST_DART_APEX"),),
            "back_dart_apex": (("back_skirt", "WAIST_DART_APEX"),),
            "front_dart_leg_left": (("front_skirt", "WAIST_DART_INNER"),),
            "front_dart_leg_right": (("front_skirt", "WAIST_DART_OUTER"),),
            "back_dart_leg_left": (("back_skirt", "WAIST_DART_INNER"),),
            "back_dart_leg_right": (("back_skirt", "WAIST_DART_OUTER"),),
            "slit_end": (("back_skirt", "SLIT_END"),),
        }
        path_queries = {
            "waistline": (
                ("front_skirt", ("waist_inner", "waist_outer")),
                ("back_skirt", ("waist_inner", "waist_outer")),
            ),
            "side_seam": (
                ("front_skirt", ("side_seam",)),
                ("back_skirt", ("side_seam",)),
            ),
            "center_seam": (("back_skirt", ("center_back",)),),
            "hemline": (
                ("front_skirt", ("hemline",)),
                ("back_skirt", ("hemline",)),
            ),
            "front_dart_leg": (("front_skirt", ("dart_leg_inner", "dart_leg_outer")),),
            "back_dart_leg": (("back_skirt", ("dart_leg_inner", "dart_leg_outer")),),
            "slit": (("back_skirt", ("slit",)),),
        }
        reference_queries = {
            "front_WL": ("front_skirt", "WL"),
            "back_WL": ("back_skirt", "WL"),
            "front_HL": ("front_skirt", "HL"),
            "back_HL": ("back_skirt", "HL"),
            "front_GRAIN": ("front_skirt", "GRAIN"),
            "back_GRAIN": ("back_skirt", "GRAIN"),
        }
        unknown_queries.update(
            {
                "center_waist",
                "side_waist",
                "side_hip",
                "hem_center",
                "hem_side",
                "dart_apex",
                "dart_leg_left",
                "dart_leg_right",
                "dart_leg",
                # The provisional boundary proves a centre-back seam, not a
                # particular zipper/closure path or endpoint.
                "closure",
                "closure_end",
            }
        )

    panels = {panel.id: panel for panel in block.panels}
    semantic_panels = {
        query: [{"panel_id": panel_id} for panel_id in panel_ids]
        for query, panel_ids in panel_queries.items()
    }
    semantic_landmarks: dict[str, list[dict[str, Any]]] = {}
    for query, sources in landmark_queries.items():
        entries = []
        for panel_id, landmark_name in sources:
            panel = panels[panel_id]
            path_name, point_index = _landmark_edge_location(
                panel, landmark_name, boundary_points
            )
            entries.append(
                {
                    "panel_id": panel_id,
                    "edge_id": path_name,
                    "point_index": point_index,
                    "source_landmark": landmark_name,
                }
            )
        semantic_landmarks[query] = entries
    semantic_paths = {
        query: [
            {"panel_id": panel_id, "edge_ids": list(path_names)}
            for panel_id, path_names in sources
        ]
        for query, sources in path_queries.items()
    }
    semantic_reference_lines = {}
    for query, (panel_id, line_name) in reference_queries.items():
        line = next(
            item for item in panels[panel_id].reference_lines if item.name == line_name
        )
        semantic_reference_lines[query] = [
            {
                "panel_id": panel_id,
                "line_name": line.name,
                "points_cm": [list(line.start_cm), list(line.end_cm)],
                "source_role": line.role,
            }
        ]

    # Fail if this local adapter drifts from the common teacher/student query
    # inventory.  A known missing component is false; an unasserted source
    # concept (for example closure type or a deprecated multi-instance dart)
    # is tri-state UNKNOWN and is masked from training.
    from benchmark.drafting_semantics.semantic_teacher_student import SEMANTIC_QUERY_INVENTORY

    allowed = {
        kind: {
            query.name
            for query in SEMANTIC_QUERY_INVENTORY
            if query.category == block.category and query.kind == kind
        }
        for kind in ("panel", "path", "landmark", "reference_line")
    }
    observed = {
        "panel": set(semantic_panels),
        "path": set(semantic_paths),
        "landmark": set(semantic_landmarks),
        "reference_line": set(semantic_reference_lines),
    }
    for kind in observed:
        if observed[kind] - allowed[kind]:
            raise ValueError(
                f"basic-block {kind} adapter drift: {sorted(observed[kind] - allowed[kind])}"
            )
    return {
        "semantic_panels": semantic_panels,
        "semantic_landmarks": semantic_landmarks,
        "semantic_paths": semantic_paths,
        "semantic_reference_lines": semantic_reference_lines,
        "semantic_query_presence": {
            query.name: (
                None if query.name in unknown_queries else query.name in observed[query.kind]
            )
            for query in SEMANTIC_QUERY_INVENTORY
            if query.category == block.category
        },
        "semantic_query_adapter": {
            "contract": "exact semantic_teacher_student query names",
            "source_geometry": "BasicBlock panel-local landmark and VectorPath names",
            "known_absent_are_false": True,
            "unknown_are_null_and_masked": True,
            "no_zipper_notch_or_seam_allowance_claim": True,
        },
        "semantic_coordinate_frame": {
            "canonical_u": "left_0_right_1",
            "canonical_v": "bottom_0_top_1",
            "source_y_axis_down": True,
        },
        "provenance_status": PROVENANCE_STATUS,
    }


def _landmark_edge_location(
    panel: Panel,
    landmark_name: str,
    boundary_points: Mapping[tuple[str, str], tuple[Point2D, ...]],
) -> tuple[str, int]:
    by_path = {path.name: path for path in panel.paths}
    for path_name in panel.boundary_order:
        path = by_path[path_name]
        if landmark_name not in path.landmark_sequence:
            continue
        sequence_index = path.landmark_sequence.index(landmark_name)
        if path.geometry_kind == "cubic_bezier":
            point_index = 0 if sequence_index == 0 else len(boundary_points[(panel.id, path_name)]) - 1
        else:
            point_index = sequence_index
        return path_name, point_index
    raise ValueError(f"landmark {panel.id}/{landmark_name} is not on a boundary path")


def _point_tuple(value: Sequence[Any]) -> Point2D:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("point must contain exactly two coordinates")
    return float(value[0]), float(value[1])


def _point(value: Point2D, label: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two coordinates")
    _finite(value[0], f"{label}[0]")
    _finite(value[1], f"{label}[1]")
    if max(abs(float(value[0])), abs(float(value[1]))) > 300.0:
        raise ValueError(f"{label} exceeds provisional panel extent")


def _finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")


def _unique(values: Iterable[str], label: str) -> set[str]:
    observed: set[str] = set()
    for value in values:
        _text(value, label)
        if value in observed:
            raise ValueError(f"duplicate {label} id: {value}")
        observed.add(value)
    return observed


__all__ = [
    "BasicBlock",
    "BasicBlockCorpus",
    "Dart",
    "DESIGN_BOUNDS",
    "Formula",
    "Landmark",
    "MEASUREMENT_BOUNDS",
    "NumericBound",
    "Panel",
    "PROVENANCE_STATUS",
    "Provenance",
    "ReferenceLine",
    "SCHEMA_VERSION",
    "SUPPORTED_CATEGORIES",
    "SeamRelation",
    "Symmetry",
    "VectorPath",
    "build_basic_block",
    "generate_corpus",
    "generate_variations",
    "load_corpus_json",
    "write_corpus_json",
]
