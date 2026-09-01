from __future__ import annotations

import json
import math
from numbers import Real
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


EDGE_ROLES = (
    "other",
    "neckline",
    "shoulder",
    "armhole",
    "center_front",
    "center_back",
    "side_seam",
    "waistline",
    "hemline",
    "dart_leg",
    "sleeve_head",
    "sleeve_underarm",
    "sleeve_hem",
    "cuff_attachment",
    "inseam",
    "outseam",
    "crotch_curve",
    "collar_attachment",
)

PANEL_ROLES = (
    "other",
    "front_bodice",
    "back_bodice",
    "front_skirt",
    "back_skirt",
    "front_pants",
    "back_pants",
    "sleeve",
    "collar",
    "cuff",
    "waistband",
)

EVIDENCE_KINDS = (
    "observed_source",
    "derived_topology",
    "derived_generator_formula",
    "recipe_reconstruction",
    "synthetic_unvalidated",
    "unavailable",
)


@dataclass(frozen=True)
class Landmark:
    name: str
    panel_id: str
    xy_cm: tuple[float, float]
    evidence: str
    confidence: float
    vertex_index: int | None = None
    training_eligible: bool = True


@dataclass(frozen=True)
class ReferenceLine:
    name: str
    panel_id: str
    points_cm: tuple[tuple[float, float], tuple[float, float]]
    evidence: str
    confidence: float
    intersects_panel: bool = True
    training_eligible: bool = True

    def validate(self, *, expected_panel_id: str | None = None) -> None:
        """Validate concrete reference-line evidence without inventing geometry.

        A ``ReferenceLine`` always stores an actual finite segment.  A line
        that exists mathematically but does not intersect its panel is valid
        (the source T-shirt HL uses this state), but it cannot be a training
        coordinate target.  Conversely, an intersecting synthetic line may
        be retained for review while remaining ineligible.
        """

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("reference line name is required")
        if not isinstance(self.panel_id, str) or not self.panel_id.strip():
            raise ValueError(f"reference line {self.name} panel_id is required")
        if expected_panel_id is not None and self.panel_id != expected_panel_id:
            raise ValueError(
                f"reference line panel mismatch: {self.name} belongs to "
                f"{self.panel_id!r}, expected {expected_panel_id!r}"
            )
        if not isinstance(self.points_cm, (tuple, list)) or len(self.points_cm) != 2:
            raise ValueError(
                f"reference line {self.name} requires exactly two endpoints"
            )
        endpoints: list[tuple[float, float]] = []
        for endpoint in self.points_cm:
            if not isinstance(endpoint, (tuple, list)) or len(endpoint) != 2:
                raise ValueError(
                    f"reference line {self.name} endpoints must be 2D points"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in endpoint
            ):
                raise ValueError(
                    f"reference line {self.name} endpoints must be finite numbers"
                )
            endpoints.append((float(endpoint[0]), float(endpoint[1])))
        if math.dist(endpoints[0], endpoints[1]) <= 1e-8:
            raise ValueError(
                f"reference line {self.name} endpoints must be nondegenerate"
            )
        if self.evidence not in EVIDENCE_KINDS:
            raise ValueError(
                f"unknown reference line evidence kind: {self.evidence}"
            )
        if self.evidence == "unavailable":
            raise ValueError(
                f"reference line {self.name} has concrete endpoints and cannot use unavailable evidence"
            )
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, Real)
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError(
                f"reference line {self.name} confidence must be finite and in [0, 1]"
            )
        if not isinstance(self.intersects_panel, bool):
            raise ValueError(
                f"reference line {self.name} intersects_panel must be boolean"
            )
        if not isinstance(self.training_eligible, bool):
            raise ValueError(
                f"reference line {self.name} training_eligible must be boolean"
            )
        if self.training_eligible and not self.intersects_panel:
            raise ValueError(
                f"reference line {self.name} cannot be training eligible without intersecting its panel"
            )
        if self.training_eligible and self.evidence == "synthetic_unvalidated":
            raise ValueError(
                f"synthetic_unvalidated reference line {self.name} cannot be training eligible"
            )
        if self.training_eligible and float(self.confidence) <= 0.0:
            raise ValueError(
                f"training-eligible reference line {self.name} requires positive confidence"
            )


@dataclass(frozen=True)
class Dart:
    panel_id: str
    kind: str
    leg_edge_ids: tuple[str, str]
    apex_cm: tuple[float, float]
    base_cm: tuple[tuple[float, float], tuple[float, float]]
    intake_cm: float
    depth_cm: float
    evidence: str = "observed_source"
    confidence: float = 1.0


@dataclass(frozen=True)
class ConstructionStep:
    order: int
    operation: str
    panel_role: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    evidence: str = "recipe_reconstruction"
    source_reference: str | None = None
    training_eligible: bool = False


@dataclass(frozen=True)
class EdgeAnnotation:
    id: str
    index: int
    endpoints: tuple[int, int]
    start_cm: tuple[float, float]
    end_cm: tuple[float, float]
    curvature_type: str
    role: str
    stitched: bool
    self_stitched: bool
    length_cm: float
    evidence: str
    confidence: float


@dataclass(frozen=True)
class PanelAnnotation:
    id: str
    role: str
    vertices_cm: tuple[tuple[float, float], ...]
    edges: tuple[EdgeAnnotation, ...]
    landmarks: tuple[Landmark, ...] = ()
    reference_lines: tuple[ReferenceLine, ...] = ()


@dataclass(frozen=True)
class DraftingSemanticRecord:
    sample_id: str
    split: str
    panels: tuple[PanelAnnotation, ...]
    darts: tuple[Dart, ...]
    measurements: dict[str, Any]
    construction_steps: tuple[ConstructionStep, ...]
    body_condition_cm: dict[str, float]
    program: dict[str, Any]
    provenance: dict[str, Any]
    production_annotations: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "drafting-semantic-1.0"

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        panel_ids = {panel.id for panel in self.panels}
        if len(panel_ids) != len(self.panels):
            raise ValueError("panel ids must be unique")
        edge_ids: set[str] = set()
        for panel in self.panels:
            if panel.role not in PANEL_ROLES:
                raise ValueError(f"unknown panel role: {panel.role}")
            for edge in panel.edges:
                if edge.id in edge_ids:
                    raise ValueError(f"duplicate edge id: {edge.id}")
                edge_ids.add(edge.id)
                if edge.role not in EDGE_ROLES:
                    raise ValueError(f"unknown edge role: {edge.role}")
                if min(edge.endpoints) < 0 or max(edge.endpoints) >= len(panel.vertices_cm):
                    raise ValueError(f"invalid endpoint in {edge.id}")
            for landmark in panel.landmarks:
                if landmark.evidence not in EVIDENCE_KINDS:
                    raise ValueError(f"unknown evidence kind: {landmark.evidence}")
                if landmark.panel_id != panel.id:
                    raise ValueError(f"landmark panel mismatch: {landmark.name}")
            reference_names: set[str] = set()
            for line in panel.reference_lines:
                line.validate(expected_panel_id=panel.id)
                if line.name in reference_names:
                    raise ValueError(
                        f"duplicate reference line name in panel {panel.id}: {line.name}"
                    )
                reference_names.add(line.name)
        for dart in self.darts:
            if dart.panel_id not in panel_ids:
                raise ValueError(f"dart references missing panel: {dart.panel_id}")
            if not set(dart.leg_edge_ids).issubset(edge_ids):
                raise ValueError(f"dart references missing edge: {dart.leg_edge_ids}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DraftingSemanticRecord":
        panels = []
        for raw_panel in value["panels"]:
            edges = []
            for edge in raw_panel["edges"]:
                edges.append(
                    EdgeAnnotation(
                        **{
                            **edge,
                            "endpoints": tuple(int(item) for item in edge["endpoints"]),
                            "start_cm": tuple(float(item) for item in edge["start_cm"]),
                            "end_cm": tuple(float(item) for item in edge["end_cm"]),
                        }
                    )
                )
            landmarks = []
            for item in raw_panel.get("landmarks", []):
                landmarks.append(Landmark(**{**item, "xy_cm": tuple(float(value) for value in item["xy_cm"])}))
            reference_lines = []
            for item in raw_panel.get("reference_lines", []):
                reference_lines.append(
                    ReferenceLine(
                        **{
                            **item,
                            "points_cm": tuple(tuple(float(value) for value in point) for point in item["points_cm"]),
                        }
                    )
                )
            panels.append(
                PanelAnnotation(
                    id=raw_panel["id"],
                    role=raw_panel["role"],
                    vertices_cm=tuple(tuple(float(v) for v in point) for point in raw_panel["vertices_cm"]),
                    edges=tuple(edges),
                    landmarks=tuple(landmarks),
                    reference_lines=tuple(reference_lines),
                )
            )
        darts = []
        for item in value.get("darts", []):
            darts.append(
                Dart(
                    **{
                        **item,
                        "leg_edge_ids": tuple(item["leg_edge_ids"]),
                        "apex_cm": tuple(float(value) for value in item["apex_cm"]),
                        "base_cm": tuple(tuple(float(value) for value in point) for point in item["base_cm"]),
                    }
                )
            )
        construction_steps = []
        for item in value.get("construction_steps", []):
            construction_steps.append(
                ConstructionStep(
                    **{
                        **item,
                        "inputs": tuple(item.get("inputs", ())),
                        "outputs": tuple(item.get("outputs", ())),
                    }
                )
            )
        record = cls(
            sample_id=value["sample_id"],
            split=value.get("split", "unknown"),
            panels=tuple(panels),
            darts=tuple(darts),
            measurements=value.get("measurements", {}),
            construction_steps=tuple(construction_steps),
            body_condition_cm={key: float(raw) for key, raw in value.get("body_condition_cm", {}).items()},
            program=value.get("program", {}),
            provenance=value.get("provenance", {}),
            production_annotations=value.get("production_annotations", {}),
            schema_version=value.get("schema_version", "drafting-semantic-1.0"),
        )
        record.validate()
        return record
