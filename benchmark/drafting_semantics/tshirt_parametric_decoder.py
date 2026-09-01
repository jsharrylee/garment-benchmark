"""Parametric T-shirt decoder built on the existing BasicBlock v3 archetype.

The vision model's contract is a bounded vector of measurements and design
parameters, never raw vertices or spline controls.  BasicBlock v3 remains the
single source of drafting formulas.  This adapter adds the pieces needed by an
inverse-pattern decoder:

* explicit, instance-aware paths (``front_armhole#left``/``#right``);
* exact reflected geometry for left/right instances;
* shared landmark references between adjacent paths;
* an auditable numerical sleeve-head constraint with configurable ease; and
* a lossless projection to the repository's canonical ``PatternDocument``.

BasicBlock's provisional/expert-review status is deliberately preserved in
the output provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from benchmark.drafting_semantics.basic_blocks import (
    BasicBlock,
    DESIGN_BOUNDS,
    MEASUREMENT_BOUNDS,
    Panel as BasicPanel,
    VectorPath,
    build_basic_block,
)
from benchmark.pattern_pipeline.schema import (
    Edge,
    Panel,
    PatternDocument,
    Stitch,
    StitchSide,
)
from benchmark.pattern_pipeline.geometry import edge_by_id, polyline_length


Point2 = tuple[float, float]

MEASUREMENT_PARAMETER_NAMES = tuple(MEASUREMENT_BOUNDS["tshirt"])
DESIGN_PARAMETER_NAMES = tuple(DESIGN_BOUNDS["tshirt"])
PARAMETER_NAMES = (*MEASUREMENT_PARAMETER_NAMES, *DESIGN_PARAMETER_NAMES, "sleeve_ease_cm")
_EXTRA_BOUNDS = {"sleeve_ease_cm": (0.0, 4.0, 0.5)}


def _bounds(name: str) -> tuple[float, float, float]:
    if name in MEASUREMENT_BOUNDS["tshirt"]:
        item = MEASUREMENT_BOUNDS["tshirt"][name]
        return float(item.low), float(item.high), float(item.default)
    if name in DESIGN_BOUNDS["tshirt"]:
        item = DESIGN_BOUNDS["tshirt"][name]
        return float(item.low), float(item.high), float(item.default)
    return _EXTRA_BOUNDS[name]


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _distance(first: Point2, second: Point2) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _close(first: Point2, second: Point2, tolerance: float = 1e-8) -> bool:
    return _distance(first, second) <= tolerance


def _mirror_x(point: Point2, axis_x_cm: float = 0.0) -> Point2:
    return (2.0 * axis_x_cm - point[0], point[1])


@dataclass(frozen=True)
class TShirtDraftParameters:
    """Bounded semantic/body vector consumed by the deterministic decoder."""

    bust_cm: float = 92.0
    waist_cm: float = 74.0
    hip_cm: float = 98.0
    neck_circumference_cm: float = 37.0
    shoulder_length_cm: float = 13.0
    back_waist_length_cm: float = 40.5
    bicep_circumference_cm: float = 31.0
    bust_point_separation_cm: float = 18.5
    shoulder_to_bust_cm: float = 26.0
    chest_ease_cm: float = 10.0
    waist_ease_cm: float = 10.0
    hip_ease_cm: float = 9.0
    body_length_cm: float = 64.0
    neck_width_cm: float = 7.6
    front_neck_depth_cm: float = 8.2
    back_neck_depth_cm: float = 2.5
    shoulder_drop_cm: float = 2.4
    armhole_depth_cm: float = 21.5
    sleeve_length_cm: float = 21.0
    bicep_ease_cm: float = 6.0
    sleeve_hem_reduction_cm: float = 2.5
    sleeve_ease_cm: float = 0.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = self.to_dict()
        if tuple(values) != PARAMETER_NAMES:
            raise RuntimeError("T-shirt parameter order drifted from PARAMETER_NAMES")
        for name, raw in values.items():
            value = _finite(raw, name)
            low, high, _ = _bounds(name)
            if not low <= value <= high:
                raise ValueError(f"{name}={value} is outside [{low}, {high}] cm")

    def to_dict(self) -> dict[str, float]:
        return {item.name: float(getattr(self, item.name)) for item in fields(self)}

    def measurement_dict(self) -> dict[str, float]:
        values = self.to_dict()
        return {name: values[name] for name in MEASUREMENT_PARAMETER_NAMES}

    def design_dict(self) -> dict[str, float]:
        values = self.to_dict()
        return {name: values[name] for name in DESIGN_PARAMETER_NAMES}

    def to_vector(self) -> tuple[float, ...]:
        """Stable parameter order for a regression head/least-squares solver."""

        values = self.to_dict()
        return tuple(values[name] for name in PARAMETER_NAMES)

    @classmethod
    def lower_bounds(cls) -> tuple[float, ...]:
        return tuple(_bounds(name)[0] for name in PARAMETER_NAMES)

    @classmethod
    def upper_bounds(cls) -> tuple[float, ...]:
        return tuple(_bounds(name)[1] for name in PARAMETER_NAMES)

    @classmethod
    def project_vector(cls, values: Sequence[Any]) -> tuple[float, ...]:
        """Box-project a model vector into the declared drafting domain."""

        if isinstance(values, (str, bytes)) or len(values) != len(PARAMETER_NAMES):
            raise ValueError(
                f"parameter vector must contain {len(PARAMETER_NAMES)} values"
            )
        result = []
        for name, raw in zip(PARAMETER_NAMES, values):
            value = _finite(raw, name)
            low, high, _ = _bounds(name)
            result.append(min(high, max(low, value)))
        return tuple(result)

    @classmethod
    def from_vector(
        cls, values: Sequence[Any], *, project: bool = False
    ) -> "TShirtDraftParameters":
        resolved = cls.project_vector(values) if project else tuple(values)
        if len(resolved) != len(PARAMETER_NAMES):
            raise ValueError(
                f"parameter vector must contain {len(PARAMETER_NAMES)} values"
            )
        return cls(**dict(zip(PARAMETER_NAMES, resolved)))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base: "TShirtDraftParameters | None" = None,
    ) -> "TShirtDraftParameters":
        if not isinstance(values, Mapping):
            raise ValueError("T-shirt parameters must be a mapping")
        unknown = sorted(set(values) - set(PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"unknown T-shirt parameters: {', '.join(unknown)}")
        merged = (base or cls()).to_dict()
        merged.update(values)
        return cls(**merged)

    def with_residual(
        self, residual_cm: Mapping[str, Any], *, project: bool = True
    ) -> "TShirtDraftParameters":
        """Add a semantic residual to the body/category archetype."""

        if not isinstance(residual_cm, Mapping):
            raise ValueError("parameter residuals must be a mapping")
        unknown = sorted(set(residual_cm) - set(PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"unknown T-shirt parameter residuals: {', '.join(unknown)}")
        values = list(self.to_vector())
        indices = {name: index for index, name in enumerate(PARAMETER_NAMES)}
        for name, raw in residual_cm.items():
            values[indices[name]] += _finite(raw, f"residual {name}")
        return type(self).from_vector(values, project=project)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "title": "TShirtDraftParameters",
            "type": "object",
            "additionalProperties": False,
            "parameterOrder": list(PARAMETER_NAMES),
            "required": list(PARAMETER_NAMES),
            "properties": {
                name: {
                    "type": "number",
                    "unit": "cm",
                    "minimum": _bounds(name)[0],
                    "maximum": _bounds(name)[1],
                    "default": _bounds(name)[2],
                    "group": (
                        "body_measurement"
                        if name in MEASUREMENT_PARAMETER_NAMES
                        else "design_parameter"
                    ),
                }
                for name in PARAMETER_NAMES
            },
        }


@dataclass(frozen=True)
class DraftSegment:
    """One line or cubic Bezier primitive in centimetres."""

    kind: str
    control_points_cm: tuple[Point2, ...]

    def __post_init__(self) -> None:
        expected = {"line": 2, "cubic_bezier": 4}
        if self.kind not in expected or len(self.control_points_cm) != expected.get(self.kind, -1):
            raise ValueError(f"invalid {self.kind!r} draft segment")
        normalized = tuple(
            (_finite(point[0], "control x"), _finite(point[1], "control y"))
            for point in self.control_points_cm
        )
        object.__setattr__(self, "control_points_cm", normalized)

    @property
    def start_cm(self) -> Point2:
        return self.control_points_cm[0]

    @property
    def end_cm(self) -> Point2:
        return self.control_points_cm[-1]

    def point(self, parameter: float) -> Point2:
        t = min(1.0, max(0.0, float(parameter)))
        if self.kind == "line":
            p0, p1 = self.control_points_cm
            return ((1 - t) * p0[0] + t * p1[0], (1 - t) * p0[1] + t * p1[1])
        p0, p1, p2, p3 = self.control_points_cm
        u = 1.0 - t
        weights = (u**3, 3 * u * u * t, 3 * u * t * t, t**3)
        return (
            sum(weight * point[0] for weight, point in zip(weights, (p0, p1, p2, p3))),
            sum(weight * point[1] for weight, point in zip(weights, (p0, p1, p2, p3))),
        )

    def sampled_points(self, samples: int = 33) -> tuple[Point2, ...]:
        if self.kind == "line":
            return self.control_points_cm
        if samples < 2:
            raise ValueError("samples must be at least two")
        return tuple(self.point(index / (samples - 1)) for index in range(samples))

    def reversed(self) -> "DraftSegment":
        return DraftSegment(self.kind, tuple(reversed(self.control_points_cm)))

    def mirrored_x(self, axis_x_cm: float = 0.0) -> "DraftSegment":
        return DraftSegment(
            self.kind,
            tuple(_mirror_x(point, axis_x_cm) for point in self.control_points_cm),
        )

    def length_cm(self, tolerance_cm: float = 1e-9) -> float:
        if self.kind == "line":
            return _distance(self.start_cm, self.end_cm)
        return _cubic_length(self.control_points_cm, tolerance_cm, 0)


def _split_cubic(points: tuple[Point2, ...]) -> tuple[tuple[Point2, ...], tuple[Point2, ...]]:
    def midpoint(first: Point2, second: Point2) -> Point2:
        return ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)

    p0, p1, p2, p3 = points
    p01, p12, p23 = midpoint(p0, p1), midpoint(p1, p2), midpoint(p2, p3)
    p012, p123 = midpoint(p01, p12), midpoint(p12, p23)
    center = midpoint(p012, p123)
    return (p0, p01, p012, center), (center, p123, p23, p3)


def _cubic_length(points: tuple[Point2, ...], tolerance_cm: float, depth: int) -> float:
    chord = _distance(points[0], points[-1])
    polygon = sum(_distance(first, second) for first, second in zip(points, points[1:]))
    if depth >= 24 or polygon - chord <= tolerance_cm:
        return 0.5 * (polygon + chord)
    left, right = _split_cubic(points)
    # ``polygon - chord`` already shrinks rapidly under subdivision.  Keeping
    # the same local tolerance avoids making every deeper branch needlessly
    # stricter while retaining deterministic sub-micrometre accuracy here.
    return _cubic_length(left, tolerance_cm, depth + 1) + _cubic_length(
        right, tolerance_cm, depth + 1
    )


@dataclass(frozen=True)
class DraftLandmark:
    id: str
    panel_id: str
    semantic_name: str
    instance: str
    xy_cm: Point2


@dataclass(frozen=True)
class DraftPath:
    id: str
    panel_id: str
    role: str
    instance: str
    start_landmark_id: str
    end_landmark_id: str
    segments: tuple[DraftSegment, ...]
    seam_role: str | None = None

    @property
    def start_cm(self) -> Point2:
        return self.segments[0].start_cm

    @property
    def end_cm(self) -> Point2:
        return self.segments[-1].end_cm

    def sampled_points(self, samples_per_cubic: int = 33) -> tuple[Point2, ...]:
        points: list[Point2] = []
        for segment in self.segments:
            sampled = segment.sampled_points(samples_per_cubic)
            points.extend(sampled if not points else sampled[1:])
        return tuple(points)

    def length_cm(self, tolerance_cm: float = 1e-9) -> float:
        return sum(segment.length_cm(tolerance_cm) for segment in self.segments)


@dataclass(frozen=True)
class CanonicalPanel:
    id: str
    role: str
    path_ids: tuple[str, ...]


@dataclass(frozen=True)
class SymmetryRelation:
    id: str
    source_panel_id: str
    target_panel_id: str
    path_pairs: tuple[tuple[str, str], ...]
    landmark_pairs: tuple[tuple[str, str], ...]
    axis: str = "x"
    axis_coordinate_cm: float = 0.0
    reverse_parameter: bool = True


@dataclass(frozen=True)
class SleeveHeadConstraintReceipt:
    id: str
    equation: str
    armhole_path_ids: tuple[str, str]
    sleeve_head_path_ids: tuple[str, str]
    mirrored_sleeve_head_path_ids: tuple[str, str]
    front_armhole_length_cm: float
    back_armhole_length_cm: float
    sleeve_ease_cm: float
    target_length_cm: float
    actual_length_cm: float
    residual_cm: float
    initial_basic_block_cap_height_cm: float
    solved_cap_height_cm: float
    iterations: int
    tolerance_cm: float
    converged: bool
    solver: str = "bracketed_bisection_on_shared_cap_height"


@dataclass(frozen=True)
class SampledSleeveConstraintReceipt:
    """Audit the sampled polyline that downstream PatternDocument users see."""

    equation: str
    samples_per_cubic: int
    front_armhole_length_cm: float
    back_armhole_length_cm: float
    front_sleeve_head_length_cm: float
    back_sleeve_head_length_cm: float
    sleeve_ease_cm: float
    target_length_cm: float
    actual_length_cm: float
    residual_cm: float
    tolerance_cm: float
    maximum_individual_armhole_relative_mismatch: float

    @property
    def passed(self) -> bool:
        return abs(self.residual_cm) <= self.tolerance_cm

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def audit_sampled_tshirt_document_sleeve_constraint(
    document: PatternDocument,
    *,
    sleeve_ease_cm: float,
    samples_per_cubic: int,
    tolerance_cm: float = 0.005,
) -> SampledSleeveConstraintReceipt:
    """Check the additive sleeve equation after cubic curves become polylines.

    Exact P0--P3 geometry is retained in ``parametric_path_geometry`` while the
    canonical boundary uses finite point samples.  This receipt prevents the
    analytic solver tolerance from being misreported as a polyline tolerance.
    """

    if samples_per_cubic < 4:
        raise ValueError("samples_per_cubic must be at least four")
    ease = _finite(sleeve_ease_cm, "sleeve_ease_cm")
    tolerance = _finite(tolerance_cm, "tolerance_cm")
    if tolerance <= 0.0:
        raise ValueError("tolerance_cm must be positive")
    panels = {panel.id: panel for panel in document.panels}

    def length(panel_id: str, edge_id: str) -> float:
        panel = panels.get(panel_id)
        edge = edge_by_id(panel, edge_id) if panel is not None else None
        if edge is None:
            raise ValueError(f"missing sampled edge {panel_id}/{edge_id}")
        return polyline_length(edge.points)

    front_armhole = length("front", "front_armhole#right")
    back_armhole = length("back", "back_armhole#right")
    front_cap = length("sleeve#right", "front_sleeve_head#right")
    back_cap = length("sleeve#right", "back_sleeve_head#right")
    target = front_armhole + back_armhole + ease
    actual = front_cap + back_cap
    mismatches = (
        abs(front_cap - front_armhole) / max(front_cap, front_armhole),
        abs(back_cap - back_armhole) / max(back_cap, back_armhole),
    )
    return SampledSleeveConstraintReceipt(
        equation="L(sampled sleeve head) = L(sampled armholes) + sleeve_ease",
        samples_per_cubic=int(samples_per_cubic),
        front_armhole_length_cm=front_armhole,
        back_armhole_length_cm=back_armhole,
        front_sleeve_head_length_cm=front_cap,
        back_sleeve_head_length_cm=back_cap,
        sleeve_ease_cm=ease,
        target_length_cm=target,
        actual_length_cm=actual,
        residual_cm=actual - target,
        tolerance_cm=tolerance,
        maximum_individual_armhole_relative_mismatch=max(mismatches),
    )


@dataclass(frozen=True)
class CanonicalPatternGraph:
    pattern_id: str
    parameters: TShirtDraftParameters
    archetype_block: BasicBlock
    panels: tuple[CanonicalPanel, ...]
    landmarks: tuple[DraftLandmark, ...]
    paths: tuple[DraftPath, ...]
    stitches: tuple[Stitch, ...]
    symmetry_relations: tuple[SymmetryRelation, ...]
    sleeve_head_constraint: SleeveHeadConstraintReceipt
    samples_per_cubic: int = 33
    schema_version: str = "parametric-tshirt-graph/v1"

    def landmark(self, landmark_id: str) -> DraftLandmark:
        try:
            return next(item for item in self.landmarks if item.id == landmark_id)
        except StopIteration as error:
            raise KeyError(landmark_id) from error

    def path(self, path_id: str) -> DraftPath:
        try:
            return next(item for item in self.paths if item.id == path_id)
        except StopIteration as error:
            raise KeyError(path_id) from error

    def validate(self) -> None:
        self.archetype_block.validate()
        panel_ids = [item.id for item in self.panels]
        landmark_ids = [item.id for item in self.landmarks]
        path_ids = [item.id for item in self.paths]
        for label, values in (
            ("panel", panel_ids),
            ("landmark", landmark_ids),
            ("path", path_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"canonical {label} ids must be unique")
        landmarks = {item.id: item for item in self.landmarks}
        paths = {item.id: item for item in self.paths}
        owned: list[str] = []
        for panel in self.panels:
            ordered = [paths[path_id] for path_id in panel.path_ids]
            owned.extend(panel.path_ids)
            for path, following in zip(ordered, ordered[1:] + ordered[:1]):
                if path.panel_id != panel.id:
                    raise ValueError(f"path {path.id} has the wrong panel owner")
                if not _close(path.end_cm, following.start_cm):
                    raise ValueError(f"open panel boundary: {path.id}->{following.id}")
        if sorted(owned) != sorted(path_ids):
            raise ValueError("every path must belong to exactly one panel")
        for path in self.paths:
            if "#" not in path.id:
                raise ValueError(f"path id is not instance-aware: {path.id}")
            if not path.segments:
                raise ValueError(f"path {path.id} has no geometry")
            for segment, following in zip(path.segments, path.segments[1:]):
                if not _close(segment.end_cm, following.start_cm):
                    raise ValueError(f"path {path.id} has disconnected primitives")
            start, end = landmarks[path.start_landmark_id], landmarks[path.end_landmark_id]
            if start.panel_id != path.panel_id or end.panel_id != path.panel_id:
                raise ValueError(f"path {path.id} landmark ownership mismatch")
            if not _close(start.xy_cm, path.start_cm) or not _close(end.xy_cm, path.end_cm):
                raise ValueError(f"path {path.id} does not meet its shared landmarks")
        for relation in self.symmetry_relations:
            for source_id, target_id in relation.landmark_pairs:
                expected = _mirror_x(landmarks[source_id].xy_cm, relation.axis_coordinate_cm)
                if not _close(expected, landmarks[target_id].xy_cm):
                    raise ValueError(f"landmark symmetry failed: {source_id}->{target_id}")
            for source_id, target_id in relation.path_pairs:
                source, target = paths[source_id], paths[target_id]
                expected = tuple(
                    segment.mirrored_x(relation.axis_coordinate_cm).reversed()
                    for segment in reversed(source.segments)
                )
                if expected != target.segments:
                    raise ValueError(f"path symmetry failed: {source_id}->{target_id}")
        receipt = self.sleeve_head_constraint
        armholes = sum(self.path(path_id).length_cm() for path_id in receipt.armhole_path_ids)
        sleeve = sum(self.path(path_id).length_cm() for path_id in receipt.sleeve_head_path_ids)
        target = armholes + receipt.sleeve_ease_cm
        if abs(target - receipt.target_length_cm) > receipt.tolerance_cm:
            raise ValueError("sleeve-head target receipt is inconsistent")
        if abs(sleeve - receipt.actual_length_cm) > receipt.tolerance_cm:
            raise ValueError("sleeve-head actual receipt is inconsistent")
        if abs(sleeve - target) > receipt.tolerance_cm or not receipt.converged:
            raise ValueError("sleeve-head constraint did not converge")

    def _occurrences(self) -> dict[str, list[dict[str, Any]]]:
        result = {landmark.id: [] for landmark in self.landmarks}
        for path in self.paths:
            result[path.start_landmark_id].append(
                {"panel_id": path.panel_id, "edge_id": path.id, "point_index": 0}
            )
            result[path.end_landmark_id].append(
                {"panel_id": path.panel_id, "edge_id": path.id, "point_index": -1}
            )
        return result

    def to_pattern_document(self) -> PatternDocument:
        """Convert exact graph geometry to the existing canonical document."""

        self.validate()
        archetype_annotations = self.archetype_block.to_pattern_document(
            curve_samples=max(4, self.samples_per_cubic - 1)
        ).annotations
        paths = {item.id: item for item in self.paths}
        panels = tuple(
            Panel(
                panel.id,
                tuple(
                    Edge(path_id, paths[path_id].sampled_points(self.samples_per_cubic), confidence=1.0)
                    for path_id in panel.path_ids
                ),
                confidence=1.0,
            )
            for panel in self.panels
        )
        occurrences = self._occurrences()
        exact_landmarks = {
            landmark.id: [{
                **occurrences[landmark.id][0],
                "semantic_name": landmark.semantic_name,
                "instance": landmark.instance,
            }]
            for landmark in self.landmarks
            if occurrences[landmark.id]
        }
        exact_paths = {
            path.id: [{
                "panel_id": path.panel_id,
                "edge_ids": [path.id],
                "role": path.role,
                "instance": path.instance,
                "seam_role": path.seam_role,
            }]
            for path in self.paths
        }

        # The positive-x instance is the exact BasicBlock v3 half-pattern, so
        # legacy semantic target names can remain usable without averaging two
        # reflected copies.  Instance-specific names remain authoritative.
        legacy_path_ids: dict[str, tuple[str, ...]] = {
            "front_neckline": ("front_neckline#right",),
            "back_neckline": ("back_neckline#right",),
            "front_shoulder": ("front_shoulder#right",),
            "back_shoulder": ("back_shoulder#right",),
            "front_armhole": ("front_armhole#right",),
            "back_armhole": ("back_armhole#right",),
            "front_side_seam": ("front_side_seam#right",),
            "back_side_seam": ("back_side_seam#right",),
            "front_hemline": ("front_hem#right",),
            "back_hemline": ("back_hem#right",),
            "sleeve_head": ("front_sleeve_head#right", "back_sleeve_head#right"),
            "sleeve_hem": ("sleeve_hem#right",),
        }
        semantic_paths = dict(exact_paths)
        for alias, edge_ids in legacy_path_ids.items():
            panel_id = paths[edge_ids[0]].panel_id
            semantic_paths[alias] = [{"panel_id": panel_id, "edge_ids": list(edge_ids)}]
        semantic_paths["sleeve_underarm"] = [
            {"panel_id": "sleeve#right", "edge_ids": ["front_sleeve_underarm#right"]},
            {"panel_id": "sleeve#right", "edge_ids": ["back_sleeve_underarm#right"]},
        ]

        legacy_landmarks = {
            "FNP": "front_FNP#center",
            "BNP": "back_BNP#center",
            "SNP_front": "front_SNP#right",
            "SNP_back": "back_SNP#right",
            "SP_front": "front_SP#right",
            "SP_back": "back_SP#right",
            "front_underarm": "front_UNDERARM#right",
            "back_underarm": "back_UNDERARM#right",
            "sleeve_cap_apex": "sleeve_CAP_TOP#right",
        }
        semantic_landmarks = dict(exact_landmarks)
        for alias, landmark_id in legacy_landmarks.items():
            semantic_landmarks[alias] = [dict(exact_landmarks[landmark_id][0])]

        geometry = {
            path.id: {
                "panel_id": path.panel_id,
                "role": path.role,
                "instance": path.instance,
                "start_landmark_id": path.start_landmark_id,
                "end_landmark_id": path.end_landmark_id,
                "segments": [asdict(segment) for segment in path.segments],
                "length_cm": path.length_cm(),
            }
            for path in self.paths
        }
        vector = self.parameters.to_vector()
        return PatternDocument(
            pattern_id=self.pattern_id,
            generator="BasicBlock v3 + semantic parametric decoder",
            panels=panels,
            stitches=self.stitches,
            provenance={
                **asdict(self.archetype_block.provenance),
                "decoder": "semantic_parameters_plus_drafting_and_seam_solver",
                "raw_control_points_predicted": False,
                "archetype_schema_version": self.archetype_block.schema_version,
            },
            annotations={
                "parametric_graph_schema_version": self.schema_version,
                "semantic_parameters_cm": self.parameters.to_dict(),
                "bounded_parameter_vector": {
                    "names": list(PARAMETER_NAMES),
                    "values": list(vector),
                    "lower_bounds": list(self.parameters.lower_bounds()),
                    "upper_bounds": list(self.parameters.upper_bounds()),
                },
                "semantic_panels": {
                    "front_bodice": [{"panel_id": "front"}],
                    "back_bodice": [{"panel_id": "back"}],
                    "sleeve": [{"panel_id": "sleeve#right", "mirrored_panel_id": "sleeve#left"}],
                },
                "semantic_landmarks": semantic_landmarks,
                "shared_landmark_occurrences": occurrences,
                "semantic_paths": semantic_paths,
                "semantic_reference_lines": archetype_annotations[
                    "semantic_reference_lines"
                ],
                "semantic_query_presence": archetype_annotations[
                    "semantic_query_presence"
                ],
                "semantic_query_adapter": {
                    **archetype_annotations["semantic_query_adapter"],
                    "instance_policy": (
                        "legacy queries select the positive-x BasicBlock instance; "
                        "#left/#right names address exact physical instances"
                    ),
                },
                "provenance_status": archetype_annotations["provenance_status"],
                "parametric_path_geometry": geometry,
                "symmetry_relations": [asdict(item) for item in self.symmetry_relations],
                "seam_constraints": [asdict(self.sleeve_head_constraint)],
                "edge_labels": {
                    f"{path.panel_id}/{path.id}": path.role for path in self.paths
                },
                "semantic_coordinate_frame": {
                    "units": "cm",
                    "x_axis": "fold_to_side_on_positive_instance",
                    "y_axis": "shoulder_to_hem",
                    "source_y_axis_down": True,
                    "symmetry_axis_x_cm": 0.0,
                },
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pattern_id": self.pattern_id,
            "parameters": self.parameters.to_dict(),
            "archetype_block": self.archetype_block.to_dict(),
            "panels": [asdict(item) for item in self.panels],
            "landmarks": [asdict(item) for item in self.landmarks],
            "paths": [asdict(item) for item in self.paths],
            "stitches": [asdict(item) for item in self.stitches],
            "symmetry_relations": [asdict(item) for item in self.symmetry_relations],
            "sleeve_head_constraint": asdict(self.sleeve_head_constraint),
            "pattern_document": self.to_pattern_document().to_dict(),
        }


class SleeveHeadConstraintError(ValueError):
    pass


def _segments_for_path(
    panel: BasicPanel,
    path: VectorPath,
    coordinates: Mapping[str, Point2],
    *,
    controls: tuple[Point2, Point2] | None = None,
) -> tuple[DraftSegment, ...]:
    anchors = tuple(coordinates[name] for name in path.landmark_sequence)
    if path.geometry_kind == "cubic_bezier":
        first, second = controls if controls is not None else path.control_points_cm
        return (DraftSegment("cubic_bezier", (anchors[0], first, second, anchors[-1])),)
    return tuple(
        DraftSegment("line", (first, second))
        for first, second in zip(anchors, anchors[1:])
    )


def _scaled_sleeve_geometry(
    sleeve: BasicPanel, cap_height_cm: float
) -> tuple[dict[str, Point2], dict[str, tuple[Point2, Point2]]]:
    original = {item.name: item.xy_cm for item in sleeve.landmarks}
    apex_y = original["CAP_TOP"][1]
    original_cap_height = original["BACK_UNDERARM"][1] - apex_y
    if original_cap_height <= 0.0:
        raise SleeveHeadConstraintError("BasicBlock sleeve cap height is degenerate")
    coordinates: dict[str, Point2] = {}
    for name, point in original.items():
        if name.endswith("UNDERARM"):
            y = apex_y + cap_height_cm
        elif name.endswith("HEM"):
            y = point[1] + cap_height_cm - original_cap_height
        else:
            y = point[1]
        coordinates[name] = (point[0], y)
    controls: dict[str, tuple[Point2, Point2]] = {}
    for path in sleeve.paths:
        if path.geometry_kind != "cubic_bezier":
            continue
        transformed = []
        for x, y in path.control_points_cm:
            ratio = (y - apex_y) / original_cap_height
            transformed.append((x, apex_y + ratio * cap_height_cm))
        controls[path.name] = (transformed[0], transformed[1])
    return coordinates, controls


def _solve_shared_cap_height(
    sleeve: BasicPanel,
    target_length_cm: float,
    *,
    tolerance_cm: float,
    maximum_iterations: int,
) -> tuple[float, float, int]:
    paths = {item.name: item for item in sleeve.paths}

    def total(cap_height: float) -> float:
        coordinates, controls = _scaled_sleeve_geometry(sleeve, cap_height)
        return sum(
            _segments_for_path(
                sleeve,
                paths[name],
                coordinates,
                controls=controls[name],
            )[0].length_cm()
            for name in ("sleeve_head_front", "sleeve_head_back")
        )

    lower, upper = 0.0, max(1.0, sleeve.landmark("BACK_UNDERARM").xy_cm[1])
    minimum = total(lower)
    if target_length_cm < minimum - tolerance_cm:
        raise SleeveHeadConstraintError(
            "requested sleeve head is shorter than the fixed bicep-width minimum "
            f"(target={target_length_cm:.6f}, minimum={minimum:.6f})"
        )
    while total(upper) < target_length_cm:
        upper *= 2.0
        if upper > 80.0:
            raise SleeveHeadConstraintError("could not bracket sleeve-cap height")
    for iteration in range(1, maximum_iterations + 1):
        middle = 0.5 * (lower + upper)
        actual = total(middle)
        if abs(actual - target_length_cm) <= tolerance_cm:
            return middle, actual, iteration
        if actual < target_length_cm:
            lower = middle
        else:
            upper = middle
    raise SleeveHeadConstraintError("sleeve-head constraint solver did not converge")


class TShirtParametricDraftingDecoder:
    def __init__(
        self,
        *,
        samples_per_cubic: int = 33,
        constraint_tolerance_cm: float = 1e-6,
        maximum_solver_iterations: int = 80,
    ) -> None:
        if samples_per_cubic < 4:
            raise ValueError("samples_per_cubic must be at least four")
        if constraint_tolerance_cm <= 0.0 or not math.isfinite(constraint_tolerance_cm):
            raise ValueError("constraint_tolerance_cm must be finite and positive")
        if maximum_solver_iterations <= 0:
            raise ValueError("maximum_solver_iterations must be positive")
        self.samples_per_cubic = int(samples_per_cubic)
        self.constraint_tolerance_cm = float(constraint_tolerance_cm)
        self.maximum_solver_iterations = int(maximum_solver_iterations)

    def decode(
        self,
        parameters: TShirtDraftParameters | Mapping[str, Any] | None = None,
        *,
        parameter_residuals_cm: Mapping[str, Any] | None = None,
        pattern_id: str = "parametric_tshirt",
    ) -> CanonicalPatternGraph:
        if parameters is None:
            resolved = TShirtDraftParameters()
        elif isinstance(parameters, TShirtDraftParameters):
            resolved = parameters
        elif isinstance(parameters, Mapping):
            resolved = TShirtDraftParameters.from_mapping(parameters)
        else:
            raise ValueError("parameters must be a TShirtDraftParameters or mapping")
        if parameter_residuals_cm is not None:
            resolved = resolved.with_residual(parameter_residuals_cm)
        if not isinstance(pattern_id, str) or not pattern_id.strip():
            raise ValueError("pattern_id is required")

        block = build_basic_block(
            "tshirt",
            measurements=resolved.measurement_dict(),
            design=resolved.design_dict(),
            sample_id=f"{pattern_id.strip()}_archetype",
            metadata={"decoder": "tshirt_parametric_decoder"},
        )
        landmarks: list[DraftLandmark] = []
        paths: list[DraftPath] = []
        panels: list[CanonicalPanel] = []
        symmetries: list[SymmetryRelation] = []

        def add_landmark(
            landmark_id: str,
            panel_id: str,
            semantic_name: str,
            instance: str,
            point: Point2,
        ) -> None:
            landmarks.append(DraftLandmark(landmark_id, panel_id, semantic_name, instance, point))

        def add_path(
            path_id: str,
            panel_id: str,
            role: str,
            instance: str,
            start_id: str,
            end_id: str,
            segments: tuple[DraftSegment, ...],
            seam_role: str | None = None,
        ) -> DraftPath:
            result = DraftPath(
                path_id,
                panel_id,
                role,
                instance,
                start_id,
                end_id,
                segments,
                seam_role,
            )
            paths.append(result)
            return result

        def mirrored(
            source: DraftPath,
            target_id: str,
            panel_id: str,
            start_id: str,
            end_id: str,
        ) -> DraftPath:
            return add_path(
                target_id,
                panel_id,
                source.role,
                "left",
                start_id,
                end_id,
                tuple(
                    segment.mirrored_x().reversed()
                    for segment in reversed(source.segments)
                ),
                source.seam_role,
            )

        body_paths: dict[str, DraftPath] = {}
        for body in (block.panel("front"), block.panel("back")):
            coordinates = {item.name: item.xy_cm for item in body.landmarks}
            right_landmarks: dict[str, str] = {}
            left_landmarks: dict[str, str] = {}
            landmark_pairs: list[tuple[str, str]] = []
            for source in body.landmarks:
                if abs(source.xy_cm[0]) <= 1e-9:
                    landmark_id = f"{body.id}_{source.name}#center"
                    add_landmark(landmark_id, body.id, source.name, "center", source.xy_cm)
                    right_landmarks[source.name] = landmark_id
                    left_landmarks[source.name] = landmark_id
                else:
                    right_id = f"{body.id}_{source.name}#right"
                    left_id = f"{body.id}_{source.name}#left"
                    add_landmark(right_id, body.id, source.name, "right", source.xy_cm)
                    add_landmark(left_id, body.id, source.name, "left", _mirror_x(source.xy_cm))
                    right_landmarks[source.name] = right_id
                    left_landmarks[source.name] = left_id
                    landmark_pairs.append((right_id, left_id))

            source_paths = {item.name: item for item in body.paths}
            boundary_names = tuple(
                name
                for name in body.boundary_order
                if name not in {"center_front", "center_back"}
            )
            right: dict[str, DraftPath] = {}
            left: dict[str, DraftPath] = {}
            for name in boundary_names:
                source = source_paths[name]
                stem_role = "hem" if source.role == "hemline" else source.role
                stem = f"{body.id}_{stem_role}"
                right[name] = add_path(
                    f"{stem}#right",
                    body.id,
                    source.role,
                    "right",
                    right_landmarks[source.landmark_sequence[0]],
                    right_landmarks[source.landmark_sequence[-1]],
                    _segments_for_path(body, source, coordinates),
                )
                left[name] = mirrored(
                    right[name],
                    f"{stem}#left",
                    body.id,
                    left_landmarks[source.landmark_sequence[-1]],
                    left_landmarks[source.landmark_sequence[0]],
                )
                body_paths[right[name].id] = right[name]
                body_paths[left[name].id] = left[name]
            ordered = tuple(right[name].id for name in boundary_names) + tuple(
                left[name].id for name in reversed(boundary_names)
            )
            panels.append(CanonicalPanel(body.id, body.role, ordered))
            symmetries.append(
                SymmetryRelation(
                    f"{body.id}_left_right_symmetry",
                    body.id,
                    body.id,
                    tuple((right[name].id, left[name].id) for name in boundary_names),
                    tuple(landmark_pairs),
                )
            )

        front_armhole = body_paths["front_armhole#right"].length_cm()
        back_armhole = body_paths["back_armhole#right"].length_cm()
        sleeve_target = front_armhole + back_armhole + resolved.sleeve_ease_cm
        source_sleeve = block.panel("sleeve")
        initial_height = source_sleeve.landmark("BACK_UNDERARM").xy_cm[1]
        solved_height, _, iterations = _solve_shared_cap_height(
            source_sleeve,
            sleeve_target,
            tolerance_cm=self.constraint_tolerance_cm,
            maximum_iterations=self.maximum_solver_iterations,
        )
        sleeve_coordinates, sleeve_controls = _scaled_sleeve_geometry(
            source_sleeve, solved_height
        )
        right_panel, left_panel = "sleeve#right", "sleeve#left"
        right_landmarks = {}
        left_landmarks = {}
        landmark_pairs = []
        for source in source_sleeve.landmarks:
            right_id = f"sleeve_{source.name}#right"
            left_id = f"sleeve_{source.name}#left"
            point = sleeve_coordinates[source.name]
            add_landmark(right_id, right_panel, source.name, "right", point)
            add_landmark(left_id, left_panel, source.name, "left", _mirror_x(point))
            right_landmarks[source.name] = right_id
            left_landmarks[source.name] = left_id
            landmark_pairs.append((right_id, left_id))

        sleeve_stems = {
            "sleeve_head_back": "back_sleeve_head",
            "sleeve_underarm_back": "back_sleeve_underarm",
            "sleeve_hem": "sleeve_hem",
            "sleeve_underarm_front": "front_sleeve_underarm",
            "sleeve_head_front": "front_sleeve_head",
        }
        source_paths = {item.name: item for item in source_sleeve.paths}
        right_sleeve_paths: dict[str, DraftPath] = {}
        left_sleeve_paths: dict[str, DraftPath] = {}
        for name in source_sleeve.boundary_order:
            source = source_paths[name]
            seam_role = (
                "front_armhole" if name == "sleeve_head_front"
                else "back_armhole" if name == "sleeve_head_back"
                else None
            )
            right_path = add_path(
                f"{sleeve_stems[name]}#right",
                right_panel,
                source.role,
                "right",
                right_landmarks[source.landmark_sequence[0]],
                right_landmarks[source.landmark_sequence[-1]],
                _segments_for_path(
                    source_sleeve,
                    source,
                    sleeve_coordinates,
                    controls=sleeve_controls.get(name),
                ),
                seam_role,
            )
            right_sleeve_paths[name] = right_path
            left_sleeve_paths[name] = mirrored(
                right_path,
                f"{sleeve_stems[name]}#left",
                left_panel,
                left_landmarks[source.landmark_sequence[-1]],
                left_landmarks[source.landmark_sequence[0]],
            )
        panels.append(
            CanonicalPanel(
                right_panel,
                "sleeve",
                tuple(right_sleeve_paths[name].id for name in source_sleeve.boundary_order),
            )
        )
        panels.append(
            CanonicalPanel(
                left_panel,
                "sleeve",
                tuple(
                    left_sleeve_paths[name].id
                    for name in reversed(source_sleeve.boundary_order)
                ),
            )
        )
        symmetries.append(
            SymmetryRelation(
                "sleeve_left_right_symmetry",
                right_panel,
                left_panel,
                tuple(
                    (right_sleeve_paths[name].id, left_sleeve_paths[name].id)
                    for name in source_sleeve.boundary_order
                ),
                tuple(landmark_pairs),
            )
        )
        front_cap = right_sleeve_paths["sleeve_head_front"]
        back_cap = right_sleeve_paths["sleeve_head_back"]
        left_front_cap = left_sleeve_paths["sleeve_head_front"]
        left_back_cap = left_sleeve_paths["sleeve_head_back"]
        actual = front_cap.length_cm() + back_cap.length_cm()
        receipt = SleeveHeadConstraintReceipt(
            "sleeve_head_matches_armholes_with_ease",
            "L(sleeve_head) = L(front_armhole) + L(back_armhole) + sleeve_ease",
            ("front_armhole#right", "back_armhole#right"),
            (front_cap.id, back_cap.id),
            (left_front_cap.id, left_back_cap.id),
            front_armhole,
            back_armhole,
            resolved.sleeve_ease_cm,
            sleeve_target,
            actual,
            actual - sleeve_target,
            initial_height,
            solved_height,
            iterations,
            self.constraint_tolerance_cm,
            abs(actual - sleeve_target) <= self.constraint_tolerance_cm,
        )

        stitches = tuple(
            Stitch(stitch_id, StitchSide(*first), StitchSide(*second, reversed=reverse_second))
            for stitch_id, first, second, reverse_second in (
                ("shoulder_seam#right", ("front", "front_shoulder#right"), ("back", "back_shoulder#right"), True),
                ("shoulder_seam#left", ("front", "front_shoulder#left"), ("back", "back_shoulder#left"), True),
                ("side_seam#right", ("front", "front_side_seam#right"), ("back", "back_side_seam#right"), True),
                ("side_seam#left", ("front", "front_side_seam#left"), ("back", "back_side_seam#left"), True),
                ("front_armhole_seam#right", ("front", "front_armhole#right"), (right_panel, front_cap.id), False),
                ("back_armhole_seam#right", ("back", "back_armhole#right"), (right_panel, back_cap.id), True),
                ("front_armhole_seam#left", ("front", "front_armhole#left"), (left_panel, left_front_cap.id), False),
                ("back_armhole_seam#left", ("back", "back_armhole#left"), (left_panel, left_back_cap.id), True),
                ("sleeve_underarm_seam#right", (right_panel, "front_sleeve_underarm#right"), (right_panel, "back_sleeve_underarm#right"), True),
                ("sleeve_underarm_seam#left", (left_panel, "front_sleeve_underarm#left"), (left_panel, "back_sleeve_underarm#left"), True),
            )
        )
        graph = CanonicalPatternGraph(
            pattern_id.strip(),
            resolved,
            block,
            tuple(panels),
            tuple(landmarks),
            tuple(paths),
            stitches,
            tuple(symmetries),
            receipt,
            self.samples_per_cubic,
        )
        graph.validate()
        return graph

    def decode_document(
        self,
        parameters: TShirtDraftParameters | Mapping[str, Any] | None = None,
        *,
        parameter_residuals_cm: Mapping[str, Any] | None = None,
        pattern_id: str = "parametric_tshirt",
    ) -> PatternDocument:
        return self.decode(
            parameters,
            parameter_residuals_cm=parameter_residuals_cm,
            pattern_id=pattern_id,
        ).to_pattern_document()


def decode_tshirt_pattern(
    parameters: TShirtDraftParameters | Mapping[str, Any] | None = None,
    *,
    parameter_residuals_cm: Mapping[str, Any] | None = None,
    pattern_id: str = "parametric_tshirt",
) -> CanonicalPatternGraph:
    return TShirtParametricDraftingDecoder().decode(
        parameters,
        parameter_residuals_cm=parameter_residuals_cm,
        pattern_id=pattern_id,
    )


__all__ = [
    "CanonicalPanel",
    "CanonicalPatternGraph",
    "DESIGN_PARAMETER_NAMES",
    "DraftLandmark",
    "DraftPath",
    "DraftSegment",
    "MEASUREMENT_PARAMETER_NAMES",
    "PARAMETER_NAMES",
    "SampledSleeveConstraintReceipt",
    "SleeveHeadConstraintError",
    "SleeveHeadConstraintReceipt",
    "SymmetryRelation",
    "TShirtDraftParameters",
    "TShirtParametricDraftingDecoder",
    "audit_sampled_tshirt_document_sleeve_constraint",
    "decode_tshirt_pattern",
]
