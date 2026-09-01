"""Evidence-graded lower-body drafting landmarks.

The GarmentCodeData v2 semantic records contain exact panel boundary geometry
and darts, but they do not contain a patternmaker's complete named-point
ontology.  This module turns only recoverable geometry into pants/skirt
landmarks.  Every expected field is emitted: unsupported fields are explicit
``available=False`` observations rather than invented coordinates.

Two distinctions are important for downstream learning:

* ``derived_topology`` targets (for example a waist/outseam intersection) are
  suitable pseudo-ground truth inside the GarmentCode domain;
* ``synthetic_unvalidated`` targets (currently the conventional half-leg knee
  level) are retained for review but are not training eligible.

Skirt panels are intentionally canonicalised to the exchangeable role
``skirt_panel``.  A source ``front_skirt``/``back_skirt`` name is retained as
metadata, but it is not treated as visible truth unless a discriminating dart,
slit, zipper, or notch is actually present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .schema import DraftingSemanticRecord, EdgeAnnotation, PanelAnnotation


PANTS_PANEL_ROLES = frozenset({"front_pants", "back_pants"})
SKIRT_PANEL_ROLES = frozenset({"front_skirt", "back_skirt"})

PANTS_SHARED_LANDMARKS = (
    "SIDE_WAIST",
    "SIDE_HIP",
    "CROTCH_POINT",
    "KNEE_SIDE",
    "KNEE_INSEAM",
    "HEM_SIDE",
    "HEM_INSEAM",
)
PANTS_FRONT_LANDMARKS = ("CF_WAIST", "CF_HIP", *PANTS_SHARED_LANDMARKS)
PANTS_BACK_LANDMARKS = ("CB_WAIST", "CB_HIP", *PANTS_SHARED_LANDMARKS)
PANTS_REFERENCE_LINES = ("WL", "HL", "CL", "KNEE_LINE", "GRAIN")

SKIRT_LANDMARKS = (
    "WAIST_LEFT",
    "WAIST_RIGHT",
    "HIP_LEFT",
    "HIP_RIGHT",
    "HEM_LEFT",
    "HEM_RIGHT",
    "SLIT_TOP",
    "SLIT_HEM_LEFT",
    "SLIT_HEM_RIGHT",
)
SKIRT_REFERENCE_LINES = ("WL", "HL", "GRAIN")
LOWER_BODY_FEATURES = ("dart", "slit", "zipper", "notches")

_EPSILON = 1e-8


def _finite_point(value: tuple[float, float] | None) -> bool:
    return value is not None and len(value) == 2 and all(math.isfinite(float(item)) for item in value)


@dataclass(frozen=True)
class PointObservation:
    """One named point, including an explicit unavailable state."""

    name: str
    panel_id: str
    available: bool
    xy_cm: tuple[float, float] | None
    evidence: str
    confidence: float
    source_ids: tuple[str, ...] = ()
    training_eligible: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available != _finite_point(self.xy_cm):
            raise ValueError(f"{self.name}: available and xy_cm disagree")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.name}: confidence must be in [0, 1]")
        if not self.available and not self.reason:
            raise ValueError(f"{self.name}: unavailable observations require a reason")


@dataclass(frozen=True)
class LineObservation:
    """One construction/reference line with evidence and applicability."""

    name: str
    panel_id: str
    available: bool
    points_cm: tuple[tuple[float, float], tuple[float, float]] | None
    evidence: str
    confidence: float
    intersects_panel: bool = False
    source_ids: tuple[str, ...] = ()
    training_eligible: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        valid = self.points_cm is not None and len(self.points_cm) == 2 and all(
            _finite_point(point) for point in self.points_cm
        )
        if self.available != valid:
            raise ValueError(f"{self.name}: available and points_cm disagree")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.name}: confidence must be in [0, 1]")
        if not self.available and not self.reason:
            raise ValueError(f"{self.name}: unavailable observations require a reason")


@dataclass(frozen=True)
class DartObservation:
    """A source-observed or topology-derived dart."""

    name: str
    panel_id: str
    apex_cm: tuple[float, float]
    leg_a_cm: tuple[float, float]
    leg_b_cm: tuple[float, float]
    intake_cm: float
    depth_cm: float
    kind: str
    evidence: str
    confidence: float
    source_edge_ids: tuple[str, str]
    training_eligible: bool


@dataclass(frozen=True)
class FeatureObservation:
    """Availability of a non-coordinate construction feature."""

    name: str
    available: bool
    evidence: str
    confidence: float
    training_eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.name}: confidence must be in [0, 1]")
        if not self.available and not self.reason:
            raise ValueError(f"{self.name}: unavailable observations require a reason")


@dataclass(frozen=True)
class LowerBodyPanelSemantics:
    panel_id: str
    garment_kind: str
    source_role: str
    canonical_role: str
    front_back_exchangeable: bool
    landmarks: tuple[PointObservation, ...]
    reference_lines: tuple[LineObservation, ...]
    darts: tuple[DartObservation, ...]
    features: tuple[FeatureObservation, ...]

    def landmark(self, name: str) -> PointObservation:
        return next(item for item in self.landmarks if item.name == name)

    def reference_line(self, name: str) -> LineObservation:
        return next(item for item in self.reference_lines if item.name == name)

    def feature(self, name: str) -> FeatureObservation:
        return next(item for item in self.features if item.name == name)


@dataclass(frozen=True)
class LowerBodySemanticRecord:
    sample_id: str
    split: str
    panels: tuple[LowerBodyPanelSemantics, ...]
    # 1.1 adds the explicit pants crotch construction line (CL).  Historical
    # 1.0 manifests remain historical and must be regenerated before reuse.
    ontology_version: str = "lower-body-landmarks-1.1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point(value: Sequence[float]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _scale(a: tuple[float, float], value: float) -> tuple[float, float]:
    return a[0] * value, a[1] * value


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _midpoint(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _unit(value: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(*value)
    if length <= _EPSILON:
        return None
    return value[0] / length, value[1] / length


def _unique_points(points: Iterable[tuple[float, float]], tolerance: float = 1e-6) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for point in points:
        value = _point(point)
        if not any(math.dist(value, other) <= tolerance for other in output):
            output.append(value)
    return output


def _farthest_pair(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    unique = _unique_points(points)
    if len(unique) < 2:
        return None
    return max(
        ((first, second) for index, first in enumerate(unique) for second in unique[index + 1 :]),
        key=lambda pair: (math.dist(*pair), pair),
    )


def _edge_points(panel: PanelAnnotation, edge: EdgeAnnotation) -> tuple[tuple[float, float], tuple[float, float]]:
    return _point(panel.vertices_cm[edge.endpoints[0]]), _point(panel.vertices_cm[edge.endpoints[1]])


def _source_landmark(panel: PanelAnnotation, aliases: Iterable[str]) -> PointObservation | None:
    wanted = {name.upper() for name in aliases}
    candidates = [item for item in panel.landmarks if item.name.upper() in wanted]
    if not candidates:
        return None
    item = sorted(candidates, key=lambda value: (-value.confidence, value.name, value.xy_cm))[0]
    return PointObservation(
        name=sorted(wanted)[0],
        panel_id=panel.id,
        available=True,
        xy_cm=_point(item.xy_cm),
        evidence=item.evidence,
        confidence=float(item.confidence),
        source_ids=(item.name,),
        training_eligible=bool(item.training_eligible),
    )


def _available_point(
    name: str,
    panel: PanelAnnotation,
    xy_cm: tuple[float, float],
    *,
    evidence: str,
    confidence: float,
    source_ids: Iterable[str] = (),
    training_eligible: bool = True,
) -> PointObservation:
    source = _source_landmark(panel, (name,))
    if source is not None:
        return PointObservation(**{**asdict(source), "name": name})
    return PointObservation(
        name=name,
        panel_id=panel.id,
        available=True,
        xy_cm=_point(xy_cm),
        evidence=evidence,
        confidence=confidence,
        source_ids=tuple(sorted(str(item) for item in source_ids)),
        training_eligible=training_eligible,
    )


def _unavailable_point(name: str, panel: PanelAnnotation, reason: str) -> PointObservation:
    return PointObservation(name, panel.id, False, None, "unavailable", 0.0, reason=reason)


def _available_line(
    name: str,
    panel: PanelAnnotation,
    points: tuple[tuple[float, float], tuple[float, float]],
    *,
    evidence: str,
    confidence: float,
    intersects_panel: bool,
    source_ids: Iterable[str] = (),
    training_eligible: bool = True,
) -> LineObservation:
    existing = [item for item in panel.reference_lines if item.name.upper() == name.upper()]
    if existing:
        item = sorted(existing, key=lambda value: -value.confidence)[0]
        return LineObservation(
            name,
            panel.id,
            True,
            tuple(_point(point) for point in item.points_cm),
            item.evidence,
            float(item.confidence),
            bool(item.intersects_panel),
            (item.name,),
            bool(item.training_eligible),
        )
    return LineObservation(
        name,
        panel.id,
        True,
        tuple(_point(point) for point in points),
        evidence,
        confidence,
        intersects_panel,
        tuple(sorted(str(item) for item in source_ids)),
        training_eligible,
    )


def _unavailable_line(name: str, panel: PanelAnnotation, reason: str) -> LineObservation:
    return LineObservation(name, panel.id, False, None, "unavailable", 0.0, reason=reason)


def _candidate_waist_points(panel: PanelAnnotation) -> tuple[list[tuple[float, float]], tuple[str, ...], str, float]:
    waist_edges = [edge for edge in panel.edges if edge.role == "waistline"]
    if waist_edges:
        pair = _farthest_pair([point for edge in waist_edges for point in _edge_points(panel, edge)])
        if pair is not None:
            return list(pair), tuple(edge.id for edge in waist_edges), "derived_topology", 0.95

    # GarmentCode serialises lower-body panels upright (waist at maximum Y),
    # including older records whose pants waist segments were mislabeled as
    # hemline.  The fallback is deliberately lower confidence and explicit.
    vertices = [_point(value) for value in panel.vertices_cm]
    if not vertices:
        return [], (), "unavailable", 0.0
    min_y, max_y = min(point[1] for point in vertices), max(point[1] for point in vertices)
    tolerance = max((max_y - min_y) * 0.025, 1e-5)
    top = [point for point in vertices if max_y - point[1] <= tolerance]
    pair = _farthest_pair(top)
    if pair is None:
        return [], (), "unavailable", 0.0
    source_ids = tuple(
        edge.id
        for edge in panel.edges
        if all(max_y - point[1] <= tolerance for point in _edge_points(panel, edge))
    )
    return list(pair), source_ids, "derived_topology", 0.68


def _candidate_hem_points(
    panel: PanelAnnotation,
    waist_mid: tuple[float, float],
) -> tuple[list[tuple[float, float]], tuple[str, ...]]:
    hem_edges = [edge for edge in panel.edges if edge.role == "hemline"]
    if hem_edges:
        distances = [math.dist(_midpoint(_edge_points(panel, edge)), waist_mid) for edge in hem_edges]
        maximum = max(distances)
        tolerance = max(maximum * 0.05, 1e-5)
        selected = [edge for edge, distance in zip(hem_edges, distances) if maximum - distance <= tolerance]
        pair = _farthest_pair([point for edge in selected for point in _edge_points(panel, edge)])
        if pair is not None:
            return list(pair), tuple(edge.id for edge in selected)

    vertices = [_point(value) for value in panel.vertices_cm]
    if len(vertices) < 2:
        return [], ()
    distances = [math.dist(point, waist_mid) for point in vertices]
    ranked = sorted(zip(distances, vertices), key=lambda item: (-item[0], item[1]))
    pair = _farthest_pair([point for _, point in ranked[: max(2, min(4, len(ranked)))]] )
    return (list(pair), ()) if pair is not None else ([], ())


@dataclass(frozen=True)
class _PanelFrame:
    waist_points: tuple[tuple[float, float], tuple[float, float]]
    hem_points: tuple[tuple[float, float], tuple[float, float]]
    waist_mid: tuple[float, float]
    hem_mid: tuple[float, float]
    grain: tuple[float, float]
    cross: tuple[float, float]
    waist_source_ids: tuple[str, ...]
    hem_source_ids: tuple[str, ...]
    waist_evidence: str
    waist_confidence: float


def _panel_frame(panel: PanelAnnotation) -> _PanelFrame | None:
    waist, waist_ids, waist_evidence, waist_confidence = _candidate_waist_points(panel)
    if len(waist) != 2:
        return None
    waist_mid = _midpoint(waist)
    hem, hem_ids = _candidate_hem_points(panel, waist_mid)
    if len(hem) != 2:
        return None
    hem_mid = _midpoint(hem)
    grain = _unit(_sub(waist_mid, hem_mid))
    if grain is None:
        return None
    cross = (-grain[1], grain[0])
    # Keep LEFT/RIGHT deterministic in the source Cartesian frame.
    if abs(cross[0]) >= abs(cross[1]):
        if cross[0] < 0:
            cross = _scale(cross, -1.0)
    elif cross[1] < 0:
        cross = _scale(cross, -1.0)
    waist_sorted = tuple(sorted(waist, key=lambda point: (_dot(point, cross), point)))
    hem_sorted = tuple(sorted(hem, key=lambda point: (_dot(point, cross), point)))
    return _PanelFrame(
        waist_points=waist_sorted,  # type: ignore[arg-type]
        hem_points=hem_sorted,  # type: ignore[arg-type]
        waist_mid=waist_mid,
        hem_mid=hem_mid,
        grain=grain,
        cross=cross,
        waist_source_ids=tuple(sorted(waist_ids)),
        hem_source_ids=tuple(sorted(hem_ids)),
        waist_evidence=waist_evidence,
        waist_confidence=waist_confidence,
    )


def _intersections_at_level(
    panel: PanelAnnotation,
    grain: tuple[float, float],
    level: float,
) -> list[tuple[tuple[float, float], str]]:
    intersections: list[tuple[tuple[float, float], str]] = []
    for edge in panel.edges:
        first, second = _edge_points(panel, edge)
        a, b = _dot(first, grain), _dot(second, grain)
        if abs(a - b) <= _EPSILON:
            if abs(level - a) <= 1e-6:
                intersections.extend(((first, edge.id), (second, edge.id)))
            continue
        amount = (level - a) / (b - a)
        if -1e-7 <= amount <= 1.0 + 1e-7:
            point = _add(first, _scale(_sub(second, first), min(max(amount, 0.0), 1.0)))
            intersections.append((point, edge.id))
    output: list[tuple[tuple[float, float], str]] = []
    for point, edge_id in intersections:
        if not any(math.dist(point, other) <= 1e-6 for other, _ in output):
            output.append((point, edge_id))
    return output


def _line_span_at_level(
    panel: PanelAnnotation,
    frame: _PanelFrame,
    level: float,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[str, ...]] | None:
    points = _intersections_at_level(panel, frame.grain, level)
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda item: (_dot(item[0], frame.cross), item[0], item[1]))
    return (ordered[0][0], ordered[-1][0]), tuple(sorted({ordered[0][1], ordered[-1][1]}))


def _hip_depth(record: DraftingSemanticRecord) -> tuple[float | None, str, float]:
    for key in ("garmentcode_waist_to_hip_cm", "waist_to_hip_cm"):
        value = record.measurements.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            return float(value), "derived_generator_formula", 0.92
    value = record.body_condition_cm.get("hips_line")
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
        return float(value), "derived_generator_formula", 0.9
    return None, "unavailable", 0.0


def _vertex_index(panel: PanelAnnotation, point: tuple[float, float]) -> int | None:
    candidates = [
        (math.dist(_point(value), point), index)
        for index, value in enumerate(panel.vertices_cm)
    ]
    if not candidates:
        return None
    distance, index = min(candidates)
    return index if distance <= 1e-5 else None


def _incident_roles(panel: PanelAnnotation, point: tuple[float, float]) -> set[str]:
    index = _vertex_index(panel, point)
    if index is None:
        return set()
    return {edge.role for edge in panel.edges if index in edge.endpoints}


def _waist_to_hem_chain(
    panel: PanelAnnotation,
    frame: _PanelFrame,
    start: tuple[float, float],
) -> tuple[tuple[float, float], ...] | None:
    """Return the non-waist boundary chain from one waist end to the hem."""

    start_index = _vertex_index(panel, start)
    stop_indices = {
        index
        for point in frame.hem_points
        if (index := _vertex_index(panel, point)) is not None
    }
    if start_index is None or not stop_indices:
        return None
    waist_level = _dot(frame.waist_mid, frame.grain)
    hem_level = _dot(frame.hem_mid, frame.grain)
    tolerance = max(abs(waist_level - hem_level) * 0.025, 1e-5)
    graph: dict[int, list[int]] = {}
    for edge in panel.edges:
        first, second = _edge_points(panel, edge)
        levels = _dot(first, frame.grain), _dot(second, frame.grain)
        on_waist = all(abs(level - waist_level) <= tolerance for level in levels)
        on_hem = all(abs(level - hem_level) <= tolerance for level in levels)
        if edge.role == "dart_leg" or on_waist or on_hem:
            continue
        a, b = edge.endpoints
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    queue: list[tuple[int, tuple[int, ...]]] = [(start_index, (start_index,))]
    while queue:
        current, path = queue.pop(0)
        if current in stop_indices and current != start_index:
            return tuple(_point(panel.vertices_cm[index]) for index in path)
        for neighbor in sorted(graph.get(current, ())):
            if neighbor not in path:
                queue.append((neighbor, (*path, neighbor)))
    return None


def _crotch_chain_score(
    chain: tuple[tuple[float, float], ...] | None,
    frame: _PanelFrame,
) -> float:
    if chain is None or len(chain) < 2:
        return -math.inf
    cross_values = [_dot(point, frame.cross) for point in chain]
    low, high = sorted((cross_values[0], cross_values[-1]))
    excursion = max(max(cross_values) - high, low - min(cross_values), 0.0)
    chord = max(math.dist(chain[0], chain[-1]), 1e-6)
    path_length = sum(math.dist(first, second) for first, second in zip(chain, chain[1:]))
    panel_width = max(
        max(_dot(_point(point), frame.cross) for point in panel_points)
        - min(_dot(_point(point), frame.cross) for point in panel_points),
        1e-6,
    ) if (panel_points := chain) else 1.0
    return excursion / panel_width + max(path_length / chord - 1.0, 0.0)


def _pants_waist_assignment(
    panel: PanelAnnotation,
    frame: _PanelFrame,
) -> tuple[tuple[float, float], tuple[float, float], float, str]:
    first, second = frame.waist_points
    first_roles, second_roles = _incident_roles(panel, first), _incident_roles(panel, second)
    center_roles = {"crotch_curve", "center_front", "center_back", "inseam"}
    side_roles = {"outseam", "side_seam"}
    if first_roles & center_roles and second_roles & side_roles:
        return first, second, 0.97, "semantic boundary intersections"
    if second_roles & center_roles and first_roles & side_roles:
        return second, first, 0.97, "semantic boundary intersections"

    first_score = _crotch_chain_score(_waist_to_hem_chain(panel, frame, first), frame)
    second_score = _crotch_chain_score(_waist_to_hem_chain(panel, frame, second), frame)
    if math.isfinite(first_score) and math.isfinite(second_score) and abs(first_score - second_score) > 1e-4:
        center = first if first_score > second_score else second
        side = second if center == first else first
        return center, side, 0.72, "crotch-chain boundary excursion"

    # Last-resort GarmentCode placement convention.  It remains available for
    # legacy records but is deliberately too low-confidence to masquerade as
    # source-observed drafting truth.
    centroid = _midpoint([_point(value) for value in panel.vertices_cm])
    center = max(
        (first, second),
        key=lambda point: (abs(_dot(_sub(point, centroid), frame.cross)), point),
    )
    side = second if center == first else first
    return center, side, 0.52, "ambiguous geometry; panel-centroid placement fallback"


def _shared_role_vertex(panel: PanelAnnotation, first_role: str, second_role: str) -> tuple[tuple[float, float], tuple[str, ...]] | None:
    first = [edge for edge in panel.edges if edge.role == first_role]
    second = [edge for edge in panel.edges if edge.role == second_role]
    candidates: list[tuple[tuple[float, float], tuple[str, ...]]] = []
    for left in first:
        for right in second:
            shared = set(left.endpoints) & set(right.endpoints)
            for index in shared:
                candidates.append((_point(panel.vertices_cm[index]), tuple(sorted((left.id, right.id)))))
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0] if candidates else None


def _geometric_crotch(
    panel: PanelAnnotation,
    frame: _PanelFrame,
    center_waist: tuple[float, float],
    side_waist: tuple[float, float],
) -> tuple[tuple[float, float], tuple[str, ...]] | None:
    direction = 1.0 if _dot(_sub(center_waist, side_waist), frame.cross) >= 0 else -1.0
    waist_cross = _dot(center_waist, frame.cross)
    waist_level = _dot(frame.waist_mid, frame.grain)
    hem_level = _dot(frame.hem_mid, frame.grain)
    candidates = []
    for index, raw in enumerate(panel.vertices_cm):
        point = _point(raw)
        level = _dot(point, frame.grain)
        if level >= waist_level - 1e-6 or level <= hem_level + 1e-6:
            continue
        excursion = direction * (_dot(point, frame.cross) - waist_cross)
        candidates.append((excursion, level, point, index))
    if not candidates:
        return None
    _, _, point, index = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    source_ids = tuple(sorted(edge.id for edge in panel.edges if index in edge.endpoints))
    return point, source_ids


def _dart_observations(record: DraftingSemanticRecord, panel: PanelAnnotation) -> tuple[DartObservation, ...]:
    darts = [dart for dart in record.darts if dart.panel_id == panel.id]
    ordered = sorted(darts, key=lambda item: (item.apex_cm, item.base_cm, item.leg_edge_ids))
    return tuple(
        DartObservation(
            name=f"DART_{index + 1}",
            panel_id=panel.id,
            apex_cm=_point(dart.apex_cm),
            leg_a_cm=_point(dart.base_cm[0]),
            leg_b_cm=_point(dart.base_cm[1]),
            intake_cm=float(dart.intake_cm),
            depth_cm=float(dart.depth_cm),
            kind=dart.kind,
            evidence=dart.evidence,
            confidence=float(dart.confidence),
            source_edge_ids=tuple(dart.leg_edge_ids),
            training_eligible=dart.evidence not in {"synthetic_unvalidated", "unavailable"},
        )
        for index, dart in enumerate(ordered)
    )


def _production_feature(
    record: DraftingSemanticRecord,
    singular: str,
    plural: str,
) -> FeatureObservation:
    raw = record.production_annotations.get(f"source_{plural}")
    if isinstance(raw, Mapping) and bool(raw.get("available")):
        return FeatureObservation(singular, True, "observed_source", 1.0, True)
    reason = None
    if isinstance(raw, Mapping):
        reason = str(raw.get("reason") or "source annotation reports unavailable")
    return FeatureObservation(
        singular,
        False,
        "unavailable",
        0.0,
        False,
        reason or f"{singular} is not encoded by the source record",
    )


def _slit_landmarks(
    panel: PanelAnnotation,
    frame: _PanelFrame | None,
) -> dict[str, PointObservation]:
    result: dict[str, PointObservation] = {}
    aliases = {
        "SLIT_TOP": {"SLIT_TOP", "SLIT_END", "SLIT_APEX"},
        "SLIT_HEM_LEFT": {"SLIT_HEM_LEFT", "SLIT_LEFT"},
        "SLIT_HEM_RIGHT": {"SLIT_HEM_RIGHT", "SLIT_RIGHT"},
    }
    for canonical, names in aliases.items():
        candidates = [item for item in panel.landmarks if item.name.upper() in names]
        if candidates:
            item = sorted(candidates, key=lambda value: (-value.confidence, value.name, value.xy_cm))[0]
            result[canonical] = PointObservation(
                canonical,
                panel.id,
                True,
                _point(item.xy_cm),
                item.evidence,
                float(item.confidence),
                (item.name,),
                bool(item.training_eligible),
            )

    # Explicitly named slit edges are source evidence.  Generic concavities are
    # never treated as slits because a dart, vent, or decorative cut can have
    # the same geometry.
    slit_edges = [edge for edge in panel.edges if "slit" in edge.id.lower()]
    if slit_edges and "SLIT_TOP" not in result:
        points = _unique_points(point for edge in slit_edges for point in _edge_points(panel, edge))
        if points:
            # The slit top is the point farthest from the panel's lowest Y hem.
            top = max(points, key=lambda point: (point[1], point))
            result["SLIT_TOP"] = _available_point(
                "SLIT_TOP",
                panel,
                top,
                evidence="observed_source",
                confidence=0.9,
                source_ids=(edge.id for edge in slit_edges),
            )

    # Pencil-skirt vents in GarmentCode are represented by two unstitched
    # boundary edges that leave and return to the hem.  This is observable
    # topology (unlike a generic concavity), and it is distinguishable from a
    # waist dart because its two bases lie on the recovered hem and its edges
    # are not self-stitched.  Never infer a slit from shape alone when these
    # boundary conditions are absent.
    if frame is not None and "SLIT_TOP" not in result:
        hem_level = _dot(frame.hem_mid, frame.grain)
        waist_level = _dot(frame.waist_mid, frame.grain)
        span = max(waist_level - hem_level, 1e-6)
        tolerance = max(span * 0.025, 1e-5)
        candidates: list[
            tuple[
                float,
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[str, str],
            ]
        ] = []
        eligible = [
            edge
            for edge in panel.edges
            if edge.role == "other" and not edge.stitched and not edge.self_stitched
        ]
        for index, first in enumerate(eligible):
            for second in eligible[index + 1 :]:
                shared = set(first.endpoints) & set(second.endpoints)
                if len(shared) != 1:
                    continue
                apex_index = next(iter(shared))
                first_base = next(value for value in first.endpoints if value != apex_index)
                second_base = next(value for value in second.endpoints if value != apex_index)
                apex = _point(panel.vertices_cm[apex_index])
                bases = (_point(panel.vertices_cm[first_base]), _point(panel.vertices_cm[second_base]))
                if any(abs(_dot(base, frame.grain) - hem_level) > tolerance for base in bases):
                    continue
                depth = _dot(apex, frame.grain) - hem_level
                if depth <= span * 0.025:
                    continue
                left, right = sorted(bases, key=lambda point: (_dot(point, frame.cross), point))
                candidates.append((depth, apex, left, right, tuple(sorted((first.id, second.id)))))
        if candidates:
            _, apex, left, right, source_ids = max(
                candidates, key=lambda item: (item[0], item[1], item[2], item[3], item[4])
            )
            for name, value in (
                ("SLIT_TOP", apex),
                ("SLIT_HEM_LEFT", left),
                ("SLIT_HEM_RIGHT", right),
            ):
                result[name] = _available_point(
                    name,
                    panel,
                    value,
                    evidence="derived_topology",
                    confidence=0.92,
                    source_ids=source_ids,
                )
    return result


def _pants_panel(record: DraftingSemanticRecord, panel: PanelAnnotation) -> LowerBodyPanelSemantics:
    center_prefix = "CF" if panel.role == "front_pants" else "CB"
    names = PANTS_FRONT_LANDMARKS if center_prefix == "CF" else PANTS_BACK_LANDMARKS
    frame = _panel_frame(panel)
    if frame is None:
        reason = "waist and hem boundary frame could not be recovered"
        landmarks = tuple(_unavailable_point(name, panel, reason) for name in names)
        lines = tuple(_unavailable_line(name, panel, reason) for name in PANTS_REFERENCE_LINES)
    else:
        center_waist, side_waist, assignment_confidence, assignment_reason = _pants_waist_assignment(panel, frame)
        center_name = f"{center_prefix}_WAIST"
        center_hip_name = f"{center_prefix}_HIP"
        waist_confidence = min(frame.waist_confidence, assignment_confidence)
        point_map: dict[str, PointObservation] = {
            center_name: _available_point(
                center_name,
                panel,
                center_waist,
                evidence=frame.waist_evidence,
                confidence=waist_confidence,
                source_ids=frame.waist_source_ids,
            ),
            "SIDE_WAIST": _available_point(
                "SIDE_WAIST",
                panel,
                side_waist,
                evidence=frame.waist_evidence,
                confidence=waist_confidence,
                source_ids=frame.waist_source_ids,
            ),
        }

        waist_level = _dot(frame.waist_mid, frame.grain)
        hem_level = _dot(frame.hem_mid, frame.grain)
        wl_span = _line_span_at_level(panel, frame, waist_level)
        line_map: dict[str, LineObservation] = {}
        if wl_span is not None:
            line_map["WL"] = _available_line(
                "WL",
                panel,
                wl_span[0],
                evidence=frame.waist_evidence,
                confidence=frame.waist_confidence,
                intersects_panel=True,
                source_ids=(*frame.waist_source_ids, *wl_span[1]),
            )
        else:
            line_map["WL"] = _unavailable_line("WL", panel, "waist boundary has fewer than two intersections")

        hip_depth, hip_evidence, hip_confidence = _hip_depth(record)
        if hip_depth is None:
            point_map[center_hip_name] = _unavailable_point(
                center_hip_name, panel, "waist-to-hip depth is absent from measurements/body condition"
            )
            point_map["SIDE_HIP"] = _unavailable_point(
                "SIDE_HIP", panel, "waist-to-hip depth is absent from measurements/body condition"
            )
            line_map["HL"] = _unavailable_line(
                "HL", panel, "waist-to-hip depth is absent from measurements/body condition"
            )
        else:
            hip_level = waist_level - hip_depth
            hip_span = _line_span_at_level(panel, frame, hip_level)
            if hip_span is None:
                reason = "derived hip level does not intersect this panel twice"
                point_map[center_hip_name] = _unavailable_point(center_hip_name, panel, reason)
                point_map["SIDE_HIP"] = _unavailable_point("SIDE_HIP", panel, reason)
                hip_center = _add(frame.waist_mid, _scale(frame.grain, -hip_depth))
                half_width = math.dist(*frame.waist_points) / 2.0
                line_map["HL"] = _available_line(
                    "HL",
                    panel,
                    (
                        _add(hip_center, _scale(frame.cross, -half_width)),
                        _add(hip_center, _scale(frame.cross, half_width)),
                    ),
                    evidence=hip_evidence,
                    confidence=hip_confidence,
                    intersects_panel=False,
                    training_eligible=False,
                )
            else:
                hip_points = sorted(hip_span[0], key=lambda point: (_dot(point, frame.cross), point))
                center_sign = 1.0 if _dot(_sub(center_waist, side_waist), frame.cross) >= 0 else -1.0
                center_hip = max(hip_points, key=lambda point: center_sign * _dot(point, frame.cross))
                side_hip = hip_points[0] if center_hip == hip_points[-1] else hip_points[-1]
                point_map[center_hip_name] = _available_point(
                    center_hip_name,
                    panel,
                    center_hip,
                    evidence=hip_evidence,
                    confidence=min(hip_confidence, assignment_confidence),
                    source_ids=hip_span[1],
                )
                point_map["SIDE_HIP"] = _available_point(
                    "SIDE_HIP",
                    panel,
                    side_hip,
                    evidence=hip_evidence,
                    confidence=min(hip_confidence, assignment_confidence),
                    source_ids=hip_span[1],
                )
                line_map["HL"] = _available_line(
                    "HL",
                    panel,
                    hip_span[0],
                    evidence=hip_evidence,
                    confidence=hip_confidence,
                    intersects_panel=True,
                    source_ids=hip_span[1],
                )

        crotch = _shared_role_vertex(panel, "crotch_curve", "inseam")
        crotch_confidence = 0.97
        crotch_reason = "semantic crotch-curve/inseam intersection"
        if crotch is None:
            crotch = _geometric_crotch(panel, frame, center_waist, side_waist)
            crotch_confidence = 0.62
            crotch_reason = assignment_reason
        if crotch is None:
            point_map["CROTCH_POINT"] = _unavailable_point(
                "CROTCH_POINT", panel, "no semantic intersection or geometric crotch extremum was recoverable"
            )
            line_map["KNEE_LINE"] = _unavailable_line(
                "KNEE_LINE", panel, "crotch point is required to derive a conventional knee level"
            )
            line_map["CL"] = _unavailable_line(
                "CL", panel, "crotch point is required to derive the crotch line"
            )
            point_map["KNEE_SIDE"] = _unavailable_point(
                "KNEE_SIDE", panel, "crotch point is required to derive a conventional knee level"
            )
            point_map["KNEE_INSEAM"] = _unavailable_point(
                "KNEE_INSEAM", panel, "crotch point is required to derive a conventional knee level"
            )
        else:
            point_map["CROTCH_POINT"] = _available_point(
                "CROTCH_POINT",
                panel,
                crotch[0],
                evidence="derived_topology",
                confidence=crotch_confidence,
                source_ids=crotch[1],
            )
            crotch_level = _dot(crotch[0], frame.grain)
            crotch_span = _line_span_at_level(panel, frame, crotch_level)
            if crotch_span is None:
                line_map["CL"] = _unavailable_line(
                    "CL", panel, "crotch level does not intersect the panel twice"
                )
            else:
                line_map["CL"] = _available_line(
                    "CL",
                    panel,
                    crotch_span[0],
                    evidence="derived_topology",
                    confidence=crotch_confidence,
                    intersects_panel=True,
                    source_ids=(*crotch[1], *crotch_span[1]),
                )
            knee_level = (crotch_level + hem_level) / 2.0
            knee_span = _line_span_at_level(panel, frame, knee_level)
            if knee_span is None:
                reason = "conventional half-leg knee level does not intersect twice"
                line_map["KNEE_LINE"] = _unavailable_line("KNEE_LINE", panel, reason)
                point_map["KNEE_SIDE"] = _unavailable_point("KNEE_SIDE", panel, reason)
                point_map["KNEE_INSEAM"] = _unavailable_point("KNEE_INSEAM", panel, reason)
            else:
                knee_points = sorted(knee_span[0], key=lambda point: (_dot(point, frame.cross), point))
                center_sign = 1.0 if _dot(_sub(center_waist, side_waist), frame.cross) >= 0 else -1.0
                inseam = max(knee_points, key=lambda point: center_sign * _dot(point, frame.cross))
                side = knee_points[0] if inseam == knee_points[-1] else knee_points[-1]
                for name, value in (("KNEE_SIDE", side), ("KNEE_INSEAM", inseam)):
                    point_map[name] = _available_point(
                        name,
                        panel,
                        value,
                        evidence="synthetic_unvalidated",
                        confidence=0.45,
                        source_ids=knee_span[1],
                        training_eligible=False,
                    )
                line_map["KNEE_LINE"] = _available_line(
                    "KNEE_LINE",
                    panel,
                    knee_span[0],
                    evidence="synthetic_unvalidated",
                    confidence=0.45,
                    intersects_panel=True,
                    source_ids=knee_span[1],
                    training_eligible=False,
                )

        hem_left, hem_right = frame.hem_points
        center_sign = 1.0 if _dot(_sub(center_waist, side_waist), frame.cross) >= 0 else -1.0
        inseam_hem = max((hem_left, hem_right), key=lambda point: center_sign * _dot(point, frame.cross))
        side_hem = hem_left if inseam_hem == hem_right else hem_right
        point_map["HEM_SIDE"] = _available_point(
            "HEM_SIDE",
            panel,
            side_hem,
            evidence="derived_topology",
            confidence=0.88,
            source_ids=frame.hem_source_ids,
        )
        point_map["HEM_INSEAM"] = _available_point(
            "HEM_INSEAM",
            panel,
            inseam_hem,
            evidence="derived_topology",
            confidence=0.88,
            source_ids=frame.hem_source_ids,
        )
        grain_points = (
            _add(frame.hem_mid, _scale(_sub(frame.waist_mid, frame.hem_mid), 0.15)),
            _add(frame.hem_mid, _scale(_sub(frame.waist_mid, frame.hem_mid), 0.85)),
        )
        line_map["GRAIN"] = _available_line(
            "GRAIN",
            panel,
            grain_points,
            evidence="derived_topology",
            confidence=0.65,
            intersects_panel=True,
            training_eligible=True,
        )
        landmarks = tuple(
            point_map.get(name, _unavailable_point(name, panel, f"{name} was not recoverable")) for name in names
        )
        lines = tuple(
            line_map.get(name, _unavailable_line(name, panel, f"{name} was not recoverable"))
            for name in PANTS_REFERENCE_LINES
        )

    darts = _dart_observations(record, panel)
    features = (
        FeatureObservation(
            "dart",
            bool(darts),
            darts[0].evidence if darts else "unavailable",
            max((item.confidence for item in darts), default=0.0),
            any(item.training_eligible for item in darts),
            None if darts else "no source Dart object is attached to this panel",
        ),
        FeatureObservation("slit", False, "unavailable", 0.0, False, "pants slit is outside this ontology"),
        _production_feature(record, "zipper", "zippers"),
        _production_feature(record, "notches", "notches"),
    )
    return LowerBodyPanelSemantics(
        panel.id,
        "pants",
        panel.role,
        panel.role,
        False,
        landmarks,
        lines,
        darts,
        features,
    )


def _skirt_panel(record: DraftingSemanticRecord, panel: PanelAnnotation) -> LowerBodyPanelSemantics:
    frame = _panel_frame(panel)
    point_map: dict[str, PointObservation] = {}
    line_map: dict[str, LineObservation] = {}
    if frame is None:
        reason = "waist and hem boundary frame could not be recovered"
        for name in SKIRT_LANDMARKS:
            point_map[name] = _unavailable_point(name, panel, reason)
        for name in SKIRT_REFERENCE_LINES:
            line_map[name] = _unavailable_line(name, panel, reason)
    else:
        left_waist, right_waist = frame.waist_points
        left_hem, right_hem = frame.hem_points
        for name, value, ids, confidence in (
            ("WAIST_LEFT", left_waist, frame.waist_source_ids, frame.waist_confidence),
            ("WAIST_RIGHT", right_waist, frame.waist_source_ids, frame.waist_confidence),
            ("HEM_LEFT", left_hem, frame.hem_source_ids, 0.9),
            ("HEM_RIGHT", right_hem, frame.hem_source_ids, 0.9),
        ):
            point_map[name] = _available_point(
                name,
                panel,
                value,
                evidence="derived_topology",
                confidence=confidence,
                source_ids=ids,
            )
        waist_level = _dot(frame.waist_mid, frame.grain)
        wl_span = _line_span_at_level(panel, frame, waist_level)
        if wl_span is not None:
            line_map["WL"] = _available_line(
                "WL",
                panel,
                wl_span[0],
                evidence=frame.waist_evidence,
                confidence=frame.waist_confidence,
                intersects_panel=True,
                source_ids=(*frame.waist_source_ids, *wl_span[1]),
            )
        else:
            line_map["WL"] = _unavailable_line("WL", panel, "waist line does not intersect twice")

        hip_depth, hip_evidence, hip_confidence = _hip_depth(record)
        if hip_depth is None:
            reason = "waist-to-hip depth is absent from measurements/body condition"
            point_map["HIP_LEFT"] = _unavailable_point("HIP_LEFT", panel, reason)
            point_map["HIP_RIGHT"] = _unavailable_point("HIP_RIGHT", panel, reason)
            line_map["HL"] = _unavailable_line("HL", panel, reason)
        else:
            hip_level = waist_level - hip_depth
            hip_span = _line_span_at_level(panel, frame, hip_level)
            if hip_span is None:
                reason = "derived hip level does not intersect this skirt panel twice"
                point_map["HIP_LEFT"] = _unavailable_point("HIP_LEFT", panel, reason)
                point_map["HIP_RIGHT"] = _unavailable_point("HIP_RIGHT", panel, reason)
                line_map["HL"] = _unavailable_line("HL", panel, reason)
            else:
                left_hip, right_hip = sorted(
                    hip_span[0], key=lambda point: (_dot(point, frame.cross), point)
                )
                point_map["HIP_LEFT"] = _available_point(
                    "HIP_LEFT",
                    panel,
                    left_hip,
                    evidence=hip_evidence,
                    confidence=hip_confidence,
                    source_ids=hip_span[1],
                )
                point_map["HIP_RIGHT"] = _available_point(
                    "HIP_RIGHT",
                    panel,
                    right_hip,
                    evidence=hip_evidence,
                    confidence=hip_confidence,
                    source_ids=hip_span[1],
                )
                line_map["HL"] = _available_line(
                    "HL",
                    panel,
                    hip_span[0],
                    evidence=hip_evidence,
                    confidence=hip_confidence,
                    intersects_panel=True,
                    source_ids=hip_span[1],
                )
        grain_points = (
            _add(frame.hem_mid, _scale(_sub(frame.waist_mid, frame.hem_mid), 0.15)),
            _add(frame.hem_mid, _scale(_sub(frame.waist_mid, frame.hem_mid), 0.85)),
        )
        line_map["GRAIN"] = _available_line(
            "GRAIN",
            panel,
            grain_points,
            evidence="derived_topology",
            confidence=0.65,
            intersects_panel=True,
            training_eligible=True,
        )

    slit_points = _slit_landmarks(panel, frame)
    for name in ("SLIT_TOP", "SLIT_HEM_LEFT", "SLIT_HEM_RIGHT"):
        point_map[name] = slit_points.get(
            name,
            _unavailable_point(name, panel, "no explicitly named source slit landmark/edge exists"),
        )
    landmarks = tuple(point_map[name] for name in SKIRT_LANDMARKS)
    lines = tuple(line_map[name] for name in SKIRT_REFERENCE_LINES)
    darts = _dart_observations(record, panel)
    slit_available = bool(slit_points)
    zipper = _production_feature(record, "zipper", "zippers")
    notches = _production_feature(record, "notches", "notches")
    features = (
        FeatureObservation(
            "dart",
            bool(darts),
            darts[0].evidence if darts else "unavailable",
            max((item.confidence for item in darts), default=0.0),
            any(item.training_eligible for item in darts),
            None if darts else "no source Dart object is attached to this panel",
        ),
        FeatureObservation(
            "slit",
            slit_available,
            "observed_source" if slit_available else "unavailable",
            max((item.confidence for item in slit_points.values()), default=0.0),
            any(item.training_eligible for item in slit_points.values()),
            None if slit_available else "no explicitly named source slit landmark/edge exists",
        ),
        zipper,
        notches,
    )
    discriminating_cue = bool(darts) or slit_available or zipper.available or notches.available
    return LowerBodyPanelSemantics(
        panel.id,
        "skirt",
        panel.role,
        "skirt_panel",
        not discriminating_cue,
        landmarks,
        lines,
        darts,
        features,
    )


def extract_lower_body_semantics(record: DraftingSemanticRecord) -> LowerBodySemanticRecord:
    """Extract deterministic pants/skirt targets from a semantic record.

    Panels outside the lower-body ontology are intentionally ignored.  Output
    order follows source panel order so it is stable across repeated runs.
    """

    panels: list[LowerBodyPanelSemantics] = []
    for panel in record.panels:
        if panel.role in PANTS_PANEL_ROLES:
            panels.append(_pants_panel(record, panel))
        elif panel.role in SKIRT_PANEL_ROLES:
            panels.append(_skirt_panel(record, panel))
    return LowerBodySemanticRecord(record.sample_id, record.split, tuple(panels))


__all__ = [
    "PANTS_FRONT_LANDMARKS",
    "PANTS_BACK_LANDMARKS",
    "PANTS_REFERENCE_LINES",
    "SKIRT_LANDMARKS",
    "SKIRT_REFERENCE_LINES",
    "PointObservation",
    "LineObservation",
    "DartObservation",
    "FeatureObservation",
    "LowerBodyPanelSemantics",
    "LowerBodySemanticRecord",
    "extract_lower_body_semantics",
]
