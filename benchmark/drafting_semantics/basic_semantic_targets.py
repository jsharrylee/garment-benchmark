"""Auditable semantic-query targets for common basic garments.

This module is the data-integration layer between exact/provisional vector
patterns and :mod:`semantic_teacher_student`.  It deliberately keeps three
different ideas separate:

``presence``
    Whether the source contains an element.
``query_applicability``
    Whether that presence/absence is reliable enough to supervise a model.
``coordinate_mask``
    Which individual coordinate channels are actually recoverable.

That distinction matters for GarmentCodeData v2.  It contains exact boundary
endpoints and arc lengths, but many source curves do not expose their Bezier
controls in :class:`DraftingSemanticRecord`.  Such a path still supervises
endpoints and length; curve depth and tangents remain masked.  Likewise, an
unencoded zipper, closure, slit, or notch is *unknown*, not a fabricated
negative example.

The provisional common-pattern blocks have a stronger local contract.  Their
``PatternDocument`` annotations explicitly enumerate present and absent
queries and contain densely sampled curves, so every declared path coordinate
can be supervised.  Their source remains visibly distinct from GarmentCode
and their provenance remains ``PROVISIONAL_EXPERT_REVIEW``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .lower_body_semantics import extract_lower_body_semantics
from .schema import DraftingSemanticRecord, EdgeAnnotation, PanelAnnotation
from .semantic_teacher_student import (
    CATEGORY_NAMES,
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INDEX,
    SEMANTIC_QUERY_INVENTORY,
    SEMANTIC_QUERY_KEYS,
)


_EPSILON = 1e-8
_POINT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GarmentFrame:
    """Axis-aligned, centimetre-valued frame shared by every target query.

    GarmentCode and the provisional blocks both store panel-local drafting
    coordinates with stable Cartesian axes.  The union box of all relevant
    panels therefore provides a deterministic same-garment frame.  Normalized
    points use ``u=(x-min_x)/width`` and a canonical y-up ``v`` with zero at
    the hem/bottom and one at the neck/waist/top.  The source-axis flag makes
    the y-down provisional blocks agree with y-up GarmentCode records.  Scalar
    path geometry is divided by ``scale_cm=max(width,height)`` so aspect-ratio
    changes are retained instead of being hidden by independent axis scaling.
    """

    min_x_cm: float
    min_y_cm: float
    max_x_cm: float
    max_y_cm: float
    source_y_axis_down: bool = False

    @property
    def width_cm(self) -> float:
        return self.max_x_cm - self.min_x_cm

    @property
    def height_cm(self) -> float:
        return self.max_y_cm - self.min_y_cm

    @property
    def scale_cm(self) -> float:
        return max(self.width_cm, self.height_cm)

    def normalize_point(self, point: Sequence[float]) -> tuple[float, float]:
        raw_v = (float(point[1]) - self.min_y_cm) / self.height_cm
        return (
            (float(point[0]) - self.min_x_cm) / self.width_cm,
            1.0 - raw_v if self.source_y_axis_down else raw_v,
        )

    def canonical_point(self, point: Sequence[float]) -> tuple[float, float]:
        """Return a centimetre point in the canonical y-up drafting axes."""

        return float(point[0]), -float(point[1]) if self.source_y_axis_down else float(point[1])

    def canonical_vector(self, vector: Sequence[float]) -> tuple[float, float]:
        return float(vector[0]), -float(vector[1]) if self.source_y_axis_down else float(vector[1])

    @classmethod
    def from_points(
        cls,
        points: Iterable[Sequence[float]],
        *,
        source_y_axis_down: bool = False,
    ) -> "GarmentFrame":
        values = np.asarray(tuple((float(point[0]), float(point[1])) for point in points), dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != (2,) or not len(values):
            raise ValueError("a garment frame requires at least one 2D point")
        if not np.isfinite(values).all():
            raise ValueError("garment-frame points must be finite")
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        if float(maximum[0] - minimum[0]) <= _EPSILON:
            raise ValueError("garment frame has degenerate width")
        if float(maximum[1] - minimum[1]) <= _EPSILON:
            raise ValueError("garment frame has degenerate height")
        return cls(
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
            bool(source_y_axis_down),
        )


@dataclass(frozen=True)
class BasicSemanticTarget:
    """One fixed-inventory semantic target suitable for teacher/student loss."""

    sample_id: str
    category: str
    category_id: int
    source: str
    provenance_status: str
    frame: GarmentFrame
    query_applicability: np.ndarray
    presence: np.ndarray
    coordinates: np.ndarray
    coordinate_mask: np.ndarray
    evidence: tuple[str, ...]

    @property
    def query_mask(self) -> np.ndarray:
        """Alias used by :func:`semantic_distillation_loss`."""

        return self.query_applicability

    @property
    def presence_targets(self) -> np.ndarray:
        return self.presence

    @property
    def coordinate_targets(self) -> np.ndarray:
        return self.coordinates

    def validate(self) -> None:
        query_count = len(SEMANTIC_QUERY_INVENTORY)
        if self.category not in CATEGORY_NAMES:
            raise ValueError(f"unsupported semantic target category: {self.category!r}")
        if self.category_id != CATEGORY_NAMES.index(self.category):
            raise ValueError("category_id does not match category")
        if self.query_applicability.shape != (query_count,):
            raise ValueError("query_applicability has the wrong shape")
        if self.presence.shape != (query_count,):
            raise ValueError("presence has the wrong shape")
        expected = (query_count, MAX_COORDINATE_DIM)
        if self.coordinates.shape != expected or self.coordinate_mask.shape != expected:
            raise ValueError("coordinate target tensors have the wrong shape")
        if len(self.evidence) != query_count:
            raise ValueError("evidence must contain one entry per semantic query")
        for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
            if query.category != self.category and (
                bool(self.query_applicability[index])
                or bool(self.presence[index])
                or bool(self.coordinate_mask[index].any())
            ):
                raise ValueError("a target activated a query from another garment category")
            declared = len(query.coordinate_names)
            if bool(self.coordinate_mask[index, declared:].any()):
                raise ValueError("coordinate mask activates an undeclared channel")
        if bool((self.coordinate_mask & ~self.query_applicability[:, None]).any()):
            raise ValueError("coordinates cannot supervise an inapplicable query")
        if bool((self.coordinate_mask & ~(self.presence > 0.5)[:, None]).any()):
            raise ValueError("coordinates cannot supervise an absent element")
        if not np.isfinite(self.coordinates[self.coordinate_mask]).all():
            raise ValueError("active coordinates must be finite")


class _TargetWriter:
    def __init__(self, sample_id: str, category: str, source: str, provenance_status: str, frame: GarmentFrame):
        count = len(SEMANTIC_QUERY_INVENTORY)
        self.sample_id = sample_id
        self.category = category
        self.source = source
        self.provenance_status = provenance_status
        self.frame = frame
        self.applicability = np.zeros((count,), dtype=np.bool_)
        self.presence = np.zeros((count,), dtype=np.float32)
        self.coordinates = np.full((count, MAX_COORDINATE_DIM), np.nan, dtype=np.float32)
        self.coordinate_mask = np.zeros((count, MAX_COORDINATE_DIM), dtype=np.bool_)
        self.evidence = ["category_inapplicable"] * count

    def _index(self, kind: str, name: str) -> int:
        key = f"{self.category}:{kind}:{name}"
        if key not in SEMANTIC_QUERY_INDEX:
            raise KeyError(f"semantic query does not exist: {key}")
        return SEMANTIC_QUERY_INDEX[key]

    def absent(self, kind: str, name: str, *, evidence: str) -> None:
        index = self._index(kind, name)
        self.applicability[index] = True
        self.presence[index] = 0.0
        self.evidence[index] = evidence

    def unknown(self, kind: str, name: str, *, evidence: str) -> None:
        index = self._index(kind, name)
        self.applicability[index] = False
        self.presence[index] = 0.0
        self.evidence[index] = evidence

    def present(
        self,
        kind: str,
        name: str,
        values: Mapping[str, float] | None,
        *,
        evidence: str,
        training_eligible: bool = True,
    ) -> None:
        index = self._index(kind, name)
        self.presence[index] = 1.0
        self.applicability[index] = bool(training_eligible)
        self.evidence[index] = evidence
        if not training_eligible or values is None:
            return
        query = SEMANTIC_QUERY_INVENTORY[index]
        for channel, coordinate_name in enumerate(query.coordinate_names):
            value = values.get(coordinate_name)
            if value is None or not math.isfinite(float(value)):
                continue
            self.coordinates[index, channel] = float(value)
            self.coordinate_mask[index, channel] = True

    def finish(self) -> BasicSemanticTarget:
        result = BasicSemanticTarget(
            sample_id=self.sample_id,
            category=self.category,
            category_id=CATEGORY_NAMES.index(self.category),
            source=self.source,
            provenance_status=self.provenance_status,
            frame=self.frame,
            query_applicability=self.applicability,
            presence=self.presence,
            coordinates=self.coordinates,
            coordinate_mask=self.coordinate_mask,
            evidence=tuple(self.evidence),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class _Segment:
    source_id: str
    points_cm: tuple[tuple[float, float], ...]
    length_cm: float
    start_tangent_reliable: bool
    end_tangent_reliable: bool


def _point(value: Sequence[float]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _same_point(first: Sequence[float], second: Sequence[float]) -> bool:
    return math.dist(_point(first), _point(second)) <= _POINT_TOLERANCE


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(_point(first), _point(second)) for first, second in zip(points, points[1:]))


def _terminal_points(segments: Sequence[_Segment]) -> tuple[tuple[float, float], ...]:
    endpoints = [_point(segment.points_cm[0]) for segment in segments]
    endpoints.extend(_point(segment.points_cm[-1]) for segment in segments)
    groups: list[list[tuple[float, float]]] = []
    for point in endpoints:
        for group in groups:
            if _same_point(point, group[0]):
                group.append(point)
                break
        else:
            groups.append([point])
    terminals = [group[0] for group in groups if len(group) == 1]
    return tuple(terminals if len(terminals) >= 2 else (group[0] for group in groups))


def _farthest_pair(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("a path requires at least two distinct endpoints")
    return max(
        ((first, second) for index, first in enumerate(points) for second in points[index + 1 :]),
        key=lambda pair: (math.dist(pair[0], pair[1]), pair),
    )


def _endpoint_tangent(
    segments: Sequence[_Segment],
    endpoint: tuple[float, float],
    *,
    outgoing: bool,
) -> tuple[float, float] | None:
    candidates: list[tuple[str, tuple[float, float], bool]] = []
    for segment in segments:
        points = segment.points_cm
        if len(points) < 2:
            continue
        if _same_point(points[0], endpoint):
            delta = (points[1][0] - points[0][0], points[1][1] - points[0][1])
            candidates.append((segment.source_id, delta, segment.start_tangent_reliable))
        if _same_point(points[-1], endpoint):
            # Vector pointing from the endpoint into the path interior.
            delta = (points[-2][0] - points[-1][0], points[-2][1] - points[-1][1])
            candidates.append((segment.source_id, delta, segment.end_tangent_reliable))
    eligible = sorted((item for item in candidates if item[2]), key=lambda item: item[0])
    if not eligible:
        return None
    delta = eligible[0][1]
    length = math.hypot(*delta)
    if length <= _EPSILON:
        return None
    inward = (delta[0] / length, delta[1] / length)
    # Traversal starts by going inward from start and finishes by going out of
    # the interior into end.
    return inward if outgoing else (-inward[0], -inward[1])


def _path_values(
    segments: Sequence[_Segment],
    frame: GarmentFrame,
    *,
    depth_reliable: bool,
) -> dict[str, float]:
    terminals = _terminal_points(segments)
    start, end = _farthest_pair(terminals)
    start_uv, end_uv = frame.normalize_point(start), frame.normalize_point(end)
    if (end_uv[0], end_uv[1]) < (start_uv[0], start_uv[1]):
        start, end = end, start
        start_uv, end_uv = end_uv, start_uv
    values: dict[str, float] = {
        "start_u": start_uv[0],
        "start_v": start_uv[1],
        "end_u": end_uv[0],
        "end_v": end_uv[1],
        "arc_length_norm": sum(float(segment.length_cm) for segment in segments) / frame.scale_cm,
    }

    canonical_start = frame.canonical_point(start)
    canonical_end = frame.canonical_point(end)
    chord = (
        canonical_end[0] - canonical_start[0],
        canonical_end[1] - canonical_start[1],
    )
    chord_length = math.hypot(*chord)
    if depth_reliable and chord_length > _EPSILON:
        normal = (-chord[1] / chord_length, chord[0] / chord_length)
        distances = [
            (frame.canonical_point(point)[0] - canonical_start[0]) * normal[0]
            + (frame.canonical_point(point)[1] - canonical_start[1]) * normal[1]
            for segment in segments
            for point in segment.points_cm
        ]
        signed = max(distances, key=lambda value: (abs(value), value), default=0.0)
        values["signed_depth_norm"] = float(signed) / frame.scale_cm

    start_tangent = _endpoint_tangent(segments, start, outgoing=True)
    end_tangent = _endpoint_tangent(segments, end, outgoing=False)
    if start_tangent is not None:
        start_tangent = frame.canonical_vector(start_tangent)
        values["start_tangent_angle_norm"] = math.atan2(start_tangent[1], start_tangent[0]) / math.pi
    if end_tangent is not None:
        end_tangent = frame.canonical_vector(end_tangent)
        values["end_tangent_angle_norm"] = math.atan2(end_tangent[1], end_tangent[0]) / math.pi
    return values


def _choose_path_values(
    candidates: Sequence[tuple[str, tuple[_Segment, ...], bool]], frame: GarmentFrame
) -> tuple[dict[str, float], str] | None:
    if not candidates:
        return None
    # One query represents one canonical panel path.  Primitive pieces inside
    # that panel are aggregated, but duplicated/mirrored panel paths are not
    # spuriously summed.  Longest-path selection is deterministic and retains
    # child ids in evidence for audit.
    panel_id, segments, depth_reliable = max(
        candidates,
        key=lambda item: (
            sum(segment.length_cm for segment in item[1]),
            item[0],
            tuple(segment.source_id for segment in item[1]),
        ),
    )
    evidence = f"merged_semantic_path:{panel_id}:" + ",".join(segment.source_id for segment in segments)
    return _path_values(segments, frame, depth_reliable=depth_reliable), evidence


def _panel_box_values(points: Iterable[Sequence[float]], frame: GarmentFrame) -> dict[str, float]:
    values = np.asarray(tuple(_point(point) for point in points), dtype=np.float64)
    if not len(values):
        raise ValueError("panel target has no points")
    minimum, maximum = values.min(axis=0), values.max(axis=0)
    center = 0.5 * (minimum + maximum)
    center_uv = frame.normalize_point(center)
    return {
        "center_u": center_uv[0],
        "center_v": center_uv[1],
        "width": float(maximum[0] - minimum[0]) / frame.width_cm,
        "height": float(maximum[1] - minimum[1]) / frame.height_cm,
    }


def _landmark_values(point: Sequence[float], frame: GarmentFrame) -> dict[str, float]:
    u, v = frame.normalize_point(point)
    return {"u": u, "v": v}


def _reference_line_values(
    points: Sequence[Sequence[float]], frame: GarmentFrame
) -> dict[str, float]:
    if len(points) != 2:
        raise ValueError("a semantic reference line requires exactly two endpoints")
    first, second = frame.normalize_point(points[0]), frame.normalize_point(points[1])
    if second < first:
        first, second = second, first
    return {
        "start_u": first[0],
        "start_v": first[1],
        "end_u": second[0],
        "end_v": second[1],
    }


def _record_program_values(record: DraftingSemanticRecord) -> tuple[Any, Any]:
    upper = record.program.get("upper_type")
    design = record.program.get("design_values", {})
    bottom = design.get("meta.bottom") if isinstance(design, Mapping) else None
    return upper, bottom


def panel_role_counts(record: DraftingSemanticRecord) -> dict[str, int]:
    output: dict[str, int] = {}
    for panel in record.panels:
        output[panel.role] = output.get(panel.role, 0) + 1
    return output


def is_common_basic_tshirt(record: DraftingSemanticRecord) -> bool:
    """Match the frozen common-top definition used by the prior audit."""

    upper, bottom = _record_program_values(record)
    counts = panel_role_counts(record)
    return (
        upper == "Shirt"
        and bottom is None
        and not record.darts
        and set(counts) <= {"front_bodice", "back_bodice", "sleeve"}
        and counts.get("front_bodice") == 2
        and counts.get("back_bodice") == 2
        and counts.get("sleeve") == 4
    )


def is_common_basic_pants(record: DraftingSemanticRecord) -> bool:
    """Match the frozen straight-pants topology definition used by the audit."""

    upper, bottom = _record_program_values(record)
    counts = panel_role_counts(record)
    return (
        upper is None
        and bottom == "Pants"
        and set(counts) <= {"front_pants", "back_pants", "waistband"}
        and counts.get("front_pants") == 2
        and counts.get("back_pants") == 2
        and counts.get("waistband", 0) in {0, 2}
    )


def is_common_basic_skirt(record: DraftingSemanticRecord) -> bool:
    """Match the frozen two-panel Skirt2/PencilSkirt audit definition."""

    upper, bottom = _record_program_values(record)
    counts = panel_role_counts(record)
    return (
        upper is None
        and bottom in {"Skirt2", "PencilSkirt"}
        and set(counts) <= {"front_skirt", "back_skirt", "waistband"}
        and counts.get("front_skirt") == 1
        and counts.get("back_skirt") == 1
        and counts.get("waistband", 0) in {0, 2}
    )


def common_basic_category(record: DraftingSemanticRecord) -> str | None:
    matches = tuple(
        category
        for category, predicate in (
            ("tshirt", is_common_basic_tshirt),
            ("pants", is_common_basic_pants),
            ("skirt", is_common_basic_skirt),
        )
        if predicate(record)
    )
    if len(matches) > 1:
        raise ValueError("record matches more than one common-basic definition")
    return matches[0] if matches else None


def filter_common_basic_records(
    records: Iterable[DraftingSemanticRecord], category: str | None = None
) -> tuple[DraftingSemanticRecord, ...]:
    if category is not None and category not in CATEGORY_NAMES:
        raise ValueError(f"unsupported category: {category!r}")
    return tuple(
        record
        for record in records
        if common_basic_category(record) is not None
        and (category is None or common_basic_category(record) == category)
    )


def _infer_simple_category(record: DraftingSemanticRecord) -> str:
    upper, bottom = _record_program_values(record)
    if upper is not None and bottom is None:
        return "tshirt"
    if upper is None and bottom == "Pants":
        return "pants"
    if upper is None and bottom in {"Skirt2", "PencilSkirt"}:
        return "skirt"
    raise ValueError("record is not a standalone T-shirt, pants, or skirt sample")


_PANEL_ROLES: dict[str, dict[str, frozenset[str]]] = {
    "tshirt": {
        "front_bodice": frozenset({"front_bodice"}),
        "back_bodice": frozenset({"back_bodice"}),
        "sleeve": frozenset({"sleeve"}),
        "neckband": frozenset({"collar"}),
    },
    "pants": {
        "front_pants": frozenset({"front_pants"}),
        "back_pants": frozenset({"back_pants"}),
        "waistband": frozenset({"waistband"}),
    },
    "skirt": {
        "skirt_panel": frozenset({"front_skirt", "back_skirt"}),
        "waistband": frozenset({"waistband"}),
    },
}


# query -> (eligible panel roles, eligible exact edge roles).  Unsupported
# construction features are handled explicitly below rather than guessed from
# ``other`` edges.
_PATH_SPECS: dict[str, dict[str, tuple[frozenset[str], frozenset[str]]]] = {
    "tshirt": {
        "front_neckline": (frozenset({"front_bodice"}), frozenset({"neckline"})),
        "back_neckline": (frozenset({"back_bodice"}), frozenset({"neckline"})),
        "front_shoulder": (frozenset({"front_bodice"}), frozenset({"shoulder"})),
        "back_shoulder": (frozenset({"back_bodice"}), frozenset({"shoulder"})),
        "front_armhole": (frozenset({"front_bodice"}), frozenset({"armhole"})),
        "back_armhole": (frozenset({"back_bodice"}), frozenset({"armhole"})),
        "front_side_seam": (frozenset({"front_bodice"}), frozenset({"side_seam"})),
        "back_side_seam": (frozenset({"back_bodice"}), frozenset({"side_seam"})),
        "front_hemline": (frozenset({"front_bodice"}), frozenset({"hemline", "waistline"})),
        "back_hemline": (frozenset({"back_bodice"}), frozenset({"hemline", "waistline"})),
        "sleeve_head": (frozenset({"sleeve"}), frozenset({"sleeve_head"})),
        "sleeve_underarm": (frozenset({"sleeve"}), frozenset({"sleeve_underarm"})),
        "sleeve_hem": (frozenset({"sleeve"}), frozenset({"sleeve_hem"})),
        "neckband_attachment": (
            frozenset({"collar", "front_bodice", "back_bodice"}),
            frozenset({"collar_attachment"}),
        ),
    },
    "pants": {
        "front_waistline": (frozenset({"front_pants"}), frozenset({"waistline"})),
        "back_waistline": (frozenset({"back_pants"}), frozenset({"waistline"})),
        "side_seam": (
            frozenset({"front_pants", "back_pants"}),
            frozenset({"side_seam", "outseam"}),
        ),
        "inseam": (frozenset({"front_pants", "back_pants"}), frozenset({"inseam"})),
        "front_crotch_curve": (frozenset({"front_pants"}), frozenset({"crotch_curve"})),
        "back_crotch_curve": (frozenset({"back_pants"}), frozenset({"crotch_curve"})),
        "hemline": (frozenset({"front_pants", "back_pants"}), frozenset({"hemline"})),
        "front_dart_leg": (frozenset({"front_pants"}), frozenset({"dart_leg"})),
        "back_dart_leg": (frozenset({"back_pants"}), frozenset({"dart_leg"})),
        # Legacy compatibility query; targets are deliberately UNKNOWN.
        "dart_leg": (frozenset({"front_pants", "back_pants"}), frozenset({"dart_leg"})),
    },
    "skirt": {
        "waistline": (frozenset({"front_skirt", "back_skirt"}), frozenset({"waistline"})),
        "side_seam": (frozenset({"front_skirt", "back_skirt"}), frozenset({"side_seam"})),
        "center_seam": (
            frozenset({"front_skirt", "back_skirt"}),
            frozenset({"center_front", "center_back"}),
        ),
        "hemline": (frozenset({"front_skirt", "back_skirt"}), frozenset({"hemline"})),
        "front_dart_leg": (frozenset({"front_skirt"}), frozenset({"dart_leg"})),
        "back_dart_leg": (frozenset({"back_skirt"}), frozenset({"dart_leg"})),
        # Legacy compatibility query; targets are deliberately UNKNOWN.
        "dart_leg": (frozenset({"front_skirt", "back_skirt"}), frozenset({"dart_leg"})),
    },
}


def _shortest_boundary_bridge(
    panel: PanelAnnotation,
    first_roles: frozenset[str],
    second_roles: frozenset[str],
) -> tuple[EdgeAnnotation, ...]:
    """Return the shortest non-empty boundary gap between two role runs.

    Legacy GarmentCode exports occasionally labelled the front shoulder as a
    side seam.  The shoulder is nevertheless topologically identifiable as
    the short boundary chain between neckline and armhole.  Both directions
    around the closed panel are considered so the long centre/hem/side route
    is not selected.
    """

    edges = tuple(sorted(panel.edges, key=lambda item: (item.index, item.id)))
    if not edges:
        return ()
    candidates: list[tuple[float, tuple[str, ...], tuple[EdgeAnnotation, ...]]] = []
    for position, edge in enumerate(edges):
        if edge.role not in first_roles:
            continue
        for direction in (-1, 1):
            bridge: list[EdgeAnnotation] = []
            for amount in range(1, len(edges) + 1):
                current = edges[(position + direction * amount) % len(edges)]
                if current.role in first_roles:
                    break
                if current.role in second_roles:
                    if bridge:
                        ordered = tuple(bridge if direction > 0 else reversed(bridge))
                        candidates.append(
                            (
                                sum(float(item.length_cm) for item in ordered),
                                tuple(item.id for item in ordered),
                                ordered,
                            )
                        )
                    break
                bridge.append(current)
    return min(candidates, default=(0.0, (), ()), key=lambda item: (item[0], item[1]))[2]


def _top_boundary_edges(panel: PanelAnnotation) -> tuple[EdgeAnnotation, ...]:
    """Recover an upright lower-body waist boundary from source geometry."""

    if not panel.vertices_cm:
        return ()
    values = tuple(_point(value) for value in panel.vertices_cm)
    minimum_y = min(value[1] for value in values)
    maximum_y = max(value[1] for value in values)
    tolerance = max((maximum_y - minimum_y) * 0.025, 1e-5)
    return tuple(
        edge
        for edge in panel.edges
        if edge.role != "dart_leg"
        and all(maximum_y - point[1] <= tolerance for point in (edge.start_cm, edge.end_cm))
    )


def resolved_common_basic_edge_roles(
    record: DraftingSemanticRecord,
) -> dict[str, str | None]:
    """Resolve defensible legacy roles and mask unresolved edges.

    The returned mapping is safe for primitive edge-role supervision.  A
    ``None`` value means UNKNOWN and must map to the loss ignore index, never
    to the negative ``other`` class.
    """

    category = common_basic_category(record) or _infer_simple_category(record)
    resolved: dict[str, str | None] = {
        edge.id: (None if edge.role == "other" else edge.role)
        for panel in record.panels
        for edge in panel.edges
    }
    if category == "tshirt":
        for panel in record.panels:
            if panel.role not in {"front_bodice", "back_bodice"}:
                continue
            if any(edge.role == "shoulder" for edge in panel.edges):
                continue
            bridge = _shortest_boundary_bridge(
                panel, frozenset({"neckline"}), frozenset({"armhole"})
            )
            for edge in bridge:
                resolved[edge.id] = "shoulder"
    elif category in {"pants", "skirt"}:
        roles = (
            {"front_pants", "back_pants"}
            if category == "pants"
            else {"front_skirt", "back_skirt"}
        )
        for panel in record.panels:
            if panel.role not in roles:
                continue
            for edge in _top_boundary_edges(panel):
                resolved[edge.id] = "waistline"
    return resolved


def _gcd_segments(edges: Sequence[EdgeAnnotation]) -> tuple[_Segment, ...]:
    return tuple(
        _Segment(
            source_id=edge.id,
            points_cm=(_point(edge.start_cm), _point(edge.end_cm)),
            length_cm=float(edge.length_cm),
            start_tangent_reliable=edge.curvature_type == "line",
            end_tangent_reliable=edge.curvature_type == "line",
        )
        for edge in sorted(edges, key=lambda item: (item.index, item.id))
    )


def _gcd_path_candidates(
    panels: Sequence[PanelAnnotation],
    panel_roles: frozenset[str],
    edge_roles: frozenset[str],
    resolved_roles: Mapping[str, str | None] | None = None,
) -> tuple[tuple[str, tuple[_Segment, ...], bool], ...]:
    output = []
    for panel in panels:
        if panel.role not in panel_roles:
            continue
        edges = tuple(
            edge
            for edge in panel.edges
            if (
                resolved_roles.get(edge.id, edge.role)
                if resolved_roles is not None
                else edge.role
            )
            in edge_roles
        )
        if edges:
            output.append(
                (
                    panel.id,
                    _gcd_segments(edges),
                    all(edge.curvature_type == "line" for edge in edges),
                )
            )
    return tuple(output)


def _structural_path_absence_is_known(
    record: DraftingSemanticRecord,
    category: str,
    query: str,
) -> bool:
    """Return true only for a source-exhaustive structural negative.

    Missing edge-role labels are never sufficient evidence of absence.  This
    deliberately avoids converting legacy ontology gaps into negative labels.
    """

    if query in {"front_dart_leg", "back_dart_leg"}:
        role_prefix = query.removesuffix("_dart_leg")
        wanted = {
            ("pants", "front"): "front_pants",
            ("pants", "back"): "back_pants",
            ("skirt", "front"): "front_skirt",
            ("skirt", "back"): "back_skirt",
        }[(category, role_prefix)]
        panel_roles = {panel.id: panel.role for panel in record.panels}
        return not any(panel_roles.get(dart.panel_id) == wanted for dart in record.darts)
    if category == "tshirt" and query == "neckband_attachment":
        return not any(panel.role == "collar" for panel in record.panels)
    return False


def _canonical_source_dart(
    record: DraftingSemanticRecord,
    panel_role: str,
    frame: GarmentFrame,
) -> Any | None:
    """Choose a stable mirrored instance without silently merging darts.

    GarmentCode commonly stores mirrored copies (and sometimes two physical
    waist darts) under one front/back panel role.  The fixed query represents
    the canonical positive-u instance.  This is explicit deterministic
    sampling, not a claim that the other physical darts are absent.
    """

    roles = {panel.id: panel.role for panel in record.panels}
    candidates = [dart for dart in record.darts if roles.get(dart.panel_id) == panel_role]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda dart: (
            frame.normalize_point(dart.apex_cm)[0],
            float(dart.confidence),
            dart.panel_id,
            tuple(dart.leg_edge_ids),
        ),
    )


def _source_dart_path_candidate(
    record: DraftingSemanticRecord,
    panel_role: str,
    frame: GarmentFrame,
) -> tuple[str, tuple[_Segment, ...], bool] | None:
    dart = _canonical_source_dart(record, panel_role, frame)
    if dart is None:
        return None
    panels = {panel.id: panel for panel in record.panels}
    edges = {edge.id: edge for edge in panels[dart.panel_id].edges}
    try:
        selected = tuple(edges[edge_id] for edge_id in dart.leg_edge_ids)
    except KeyError as error:  # record.validate normally catches this
        raise ValueError(f"dart references missing edge: {error.args[0]}") from error
    return dart.panel_id, _gcd_segments(selected), True


def _landmark_candidates(
    panels: Sequence[PanelAnnotation], panel_roles: frozenset[str], names: frozenset[str]
) -> list[tuple[tuple[float, float], bool, float, str, str]]:
    output = []
    for panel in panels:
        if panel.role not in panel_roles:
            continue
        for landmark in panel.landmarks:
            if landmark.name.upper() not in names:
                continue
            output.append(
                (
                    _point(landmark.xy_cm),
                    bool(landmark.training_eligible),
                    float(landmark.confidence),
                    landmark.evidence,
                    f"{panel.id}/{landmark.name}",
                )
            )
    return output


def _shared_resolved_role_points(
    panels: Sequence[PanelAnnotation],
    panel_role: str,
    first_role: str,
    second_role: str,
    resolved_roles: Mapping[str, str | None],
) -> list[tuple[tuple[float, float], bool, float, str, str]]:
    """Return shared endpoints after the auditable legacy-role repair."""

    output = []
    for panel in panels:
        if panel.role != panel_role:
            continue
        first = tuple(
            edge for edge in panel.edges if resolved_roles.get(edge.id) == first_role
        )
        second = tuple(
            edge for edge in panel.edges if resolved_roles.get(edge.id) == second_role
        )
        for left in first:
            for right in second:
                for candidate in (left.start_cm, left.end_cm):
                    if _same_point(candidate, right.start_cm) or _same_point(candidate, right.end_cm):
                        output.append(
                            (
                                _point(candidate),
                                True,
                                min(float(left.confidence), float(right.confidence)) * 0.9,
                                "derived_legacy_role_topology",
                                f"{left.id}+{right.id}",
                            )
                        )
    return output


def _select_landmark_candidate(
    candidates: Sequence[tuple[tuple[float, float], bool, float, str, str]], frame: GarmentFrame
) -> tuple[tuple[float, float], bool, str] | None:
    if not candidates:
        return None
    # Prefer training-eligible/high-confidence evidence, then the canonical
    # positive-x half when mirrored observations exist.
    selected = max(
        candidates,
        key=lambda item: (
            bool(item[1]),
            float(item[2]),
            frame.normalize_point(item[0])[0],
            item[4],
        ),
    )
    return selected[0], bool(selected[1]), f"{selected[3]}:{selected[4]}"


def _write_landmark_candidates(
    writer: _TargetWriter,
    name: str,
    candidates: Sequence[tuple[tuple[float, float], bool, float, str, str]],
) -> None:
    selected = _select_landmark_candidate(candidates, writer.frame)
    if selected is None:
        writer.unknown("landmark", name, evidence="named point unavailable in source adapter")
        return
    point, eligible, evidence = selected
    writer.present(
        "landmark",
        name,
        _landmark_values(point, writer.frame),
        evidence=evidence,
        training_eligible=eligible,
    )


def _write_tshirt_landmarks(writer: _TargetWriter, record: DraftingSemanticRecord) -> None:
    panels = record.panels
    resolved_roles = resolved_common_basic_edge_roles(record)
    direct = {
        "FNP": (frozenset({"front_bodice"}), frozenset({"FNP"})),
        "BNP": (frozenset({"back_bodice"}), frozenset({"BNP"})),
        "SNP_front": (frozenset({"front_bodice"}), frozenset({"SNP"})),
        "SNP_back": (frozenset({"back_bodice"}), frozenset({"SNP"})),
        "SP_front": (frozenset({"front_bodice"}), frozenset({"SP"})),
        "SP_back": (frozenset({"back_bodice"}), frozenset({"SP"})),
    }
    for query, (roles, names) in direct.items():
        candidates = _landmark_candidates(panels, roles, names)
        if query in {"SNP_front", "SNP_back", "SP_front", "SP_back"}:
            side = "front" if query.endswith("front") else "back"
            panel_role = f"{side}_bodice"
            first, second = (
                ("neckline", "shoulder") if query.startswith("SNP") else ("shoulder", "armhole")
            )
            candidates.extend(
                _shared_resolved_role_points(
                    panels, panel_role, first, second, resolved_roles
                )
            )
        _write_landmark_candidates(writer, query, candidates)
    _write_landmark_candidates(
        writer,
        "front_underarm",
        _shared_resolved_role_points(
            panels, "front_bodice", "armhole", "side_seam", resolved_roles
        ),
    )
    _write_landmark_candidates(
        writer,
        "back_underarm",
        _shared_resolved_role_points(
            panels, "back_bodice", "armhole", "side_seam", resolved_roles
        ),
    )

    cap_points: list[tuple[tuple[float, float], bool, float, str, str]] = []
    for panel in panels:
        if panel.role != "sleeve":
            continue
        for edge in panel.edges:
            if edge.role != "sleeve_head":
                continue
            for suffix, point in (("start", edge.start_cm), ("end", edge.end_cm)):
                cap_points.append(
                    (_point(point), True, float(edge.confidence) * 0.8, "derived_topology", f"{edge.id}/{suffix}")
                )
    if cap_points:
        # Sleeve halves meet nearer the garment x-axis than their underarm
        # endpoints.  This is a topology-derived target, not a curve-control
        # claim.
        chosen = min(
            cap_points,
            key=lambda item: (
                abs(writer.frame.normalize_point(item[0])[0] - 0.5),
                -writer.frame.normalize_point(item[0])[0],
                item[4],
            ),
        )
        writer.present(
            "landmark",
            "sleeve_cap_apex",
            _landmark_values(chosen[0], writer.frame),
            evidence=f"{chosen[3]}:{chosen[4]}",
        )
    else:
        writer.unknown("landmark", "sleeve_cap_apex", evidence="sleeve-head topology unavailable")


def _lower_point_candidates(
    lower_panels: Sequence[Any],
    *,
    source_role: str | None,
    name: str,
) -> list[tuple[tuple[float, float], bool, float, str, str]]:
    output = []
    for panel in lower_panels:
        if source_role is not None and panel.source_role != source_role:
            continue
        try:
            point = panel.landmark(name)
        except StopIteration:
            continue
        if not point.available or point.xy_cm is None:
            continue
        output.append(
            (
                _point(point.xy_cm),
                bool(point.training_eligible),
                float(point.confidence),
                point.evidence,
                f"{panel.panel_id}/{name}",
            )
        )
    return output


def _source_reference_line_candidates(
    panels: Sequence[PanelAnnotation], *, source_role: str, name: str
) -> list[tuple[tuple[tuple[float, float], tuple[float, float]], bool, float, str, str]]:
    output = []
    for panel in panels:
        if panel.role != source_role:
            continue
        for line in panel.reference_lines:
            if line.name.upper() != name.upper():
                continue
            output.append(
                (
                    tuple(_point(point) for point in line.points_cm),
                    bool(line.training_eligible),
                    float(line.confidence),
                    line.evidence,
                    f"{panel.id}/{line.name}",
                )
            )
    return output


def _lower_reference_line_candidates(
    lower_panels: Sequence[Any], *, source_role: str, name: str
) -> list[tuple[tuple[tuple[float, float], tuple[float, float]], bool, float, str, str]]:
    output = []
    for panel in lower_panels:
        if panel.source_role != source_role:
            continue
        try:
            line = panel.reference_line(name)
        except StopIteration:
            continue
        if not line.available or line.points_cm is None:
            continue
        output.append(
            (
                tuple(_point(point) for point in line.points_cm),
                bool(line.training_eligible),
                float(line.confidence),
                line.evidence,
                f"{panel.panel_id}/{name}",
            )
        )
    return output


def _write_reference_line_candidates(
    writer: _TargetWriter,
    query: str,
    candidates: Sequence[
        tuple[tuple[tuple[float, float], tuple[float, float]], bool, float, str, str]
    ],
    *,
    promote_provisional: str | None = None,
) -> None:
    if not candidates:
        writer.unknown(
            "reference_line", query, evidence="reference/construction line unavailable"
        )
        return
    eligible = [item for item in candidates if item[1]]
    if not eligible and promote_provisional is None:
        evidence = ",".join(sorted(f"{item[3]}:{item[4]}" for item in candidates))
        # Presence and coordinate supervision are independent.  The source
        # really contains this construction line, but its provenance says it
        # is not suitable as a coordinate target.  Preserve that observation
        # as present while masking both the query loss and all coordinates.
        # Treating it as UNKNOWN/absent would silently invert source evidence
        # (notably the stored, unvalidated T-shirt HL annotations).
        writer.present(
            "reference_line",
            query,
            None,
            evidence=f"stored line is not training eligible:{evidence}",
            training_eligible=False,
        )
        return
    pool = eligible or list(candidates)
    selected = max(
        pool,
        key=lambda item: (
            float(item[2]),
            sum(writer.frame.normalize_point(point)[0] for point in item[0]) / 2.0,
            item[4],
        ),
    )
    marker = "" if promote_provisional is None else f"{promote_provisional}:"
    writer.present(
        "reference_line",
        query,
        _reference_line_values(selected[0], writer.frame),
        evidence=f"{marker}{selected[3]}:{selected[4]}",
        training_eligible=True,
    )


def _write_record_reference_lines(
    writer: _TargetWriter, record: DraftingSemanticRecord, lower: Any | None
) -> None:
    if writer.category == "tshirt":
        specs = (
            ("front_BL", "front_bodice", "BL"),
            ("back_BL", "back_bodice", "BL"),
            ("front_WL", "front_bodice", "WL"),
            ("back_WL", "back_bodice", "WL"),
            ("front_HL", "front_bodice", "HL"),
            ("back_HL", "back_bodice", "HL"),
        )
        for query, role, source_name in specs:
            _write_reference_line_candidates(
                writer,
                query,
                _source_reference_line_candidates(
                    record.panels, source_role=role, name=source_name
                ),
            )
        return

    assert lower is not None
    if writer.category == "pants":
        specs = tuple(
            (f"{prefix}_{query_name}", role, source_name)
            for prefix, role in (("front", "front_pants"), ("back", "back_pants"))
            for query_name, source_name in (
                ("WL", "WL"),
                ("HL", "HL"),
                ("KL", "KNEE_LINE"),
                ("CL", "CL"),
                ("GRAIN", "GRAIN"),
            )
        )
    else:
        specs = tuple(
            (f"{prefix}_{name}", role, name)
            for prefix, role in (("front", "front_skirt"), ("back", "back_skirt"))
            for name in ("WL", "HL", "GRAIN")
        )
    for query, role, source_name in specs:
        promote = (
            "PROVISIONAL_CONVENTIONAL_HALF_LEG_KNEE_LINE"
            if writer.category == "pants" and source_name == "KNEE_LINE"
            else None
        )
        _write_reference_line_candidates(
            writer,
            query,
            _lower_reference_line_candidates(
                lower.panels, source_role=role, name=source_name
            ),
            promote_provisional=promote,
        )


def _write_pants_landmarks(writer: _TargetWriter, record: DraftingSemanticRecord, lower: Any) -> None:
    mapping = {
        "CF_waist": ("front_pants", "CF_WAIST"),
        "CB_waist": ("back_pants", "CB_WAIST"),
        "front_side_waist": ("front_pants", "SIDE_WAIST"),
        "back_side_waist": ("back_pants", "SIDE_WAIST"),
        "front_side_hip": ("front_pants", "SIDE_HIP"),
        "back_side_hip": ("back_pants", "SIDE_HIP"),
        "front_center_hip": ("front_pants", "CF_HIP"),
        "back_center_hip": ("back_pants", "CB_HIP"),
        "front_crotch_point": ("front_pants", "CROTCH_POINT"),
        "back_crotch_point": ("back_pants", "CROTCH_POINT"),
        "front_hem_in": ("front_pants", "HEM_INSEAM"),
        "front_hem_out": ("front_pants", "HEM_SIDE"),
        "back_hem_in": ("back_pants", "HEM_INSEAM"),
        "back_hem_out": ("back_pants", "HEM_SIDE"),
    }
    for query, (role, name) in mapping.items():
        _write_landmark_candidates(
            writer,
            query,
            _lower_point_candidates(lower.panels, source_role=role, name=name),
        )
    # The lower-body adapter defines knee on the conventional halfway point
    # along each leg edge.  It previously emitted presence=1 but then masked
    # every coordinate as `synthetic_unvalidated`, an internally inconsistent
    # target.  The experiment explicitly accepts provisional generator truth,
    # so expose the points with an unmistakable provenance marker.
    knee_mapping = {
        "front_knee_in": ("front_pants", "KNEE_INSEAM"),
        "front_knee_out": ("front_pants", "KNEE_SIDE"),
        "back_knee_in": ("back_pants", "KNEE_INSEAM"),
        "back_knee_out": ("back_pants", "KNEE_SIDE"),
    }
    for query, (role, name) in knee_mapping.items():
        candidates = _lower_point_candidates(lower.panels, source_role=role, name=name)
        selected = _select_landmark_candidate(candidates, writer.frame)
        if selected is None:
            writer.unknown(
                "landmark", query, evidence="conventional half-leg knee could not be derived"
            )
            continue
        point, _, evidence = selected
        writer.present(
            "landmark",
            query,
            _landmark_values(point, writer.frame),
            evidence=f"PROVISIONAL_CONVENTIONAL_HALF_LEG_KNEE:{evidence}",
            training_eligible=True,
        )

    _write_source_role_dart_landmarks(
        writer,
        record,
        (("front", "front_pants"), ("back", "back_pants")),
    )
    writer.unknown(
        "landmark",
        "dart_apex",
        evidence="deprecated combined dart query; use front_dart_apex/back_dart_apex",
    )


def _eligible_pair(panel: Any, first: str, second: str) -> tuple[Any, Any] | None:
    try:
        left, right = panel.landmark(first), panel.landmark(second)
    except StopIteration:
        return None
    if not left.available or not right.available or left.xy_cm is None or right.xy_cm is None:
        return None
    return left, right


def _write_source_role_dart_landmarks(
    writer: _TargetWriter,
    record: DraftingSemanticRecord,
    role_specs: Sequence[tuple[str, str]],
) -> None:
    """Write one deterministic canonical dart for each front/back role."""

    for prefix, panel_role in role_specs:
        dart = _canonical_source_dart(record, panel_role, writer.frame)
        names = (
            f"{prefix}_dart_apex",
            f"{prefix}_dart_leg_left",
            f"{prefix}_dart_leg_right",
        )
        if dart is None:
            for name in names:
                writer.absent(
                    "landmark",
                    name,
                    evidence=f"source Dart tuple has no {panel_role} dart",
                )
            continue
        bases = sorted(
            (_point(dart.base_cm[0]), _point(dart.base_cm[1])),
            key=lambda point: writer.frame.normalize_point(point)[0],
        )
        evidence = (
            "canonical_positive_u_instance_of_source_dart_tuple:"
            f"{dart.evidence}:{dart.panel_id}/{'+' .join(dart.leg_edge_ids)}"
        )
        for name, point in zip(names, (dart.apex_cm, bases[0], bases[1]), strict=True):
            writer.present(
                "landmark",
                name,
                _landmark_values(point, writer.frame),
                evidence=evidence,
            )


def _write_skirt_landmarks(writer: _TargetWriter, record: DraftingSemanticRecord, lower: Any) -> None:
    derived: dict[str, list[tuple[tuple[float, float], bool, float, str, str]]] = {
        name: []
        for prefix in ("front", "back")
        for name in (
            f"{prefix}_center_waist",
            f"{prefix}_side_waist",
            f"{prefix}_side_hip",
            f"{prefix}_center_hip",
            f"{prefix}_hem_center",
            f"{prefix}_hem_side",
        )
    }
    for panel in lower.panels:
        if panel.source_role not in {"front_skirt", "back_skirt"}:
            continue
        prefix = "front" if panel.source_role == "front_skirt" else "back"
        waist = _eligible_pair(panel, "WAIST_LEFT", "WAIST_RIGHT")
        hip = _eligible_pair(panel, "HIP_LEFT", "HIP_RIGHT")
        hem = _eligible_pair(panel, "HEM_LEFT", "HEM_RIGHT")
        if waist:
            left, right = waist
            midpoint = ((left.xy_cm[0] + right.xy_cm[0]) / 2.0, (left.xy_cm[1] + right.xy_cm[1]) / 2.0)
            eligible = bool(left.training_eligible and right.training_eligible)
            confidence = min(float(left.confidence), float(right.confidence))
            evidence = f"derived_topology:{panel.panel_id}/WAIST_LEFT+WAIST_RIGHT"
            derived[f"{prefix}_center_waist"].append((midpoint, eligible, confidence, evidence, panel.panel_id))
            side = max((_point(left.xy_cm), _point(right.xy_cm)), key=lambda value: (value[0], value[1]))
            derived[f"{prefix}_side_waist"].append((side, eligible, confidence, evidence, panel.panel_id))
        if hip:
            left, right = hip
            midpoint = (
                (left.xy_cm[0] + right.xy_cm[0]) / 2.0,
                (left.xy_cm[1] + right.xy_cm[1]) / 2.0,
            )
            derived[f"{prefix}_center_hip"].append(
                (
                    midpoint,
                    bool(left.training_eligible and right.training_eligible),
                    min(float(left.confidence), float(right.confidence)),
                    f"derived_topology:{panel.panel_id}/HIP_LEFT+HIP_RIGHT/midpoint",
                    panel.panel_id,
                )
            )
            side = max((_point(left.xy_cm), _point(right.xy_cm)), key=lambda value: (value[0], value[1]))
            derived[f"{prefix}_side_hip"].append(
                (
                    side,
                    bool(left.training_eligible and right.training_eligible),
                    min(float(left.confidence), float(right.confidence)),
                    f"derived_topology:{panel.panel_id}/HIP_LEFT+HIP_RIGHT",
                    panel.panel_id,
                )
            )
        if hem:
            left, right = hem
            midpoint = ((left.xy_cm[0] + right.xy_cm[0]) / 2.0, (left.xy_cm[1] + right.xy_cm[1]) / 2.0)
            eligible = bool(left.training_eligible and right.training_eligible)
            confidence = min(float(left.confidence), float(right.confidence))
            evidence = f"derived_topology:{panel.panel_id}/HEM_LEFT+HEM_RIGHT"
            derived[f"{prefix}_hem_center"].append((midpoint, eligible, confidence, evidence, panel.panel_id))
            side = max((_point(left.xy_cm), _point(right.xy_cm)), key=lambda value: (value[0], value[1]))
            derived[f"{prefix}_hem_side"].append((side, eligible, confidence, evidence, panel.panel_id))
    for query, candidates in derived.items():
        _write_landmark_candidates(writer, query, candidates)

    for query in ("center_waist", "side_waist", "side_hip", "hem_center", "hem_side"):
        writer.unknown(
            "landmark",
            query,
            evidence="deprecated exchangeable multi-panel landmark; use front/back query",
        )

    _write_source_role_dart_landmarks(
        writer,
        record,
        (("front", "front_skirt"), ("back", "back_skirt")),
    )
    for query in ("dart_apex", "dart_leg_left", "dart_leg_right"):
        writer.unknown(
            "landmark",
            query,
            evidence="deprecated combined dart query; use front/back dart queries",
        )

    slit_candidates = _lower_point_candidates(lower.panels, source_role=None, name="SLIT_TOP")
    _write_landmark_candidates(writer, "slit_end", slit_candidates)
    # A zipper availability bit does not reveal its endpoint.  Keep this point
    # unknown even when a production annotation says a zipper exists.
    writer.unknown("landmark", "closure_end", evidence="closure endpoint coordinate is not encoded")


def _write_record_paths(writer: _TargetWriter, record: DraftingSemanticRecord, lower: Any | None) -> None:
    resolved_roles = resolved_common_basic_edge_roles(record)
    for query, (panel_roles, edge_roles) in _PATH_SPECS[writer.category].items():
        if query == "dart_leg":
            writer.unknown(
                "path",
                query,
                evidence="deprecated combined dart query; use front_dart_leg/back_dart_leg",
            )
            continue
        if query in {"front_dart_leg", "back_dart_leg"}:
            role = next(iter(panel_roles))
            candidate = _source_dart_path_candidate(record, role, writer.frame)
            candidates = () if candidate is None else (candidate,)
        else:
            candidates = _gcd_path_candidates(
                record.panels, panel_roles, edge_roles, resolved_roles
            )
        chosen = _choose_path_values(candidates, writer.frame)
        if chosen is not None:
            writer.present("path", query, chosen[0], evidence=chosen[1])
        elif _structural_path_absence_is_known(record, writer.category, query):
            writer.absent(
                "path", query, evidence="source structure exhaustively establishes absence"
            )
        else:
            writer.unknown(
                "path",
                query,
                evidence="missing/legacy edge role is UNKNOWN, not a negative label",
            )

    if writer.category == "pants":
        writer.unknown(
            "path",
            "waistband_attachment",
            evidence="GarmentCode semantic record has no waistband-attachment edge ontology",
        )
    elif writer.category == "skirt":
        assert lower is not None
        slit_available = any(panel.feature("slit").available for panel in lower.panels)
        zipper_available = any(panel.feature("zipper").available for panel in lower.panels)
        slit_segments = []
        for panel in record.panels:
            edges = tuple(edge for edge in panel.edges if "slit" in edge.id.lower())
            if edges:
                slit_segments.append((panel.id, _gcd_segments(edges), all(edge.curvature_type == "line" for edge in edges)))
        chosen = _choose_path_values(slit_segments, writer.frame)
        if chosen is not None:
            writer.present("path", "slit", chosen[0], evidence=chosen[1])
        elif slit_available:
            writer.present(
                "path",
                "slit",
                None,
                evidence="observed source slit feature without recoverable path coordinates",
            )
        else:
            writer.unknown("path", "slit", evidence="source does not encode a trustworthy slit negative")
        if zipper_available:
            writer.present(
                "path",
                "closure",
                None,
                evidence="observed source zipper feature without recoverable path coordinates",
            )
        else:
            writer.unknown("path", "closure", evidence="source does not encode a trustworthy closure negative")
        writer.unknown(
            "path",
            "waistband_attachment",
            evidence="GarmentCode semantic record has no waistband-attachment edge ontology",
        )


def semantic_target_from_drafting_record(
    record: DraftingSemanticRecord,
    *,
    category: str | None = None,
    require_common_basic: bool = False,
) -> BasicSemanticTarget:
    """Convert one GarmentCode-style semantic record to the shared inventory."""

    record.validate()
    resolved = category or _infer_simple_category(record)
    if resolved not in CATEGORY_NAMES:
        raise ValueError(f"unsupported semantic category: {resolved!r}")
    if require_common_basic and common_basic_category(record) != resolved:
        raise ValueError(f"record {record.sample_id!r} is outside the strict common-{resolved} subset")
    relevant_roles = set().union(*_PANEL_ROLES[resolved].values())
    relevant_panels = tuple(panel for panel in record.panels if panel.role in relevant_roles)
    frame = GarmentFrame.from_points(
        point for panel in relevant_panels for point in panel.vertices_cm
    )
    writer = _TargetWriter(
        record.sample_id,
        resolved,
        "garmentcode_drafting_semantic_record",
        (
            "SOURCE_EXACT_AND_DERIVED_TOPOLOGY_MIXED"
            "+PROVISIONAL_CONVENTIONAL_HALF_LEG_KNEE"
            if resolved == "pants"
            else "SOURCE_EXACT_AND_DERIVED_TOPOLOGY_MIXED"
        ),
        frame,
    )

    for query, roles in _PANEL_ROLES[resolved].items():
        panels = tuple(panel for panel in relevant_panels if panel.role in roles)
        if panels:
            writer.present(
                "panel",
                query,
                _panel_box_values(
                    (point for panel in panels for point in panel.vertices_cm), frame
                ),
                evidence="observed source panel roles:" + ",".join(panel.id for panel in panels),
            )
        else:
            writer.absent("panel", query, evidence="exhaustive source panel roles contain no matching panel")

    lower = extract_lower_body_semantics(record) if resolved in {"pants", "skirt"} else None
    _write_record_paths(writer, record, lower)
    _write_record_reference_lines(writer, record, lower)
    if resolved == "tshirt":
        _write_tshirt_landmarks(writer, record)
    elif resolved == "pants":
        _write_pants_landmarks(writer, record, lower)
    else:
        _write_skirt_landmarks(writer, record, lower)
    return writer.finish()


def _document_points(document: Any) -> tuple[tuple[float, float], ...]:
    return tuple(
        _point(point)
        for panel in document.panels
        for edge in panel.edges
        for point in edge.points
    )


def _document_panel_points(document: Any, panel_ids: Sequence[str]) -> tuple[tuple[float, float], ...]:
    wanted = set(str(value) for value in panel_ids)
    return tuple(
        _point(point)
        for panel in document.panels
        if panel.id in wanted
        for edge in panel.edges
        for point in edge.points
    )


def _document_path_candidate(document: Any, entry: Mapping[str, Any]) -> tuple[str, tuple[_Segment, ...], bool]:
    panel_id = str(entry["panel_id"])
    panels = {panel.id: panel for panel in document.panels}
    if panel_id not in panels:
        raise ValueError(f"semantic path references unknown panel: {panel_id}")
    edges = {edge.id: edge for edge in panels[panel_id].edges}
    segments = []
    for edge_id in entry.get("edge_ids", ()):
        edge_id = str(edge_id)
        if edge_id not in edges:
            raise ValueError(f"semantic path references unknown edge: {panel_id}/{edge_id}")
        points = tuple(_point(value) for value in edges[edge_id].points)
        if len(points) < 2:
            raise ValueError(f"semantic path edge has fewer than two points: {panel_id}/{edge_id}")
        segments.append(
            _Segment(
                source_id=edge_id,
                points_cm=points,
                length_cm=_polyline_length(points),
                start_tangent_reliable=True,
                end_tangent_reliable=True,
            )
        )
    if not segments:
        raise ValueError(f"semantic path contains no edges: {panel_id}")
    return panel_id, tuple(segments), True


def _document_landmark_point(document: Any, entry: Mapping[str, Any]) -> tuple[float, float]:
    if "point_cm" in entry:
        return _point(entry["point_cm"])
    panel_id, edge_id = str(entry["panel_id"]), str(entry["edge_id"])
    panels = {panel.id: panel for panel in document.panels}
    if panel_id not in panels:
        raise ValueError(f"semantic landmark references unknown panel: {panel_id}")
    edges = {edge.id: edge for edge in panels[panel_id].edges}
    if edge_id not in edges:
        raise ValueError(f"semantic landmark references unknown edge: {panel_id}/{edge_id}")
    points = edges[edge_id].points
    index = int(entry.get("point_index", 0))
    if index < 0:
        index += len(points)
    if not 0 <= index < len(points):
        raise ValueError(f"semantic landmark index is out of range: {panel_id}/{edge_id}/{index}")
    return _point(points[index])


def _document_reference_line_points(
    entry: Mapping[str, Any]
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = tuple(_point(value) for value in entry.get("points_cm", ()))
    if len(points) != 2:
        raise ValueError("semantic reference line requires two points_cm endpoints")
    return points[0], points[1]


def _as_annotation_entries(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    return tuple(value or ())


def _document_dart_specialization(
    document: Any,
    category: str,
    groups: Mapping[str, Mapping[str, Any]],
    frame: GarmentFrame,
) -> tuple[
    dict[tuple[str, str], tuple[Mapping[str, Any], ...]],
    dict[str, tuple[float, float]],
]:
    """Split a legacy combined dart annotation into front/back queries.

    The legacy basic-block adapter is retained for source compatibility, but
    training never collapses two semantically different panel darts into one
    arbitrary instance.  Path topology itself identifies the two base points
    and shared apex for each front/back panel.
    """

    if category not in {"pants", "skirt"}:
        return {}, {}
    generic_paths = _as_annotation_entries(groups["path"].get("dart_leg", ()))
    overrides: dict[tuple[str, str], tuple[Mapping[str, Any], ...]] = {}
    landmarks: dict[str, tuple[float, float]] = {}
    if category == "skirt":
        for generic in ("center_waist", "side_waist", "side_hip", "hem_center", "hem_side"):
            for prefix in ("front", "back"):
                entries = tuple(
                    entry
                    for entry in _as_annotation_entries(
                        groups["landmark"].get(generic, ())
                    )
                    if prefix in str(entry.get("panel_id", "")).lower()
                )
                if entries:
                    overrides[("landmark", f"{prefix}_{generic}")] = (entries[0],)
    for prefix in ("front", "back"):
        matches = tuple(
            entry
            for entry in generic_paths
            if prefix in str(entry.get("panel_id", "")).lower()
        )
        if not matches:
            continue
        # One provisional basic block has exactly one dart on each role.  If a
        # future adapter emits several, select by explicit positive-u apex in
        # the same deterministic convention used for GCD records.
        candidates = []
        for entry in matches:
            panel_id, segments, _ = _document_path_candidate(document, entry)
            endpoints = [segment.points_cm[0] for segment in segments]
            endpoints.extend(segment.points_cm[-1] for segment in segments)
            shared = []
            for point in endpoints:
                if sum(_same_point(point, other) for other in endpoints) > 1:
                    if not any(_same_point(point, prior) for prior in shared):
                        shared.append(point)
            terminals = _terminal_points(segments)
            if len(shared) != 1 or len(terminals) < 2:
                raise ValueError(f"legacy dart topology is not a two-leg V: {panel_id}")
            candidates.append((frame.normalize_point(shared[0])[0], panel_id, entry, shared[0], terminals))
        _, _, chosen_entry, apex, terminals = max(candidates, key=lambda item: (item[0], item[1]))
        bases = sorted(terminals, key=lambda point: frame.normalize_point(point)[0])
        overrides[("path", f"{prefix}_dart_leg")] = (chosen_entry,)
        landmarks[f"{prefix}_dart_apex"] = _point(apex)
        landmarks[f"{prefix}_dart_leg_left"] = _point(bases[0])
        landmarks[f"{prefix}_dart_leg_right"] = _point(bases[-1])
    return overrides, landmarks


def semantic_target_from_pattern_document(
    document: Any,
    *,
    category: str,
    source: str,
    provenance_status: str,
    source_y_axis_down: bool = False,
) -> BasicSemanticTarget:
    """Build targets from an explicitly annotated ``PatternDocument``.

    This generic helper is intentionally public so another reviewed basic
    block source can use the same no-leakage tensor contract without being
    converted to ``DraftingSemanticRecord`` first.
    """

    if category not in CATEGORY_NAMES:
        raise ValueError(f"unsupported semantic category: {category!r}")
    frame = GarmentFrame.from_points(
        _document_points(document), source_y_axis_down=source_y_axis_down
    )
    writer = _TargetWriter(document.pattern_id, category, source, provenance_status, frame)
    annotations = document.annotations
    explicit_presence = annotations.get("semantic_query_presence")
    if not isinstance(explicit_presence, Mapping):
        raise ValueError("PatternDocument requires explicit semantic_query_presence annotations")
    groups = {
        "panel": annotations.get("semantic_panels", {}),
        "path": annotations.get("semantic_paths", {}),
        "landmark": annotations.get("semantic_landmarks", {}),
        "reference_line": annotations.get("semantic_reference_lines", {}),
    }
    if not all(isinstance(value, Mapping) for value in groups.values()):
        raise ValueError("PatternDocument semantic annotation groups must be mappings")
    specialized_entries, specialized_landmarks = _document_dart_specialization(
        document, category, groups, frame
    )
    deprecated_queries = {
        ("path", "dart_leg"),
        ("landmark", "dart_apex"),
        ("landmark", "dart_leg_left"),
        ("landmark", "dart_leg_right"),
    }
    if category == "skirt":
        deprecated_queries.update(
            ("landmark", name)
            for name in ("center_waist", "side_waist", "side_hip", "hem_center", "hem_side")
        )

    for query in (item for item in SEMANTIC_QUERY_INVENTORY if item.category == category):
        if query.name not in explicit_presence:
            raise ValueError(f"explicit presence map omits {category}/{query.name}")
        if (query.kind, query.name) in deprecated_queries:
            writer.unknown(
                query.kind,
                query.name,
                evidence="deprecated combined dart query; role-specific query is authoritative",
            )
            continue
        raw_presence = explicit_presence[query.name]
        if (query.kind, query.name) in specialized_entries or query.name in specialized_landmarks:
            raw_presence = True
        if raw_presence is None or str(raw_presence).upper() in {"UNKNOWN", "NOT_ASSERTED"}:
            writer.unknown(query.kind, query.name, evidence="explicit annotated UNKNOWN")
            continue
        if not isinstance(raw_presence, (bool, np.bool_)):
            raise ValueError(
                f"semantic presence must be bool/None/UNKNOWN: {category}/{query.name}"
            )
        present = bool(raw_presence)
        entries = specialized_entries.get(
            (query.kind, query.name),
            _as_annotation_entries(groups[query.kind].get(query.name, ())),
        )
        if not present:
            if entries:
                raise ValueError(f"absent query has geometry annotations: {category}/{query.name}")
            writer.absent(query.kind, query.name, evidence="explicit annotated absence")
            continue
        if not entries and not (query.kind == "landmark" and query.name in specialized_landmarks):
            raise ValueError(f"present query lacks geometry annotations: {category}/{query.name}")
        if query.kind == "panel":
            points = _document_panel_points(document, tuple(str(entry["panel_id"]) for entry in entries))
            values = _panel_box_values(points, frame)
            evidence = "exact annotated panels:" + ",".join(str(entry["panel_id"]) for entry in entries)
        elif query.kind == "path":
            chosen = _choose_path_values(
                tuple(_document_path_candidate(document, entry) for entry in entries), frame
            )
            if chosen is None:
                raise ValueError(f"present path has no candidate: {category}/{query.name}")
            values, evidence = chosen
            evidence = "dense exact " + evidence
        elif query.kind == "landmark":
            # Annotation order is part of the reviewed adapter.  It chooses a
            # canonical panel when a query (for example dart_apex) has several
            # physically valid instances.
            point = specialized_landmarks.get(query.name)
            if point is None:
                point = _document_landmark_point(document, entries[0])
            values = _landmark_values(point, frame)
            evidence = (
                "exact dart topology specialization:"
                if query.name in specialized_landmarks
                else "exact annotated landmark:"
            ) + str(entries[0].get("source_landmark", query.name) if entries else query.name)
        else:
            # Reference lines are construction evidence and never boundary
            # paths.  They supervise only their normalized fixed endpoints.
            values = _reference_line_values(
                _document_reference_line_points(entries[0]), frame
            )
            evidence = "exact annotated construction line:" + str(
                entries[0].get("line_name", query.name)
            )
        writer.present(query.kind, query.name, values, evidence=evidence)
    return writer.finish()


def semantic_target_from_basic_block(block: Any, *, curve_samples: int = 24) -> BasicSemanticTarget:
    """Convert one provisional common block while preserving its quarantine."""

    block.validate()
    document = block.to_pattern_document(curve_samples=curve_samples)
    status = str(document.annotations.get("provenance_status", "PROVISIONAL_EXPERT_REVIEW"))
    if status != "PROVISIONAL_EXPERT_REVIEW":
        raise ValueError("basic-block semantic targets must remain PROVISIONAL_EXPERT_REVIEW")
    return semantic_target_from_pattern_document(
        document,
        category=block.category,
        source="provisional_common_basic_block",
        provenance_status=status,
        source_y_axis_down=True,
    )


def stack_semantic_targets(targets: Sequence[BasicSemanticTarget]) -> dict[str, Any]:
    """Stack targets into NumPy fields named for the training loss contract."""

    values = tuple(targets)
    if not values:
        raise ValueError("cannot stack an empty semantic-target sequence")
    for target in values:
        target.validate()
    return {
        "sample_ids": tuple(target.sample_id for target in values),
        "sources": tuple(target.source for target in values),
        "category_ids": np.asarray([target.category_id for target in values], dtype=np.int64),
        "query_mask": np.stack([target.query_applicability for target in values]),
        "presence_targets": np.stack([target.presence for target in values]),
        "coordinate_targets": np.stack([target.coordinates for target in values]),
        "coordinate_mask": np.stack([target.coordinate_mask for target in values]),
        "query_keys": SEMANTIC_QUERY_KEYS,
    }


__all__ = [
    "BasicSemanticTarget",
    "GarmentFrame",
    "common_basic_category",
    "filter_common_basic_records",
    "is_common_basic_pants",
    "is_common_basic_skirt",
    "is_common_basic_tshirt",
    "panel_role_counts",
    "resolved_common_basic_edge_roles",
    "semantic_target_from_basic_block",
    "semantic_target_from_drafting_record",
    "semantic_target_from_pattern_document",
    "stack_semantic_targets",
]
