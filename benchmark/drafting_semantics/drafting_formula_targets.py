"""Build formula-shaped curve targets from explicitly traced T-shirt edges.

This module never discovers a curve role from its shape or completed-panel
neighbourhood.  It consumes only semantic roles already attached by the
GarmentCode creation-event binder or by FreeSewing's author-named output
adapter.  Geometry is converted into a stable chord-normalized representation
without discarding the original primitives stored on :class:`TracedEdge`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .tshirt_schema import (
    DRAFTING_FORMULA_ROLES,
    DRAFTING_FORMULA_SCALARS,
    CurveGeometry,
    DraftingFormulaSegment,
    DraftingFormulaTarget,
    DraftingSeamRelation,
    TracedEdge,
    TracedPanel,
    TracedPoint,
)


Point2D = tuple[float, float]


def _finite_numeric(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in values.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, bool):
        return output
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if prefix:
            output[prefix] = float(value)
        return output
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            output.update(_flatten_numeric(child, child_prefix))
    return output


def _bezier_point(points: Sequence[Point2D], t: float) -> Point2D:
    work = [(float(x), float(y)) for x, y in points]
    while len(work) > 1:
        work = [
            ((1.0 - t) * first[0] + t * second[0], (1.0 - t) * first[1] + t * second[1])
            for first, second in zip(work, work[1:])
        ]
    return work[0]


def _arc_points(geometry: CurveGeometry, samples: int) -> list[Point2D]:
    if geometry.center_cm is None or geometry.radius_cm is None:
        # The stored through-point is evidence for an arc, not a Bezier
        # control.  Sampling the polyline is a conservative geometry fallback
        # while its Bezier-control mask remains false.
        return [geometry.start_cm, *geometry.control_points_cm, geometry.end_cm]
    center_x, center_y = geometry.center_cm
    start_angle = math.atan2(geometry.start_cm[1] - center_y, geometry.start_cm[0] - center_x)
    end_angle = math.atan2(geometry.end_cm[1] - center_y, geometry.end_cm[0] - center_x)
    counter_clockwise = (end_angle - start_angle) % (2.0 * math.pi)
    if geometry.control_points_cm:
        through = geometry.control_points_cm[0]
        through_angle = math.atan2(through[1] - center_y, through[0] - center_x)
        through_from_start = (through_angle - start_angle) % (2.0 * math.pi)
        delta = counter_clockwise if through_from_start <= counter_clockwise + 1e-7 else counter_clockwise - 2.0 * math.pi
    elif geometry.clockwise is True:
        delta = counter_clockwise - 2.0 * math.pi
    else:
        delta = counter_clockwise
    radius = float(geometry.radius_cm)
    return [
        (
            center_x + radius * math.cos(start_angle + delta * index / (samples - 1)),
            center_y + radius * math.sin(start_angle + delta * index / (samples - 1)),
        )
        for index in range(samples)
    ]


def _sample_geometry(geometry: CurveGeometry, *, forward: bool, samples: int = 65) -> list[Point2D]:
    normalized = geometry.kind.strip().lower().replace("-", "_")
    if "bezier" in normalized:
        controls = (geometry.start_cm, *geometry.control_points_cm, geometry.end_cm)
        points = [_bezier_point(controls, index / (samples - 1)) for index in range(samples)]
    elif normalized == "arc":
        points = _arc_points(geometry, samples)
    else:
        points = [geometry.start_cm, geometry.end_cm]
    return points if forward else list(reversed(points))


def _target_key(panel: TracedPanel, role: str) -> str:
    return f"{panel.semantic_role}.{role}"


def _components(panel: TracedPanel, role: str) -> list[list[TracedEdge]]:
    candidates = [edge for edge in panel.edges if edge.semantic_role == role]
    by_point: dict[str, list[TracedEdge]] = defaultdict(list)
    for edge in candidates:
        by_point[edge.start_point_id].append(edge)
        by_point[edge.end_point_id].append(edge)
    remaining = {edge.id: edge for edge in candidates}
    output: list[list[TracedEdge]] = []
    while remaining:
        seed = next(iter(remaining.values()))
        stack = [seed]
        ids: set[str] = set()
        while stack:
            edge = stack.pop()
            if edge.id in ids:
                continue
            ids.add(edge.id)
            for point_id in (edge.start_point_id, edge.end_point_id):
                stack.extend(item for item in by_point[point_id] if item.id not in ids)
        component = [edge for edge in candidates if edge.id in ids]
        output.append(component)
        for edge in component:
            remaining.pop(edge.id, None)
    return output


def _endpoint_priority(role: str, point: TracedPoint) -> tuple[int, str]:
    priorities = {
        "neckline": {"FNP": 0, "BNP": 0, "SNP": 1},
        "armhole": {"SP": 0},
        "sleeve_head": {},
    }[role]
    return priorities.get(str(point.canonical_name), 5), point.id


def _ordered_component(
    panel: TracedPanel, role: str, component: Sequence[TracedEdge]
) -> tuple[list[tuple[TracedEdge, bool]], bool]:
    points = {point.id: point for point in panel.points}
    by_point: dict[str, list[TracedEdge]] = defaultdict(list)
    for edge in component:
        by_point[edge.start_point_id].append(edge)
        by_point[edge.end_point_id].append(edge)
    non_branching = all(len(edges) <= 2 for edges in by_point.values())
    endpoints = [points[point_id] for point_id, edges in by_point.items() if len(edges) == 1]
    choices = endpoints or [points[point_id] for point_id in by_point]
    start = min(choices, key=lambda point: _endpoint_priority(role, point))
    current = start.id
    unused = {edge.id: edge for edge in component}
    ordered: list[tuple[TracedEdge, bool]] = []
    while unused:
        choices = [edge for edge in by_point[current] if edge.id in unused]
        if not choices:
            break
        edge = min(choices, key=lambda item: item.id)
        forward = edge.start_point_id == current
        ordered.append((edge, forward))
        unused.pop(edge.id)
        current = edge.end_point_id if forward else edge.start_point_id
    return ordered, non_branching and not unused and len(endpoints) in {0, 2}


def _normalization(start: Point2D, end: Point2D, dense: Sequence[Point2D]) -> tuple[Point2D, Point2D, float, bool]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    chord = math.hypot(dx, dy)
    chord_available = chord > 1e-8
    if chord_available:
        axis = (dx / chord, dy / chord)
        return axis, (-axis[1], axis[0]), chord, True
    xs = [point[0] for point in dense]
    ys = [point[1] for point in dense]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return (1.0, 0.0), (0.0, 1.0), scale, False


def _coordinate_in_frame(
    point: Point2D, origin: Point2D, x_axis: Point2D, y_axis: Point2D, scale: float
) -> Point2D:
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return (
        (dx * x_axis[0] + dy * x_axis[1]) / scale,
        (dx * y_axis[0] + dy * y_axis[1]) / scale,
    )


def _unit_tangent(first: Point2D, second: Point2D, x_axis: Point2D, y_axis: Point2D) -> tuple[Point2D, bool]:
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= 1e-10:
        return (0.0, 0.0), False
    dx, dy = dx / length, dy / length
    return (dx * x_axis[0] + dy * x_axis[1], dx * y_axis[0] + dy * y_axis[1]), True


def _endpoint_name(point: TracedPoint) -> str | None:
    if point.canonical_name:
        return point.canonical_name
    if point.training_eligible and point.source_name and "unbound" not in point.source_name.lower():
        return point.source_name
    return None


def _lookup(mapping: Mapping[str, Any] | None, panel: TracedPanel, role: str, default: Any) -> Any:
    if mapping is None:
        return default
    return mapping.get(_target_key(panel, role), mapping.get(role, default))


def build_drafting_formula_targets(
    panels: Sequence[TracedPanel],
    *,
    source_kind: str,
    formula_parameters: Mapping[str, Mapping[str, float]] | None = None,
    formula_references: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DraftingFormulaTarget, ...]:
    """Create masked formula targets from already-evidenced semantic paths.

    ``source_kind`` must state whether roles came from GarmentCode's runtime
    event binder or FreeSewing's completed author-named output.  This keeps the
    latter useful without upgrading it to creation-time evidence.
    """

    if source_kind not in {"garmentcode_creation_trace", "freesewing_named_output"}:
        raise ValueError(f"unsupported formula-target source: {source_kind}")
    targets: list[DraftingFormulaTarget] = []
    for panel in panels:
        point_by_id = {point.id: point for point in panel.points}
        for role in DRAFTING_FORMULA_ROLES:
            for component_index, component in enumerate(_components(panel, role)):
                ordered, continuous = _ordered_component(panel, role, component)
                if not ordered:
                    continue
                dense_segments = [
                    _sample_geometry(edge.geometry, forward=forward) for edge, forward in ordered
                ]
                dense = [*dense_segments[0]]
                for points in dense_segments[1:]:
                    dense.extend(points[1:])
                first_edge, first_forward = ordered[0]
                last_edge, last_forward = ordered[-1]
                start_point_id = first_edge.start_point_id if first_forward else first_edge.end_point_id
                end_point_id = last_edge.end_point_id if last_forward else last_edge.start_point_id
                start = dense[0]
                end = dense[-1]
                x_axis, y_axis, scale, chord_available = _normalization(start, end, dense)
                frame_points = [_coordinate_in_frame(point, start, x_axis, y_axis, scale) for point in dense]
                xs = [point[0] for point in dense]
                ys = [point[1] for point in dense]
                arc_length = sum(math.dist(first, second) for first, second in zip(dense, dense[1:]))
                chord = math.dist(start, end)
                start_tangent, start_tangent_mask = _unit_tangent(dense[0], dense[1], x_axis, y_axis)
                end_tangent, end_tangent_mask = _unit_tangent(dense[-2], dense[-1], x_axis, y_axis)

                segments: list[DraftingFormulaSegment] = []
                for (edge, forward), sampled in zip(ordered, dense_segments):
                    controls = list(edge.geometry.control_points_cm)
                    if not forward:
                        controls.reverse()
                    normalized_kind = edge.geometry.kind.strip().lower().replace("-", "_")
                    if normalized_kind == "quadratic_bezier" and len(controls) == 1:
                        control_slots = (
                            _coordinate_in_frame(controls[0], start, x_axis, y_axis, scale),
                            (0.0, 0.0),
                        )
                        control_mask = (True, False)
                    elif normalized_kind == "cubic_bezier" and len(controls) == 2:
                        control_slots = tuple(
                            _coordinate_in_frame(point, start, x_axis, y_axis, scale) for point in controls
                        )
                        control_mask = (True, True)
                    else:
                        control_slots = ((0.0, 0.0), (0.0, 0.0))
                        control_mask = (False, False)
                    parameters = _flatten_numeric(edge.geometry.parameters, "geometry")
                    parameters.update(_finite_numeric(edge.provenance.get("measurement_inputs", {})))
                    segments.append(
                        DraftingFormulaSegment(
                            edge_id=edge.id,
                            geometry_kind=edge.geometry.kind,
                            normalized_start=_coordinate_in_frame(sampled[0], start, x_axis, y_axis, scale),
                            normalized_end=_coordinate_in_frame(sampled[-1], start, x_axis, y_axis, scale),
                            normalized_bezier_controls=(control_slots[0], control_slots[1]),
                            bezier_control_mask=control_mask,
                            source_formula=edge.formula,
                            operation_id=edge.operation_id,
                            source_parameters=parameters,
                        )
                    )

                parameters: dict[str, float] = {}
                for edge, _ in ordered:
                    parameters.update(_finite_numeric(edge.provenance.get("measurement_inputs", {})))
                parameters.update(_finite_numeric(_lookup(formula_parameters, panel, role, {})))
                parameter_mask = {name: True for name in parameters}
                operations = tuple(dict.fromkeys(
                    edge.operation_id for edge, _ in ordered if edge.operation_id is not None
                ))
                start_name = _endpoint_name(point_by_id[start_point_id])
                end_name = _endpoint_name(point_by_id[end_point_id])
                all_edges_eligible = all(edge.training_eligible for edge, _ in ordered)
                if source_kind == "garmentcode_creation_trace":
                    domain = "garmentcode_runtime"
                    evidence = "creation_event_formula_and_live_geometry"
                    source_boundary = "semantic role/formula attached at intercepted creation event"
                    confidence = 1.0
                else:
                    domain = "freesewing_named_output"
                    evidence = "author_named_completed_path_and_public_source_formula"
                    source_boundary = "completed author-named path; creation-time operation DAG unavailable"
                    confidence = 0.9
                references = tuple(str(item) for item in _lookup(formula_references, panel, role, ()))
                scalar_values = {
                    "width_cm": max(xs) - min(xs),
                    "height_cm": max(ys) - min(ys),
                    "depth_cm": max(abs(point[1]) for point in frame_points) * scale,
                    "chord_cm": chord,
                    "arc_length_cm": arc_length,
                }
                endpoint_dx = abs(end[0] - start[0])
                endpoint_dy = abs(end[1] - start[1])
                if role == "neckline":
                    semantic_values = {
                        "neckline_width_cm": endpoint_dx,
                        "neckline_depth_cm": endpoint_dy,
                    }
                elif role == "armhole":
                    semantic_values = {"armhole_depth_cm": endpoint_dy}
                else:
                    semantic_values = {"sleeve_cap_height_cm": scalar_values["depth_cm"]}
                target = DraftingFormulaTarget(
                    id=f"{panel.id}.{role}.{component_index:02d}",
                    panel_id=panel.id,
                    panel_role=panel.semantic_role,
                    semantic_role=role,
                    edge_ids=tuple(edge.id for edge, _ in ordered),
                    endpoint_point_ids=(start_point_id, end_point_id),
                    endpoint_names=(start_name, end_name),
                    endpoint_name_mask=(start_name is not None, end_name is not None),
                    scalar_values=scalar_values,
                    scalar_mask={name: (chord_available if name == "chord_cm" else True)
                                 for name in DRAFTING_FORMULA_SCALARS},
                    semantic_values=semantic_values,
                    semantic_mask={name: True for name in semantic_values},
                    endpoint_tangents_unit=(start_tangent, end_tangent),
                    endpoint_tangent_mask=(start_tangent_mask, end_tangent_mask),
                    segments=tuple(segments),
                    source_formula_parameters=parameters,
                    source_parameter_mask=parameter_mask,
                    operation_ids=operations,
                    domain=domain,
                    evidence=evidence,
                    provenance={
                        "source_kind": source_kind,
                        "evidence_boundary": source_boundary,
                        "formula_references": references,
                        "path_ordering": "endpoint graph over pre-labeled semantic edges",
                        "role_inferred_from_shape": False,
                        "normalization": "origin=start endpoint; X=start-to-end chord; scale=chord",
                        "control_policy": "only actual quadratic/cubic Bezier controls are mask-valid",
                        "semantic_measurement_definitions": {
                            "neckline_width_cm": "absolute panel-X separation between FNP/BNP and SNP",
                            "neckline_depth_cm": "absolute panel-Y separation between FNP/BNP and SNP",
                            "armhole_depth_cm": "absolute panel-Y separation between SP and underarm endpoint",
                            "sleeve_cap_height_cm": (
                                "maximum perpendicular distance from this semantic path's endpoint chord; "
                                "per generator piece when the source uses half sleeves"
                            ),
                        },
                        "source_edge_evidence": [edge.evidence for edge, _ in ordered],
                        "continuous_non_branching_path": continuous,
                    },
                    training_eligible=continuous and chord_available and all_edges_eligible,
                    confidence=confidence if continuous and chord_available else 0.0,
                )
                target.validate()
                targets.append(target)
    return tuple(targets)


def build_sleeve_armhole_relation(
    targets: Sequence[DraftingFormulaTarget], *, source_kind: str
) -> tuple[DraftingSeamRelation, ...]:
    """Store the observable whole-record seam-length contract.

    Exact front/back segment pairing is intentionally not invented here.  The
    relation is aggregate and says so in provenance; later source adapters can
    add finer author- or operation-backed pairings without changing this one.
    """

    sleeves = tuple(target for target in targets if target.semantic_role == "sleeve_head")
    armholes = tuple(target for target in targets if target.semantic_role == "armhole")
    if not sleeves or not armholes:
        return ()
    sleeve_length = sum(target.scalar_values["arc_length_cm"] for target in sleeves)
    armhole_length = sum(target.scalar_values["arc_length_cm"] for target in armholes)
    available = armhole_length > 1e-8 and all(
        target.scalar_mask["arc_length_cm"] and target.training_eligible for target in (*sleeves, *armholes)
    )
    ratio = sleeve_length / armhole_length if armhole_length > 1e-8 else 0.0
    operations = tuple(dict.fromkeys(
        operation_id for target in sleeves for operation_id in target.operation_ids
    ))
    if source_kind == "garmentcode_creation_trace":
        domain = "garmentcode_runtime"
        evidence = "creation_event_even_armhole_openings_and_live_geometry"
        source_contract = "GarmentCode even_armhole_openings runtime output"
        confidence = 1.0
    elif source_kind == "freesewing_named_output":
        domain = "freesewing_named_output"
        evidence = "author_named_completed_paths_and_public_sleevecap_contract"
        source_contract = "FreeSewing Teagan stores front/back armhole lengths for inherited sleeve draft"
        confidence = 0.9
    else:
        raise ValueError(f"unsupported formula-target source: {source_kind}")
    values = {
        "sleeve_head_length_cm": sleeve_length,
        "armhole_length_cm": armhole_length,
        "ease_difference_cm": sleeve_length - armhole_length,
        "ease_ratio": ratio,
    }
    relation = DraftingSeamRelation(
        id="sleeve_head_to_armhole.aggregate",
        sleeve_head_target_ids=tuple(target.id for target in sleeves),
        armhole_target_ids=tuple(target.id for target in armholes),
        values=values,
        value_mask={name: available for name in values},
        operation_ids=operations,
        domain=domain,
        evidence=evidence,
        provenance={
            "source_kind": source_kind,
            "source_contract": source_contract,
            "pairing_scope": "aggregate_per_record",
            "front_back_armholes_distinguished_by_target.panel_role": True,
            "exact_sleeve_segment_to_front_back_pairing_asserted": False,
            "post_hoc_role_inference": False,
        },
        training_eligible=available,
        confidence=confidence if available else 0.0,
    )
    relation.validate()
    return (relation,)


def freesewing_formula_context(raw: Mapping[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, tuple[str, ...]]]:
    """Return only formula inputs explicitly visible in Teagan's public source."""

    input_values = raw.get("input", {})
    options = _finite_numeric(input_values.get("resolved_options", {}))
    absolute_options = _finite_numeric(input_values.get("resolved_absolute_options_mm", {}))
    measurements = _finite_numeric(input_values.get("resolved_measurements_mm", {}))

    def selected(prefix: str, values: Mapping[str, float], names: Sequence[str], scale: float = 1.0) -> dict[str, float]:
        return {f"{prefix}.{name}": float(values[name]) * scale for name in names if name in values}

    neckline: dict[str, float] = {}
    neckline.update(selected("option", options, ("necklineWidth", "necklineDepth", "necklineBend", "backNeckCutout")))
    neckline.update(selected("absolute_option_cm", absolute_options,
                             ("necklineWidth", "necklineDepth", "backNeckCutout"), 0.1))
    neckline.update(selected("measurement_cm", measurements, ("hpsToWaistBack", "neck"), 0.1))
    sleeve_head = selected("measurement_cm", measurements, ("biceps",), 0.1)
    sleeve_head.update(selected("option", options, ("bicepsEase", "sleeveEase", "sleeveLength")))
    parameters = {
        "front.neckline": neckline,
        "back.neckline": neckline,
        "sleeve.sleeve_head": sleeve_head,
    }
    references = {
        "front.neckline": (
            "@freesewing/teagan@4.10.1/src/front.mjs: cfNeck, cfNeckCp1, neck, neckCp2 equations",
        ),
        "back.neckline": (
            "@freesewing/teagan@4.10.1/src/back.mjs: cbNeck, cbNeckCp1, neckCp2 equations",
        ),
        "front.armhole": (
            "@freesewing/teagan@4.10.1/src/front.mjs: author-named armhole path; inherited Brian construction",
        ),
        "back.armhole": (
            "@freesewing/teagan@4.10.1/src/back.mjs: author-named armhole path; inherited Brian construction",
        ),
        "sleeve.sleeve_head": (
            "@freesewing/teagan@4.10.1/src/sleeve.mjs: joins inherited library sleeve-cap points",
            "@freesewing/library: sleeve recipe source parameters not intercepted by this adapter",
        ),
    }
    return parameters, references


__all__ = [
    "build_drafting_formula_targets",
    "build_sleeve_armhole_relation",
    "freesewing_formula_context",
]
