"""Creation-event semantic bindings for the bounded GarmentCode T-shirt.

The generic runtime recorder deliberately does not guess pattern-making
semantics.  This recipe-specific binder is invoked *inside* the recorder when
an intercepted call finishes.  It names the points and primitives from the
call's own inputs/outputs and source site; it never reads the serialized
pattern or a completed-panel edge neighbourhood.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _numeric(values: Mapping[str, Any], names: Sequence[str], prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in names:
        value = values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            output[f"{prefix}.{name}"] = float(value)
    return output


def _flatten_numeric(value: Any, prefix: str, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        output[prefix] = float(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_numeric(child, f"{prefix}.{key}" if prefix else str(key), output)


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-16:
        return math.hypot(px - ax, py - ay)
    amount = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + amount * dx), py - (ay + amount * dy))


def _snapshot_edges(event: Mapping[str, Any], stage: str = "pre_geometry") -> list[Mapping[str, Any]]:
    geometry = event.get(stage, {})
    if not isinstance(geometry, Mapping):
        return []
    interface = geometry.get("target_interface", {})
    if not isinstance(interface, Mapping):
        return []
    sequence = interface.get("edges", {})
    if not isinstance(sequence, Mapping):
        return []
    edges = sequence.get("edges", ())
    return [edge for edge in edges if isinstance(edge, Mapping)] if isinstance(edges, Sequence) else []


def _primitive_point(primitive: Mapping[str, Any], endpoint: str) -> Mapping[str, Any] | None:
    value = primitive.get(endpoint)
    return value if isinstance(value, Mapping) else None


class BasicTShirtCreationBinder:
    """Attach recipe meanings while each runtime event is being completed."""

    def __init__(self, body: Mapping[str, float], design: Mapping[str, float]) -> None:
        self.body = {str(name): float(value) for name, value in body.items()}
        self.design = {str(name): float(value) for name, value in design.items()}
        self.events: dict[str, dict[str, Any]] = {}

    def __call__(self, event: dict[str, Any]) -> None:
        point_bindings: list[dict[str, Any]] = []
        primitive_bindings: list[dict[str, Any]] = []
        operation = str(event.get("operation", ""))
        source = str(event.get("source_reference", "")).replace("\\", "/")

        if operation == "EdgeSeqFactory.from_verts" and source.endswith("/assets/garment_programs/tee.py:41"):
            self._bind_torso_base(event, "front", point_bindings, primitive_bindings)
        elif operation == "EdgeSeqFactory.from_verts" and source.endswith("/assets/garment_programs/tee.py:92"):
            self._bind_torso_base(event, "back", point_bindings, primitive_bindings)
        elif operation == "EdgeSeqFactory.from_verts" and source.endswith("/assets/garment_programs/sleeves.py:141"):
            self._bind_sleeve_base(event, point_bindings, primitive_bindings)
        elif operation == "EdgeSequence.close_loop" and source.endswith("/assets/garment_programs/sleeves.py:150"):
            self._bind_sleeve_closure(event, primitive_bindings)
        elif operation == "ArmholeCurve":
            self._bind_armhole_helper(event, point_bindings, primitive_bindings)
        elif operation == "even_armhole_openings":
            self._bind_even_armhole_openings(event, point_bindings, primitive_bindings)
        elif operation in {"Edge.subdivide_param", "Edge.subdivide_len"}:
            self._bind_subdivision(event, point_bindings, primitive_bindings)
        elif operation == "CircleNeckHalf":
            self._bind_neck_helper(event, point_bindings, primitive_bindings)
        elif operation == "cut_corner":
            self._bind_corner_cut(event, point_bindings, primitive_bindings)

        event["semantic_points"] = point_bindings
        event["semantic_primitives"] = primitive_bindings
        event["semantic_binding"] = {
            "policy": "basic_tshirt_creation_event_v1",
            "uses_completed_panel_topology": False,
            "uses_serialized_pattern": False,
            "binding_time": "event_finish_before_next_recipe_statement",
        }
        self.events[str(event["id"])] = event

    def _base_inputs(self, role: str) -> dict[str, float]:
        body_names = ("bust", "back_width", "waist_line", "_shoulder_incl")
        values = _numeric(self.body, body_names, "body")
        values.update(_numeric(self.design, ("shirt_width", "shirt_flare", "shirt_length"), "design"))
        if role == "front":
            values.update(_numeric(self.body, ("shoulder_w",), "body"))
        return values

    def _torso_equation_values(self, role: str) -> dict[str, float]:
        """Evaluate the exact scalar equations used by tee.py's base block."""

        bust = self.body["bust"]
        back_width = self.body["back_width"]
        shirt_width = self.design["shirt_width"]
        shirt_flare = self.design["shirt_flare"]
        shirt_length = self.design["shirt_length"]
        shoulder_tangent = math.tan(math.radians(self.body["_shoulder_incl"]))
        body_width = (bust - back_width) / 2.0 if role == "front" else back_width / 2.0
        body_fraction = body_width / bust
        eased_total_width = shirt_width * bust
        panel_width = body_fraction * eased_total_width
        hem_width = body_fraction * shirt_flare * eased_total_width
        shoulder_rise = shoulder_tangent * panel_width
        base_length = shirt_length * self.body["waist_line"]
        front_back_width_difference = (body_fraction - (0.5 - body_fraction)) * bust
        panel_length = (
            base_length - shoulder_tangent * front_back_width_difference
            if role == "front"
            else base_length
        )
        return {
            "derived.body_width": body_width,
            "derived.body_fraction": body_fraction,
            "derived.eased_total_width": eased_total_width,
            "derived.panel_width": panel_width,
            "derived.hem_width": hem_width,
            "derived.shoulder_tangent": shoulder_tangent,
            "derived.shoulder_rise": shoulder_rise,
            "derived.base_length": base_length,
            "derived.front_back_width_difference": front_back_width_difference,
            "derived.panel_length": panel_length,
        }

    def _bind_torso_base(
        self,
        event: Mapping[str, Any],
        role: str,
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        inputs = self._base_inputs(role)
        inputs.update(self._torso_equation_values(role))
        center = "center_front" if role == "front" else "center_back"
        formulas = (
            "P0=(0,0)",
            "P1=(-hem_width,0); hem_width=body_fraction*shirt_flare*(shirt_width*bust)",
            "P2=(-panel_width,panel_length); panel_width=body_fraction*(shirt_width*bust)",
            "P3=(0,panel_length+shoulder_tangent*panel_width)",
        )
        if role == "front":
            formulas = (
                formulas[0],
                formulas[1],
                formulas[2]
                + "; panel_length=shirt_length*waist_line-shoulder_tangent*((body_fraction-(0.5-body_fraction))*bust)",
                formulas[3],
            )
        source_names = (f"{center}_hem", "side_hem", "raw_armhole_corner", "raw_neck_corner")
        for index, item in enumerate(event.get("point_inputs", ())):
            if not isinstance(item, Mapping) or index >= len(formulas):
                continue
            points.append(
                {
                    "object_token": item.get("object_token"),
                    "source_name": source_names[index],
                    "formula": formulas[index],
                    "measurement_inputs": inputs,
                    "dependency_event_ids": ("recipe.inputs",),
                    "evidence": "creation_event_binding",
                }
            )
        roles = ("hemline", "side_seam", "shoulder", center)
        for index, primitive in enumerate(event.get("created_primitives", ())):
            if isinstance(primitive, Mapping) and index < len(roles):
                primitives.append(
                    {
                        "edge_token": primitive.get("edge_token"),
                        "semantic_role": roles[index],
                        "formula": primitive.get("construction_formula", "line through recipe vertices"),
                        "measurement_inputs": inputs,
                        "evidence": "creation_event_binding",
                    }
                )

    def _bind_sleeve_base(
        self,
        event: Mapping[str, Any],
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        values = _numeric(self.body, ("arm_length", "wrist", "_shoulder_incl"), "body")
        values.update(_numeric(self.design, ("sleeve_length", "sleeve_end_width", "sleeve_angle"), "design"))
        raw_points = [item for item in event.get("point_inputs", ()) if isinstance(item, Mapping)]
        if len(raw_points) >= 3:
            coordinates = [item.get("xy") for item in raw_points[:3]]
            if all(isinstance(value, Sequence) and len(value) >= 2 for value in coordinates):
                values.update(
                    {
                        "evaluated.end_width": abs(float(coordinates[1][1])),
                        "evaluated.sleeve_panel_length": float(coordinates[2][0]),
                        "evaluated.arm_width": abs(float(coordinates[2][1])),
                        "constant.minimum_sleeve_length": 5.0,
                    }
                )
        formulas = (
            "P0=(0,0)",
            "P1=(0,-end_width); end_width=max(sleeve_end_width*arm_width,wrist/2)",
            "P2=(panel_length,-arm_width); panel_length=max(sleeve_length*(arm_length-opening_length)+length_shift,5)",
        )
        for index, item in enumerate(event.get("point_inputs", ())):
            if isinstance(item, Mapping) and index < len(formulas):
                points.append(
                    {
                        "object_token": item.get("object_token"),
                        "source_name": ("sleeve_hem_fold", "sleeve_hem_underarm", "sleeve_underarm_head")[index],
                        "formula": formulas[index],
                        "measurement_inputs": values,
                        "dependency_event_ids": tuple(event.get("dependencies", ())) or ("recipe.inputs",),
                        "evidence": "creation_event_binding",
                    }
                )
        for index, primitive in enumerate(event.get("created_primitives", ())):
            if isinstance(primitive, Mapping) and index < 2:
                primitives.append(
                    {
                        "edge_token": primitive.get("edge_token"),
                        "semantic_role": ("sleeve_hem", "sleeve_underarm")[index],
                        "formula": primitive.get("construction_formula", "line through recipe vertices"),
                        "measurement_inputs": values,
                        "evidence": "creation_event_binding",
                    }
                )

    def _bind_sleeve_closure(self, event: Mapping[str, Any], primitives: list[dict[str, Any]]) -> None:
        values = _numeric(self.body, ("arm_length", "wrist", "_shoulder_incl"), "body")
        for primitive in event.get("created_primitives", ()):
            if isinstance(primitive, Mapping):
                primitives.append(
                    {
                        "edge_token": primitive.get("edge_token"),
                        "semantic_role": "sleeve_underarm",
                        "formula": "close_loop(last sleeve-head endpoint, P0)",
                        "measurement_inputs": values,
                        "evidence": "creation_event_binding",
                    }
                )

    @staticmethod
    def _helper_values(event: Mapping[str, Any]) -> dict[str, float]:
        output: dict[str, float] = {}
        _flatten_numeric(event.get("parameters", {}), "evaluated", output)
        return output

    def _bind_armhole_helper(
        self,
        event: Mapping[str, Any],
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        values = _numeric(
            self.body,
            ("bust", "back_width", "shoulder_w", "_shoulder_incl", "_base_sleeve_balance", "_armscye_depth"),
            "body",
        )
        values.update(_numeric(self.design, ("armhole_depth", "sleeve_angle", "armhole_smoothing"), "design"))
        values.update(self._helper_values(event))
        for index, primitive in enumerate(event.get("created_primitives", ())):
            if not isinstance(primitive, Mapping):
                continue
            role = "armhole_projection_template" if index == 0 else "sleeve_head"
            primitives.append(
                {
                    "edge_token": primitive.get("edge_token"),
                    "semantic_role": role,
                    "formula": "ArmholeCurve(incl,width,angle,bottom_angle_mix,cubic_controls)",
                    "measurement_inputs": values,
                    "evidence": "creation_event_binding",
                    "training_eligible": role == "sleeve_head",
                }
            )
            for endpoint, suffix in (("start_point", "start"), ("end_point", "end")):
                point = _primitive_point(primitive, endpoint)
                if point is not None:
                    points.append(
                        {
                            "object_token": point.get("object_token"),
                            "source_name": f"{role}_{suffix}",
                            "formula": "ArmholeCurve endpoint from evaluated incl/width/angle",
                            "measurement_inputs": values,
                            "dependency_event_ids": tuple(event.get("dependencies", ())) or ("recipe.inputs",),
                            "evidence": "creation_event_binding",
                            "training_eligible": role == "sleeve_head",
                        }
                    )

    def _bind_neck_helper(
        self,
        event: Mapping[str, Any],
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        values = _numeric(self.body, ("neck_w", "_bust_line", "_shoulder_incl"), "body")
        values.update(_numeric(self.design, ("neck_width", "front_neck_depth", "back_neck_depth"), "design"))
        values.update(self._helper_values(event))
        for primitive in event.get("created_primitives", ()):
            if not isinstance(primitive, Mapping):
                continue
            primitives.append(
                {
                    "edge_token": primitive.get("edge_token"),
                    "semantic_role": "neckline_projection_template",
                    "formula": "CircleNeckHalf: circle through (0,0),(width,0),(width/2,-depth), then half arc",
                    "measurement_inputs": values,
                    "evidence": "creation_event_binding",
                    "training_eligible": False,
                }
            )
            for endpoint, suffix in (("start_point", "center"), ("end_point", "shoulder")):
                point = _primitive_point(primitive, endpoint)
                if point is not None:
                    points.append(
                        {
                            "object_token": point.get("object_token"),
                            "source_name": f"neckline_template_{suffix}",
                            "formula": "CircleNeckHalf template endpoint",
                            "measurement_inputs": values,
                            "dependency_event_ids": tuple(event.get("dependencies", ())) or ("recipe.inputs",),
                            "evidence": "creation_event_binding",
                            "training_eligible": False,
                        }
                    )

    def _bind_even_armhole_openings(
        self,
        event: Mapping[str, Any],
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        values = _numeric(self.body, ("bust", "back_width", "_base_sleeve_balance"), "body")
        values.update(self._helper_values(event))
        for primitive in event.get("created_primitives", ()):
            if not isinstance(primitive, Mapping) or "bezier" not in str(primitive.get("kind", "")):
                continue
            primitives.append(
                {
                    "edge_token": primitive.get("edge_token"),
                    "semantic_role": "sleeve_head",
                    "formula": "even_armhole_openings(front_opening,back_opening,tol)",
                    "measurement_inputs": values,
                    "evidence": "creation_event_binding",
                }
            )
            for endpoint, suffix in (("start_point", "start"), ("end_point", "end")):
                point = _primitive_point(primitive, endpoint)
                if point is not None:
                    points.append(
                        {
                            "object_token": point.get("object_token"),
                            "source_name": f"sleeve_head_{suffix}",
                            "formula": "even_armhole_openings output endpoint",
                            "measurement_inputs": values,
                            "dependency_event_ids": tuple(event.get("dependencies", ())),
                            "evidence": "creation_event_binding",
                            "training_eligible": True,
                        }
                    )

    def _dependency_helper(self, event: Mapping[str, Any]) -> Mapping[str, Any] | None:
        for event_id in reversed(tuple(event.get("dependencies", ()))):
            dependency = self.events.get(str(event_id))
            if dependency and dependency.get("operation") in {"ArmholeCurve", "CircleNeckHalf"}:
                return dependency
        return None

    def _bind_subdivision(
        self,
        event: Mapping[str, Any],
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        """Propagate a parent curve's creation-time meaning to its exact pieces."""

        input_tokens = set(str(value) for value in event.get("inputs", ()))
        parent_binding: Mapping[str, Any] | None = None
        for dependency_id in reversed(tuple(event.get("dependencies", ()))):
            dependency = self.events.get(str(dependency_id))
            if dependency is None:
                continue
            for candidate in dependency.get("semantic_primitives", ()):
                if isinstance(candidate, Mapping) and str(candidate.get("edge_token")) in input_tokens:
                    parent_binding = candidate
                    break
            if parent_binding is not None:
                break
        if parent_binding is None:
            return
        role = str(parent_binding.get("semantic_role", "other"))
        values = {
            str(name): float(value)
            for name, value in parent_binding.get("measurement_inputs", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        values.update(self._helper_values(event))
        for primitive in event.get("created_primitives", ()):
            if not isinstance(primitive, Mapping):
                continue
            primitives.append(
                {
                    "edge_token": primitive.get("edge_token"),
                    "semantic_role": role,
                    "formula": f"{event.get('operation')}({parent_binding.get('formula', role)})",
                    "measurement_inputs": values,
                    "evidence": "creation_event_binding",
                    "derived_from_edge_token": parent_binding.get("edge_token"),
                }
            )
            for endpoint, suffix in (("start_point", "start"), ("end_point", "end")):
                point = _primitive_point(primitive, endpoint)
                if point is not None:
                    points.append(
                        {
                            "object_token": point.get("object_token"),
                            "source_name": f"{role}_subdivision_{suffix}",
                            "formula": f"{event.get('operation')} endpoint on {role}",
                            "measurement_inputs": values,
                            "dependency_event_ids": tuple(event.get("dependencies", ())),
                            "input_object_tokens": tuple(event.get("inputs", ())),
                            "evidence": "creation_event_binding",
                            "training_eligible": True,
                        }
                    )

    def _bind_corner_cut(
        self,
        event: Mapping[str, Any],
        points: list[dict[str, Any]],
        primitives: list[dict[str, Any]],
    ) -> None:
        helper = self._dependency_helper(event)
        helper_operation = None if helper is None else str(helper.get("operation"))
        panel_id = str(event.get("panel_id", ""))
        panel_role = "front" if "ftorso" in panel_id else "back" if "btorso" in panel_id else "other"
        if helper_operation == "CircleNeckHalf":
            curve_role = "neckline"
            pre_roles = ("shoulder", "center_front" if panel_role == "front" else "center_back")
        elif helper_operation == "ArmholeCurve":
            curve_role = "armhole"
            pre_roles = ("side_seam", "shoulder")
        else:
            return

        values = self._helper_values(helper or event)
        if helper_operation == "CircleNeckHalf":
            values.update(_numeric(self.body, ("neck_w", "_bust_line", "_shoulder_incl"), "body"))
            values.update(_numeric(self.design, ("neck_width", "front_neck_depth", "back_neck_depth"), "design"))
        else:
            values.update(
                _numeric(
                    self.body,
                    ("bust", "back_width", "shoulder_w", "_shoulder_incl", "_base_sleeve_balance", "_armscye_depth"),
                    "body",
                )
            )
            values.update(_numeric(self.design, ("armhole_depth", "sleeve_angle", "armhole_smoothing"), "design"))

        pre_edges = _snapshot_edges(event, "pre_geometry")
        curve_primitives: list[Mapping[str, Any]] = []
        for primitive in event.get("created_primitives", ()):
            if not isinstance(primitive, Mapping):
                continue
            kind = str(primitive.get("kind", ""))
            is_inserted_curve = (curve_role == "neckline" and kind == "circular_arc") or (
                curve_role == "armhole" and "bezier" in kind
            )
            role = curve_role if is_inserted_curve else self._role_from_pre_edge(primitive, pre_edges, pre_roles)
            if role is None:
                continue
            primitives.append(
                {
                    "edge_token": primitive.get("edge_token"),
                    "semantic_role": role,
                    "formula": (
                        "cut_corner(CircleNeckHalf(depth,width), collar_corner)"
                        if curve_role == "neckline"
                        else "cut_corner(ArmholeCurve(incl,width,angle), shoulder_corner)"
                    ),
                    "measurement_inputs": values,
                    "evidence": "creation_event_binding",
                }
            )
            if is_inserted_curve:
                curve_primitives.append(primitive)

        for primitive in curve_primitives:
            for endpoint in ("start_point", "end_point"):
                point = _primitive_point(primitive, endpoint)
                if point is None:
                    continue
                coordinate = point.get("xy")
                if not isinstance(coordinate, Sequence):
                    continue
                location_role = self._point_role_from_pre_edges(coordinate, pre_edges, pre_roles)
                canonical = None
                if curve_role == "neckline":
                    if location_role == "shoulder":
                        canonical = "SNP"
                    elif location_role in {"center_front", "center_back"}:
                        canonical = "FNP" if panel_role == "front" else "BNP"
                elif location_role == "shoulder":
                    canonical = "SP"
                points.append(
                    {
                        "object_token": point.get("object_token"),
                        "canonical_name": canonical,
                        "source_name": f"{curve_role}_{location_role or 'endpoint'}",
                        "formula": (
                            f"cut_corner({helper_operation}, target_interface).endpoint_on({location_role})"
                        ),
                        "measurement_inputs": values,
                        "dependency_event_ids": tuple(event.get("dependencies", ())),
                        "input_object_tokens": tuple(event.get("inputs", ())),
                        "evidence": "creation_event_binding",
                        "training_eligible": True,
                    }
                )

    @staticmethod
    def _point_role_from_pre_edges(
        coordinate: Sequence[float], pre_edges: Sequence[Mapping[str, Any]], roles: Sequence[str]
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for edge, role in zip(pre_edges, roles):
            start, end = edge.get("start"), edge.get("end")
            if isinstance(start, Sequence) and isinstance(end, Sequence):
                candidates.append((_point_segment_distance(coordinate, start, end), role))
        return min(candidates)[1] if candidates else None

    @classmethod
    def _role_from_pre_edge(
        cls,
        primitive: Mapping[str, Any],
        pre_edges: Sequence[Mapping[str, Any]],
        roles: Sequence[str],
    ) -> str | None:
        start = _primitive_point(primitive, "start_point")
        end = _primitive_point(primitive, "end_point")
        if start is None or end is None:
            return None
        first, second = start.get("xy"), end.get("xy")
        if not isinstance(first, Sequence) or not isinstance(second, Sequence):
            return None
        midpoint = ((float(first[0]) + float(second[0])) / 2, (float(first[1]) + float(second[1])) / 2)
        return cls._point_role_from_pre_edges(midpoint, pre_edges, roles)


def semantic_binding_maps(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return latest creation-event bindings by runtime object token."""

    point_map: dict[str, dict[str, Any]] = {}
    primitive_map: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event.get("id"))
        for raw in event.get("semantic_points", ()):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("object_token"), str):
                continue
            value = dict(raw)
            value["event_id"] = event_id
            existing = point_map.get(str(raw["object_token"]))
            # Preserve the first observed creator for ordinary points.  A later
            # cut event may legitimately promote the same runtime vertex to a
            # canonical FNP/BNP/SNP/SP binding, which takes precedence.
            if (
                existing is None
                or raw.get("canonical_name")
                or (not bool(existing.get("training_eligible", True)) and bool(raw.get("training_eligible", True)))
            ):
                point_map[str(raw["object_token"])] = value
        for raw in event.get("semantic_primitives", ()):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("edge_token"), str):
                continue
            value = dict(raw)
            value["event_id"] = event_id
            primitive_map[str(raw["edge_token"])] = value
    return point_map, primitive_map


__all__ = ["BasicTShirtCreationBinder", "semantic_binding_maps"]
