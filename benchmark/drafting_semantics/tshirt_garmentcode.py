"""Construction-truth adapter for GarmentCode's bounded basic T-shirt recipe.

This module intentionally does not annotate a serialized ``*_specification``
file.  It builds the garment while :class:`RuntimeTraceRecorder` is active,
then snapshots the still-live panels before ``assembly()`` mutates their
pivots and geometric ids.  Runtime operation ids are retained all the way to
the canonical FNP/BNP/SNP/SP and edge-role labels.

The basic ``Shirt`` recipe is a half-pattern generator.  Its eight physical
panels (left/right front/back torso and two half sleeves per side) are kept as
generator truth.  That topology is not presented as a conventional factory
T-shirt block.  BP and darts are not defined by this recipe; body reference
levels are supplied in an explicitly separate anthropometric-adapter domain.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .runtime_trace import RuntimeTraceRecorder
from .drafting_formula_targets import build_drafting_formula_targets, build_sleeve_armhole_relation
from .tshirt_creation_binding import BasicTShirtCreationBinder, semantic_binding_maps
from .tshirt_schema import (
    ConstructionOperation,
    CurveGeometry,
    DartTrace,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
    TracedPoint,
    TracedReferenceLine,
)


BODY_INPUT_FIELDS = (
    "height",
    "head_l",
    "bust",
    "waist",
    "hips",
    "back_width",
    "waist_back_width",
    "shoulder_w",
    "shoulder_incl",
    "waist_line",
    "hips_line",
    "bust_line",
    "vert_bust_line",
    "armscye_depth",
    "arm_length",
    "wrist",
    "arm_pose_angle",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_garmentcode_root() -> Path:
    return _repo_root() / "external" / "GarmentCode"


def _ensure_external_import(root: Path) -> None:
    value = str(root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite trace value")
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    return {"object_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _value(design: Mapping[str, Any], *path: str) -> Any:
    current: Any = design
    for key in path:
        current = current[key]
    return current["v"] if isinstance(current, Mapping) and "v" in current else current


def _set_value(design: dict[str, Any], path: Iterable[str], value: Any) -> None:
    keys = tuple(path)
    current = design
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]]["v"] = value


def load_basic_tshirt_design(root: Path | None = None) -> dict[str, Any]:
    """Load and constrain the official preset to the bounded symmetric lane."""

    root = (root or default_garmentcode_root()).resolve()
    design = yaml.safe_load((root / "assets/design_params/t-shirt.yaml").read_text(encoding="utf-8"))["design"]
    design = copy.deepcopy(design)
    _set_value(design, ("meta", "upper"), "Shirt")
    _set_value(design, ("meta", "wb"), None)
    _set_value(design, ("meta", "bottom"), None)
    _set_value(design, ("shirt", "strapless"), False)
    _set_value(design, ("collar", "f_collar"), "CircleNeckHalf")
    _set_value(design, ("collar", "b_collar"), "CircleNeckHalf")
    _set_value(design, ("collar", "component", "style"), None)
    _set_value(design, ("sleeve", "sleeveless"), False)
    _set_value(design, ("sleeve", "armhole_shape"), "ArmholeCurve")
    _set_value(design, ("sleeve", "standing_shoulder"), False)
    _set_value(design, ("sleeve", "connect_ruffle"), 1.0)
    _set_value(design, ("sleeve", "cuff", "type"), None)
    _set_value(design, ("left", "enable_asym"), False)
    return design


def apply_tshirt_design_values(base: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
    """Apply only the active numeric design variables used by this corpus."""

    design = copy.deepcopy(dict(base))
    paths = {
        "shirt_width": ("shirt", "width"),
        "shirt_flare": ("shirt", "flare"),
        "shirt_length": ("shirt", "length"),
        "neck_width": ("collar", "width"),
        "front_neck_depth": ("collar", "fc_depth"),
        "back_neck_depth": ("collar", "bc_depth"),
        "sleeve_length": ("sleeve", "length"),
        "armhole_depth": ("sleeve", "connecting_width"),
        "sleeve_end_width": ("sleeve", "end_width"),
        "sleeve_angle": ("sleeve", "sleeve_angle"),
        "armhole_smoothing": ("sleeve", "smoothing_coeff"),
    }
    unknown = set(values) - set(paths)
    if unknown:
        raise KeyError(f"unsupported T-shirt design values: {sorted(unknown)}")
    for name, raw in values.items():
        _set_value(design, paths[name], float(raw))
    return design


def _active_design_values(design: Mapping[str, Any]) -> dict[str, float]:
    return {
        "shirt_width": float(_value(design, "shirt", "width")),
        "shirt_flare": float(_value(design, "shirt", "flare")),
        "shirt_length": float(_value(design, "shirt", "length")),
        "neck_width": float(_value(design, "collar", "width")),
        "front_neck_depth": float(_value(design, "collar", "fc_depth")),
        "back_neck_depth": float(_value(design, "collar", "bc_depth")),
        "sleeve_length": float(_value(design, "sleeve", "length")),
        "armhole_depth": float(_value(design, "sleeve", "connecting_width")),
        "sleeve_end_width": float(_value(design, "sleeve", "end_width")),
        "sleeve_angle": float(_value(design, "sleeve", "sleeve_angle")),
        "armhole_smoothing": float(_value(design, "sleeve", "smoothing_coeff")),
    }


def _collect_panels(component: Any, panel_class: type[Any]) -> tuple[Any, ...]:
    found: dict[int, Any] = {}

    def walk(value: Any) -> None:
        if isinstance(value, panel_class):
            found[id(value)] = value
            return
        getter = getattr(value, "_get_subcomponents", None)
        if callable(getter):
            for child in getter():
                walk(child)

    walk(component)
    return tuple(sorted(found.values(), key=lambda panel: panel.name))


def _panel_role(name: str) -> str:
    lowered = name.lower()
    if "ftorso" in lowered:
        return "front"
    if "btorso" in lowered:
        return "back"
    if "sleeve" in lowered:
        return "sleeve"
    return "other"


def _panel_side(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("left") or "_left" in lowered:
        return "left"
    if lowered.startswith("right") or "_right" in lowered:
        return "right"
    return "unspecified"


def _curve_geometry(edge: Any) -> CurveGeometry:
    start = tuple(float(value) for value in edge.start)
    end = tuple(float(value) for value in edge.end)
    name = type(edge).__name__
    if name == "CurveEdge":
        curve = edge.as_curve(absolute=True)
        points = curve.bpoints()[1:-1]
        controls = tuple((float(point.real), float(point.imag)) for point in points)
        kind = "quadratic_bezier" if len(controls) == 1 else "cubic_bezier"
        return CurveGeometry(
            kind=kind,
            start_cm=start,
            end_cm=end,
            control_points_cm=controls,
            parameters={"relative_control_points": _plain(edge.control_points)},
        )
    if name == "CircleEdge":
        radius, large_arc, sweep = edge.as_radius_flag()
        midpoint = edge.midpoint()
        curve = edge.as_curve()
        center = (float(curve.center.real), float(curve.center.imag))
        start_angle = float(curve.theta)
        end_angle = float(curve.theta + curve.delta)
        return CurveGeometry(
            kind="arc",
            start_cm=start,
            end_cm=end,
            control_points_cm=((float(midpoint[0]), float(midpoint[1])),),
            center_cm=center,
            radius_cm=float(radius),
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
            clockwise=bool(curve.delta < 0.0),
            parameters={
                "radius_cm": float(radius),
                "large_arc": bool(large_arc),
                "sweep": bool(sweep),
                "relative_control_y": float(edge.control_y),
            },
        )
    return CurveGeometry(kind="line", start_cm=start, end_cm=end)


def _trace_operations(events: Iterable[Mapping[str, Any]]) -> tuple[ConstructionOperation, ...]:
    """Convert the observed object-flow trace into a recipe-rooted DAG."""

    event_values = tuple(events)
    setup_id = "recipe.inputs"
    operations: list[ConstructionOperation] = [
        ConstructionOperation(
            id=setup_id,
            order=0,
            operation="bind_body_and_design_parameters",
            outputs=("body_measurements", "active_design_parameters"),
            parameters={"recipe": "GarmentCode Shirt(fitted=False)"},
            source_reference="external/GarmentCode/assets/garment_programs/meta_garment.py",
        )
    ]
    id_map = {str(event["id"]): f"runtime.{event['id']}" for event in event_values}
    for event in event_values:
        panel_id = event.get("panel_id")
        raw_dependencies = tuple(str(item) for item in event.get("dependencies", ()))
        unknown_dependencies = [item for item in raw_dependencies if item not in id_map]
        if unknown_dependencies:
            raise ValueError(
                f"runtime event {event.get('id')} references unknown dependencies: "
                f"{unknown_dependencies}"
            )
        dependencies = [id_map[item] for item in raw_dependencies]
        # Only observed runtime roots consume recipe inputs directly.  Adding
        # this dependency to every node would make reachability validation
        # vacuous and hide broken object-flow links between drafting events.
        if not dependencies:
            dependencies.append(setup_id)
        operations.append(
            ConstructionOperation(
                id=id_map[str(event["id"])],
                order=1 + int(event.get("order", 0)),
                operation=str(event.get("operation", "unknown")),
                dependencies=tuple(dependencies),
                inputs=tuple(str(value) for value in event.get("inputs", ())),
                outputs=tuple(str(value) for value in event.get("outputs", ())),
                parameters={
                    **_plain(event.get("parameters", {})),
                    "point_inputs": _plain(event.get("point_inputs", [])),
                    "created_points": _plain(event.get("created_points", [])),
                    "created_primitives": _plain(event.get("created_primitives", [])),
                    "semantic_points": _plain(event.get("semantic_points", [])),
                    "semantic_primitives": _plain(event.get("semantic_primitives", [])),
                },
                source_reference=event.get("source_reference"),
                pre_geometry=_plain(event.get("pre_geometry", {})),
                post_geometry=_plain(event.get("post_geometry", {})),
                is_helper=bool(event.get("helper")),
                status=str(event.get("status", "unknown")),
                provenance={
                    "implementation_reference": event.get("implementation_reference"),
                    "panel_id": panel_id,
                    "object_tokens": _plain(event.get("object_tokens", {})),
                    "semantic_binding": _plain(event.get("semantic_binding", {})),
                },
            )
        )
    return tuple(operations)


def _runtime_operation_id(event_id: Any) -> str:
    value = str(event_id)
    return value if value == "recipe.inputs" else f"runtime.{value}"


def _build_panel(
    panel: Any,
    runtime: RuntimeTraceRecorder,
    point_bindings: Mapping[str, Mapping[str, Any]],
    edge_bindings: Mapping[str, Mapping[str, Any]],
) -> TracedPanel:
    """Cross-link live geometry to bindings captured at its creation event."""

    role = _panel_role(panel.name)
    vertices: list[Any] = []
    seen: set[int] = set()
    for edge in panel.edges:
        for vertex in (edge.start, edge.end):
            if id(vertex) not in seen:
                seen.add(id(vertex))
                vertices.append(vertex)
    point_ids = {id(vertex): f"{panel.name}.point_{index:02d}" for index, vertex in enumerate(vertices)}
    points: list[TracedPoint] = []
    for vertex in vertices:
        identity = id(vertex)
        token = runtime.object_token(vertex, "vertex")
        binding = point_bindings.get(token)
        event_id = None if binding is None else binding.get("event_id")
        dependency_events = () if binding is None else tuple(binding.get("dependency_event_ids", ()))
        dependencies = tuple(_runtime_operation_id(value) for value in dependency_events)
        points.append(
            TracedPoint(
                id=point_ids[identity],
                panel_id=panel.name,
                xy_cm=(float(vertex[0]), float(vertex[1])),
                formula=(
                    str(binding.get("formula"))
                    if binding is not None
                    else "UNOBSERVED_DIRECT_CONSTRUCTOR; coordinate preserved by pre-assembly snapshot"
                ),
                canonical_name=None if binding is None else binding.get("canonical_name"),
                source_name=(
                    f"{panel.name}.unbound_boundary_vertex"
                    if binding is None
                    else str(binding.get("source_name") or f"{panel.name}.creation_vertex")
                ),
                measurement_inputs=(
                    {} if binding is None else {str(k): float(v) for k, v in binding.get("measurement_inputs", {}).items()}
                ),
                dependencies=dependencies,
                operation_id=None if event_id is None else _runtime_operation_id(event_id),
                evidence="preassembly_snapshot_unbound" if binding is None else str(binding.get("evidence", "creation_event_binding")),
                training_eligible=False if binding is None else bool(binding.get("training_eligible", True)),
                confidence=0.0 if binding is None else 1.0,
                provenance={
                    "captured_before_assembly": True,
                    "source_panel": panel.name,
                    "runtime_object_token": token,
                    "semantic_binding_time": None if binding is None else "creation_event_finish",
                },
            )
        )

    traced_edges: list[TracedEdge] = []
    for index, edge in enumerate(panel.edges):
        token = runtime.object_token(edge, "edge")
        binding = edge_bindings.get(token)
        event_id = None if binding is None else binding.get("event_id")
        traced_edges.append(
            TracedEdge(
                id=f"{panel.name}.edge_{index:02d}",
                panel_id=panel.name,
                start_point_id=point_ids[id(edge.start)],
                end_point_id=point_ids[id(edge.end)],
                semantic_role="other" if binding is None else str(binding.get("semantic_role", "other")),
                geometry=_curve_geometry(edge),
                source_name=str(getattr(edge, "label", "")) or type(edge).__name__,
                formula=(
                    "UNOBSERVED_DIRECT_CONSTRUCTOR; geometry preserved by pre-assembly snapshot"
                    if binding is None
                    else str(binding.get("formula", "creation event output primitive"))
                ),
                dependencies=(point_ids[id(edge.start)], point_ids[id(edge.end)]),
                operation_id=None if event_id is None else _runtime_operation_id(event_id),
                evidence="preassembly_snapshot_unbound" if binding is None else str(binding.get("evidence", "creation_event_binding")),
                training_eligible=False if binding is None else bool(binding.get("training_eligible", True)),
                confidence=0.0 if binding is None else 1.0,
                provenance={
                    "captured_before_assembly": True,
                    "source_edge_index": index,
                    "runtime_object_token": token,
                    "semantic_binding_time": None if binding is None else "creation_event_finish",
                    "measurement_inputs": {} if binding is None else _plain(binding.get("measurement_inputs", {})),
                },
            )
        )
    coordinates = [(float(vertex[0]), float(vertex[1])) for vertex in vertices]
    return TracedPanel(
        id=panel.name,
        semantic_role=role,
        source_name=panel.name,
        operation_id=None,
        points=tuple(points),
        edges=tuple(traced_edges),
        metadata={
            "generator_topology": "left/right half pattern",
            "semantic_role_origin": "GarmentCode component class/name, not completed edge topology",
            "side": _panel_side(panel.name),
            "translation_cm": _plain(panel.translation),
            "rotation_xyz_degrees": _plain(panel.rotation.as_euler("XYZ", degrees=True)),
            "bbox_cm": {
                "min": [min(point[0] for point in coordinates), min(point[1] for point in coordinates)],
                "max": [max(point[0] for point in coordinates), max(point[1] for point in coordinates)],
            },
        },
    )


def _append_snapshot_operations(
    operations: tuple[ConstructionOperation, ...], panels: tuple[TracedPanel, ...]
) -> tuple[tuple[ConstructionOperation, ...], tuple[TracedPanel, ...]]:
    """Add post-runtime grouping nodes after, never before, observed creation."""

    output = list(operations)
    updated: list[TracedPanel] = []
    for panel in panels:
        dependencies = sorted(
            {
                item.operation_id
                for item in (*panel.points, *panel.edges)
                if item.operation_id is not None
            }
        )
        if not dependencies:
            dependencies = ["recipe.inputs"]
        operation_id = f"snapshot.{panel.id}"
        output.append(
            ConstructionOperation(
                id=operation_id,
                order=len(output),
                operation="capture_live_panel_pre_assembly",
                dependencies=tuple(dependencies),
                inputs=tuple(edge.id for edge in panel.edges),
                outputs=(panel.id,),
                parameters={"grouping_only": True, "serialized_pattern_used": False},
                source_reference="benchmark/drafting_semantics/tshirt_garmentcode.py",
                domain="preassembly_snapshot",
                evidence="observed_live_geometry",
                training_eligible=False,
                provenance={"causal_role": "post-runtime grouping; not a synthetic creation predecessor"},
            )
        )
        updated.append(replace(panel, operation_id=operation_id))
    output.append(
        ConstructionOperation(
            id="adapter.reference_levels",
            order=len(output),
            operation="derive_anthropometric_reference_levels",
            dependencies=tuple(panel.operation_id for panel in updated if panel.operation_id is not None),
            inputs=("panel_3d_placement", "height", "head_l", "waist_line", "_bust_line", "hips_line"),
            outputs=("BL", "WL", "HL"),
            parameters={"domain": "anthropometric_adapter", "not_part_of_garmentcode_recipe": True},
            source_reference="benchmark/drafting_semantics/tshirt_garmentcode.py",
            domain="anthropometric_adapter",
            evidence="derived_generator_formula",
            training_eligible=False,
        )
    )
    return tuple(output), tuple(updated)


def _creation_contract_summary(
    panels: tuple[TracedPanel, ...], operations: tuple[ConstructionOperation, ...]
) -> dict[str, Any]:
    """Fail closed when a train target cannot be tied to a creation event."""

    expected = {"front": {"FNP", "SNP", "SP"}, "back": {"BNP", "SNP", "SP"}}
    for panel in panels:
        if panel.semantic_role in expected:
            observed = {
                str(point.canonical_name)
                for point in panel.points
                if point.canonical_name is not None and point.training_eligible
            }
            if observed != expected[panel.semantic_role]:
                raise ValueError(
                    f"creation-event canonical binding mismatch for {panel.id}: "
                    f"expected {sorted(expected[panel.semantic_role])}, got {sorted(observed)}"
                )
    canonical_points = [
        point for panel in panels for point in panel.points if point.canonical_name is not None and point.training_eligible
    ]
    for point in canonical_points:
        if point.evidence != "creation_event_binding":
            raise ValueError(f"canonical point {point.id} is not creation-event evidence")
        if not point.measurement_inputs or not point.dependencies or not str(point.operation_id).startswith("runtime."):
            raise ValueError(f"canonical point {point.id} lacks input/formula/DAG linkage")
    eligible_edges = [edge for panel in panels for edge in panel.edges if edge.training_eligible]
    invalid_edges = [
        edge.id
        for edge in eligible_edges
        if edge.semantic_role == "other"
        or edge.evidence != "creation_event_binding"
        or not str(edge.operation_id).startswith("runtime.")
    ]
    if invalid_edges:
        raise ValueError(f"training edges without creation-event semantics: {invalid_edges}")

    all_edges = [edge for panel in panels for edge in panel.edges]
    all_points = [point for panel in panels for point in panel.points]
    unbound_edges = [edge.id for edge in all_edges if not edge.training_eligible]
    if unbound_edges:
        raise ValueError(
            "live pre-assembly edges lack creation-event bindings: "
            f"edges={unbound_edges}"
        )

    # The canonical points and every supervised edge must be actual outputs of
    # the operation that owns their label.  This prevents a later refactor
    # from silently replacing runtime provenance with a topology heuristic.
    for point in canonical_points:
        operation = next(item for item in operations if item.id == point.operation_id)
        created_tokens = {
            str(item.get("object_token"))
            for item in operation.parameters.get("created_points", ())
            if isinstance(item, Mapping)
        }
        if str(point.provenance.get("runtime_object_token")) not in created_tokens:
            raise ValueError(f"canonical point {point.id} is not an output of {operation.id}")
    for edge in eligible_edges:
        operation = next(item for item in operations if item.id == edge.operation_id)
        created_tokens = {
            str(item.get("edge_token"))
            for item in operation.parameters.get("created_primitives", ())
            if isinstance(item, Mapping)
        }
        if str(edge.provenance.get("runtime_object_token")) not in created_tokens:
            raise ValueError(f"semantic edge {edge.id} is not an output of {operation.id}")

    by_id = {operation.id: operation for operation in operations}
    reachable: set[str] = set()
    visiting: set[str] = set()

    def reaches_inputs(operation_id: str) -> bool:
        if operation_id == "recipe.inputs":
            return True
        if operation_id in reachable:
            return True
        if operation_id in visiting:
            raise ValueError(f"operation DAG cycle reaches {operation_id}")
        if operation_id not in by_id:
            raise ValueError(f"operation DAG dependency is missing: {operation_id}")
        visiting.add(operation_id)
        operation = by_id[operation_id]
        try:
            dependency_reachability = [reaches_inputs(dependency) for dependency in operation.dependencies]
            if dependency_reachability and all(dependency_reachability):
                reachable.add(operation_id)
                return True
            return False
        finally:
            visiting.remove(operation_id)

    disconnected = [operation.id for operation in operations if not reaches_inputs(operation.id)]
    if disconnected:
        raise ValueError(f"operation DAG contains nodes disconnected from recipe.inputs: {disconnected}")
    return {
        "canonical_creation_points": len(canonical_points),
        "training_edges": len(eligible_edges),
        "unbound_snapshot_edges": 0,
        "unbound_snapshot_points": sum(not point.training_eligible for point in all_points),
        "canonical_points_verified_as_operation_outputs": True,
        "semantic_edges_verified_as_operation_outputs": True,
        "operation_count": len(operations),
        "all_operations_reachable_from_recipe_inputs": True,
        "completed_panel_topology_used_for_semantic_labels": False,
    }


def _reference_lines(panels: Iterable[Any], body: Mapping[str, float]) -> tuple[TracedReferenceLine, ...]:
    lines: list[TracedReferenceLine] = []
    world_hps = float(body["height"]) - float(body["head_l"])
    levels = {
        "BL": world_hps - float(body["_bust_line"]),
        "WL": world_hps - float(body["waist_line"]),
        "HL": world_hps - float(body["waist_line"]) - float(body["hips_line"]),
    }
    for panel in panels:
        if _panel_role(panel.name) not in {"front", "back"}:
            continue
        vertices = [(float(vertex[0]), float(vertex[1])) for vertex in panel.edges.verts()]
        x_min = min(point[0] for point in vertices)
        x_max = max(point[0] for point in vertices)
        y_min = min(point[1] for point in vertices)
        y_max = max(point[1] for point in vertices)
        vertical_axis_world = panel.rotation.apply([0.0, 1.0, 0.0])
        if not math.isclose(abs(float(vertical_axis_world[1])), 1.0, abs_tol=1e-6):
            raise ValueError(f"panel {panel.name} local Y is not world vertical")
        sign = 1.0 if float(vertical_axis_world[1]) >= 0.0 else -1.0
        for canonical, world_y in levels.items():
            local_y = sign * (world_y - float(panel.translation[1]))
            intersects = y_min <= local_y <= y_max
            input_names = {
                "BL": ("height", "head_l", "_bust_line"),
                "WL": ("height", "head_l", "waist_line"),
                "HL": ("height", "head_l", "waist_line", "hips_line"),
            }[canonical]
            lines.append(
                TracedReferenceLine(
                    id=f"{panel.name}.{canonical}",
                    panel_id=panel.name,
                    canonical_name=canonical,
                    source_name=f"anthropometric_{canonical.lower()}",
                    geometry=CurveGeometry(
                        kind="line", start_cm=(x_min, local_y), end_cm=(x_max, local_y)
                    ),
                    formula=f"world_{canonical}_y - panel.translation_y",
                    measurement_inputs={name: float(body[name]) for name in input_names},
                    dependencies=(f"snapshot.{panel.name}",),
                    operation_id="adapter.reference_levels",
                    auxiliary=True,
                    domain="anthropometric_adapter",
                    evidence="derived_generator_formula",
                    provenance={
                        "world_y_cm": world_y,
                        "intersects_panel": intersects,
                        "not_created_by_garmentcode": True,
                    },
                    training_eligible=intersects,
                    confidence=1.0,
                )
            )
    return tuple(lines)


def generate_garmentcode_tshirt_trace(
    *,
    sample_id: str,
    split: str,
    body_values: Mapping[str, float],
    design_values: Mapping[str, float] | None = None,
    garmentcode_root: Path | None = None,
    body_id: str | None = None,
    design_id: str | None = None,
) -> TShirtTraceRecord:
    """Generate one creation-time trace without serializing or assembling it."""

    root = (garmentcode_root or default_garmentcode_root()).resolve()
    _ensure_external_import(root)
    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs.meta_garment import MetaGarment
    from pygarment.garmentcode.panel import Panel

    design_path = root / "assets/design_params/t-shirt.yaml"
    design = apply_tshirt_design_values(load_basic_tshirt_design(root), design_values or {})
    # Upstream BodyParameters currently calls ``load('')`` when constructed
    # without a path, so seed it from the official mean file and immediately
    # replace every corpus-controlled measurement.
    body = BodyParameters(root / "assets/bodies/mean_all.yaml")
    body.load_from_dict({name: float(raw) for name, raw in body_values.items()})
    active_design = _active_design_values(design)
    semantic_binder = BasicTShirtCreationBinder(body.params, active_design)
    with RuntimeTraceRecorder(
        root,
        metadata={"sample_id": sample_id, "split": split, "body_id": body_id, "design_id": design_id},
        event_enricher=semantic_binder,
    ) as runtime:
        garment = MetaGarment(sample_id, body, design)
    trace_failures = [
        {
            "event_id": event.get("id"),
            "operation": event.get("operation"),
            "status": event.get("status"),
            "capture_error": event.get("capture_error"),
            "enrichment_error": event.get("enrichment_error"),
            "error": event.get("error"),
        }
        for event in runtime.events
        if event.get("status") != "ok" or event.get("capture_error") or event.get("enrichment_error")
    ]
    if trace_failures:
        raise ValueError(f"runtime trace contains failed capture/enrichment events: {trace_failures}")
    panels_live = _collect_panels(garment, Panel)
    if len(panels_live) != 8:
        raise ValueError(f"bounded basic T-shirt must create 8 half-panels, got {len(panels_live)}")
    point_bindings, edge_bindings = semantic_binding_maps(runtime.events)
    runtime_add_dart_count = sum(event["operation"] == "Panel.add_dart" for event in runtime.events)
    if runtime_add_dart_count != 0:
        raise ValueError(
            "bounded Shirt(fitted=False) no-dart contract changed: "
            f"observed {runtime_add_dart_count} Panel.add_dart events"
        )
    operations = _trace_operations(runtime.events)
    panels = tuple(_build_panel(panel, runtime, point_bindings, edge_bindings) for panel in panels_live)
    operations, panels = _append_snapshot_operations(operations, panels)
    creation_contract = _creation_contract_summary(panels, operations)
    drafting_formula_targets = build_drafting_formula_targets(
        panels, source_kind="garmentcode_creation_trace"
    )
    drafting_seam_relations = build_sleeve_armhole_relation(
        drafting_formula_targets, source_kind="garmentcode_creation_trace"
    )
    darts = tuple(
        DartTrace(
            id=f"{panel.id}.dart_not_applicable",
            panel_id=panel.id,
            kind="none",
            applicable=False,
            applicability_reason="GarmentCode Shirt(fitted=False) does not call Panel.add_dart",
            domain="garmentcode_runtime",
            evidence="observed_runtime",
            provenance={"observed_add_dart_event_count": runtime_add_dart_count},
            training_eligible=True,
        )
        for panel in panels
        if panel.semantic_role in {"front", "back"}
    )
    record = TShirtTraceRecord(
        sample_id=sample_id,
        split=split,
        source={
            "name": "GarmentCode",
            "repository": "https://github.com/maria-korosteleva/GarmentCode",
            "revision": _git_revision(root),
            "license": "MIT",
        },
        body={name: float(raw) for name, raw in body.params.items()},
        design=active_design,
        provenance={
            "annotation_policy": "creation-event semantic binding; live pre-assembly snapshot groups geometry only",
            "preset": "assets/design_params/t-shirt.yaml",
            "preset_sha256": _sha256(design_path),
            "external_checkout": "ignored local dependency",
            "body_id": body_id,
            "design_id": design_id,
            "final_serialized_json_used_for_labels": False,
            "completed_panel_topology_used_for_canonical_or_edge_semantic_labels": False,
            "unbound_non_target_vertices_preserved_from_live_snapshot": True,
        },
        panels=panels,
        operations=operations,
        reference_lines=_reference_lines(panels_live, body.params),
        darts=darts,
        drafting_formula_targets=drafting_formula_targets,
        drafting_seam_relations=drafting_seam_relations,
        metadata={
            "generator_panel_count": 8,
            "runtime_event_count": len(runtime.events),
            "runtime_add_dart_count": runtime_add_dart_count,
            "canonical_BP_status": "NOT_DEFINED_BY_RECIPE",
            "notches_status": "NOT_CREATED_BY_GARMENTCODE_RECIPE",
            "grainline_status": "NOT_CREATED_BY_GARMENTCODE_RECIPE",
            "seam_allowance_status": "NOT_CREATED_BY_GARMENTCODE_RECIPE",
            "reference_line_domain": "anthropometric_adapter",
            "dart_applicability": "NOT_APPLICABLE",
            "creation_semantic_contract": creation_contract,
            "drafting_formula_target_policy": (
                "path-local targets derived only from creation-event-labeled live edges; "
                "no role inference from final shape"
            ),
        },
    )
    record.validate()
    return record


def load_body_yaml(path: Path) -> dict[str, float]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["body"]
    return {name: float(raw) for name, raw in value.items()}


def smoke_trace(garmentcode_root: Path | None = None) -> TShirtTraceRecord:
    root = (garmentcode_root or default_garmentcode_root()).resolve()
    return generate_garmentcode_tshirt_trace(
        sample_id="smoke_basic_tshirt",
        split="test",
        body_values=load_body_yaml(root / "assets/bodies/mean_all.yaml"),
        design_values={},
        garmentcode_root=root,
        body_id="mean_all",
        design_id="official_tshirt_preset",
    )


__all__ = [
    "BODY_INPUT_FIELDS",
    "default_garmentcode_root",
    "load_basic_tshirt_design",
    "apply_tshirt_design_values",
    "generate_garmentcode_tshirt_trace",
    "load_body_yaml",
    "smoke_trace",
]
