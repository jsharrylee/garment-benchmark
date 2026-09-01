from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import (
    ConstructionStep,
    Dart,
    DraftingSemanticRecord,
    EdgeAnnotation,
    Landmark,
    PanelAnnotation,
    ReferenceLine,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel_role(name: str) -> str:
    value = name.lower()
    if "ftorso" in value or "front_torso" in value:
        return "front_bodice"
    if "btorso" in value or "back_torso" in value:
        return "back_bodice"
    if "sleeve" in value:
        return "sleeve"
    if "collar" in value or "hood" in value:
        return "collar"
    if "cuff" in value:
        return "cuff"
    if value.startswith("wb_") or "waistband" in value:
        return "waistband"
    if "skirt" in value:
        return "front_skirt" if "front" in value or value.endswith("_f") else "back_skirt"
    if "pant" in value or "trouser" in value:
        if any(token in value for token in ("_f", "front")):
            return "front_pants"
        return "back_pants"
    return "other"


def _load_yaml_group(path: Path, key: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    group = value.get(key, value)
    if not isinstance(group, dict):
        raise ValueError(f"{path} does not contain a mapping at {key}")
    return group


def _relative_point(start: tuple[float, float], end: tuple[float, float], relative: Iterable[float]) -> tuple[float, float]:
    a, b = tuple(float(value) for value in relative)
    dx, dy = end[0] - start[0], end[1] - start[1]
    return start[0] + a * dx - b * dy, start[1] + a * dy + b * dx


def _curve_points(vertices: tuple[tuple[float, float], ...], edge: dict[str, Any], samples: int = 48) -> tuple[tuple[float, float], ...]:
    start = vertices[int(edge["endpoints"][0])]
    end = vertices[int(edge["endpoints"][1])]
    curvature = edge.get("curvature")
    if not curvature:
        return (start, end)
    if isinstance(curvature, list):
        kind, params = "quadratic", curvature
    else:
        kind, params = str(curvature.get("type", "line")), curvature.get("params", [])
    if kind == "circle" and len(params) >= 3:
        radius = abs(float(params[0]))
        chord = math.dist(start, end)
        if radius <= 1e-9 or chord > 2.0 * radius + 1e-6:
            return (start, end)
        # Arc length is all that is needed downstream. Intermediate points on
        # the chord preserve endpoints and encode the exact length separately.
        return (start, end)
    if kind not in {"quadratic", "cubic"}:
        return (start, end)
    controls = [_relative_point(start, end, item) for item in params]
    values = []
    for index in range(samples):
        t = index / max(samples - 1, 1)
        if kind == "quadratic" and controls:
            c = controls[0]
            values.append(
                (
                    (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * c[0] + t**2 * end[0],
                    (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * c[1] + t**2 * end[1],
                )
            )
        elif kind == "cubic" and len(controls) >= 2:
            c1, c2 = controls[:2]
            values.append(
                (
                    (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t**2 * c2[0] + t**3 * end[0],
                    (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t**2 * c2[1] + t**3 * end[1],
                )
            )
        else:
            return (start, end)
    return tuple(values)


def _edge_length(vertices: tuple[tuple[float, float], ...], edge: dict[str, Any]) -> float:
    start = vertices[int(edge["endpoints"][0])]
    end = vertices[int(edge["endpoints"][1])]
    curvature = edge.get("curvature")
    if isinstance(curvature, dict) and curvature.get("type") == "circle":
        params = curvature.get("params", [])
        if len(params) >= 2:
            radius = abs(float(params[0]))
            chord = min(math.dist(start, end), 2.0 * radius)
            if radius > 1e-9:
                angle = 2.0 * math.asin(max(0.0, min(1.0, chord / (2.0 * radius))))
                if bool(params[1]):
                    angle = 2.0 * math.pi - angle
                return radius * angle
    points = _curve_points(vertices, edge)
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _flatten_values(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict) and "v" in child:
            output[name] = child["v"]
        elif isinstance(child, dict):
            output.update(_flatten_values(child, name))
    return output


def _paths_between(count: int, first: int, second: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    forward = []
    current = (first + 1) % count
    while current != second:
        forward.append(current)
        current = (current + 1) % count
    backward = []
    current = (first - 1) % count
    while current != second:
        backward.append(current)
        current = (current - 1) % count
    return tuple(forward), tuple(backward)


def _shared_vertex(edges: list[dict[str, Any]], first_roles: set[int], second_roles: set[int]) -> int | None:
    candidates = []
    for first in first_roles:
        for second in second_roles:
            shared = set(int(value) for value in edges[first]["endpoints"]) & set(int(value) for value in edges[second]["endpoints"])
            candidates.extend(shared)
    return min(candidates) if candidates else None


def _point_segment_distance(point: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
    dx, dy = second[0] - first[0], second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, first)
    t = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator
    projection = first[0] + max(0.0, min(1.0, t)) * dx, first[1] + max(0.0, min(1.0, t)) * dy
    return math.dist(point, projection)


def _construction_steps(upper_type: str | None) -> tuple[ConstructionStep, ...]:
    source = "external/GarmentCode/assets/garment_programs/bodice.py"
    if upper_type == "FittedShirt":
        operations = (
            ("derive_body_block_values", "bodice", ("bust", "waist", "back_width", "shoulder_w"), ("front_width", "back_width", "waist_suppression")),
            ("construct_front_back_base_loops", "bodice", ("waist_line", "shoulder_incl"), ("base_boundaries",)),
            ("insert_front_side_dart", "front_bodice", ("bust_line", "bust_points"), ("side_dart",)),
            ("adjust_shoulder_balance", "bodice", ("shoulder_w", "shoulder_incl"), ("shoulder_edges",)),
            ("insert_waist_darts", "bodice", ("bust", "waist", "bum_points"), ("waist_darts",)),
            ("cut_neckline_and_armhole", "bodice", ("neck_w", "armscye_depth"), ("neckline", "armhole")),
            ("mirror_and_assemble", "bodice", ("panel_interfaces",), ("full_bodice", "stitches")),
        )
    elif upper_type == "Shirt":
        operations = (
            ("derive_loose_torso_values", "bodice", ("bust", "waist_line"), ("torso_width", "torso_length")),
            ("construct_front_back_base_loops", "bodice", ("torso_width", "torso_length"), ("base_boundaries",)),
            ("cut_neckline_and_armhole", "bodice", ("neck_w", "armscye_depth"), ("neckline", "armhole")),
            ("mirror_and_assemble", "bodice", ("panel_interfaces",), ("full_bodice", "stitches")),
        )
    else:
        return ()
    return tuple(
        ConstructionStep(index, operation, panel_role, tuple(inputs), tuple(outputs), source_reference=source)
        for index, (operation, panel_role, inputs, outputs) in enumerate(operations)
    )


def annotate_garmentcode_sample(
    specification_path: Path,
    body_measurements_path: Path,
    design_params_path: Path,
    *,
    split: str = "unknown",
    source_license: str = "CC BY 4.0",
    synthesize_production_marks: bool = False,
) -> DraftingSemanticRecord:
    """Create an evidence-graded drafting record from one GarmentCode sample.

    Existing edge labels and stitch pairs are treated as observed evidence.
    Named drafting points and guide lines are derived only where GarmentCode's
    topology or fitted-bodice formulas make the correspondence unambiguous.
    """

    specification_path = Path(specification_path)
    body_measurements_path = Path(body_measurements_path)
    design_params_path = Path(design_params_path)
    raw = json.loads(specification_path.read_text(encoding="utf-8"))
    pattern = raw["pattern"]
    body = _load_yaml_group(body_measurements_path, "body")
    design = _load_yaml_group(design_params_path, "design")
    sample_id = specification_path.stem.removesuffix("_specification")

    source_panels: dict[str, dict[str, Any]] = pattern["panels"]
    roles = {name: _panel_role(name) for name in source_panels}
    edge_roles: dict[tuple[str, int], str] = {}
    evidence: dict[tuple[str, int], tuple[str, float]] = {}
    stitch_map: dict[tuple[str, int], list[tuple[str, int]]] = {}
    self_pairs: list[tuple[str, int, int]] = []

    for panel_name, panel in source_panels.items():
        for edge_index, edge in enumerate(panel["edges"]):
            label = str(edge.get("label", "")).lower()
            role = "other"
            panel_role = roles[panel_name]
            if "collar" in label or "neck" in label:
                role = "neckline"
            elif "armhole" in label or "armscye" in label:
                role = "sleeve_head" if panel_role == "sleeve" else "armhole"
            elif "lower" in label or "hem" in label:
                role = "sleeve_hem" if panel_role == "sleeve" else "hemline"
            edge_roles[(panel_name, edge_index)] = role
            evidence[(panel_name, edge_index)] = ("observed_source" if role != "other" else "derived_topology", 1.0 if role != "other" else 0.75)

    for pair in pattern.get("stitches", []):
        if len(pair) < 2:
            continue
        first, second = pair[:2]
        a = str(first["panel"]), int(first["edge"])
        b = str(second["panel"]), int(second["edge"])
        stitch_map.setdefault(a, []).append(b)
        stitch_map.setdefault(b, []).append(a)
        if a[0] == b[0]:
            self_pairs.append((a[0], a[1], b[1]))
            for key in (a, b):
                edge_roles[key] = "dart_leg"
                evidence[key] = ("observed_source", 1.0)

    # The short boundary path between neckline and armhole is the shoulder.
    for panel_name, panel in source_panels.items():
        if roles[panel_name] not in {"front_bodice", "back_bodice"}:
            continue
        count = len(panel["edges"])
        neck = {index for index in range(count) if edge_roles[(panel_name, index)] == "neckline"}
        armhole = {index for index in range(count) if edge_roles[(panel_name, index)] == "armhole"}
        candidates: list[tuple[int, tuple[int, ...]]] = []
        for first in neck:
            for second in armhole:
                for path in _paths_between(count, first, second):
                    if path and all(edge_roles[(panel_name, index)] == "other" for index in path):
                        candidates.append((len(path), path))
        if candidates:
            for edge_index in min(candidates, key=lambda item: item[0])[1]:
                edge_roles[(panel_name, edge_index)] = "shoulder"
                evidence[(panel_name, edge_index)] = ("derived_topology", 0.99)

    # Stitch topology distinguishes centers, side seams and waist attachments.
    for key, others in stitch_map.items():
        panel_name, edge_index = key
        own_role = roles.get(panel_name, "other")
        if edge_roles[key] in {
            "neckline",
            "armhole",
            "shoulder",
            "dart_leg",
            "sleeve_head",
            "sleeve_hem",
        }:
            continue
        for other_panel, _ in others:
            other_role = roles.get(other_panel, "other")
            new_role = None
            if own_role == other_role == "front_bodice":
                new_role = "center_front"
            elif own_role == other_role == "back_bodice":
                new_role = "center_back"
            elif {own_role, other_role} == {"front_bodice", "back_bodice"}:
                new_role = "side_seam"
            elif own_role in {"front_bodice", "back_bodice"} and other_role in {"waistband", "front_skirt", "back_skirt", "front_pants", "back_pants"}:
                new_role = "waistline"
            elif own_role in {"front_skirt", "back_skirt", "front_pants", "back_pants"} and other_role == "waistband":
                new_role = "waistline"
            elif own_role == "sleeve" and other_role in {"front_bodice", "back_bodice"}:
                new_role = "sleeve_head"
            elif own_role in {"front_bodice", "back_bodice"} and other_role == "sleeve":
                new_role = "armhole"
            elif own_role == "sleeve" and other_role == "cuff":
                new_role = "sleeve_hem"
            elif own_role == "cuff" and other_role == "sleeve":
                new_role = "cuff_attachment"
            elif own_role == other_role == "sleeve":
                new_role = "sleeve_underarm"
            elif own_role in {"front_pants", "back_pants"} and own_role == other_role:
                new_role = "crotch_curve"
            elif own_role in {"front_skirt", "back_skirt"} and other_role in {"front_skirt", "back_skirt"}:
                new_role = "side_seam"
            elif own_role == "collar" and other_role in {"front_bodice", "back_bodice"}:
                new_role = "collar_attachment"
            if new_role:
                edge_roles[key] = new_role
                evidence[key] = ("derived_topology", 0.98)
                break

    # A trousers front/back pair normally has two long connections.  The
    # longer one is the outside leg seam and the shorter one the inseam.  This
    # is a topology/length reconstruction, not an author-provided label, so it
    # remains evidence-graded and must stay masked in expert-only evaluations.
    for panel_name, panel in source_panels.items():
        own_role = roles.get(panel_name, "other")
        if own_role not in {"front_pants", "back_pants"}:
            continue
        opposite = "back_pants" if own_role == "front_pants" else "front_pants"
        candidates: list[tuple[float, int]] = []
        vertices = tuple(tuple(float(value) for value in point) for point in panel["vertices"])
        for edge_index, edge in enumerate(panel["edges"]):
            key = panel_name, edge_index
            if edge_roles[key] != "other":
                continue
            if any(roles.get(other_panel) == opposite for other_panel, _ in stitch_map.get(key, [])):
                candidates.append((_edge_length(vertices, edge), edge_index))
        if candidates:
            candidates.sort(reverse=True)
            for rank, (_, edge_index) in enumerate(candidates):
                key = panel_name, edge_index
                edge_roles[key] = "outseam" if rank == 0 else "inseam"
                evidence[key] = ("derived_topology", 0.82 if rank == 0 else 0.78)

    # On isolated tops, waist edges can be unstitched. Recover the bottom band.
    for panel_name, panel in source_panels.items():
        role = roles[panel_name]
        if role not in {"front_bodice", "back_bodice", "front_skirt", "back_skirt", "front_pants", "back_pants"}:
            continue
        vertices = tuple(tuple(float(value) for value in point) for point in panel["vertices"])
        ys = [point[1] for point in vertices]
        span = max(max(ys) - min(ys), 1e-6)
        threshold = min(ys) + 0.08 * span
        target = "waistline" if role in {"front_bodice", "back_bodice"} else "hemline"
        for edge_index, edge in enumerate(panel["edges"]):
            key = panel_name, edge_index
            if edge_roles[key] != "other":
                continue
            a, b = (vertices[int(index)] for index in edge["endpoints"])
            if (a[1] + b[1]) / 2.0 <= threshold:
                edge_roles[key] = target
                evidence[key] = ("derived_topology", 0.8)

    darts: list[Dart] = []
    for panel_name, first_index, second_index in self_pairs:
        panel = source_panels[panel_name]
        vertices = tuple(tuple(float(value) for value in point) for point in panel["vertices"])
        first_endpoints = tuple(int(value) for value in panel["edges"][first_index]["endpoints"])
        second_endpoints = tuple(int(value) for value in panel["edges"][second_index]["endpoints"])
        shared = set(first_endpoints) & set(second_endpoints)
        if shared:
            apex_index = next(iter(shared))
            first_mouth = next(value for value in first_endpoints if value != apex_index)
            second_mouth = next(value for value in second_endpoints if value != apex_index)
        else:
            pairs = sorted((math.dist(vertices[a], vertices[b]), a, b) for a in first_endpoints for b in second_endpoints)
            _, first_near, second_near = pairs[0]
            apex = ((vertices[first_near][0] + vertices[second_near][0]) / 2.0, (vertices[first_near][1] + vertices[second_near][1]) / 2.0)
            first_mouth = next(value for value in first_endpoints if value != first_near)
            second_mouth = next(value for value in second_endpoints if value != second_near)
            apex_index = None
        if apex_index is not None:
            apex = vertices[apex_index]
        base = vertices[first_mouth], vertices[second_mouth]
        dx, dy = abs(base[1][0] - base[0][0]), abs(base[1][1] - base[0][1])
        kind = "waist_dart" if dx >= dy else "side_dart"
        darts.append(
            Dart(
                panel_id=panel_name,
                kind=kind,
                leg_edge_ids=(f"{panel_name}.edge_{first_index}", f"{panel_name}.edge_{second_index}"),
                apex_cm=apex,
                base_cm=base,
                intake_cm=math.dist(*base),
                depth_cm=_point_segment_distance(apex, *base),
            )
        )

    panel_annotations: list[PanelAnnotation] = []
    measurement_edges: dict[str, Any] = {"shoulders": {}, "neckline_arc_cm": {}}
    bust_level = float(body.get("waist_line", 0.0)) - float(body.get("_bust_line", body.get("bust_line", 0.0)))
    for panel_name, panel in source_panels.items():
        vertices = tuple(tuple(float(value) for value in point) for point in panel["vertices"])
        raw_edges = list(panel["edges"])
        annotations = []
        for edge_index, edge in enumerate(raw_edges):
            start_index, end_index = (int(value) for value in edge["endpoints"])
            edge_evidence, confidence = evidence[(panel_name, edge_index)]
            curvature = edge.get("curvature")
            if isinstance(curvature, dict):
                curvature_type = str(curvature.get("type", "line"))
            elif isinstance(curvature, list):
                curvature_type = "quadratic"
            else:
                curvature_type = "line"
            annotations.append(
                EdgeAnnotation(
                    id=f"{panel_name}.edge_{edge_index}",
                    index=edge_index,
                    endpoints=(start_index, end_index),
                    start_cm=vertices[start_index],
                    end_cm=vertices[end_index],
                    curvature_type=curvature_type,
                    role=edge_roles[(panel_name, edge_index)],
                    stitched=(panel_name, edge_index) in stitch_map,
                    self_stitched=any(other_panel == panel_name for other_panel, _ in stitch_map.get((panel_name, edge_index), [])),
                    length_cm=_edge_length(vertices, edge),
                    evidence=edge_evidence,
                    confidence=confidence,
                )
            )

        role_sets = {
            role: {index for index in range(len(raw_edges)) if edge_roles[(panel_name, index)] == role}
            for role in {"neckline", "shoulder", "armhole", "center_front", "center_back"}
        }
        landmarks: list[Landmark] = []
        panel_role = roles[panel_name]
        center_role = "center_front" if panel_role == "front_bodice" else "center_back"
        if panel_role in {"front_bodice", "back_bodice"}:
            center_neck = _shared_vertex(raw_edges, role_sets["neckline"], role_sets[center_role])
            side_neck = _shared_vertex(raw_edges, role_sets["neckline"], role_sets["shoulder"])
            shoulder_point = _shared_vertex(raw_edges, role_sets["shoulder"], role_sets["armhole"])
            if center_neck is not None:
                name = "FNP" if panel_role == "front_bodice" else "BNP"
                landmarks.append(Landmark(name, panel_name, vertices[center_neck], "derived_topology", 0.99, center_neck))
            if side_neck is not None:
                landmarks.append(Landmark("SNP", panel_name, vertices[side_neck], "derived_topology", 0.99, side_neck))
            if shoulder_point is not None:
                landmarks.append(Landmark("SP", panel_name, vertices[shoulder_point], "derived_topology", 0.99, shoulder_point))
            if panel_role == "front_bodice" and "bust_points" in body:
                xs = [point[0] for point in vertices]
                sign = 1.0 if max(xs) >= abs(min(xs)) else -1.0
                landmarks.append(
                    Landmark(
                        "BP",
                        panel_name,
                        (sign * float(body["bust_points"]) / 2.0, bust_level),
                        "derived_generator_formula",
                        0.95,
                        training_eligible=False,
                    )
                )

        lines: list[ReferenceLine] = []
        if panel_role in {"front_bodice", "back_bodice"}:
            xs = [point[0] for point in vertices]
            for name, level, intersects, confidence in (
                ("WL", 0.0, True, 1.0),
                ("BL", bust_level, True, 0.98),
                ("HL", -float(body.get("hips_line", 0.0)), False, 0.95),
            ):
                lines.append(
                    ReferenceLine(
                        name,
                        panel_name,
                        ((min(xs), level), (max(xs), level)),
                        "derived_generator_formula",
                        confidence,
                        intersects_panel=intersects,
                        training_eligible=intersects,
                    )
                )

        shoulder_edges = [edge for edge in annotations if edge.role == "shoulder"]
        if shoulder_edges:
            length = sum(edge.length_cm for edge in shoulder_edges)
            first, last = shoulder_edges[0].start_cm, shoulder_edges[-1].end_cm
            angle = math.degrees(math.atan2(last[1] - first[1], last[0] - first[0]))
            measurement_edges["shoulders"][panel_name] = {"length_cm": length, "angle_deg": angle}
        neckline_edges = [edge for edge in annotations if edge.role == "neckline"]
        if neckline_edges:
            measurement_edges["neckline_arc_cm"][panel_name] = sum(edge.length_cm for edge in neckline_edges)
        panel_annotations.append(PanelAnnotation(panel_name, panel_role, vertices, tuple(annotations), tuple(landmarks), tuple(lines)))

    upper_type = design.get("meta", {}).get("upper", {}).get("v")
    measurement_edges.update(
        {
            "garmentcode_back_waist_length_cm": float(body.get("waist_line", 0.0)),
            "garmentcode_waist_to_hip_cm": float(body.get("hips_line", 0.0)),
            "body_shoulder_slope_deg": float(body.get("_shoulder_incl", body.get("shoulder_incl", 0.0))),
            "body_prediction": {
                "enabled": False,
                "reason": "caller must enable only after a body-variance audit; batch_0/default_body is constant",
            },
        }
    )
    body_fields = (
        "waist_line",
        "hips_line",
        "back_width",
        "shoulder_w",
        "_shoulder_incl",
        "neck_w",
        "armscye_depth",
        "bust_points",
        "bust",
        "waist",
        "hips",
    )
    body_condition = {name: float(body[name]) for name in body_fields if name in body}

    production: dict[str, Any] = {
        "source_notches": {"available": False, "reason": "not encoded by GarmentCodeData v2 specification"},
        "source_grainlines": {"available": False, "reason": "not encoded by GarmentCodeData v2 specification"},
        "source_seam_allowances": {"available": False, "reason": "not encoded by GarmentCodeData v2 specification"},
    }
    if synthesize_production_marks:
        grainlines = []
        for panel in panel_annotations:
            xs = [point[0] for point in panel.vertices_cm]
            ys = [point[1] for point in panel.vertices_cm]
            x = (min(xs) + max(xs)) / 2.0
            grainlines.append(
                {
                    "panel_id": panel.id,
                    "points_cm": [[x, min(ys) + 0.2 * (max(ys) - min(ys))], [x, max(ys) - 0.2 * (max(ys) - min(ys))]],
                    "evidence": "synthetic_unvalidated",
                    "training_eligible": False,
                }
            )
        production["synthetic_grainlines"] = grainlines
        production["synthetic_notches"] = [
            {
                "stitch_index": index,
                "sides": pair[:2],
                "fraction": 0.5,
                "evidence": "synthetic_unvalidated",
                "training_eligible": False,
            }
            for index, pair in enumerate(pattern.get("stitches", []))
            if len(pair) >= 2 and pair[0]["panel"] != pair[1]["panel"]
        ]
        production["synthetic_seam_allowance_cm"] = {
            "value": 1.0,
            "evidence": "synthetic_unvalidated",
            "training_eligible": False,
        }

    record = DraftingSemanticRecord(
        sample_id=sample_id,
        split=split,
        panels=tuple(panel_annotations),
        darts=tuple(darts),
        measurements=measurement_edges,
        construction_steps=_construction_steps(upper_type),
        body_condition_cm=body_condition,
        program={
            "upper_type": upper_type,
            "design_values": _flatten_values(design),
            "active_mask_status": "stored_unfiltered; downstream program model must mask inactive branches",
        },
        provenance={
            "dataset": "GarmentCodeData v2",
            "license": source_license,
            "specification_sha256": _sha256(specification_path),
            "body_measurements_sha256": _sha256(body_measurements_path),
            "design_params_sha256": _sha256(design_params_path),
            "annotation_policy": "observed labels/stitches plus evidence-graded GarmentCode topology and formula derivations",
        },
        production_annotations=production,
    )
    record.validate()
    return record
