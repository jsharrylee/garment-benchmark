"""Convert official FreeSewing Teagan named output to the T-shirt trace schema.

Unlike the GarmentCode adapter, this module does **not** claim a creation-time
operation DAG: the installed package exposes the completed named points,
paths, snippets, and production annotations.  Those remain a distinct
``freesewing_named_output`` domain and are used as a strict source/recipe
holdout after the GarmentCode model is frozen.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .drafting_formula_targets import (
    build_drafting_formula_targets,
    build_sleeve_armhole_relation,
    freesewing_formula_context,
)

from .tshirt_schema import (
    ConstructionOperation,
    CurveGeometry,
    DartTrace,
    Grainline,
    NamedPath,
    Notch,
    SeamAllowance,
    TShirtTraceRecord,
    TracedEdge,
    TracedPanel,
    TracedPoint,
    TracedReferenceLine,
)


def _cm(point: Mapping[str, Any]) -> tuple[float, float]:
    return float(point["x_mm"]) / 10.0, float(point["y_mm"]) / 10.0


def _coordinate_key(point: Mapping[str, Any]) -> tuple[float, float]:
    x, y = _cm(point)
    return round(x, 7), round(y, 7)


def _edge_role(part: str, edge_index: int) -> str:
    if part in {"teagan.front", "teagan.back"}:
        return (
            "hemline",
            "side_seam",
            "armhole",
            "armhole",
            "shoulder",
            "neckline",
            "center_front" if part.endswith("front") else "center_back",
        )[edge_index]
    if part == "teagan.sleeve":
        return (
            "sleeve_hem",
            "sleeve_underarm",
            "sleeve_head",
            "sleeve_head",
            "sleeve_head",
            "sleeve_head",
            "sleeve_head",
            "sleeve_underarm",
        )[edge_index]
    raise ValueError(f"unsupported visible Teagan part: {part}")


def _edge_geometry(operation: Mapping[str, Any]) -> CurveGeometry:
    start = _cm(operation["from"])
    end = _cm(operation["to"])
    if operation["type"] == "line":
        return CurveGeometry(kind="line", start_cm=start, end_cm=end)
    if operation["type"] == "curve":
        controls = tuple(_cm(operation[name]) for name in ("cp1", "cp2") if operation.get(name))
        if len(controls) == 1:
            return CurveGeometry(kind="quadratic_bezier", start_cm=start, end_cm=end, control_points_cm=controls)
        return CurveGeometry(kind="cubic_bezier", start_cm=start, end_cm=end, control_points_cm=controls)
    raise ValueError(f"operation does not create a boundary edge: {operation['type']}")


def _canonical_coordinates(raw: Mapping[str, Any]) -> dict[tuple[str, tuple[float, float]], tuple[str, str, str]]:
    output: dict[tuple[str, tuple[float, float]], tuple[str, str, str]] = {}
    landmarks = raw["canonical_semantics"]["landmarks"]
    for canonical in ("FNP", "BNP", "SNP", "SP"):
        for item in landmarks.get(canonical, []):
            output[(item["part"], _coordinate_key(item["coordinate"]))] = (
                canonical,
                item["source_point_name"],
                item["evidence"],
            )
    return output


def _visible_part(raw: Mapping[str, Any], name: str, canonical: Mapping[Any, Any]) -> TracedPanel:
    seam = raw["parts"][name]["paths"]["seam"]
    edge_operations = [item for item in seam["operations"] if item["type"] in {"line", "curve"}]
    expected = 7 if name != "teagan.sleeve" else 8
    if len(edge_operations) != expected:
        raise ValueError(f"unexpected {name} seam topology: {len(edge_operations)} edges")
    points_by_key: dict[tuple[float, float], Mapping[str, Any]] = {}
    for operation in edge_operations:
        for endpoint in (operation["from"], operation["to"]):
            points_by_key.setdefault(_coordinate_key(endpoint), endpoint)
    ordered_keys = list(points_by_key)
    point_ids = {key: f"{name}.point_{index:02d}" for index, key in enumerate(ordered_keys)}
    operation_id = f"freesewing.named_path.{name}.seam"
    points: list[TracedPoint] = []
    for key in ordered_keys:
        source = points_by_key[key]
        match = canonical.get((name, key))
        canonical_name = match[0] if match else None
        source_name = match[1] if match else source.get("source_name")
        refs = source.get("point_refs", [])
        if not source_name and refs:
            source_name = refs[0]
        points.append(
            TracedPoint(
                id=point_ids[key],
                panel_id=name,
                xy_cm=key,
                formula="author-named Teagan point/path endpoint",
                canonical_name=canonical_name,
                source_name=source_name or f"seam_endpoint_{len(points):02d}",
                measurement_inputs={},
                operation_id=operation_id,
                domain="freesewing_named_output",
                evidence=match[2] if match else "observed_source_path",
                provenance={"point_refs": refs, "source_package": "@freesewing/teagan@4.10.1"},
            )
        )
    edges: list[TracedEdge] = []
    for index, operation in enumerate(edge_operations):
        start_key = _coordinate_key(operation["from"])
        end_key = _coordinate_key(operation["to"])
        edges.append(
            TracedEdge(
                id=f"{name}.edge_{index:02d}",
                panel_id=name,
                start_point_id=point_ids[start_key],
                end_point_id=point_ids[end_key],
                semantic_role=_edge_role(name, index),
                geometry=_edge_geometry(operation),
                source_name=f"seam.operation_{operation['index']}",
                formula=f"FreeSewing named path operation {operation['type']}",
                dependencies=(point_ids[start_key], point_ids[end_key]),
                operation_id=operation_id,
                domain="freesewing_named_output",
                evidence="observed_source_path",
                provenance={"source_path": "seam", "source_operation_index": operation["index"]},
            )
        )
    role = name.removeprefix("teagan.")
    return TracedPanel(
        id=name,
        semantic_role=role,
        points=tuple(points),
        edges=tuple(edges),
        source_name=name,
        operation_id=operation_id,
        metadata={
            "full_pattern_piece": True,
            "coordinate_unit": "cm converted from FreeSewing mm",
            "source_coordinate_orientation": "native FreeSewing",
        },
    )


def _reference_lines(raw: Mapping[str, Any], panels: Mapping[str, TracedPanel]) -> tuple[TracedReferenceLine, ...]:
    output: list[TracedReferenceLine] = []
    for canonical, group in raw["canonical_semantics"]["horizontal_levels"].items():
        for item in group["instances"]:
            part = item["part"]
            panel = panels[part]
            xs = [point.xy_cm[0] for point in panel.points]
            y = _cm(item["coordinate"])[1]
            output.append(
                TracedReferenceLine(
                    id=f"{part}.{canonical}",
                    panel_id=part,
                    canonical_name=canonical,
                    source_name=item["source_point_name"],
                    geometry=CurveGeometry(kind="line", start_cm=(min(xs), y), end_cm=(max(xs), y)),
                    formula=f"FreeSewing named construction level from {group['source_measurement']}",
                    measurement_inputs={
                        group["source_measurement"]: float(
                            raw["input"]["resolved_measurements_mm"][group["source_measurement"]]
                        )
                        / 10.0
                    },
                    operation_id=f"freesewing.named_path.{part}.seam",
                    auxiliary=True,
                    domain="freesewing_named_output",
                    evidence=item["evidence"],
                    provenance={
                        "status": group["status"],
                        "meaning": group["meaning"],
                        "BL_is_proxy": canonical == "BL",
                    },
                    training_eligible=canonical != "BL",
                    confidence=0.7 if canonical == "BL" else 1.0,
                )
            )
    return tuple(output)


def _bezier_point(geometry: CurveGeometry, t: float) -> tuple[float, float]:
    points = (geometry.start_cm, *geometry.control_points_cm, geometry.end_cm)
    if len(points) == 2:
        return ((1 - t) * points[0][0] + t * points[1][0], (1 - t) * points[0][1] + t * points[1][1])
    work = [tuple(point) for point in points]
    while len(work) > 1:
        work = [
            ((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]) for a, b in zip(work, work[1:])
        ]
    return work[0]


def _nearest_edge(panel: TracedPanel, xy: tuple[float, float]) -> str:
    choices = []
    for edge in panel.edges:
        for index in range(41):
            point = _bezier_point(edge.geometry, index / 40.0)
            choices.append((math.dist(xy, point), edge.id))
    return min(choices)[1]


def _notches(raw: Mapping[str, Any], panels: Mapping[str, TracedPanel]) -> tuple[Notch, ...]:
    output = []
    for index, item in enumerate(raw["production_semantics"]["notches"]["items"]):
        panel = panels[item["part"]]
        xy = _cm(item["anchor"])
        output.append(
            Notch(
                id=f"{item['part']}.notch_{index:02d}",
                panel_id=item["part"],
                edge_id=_nearest_edge(panel, xy),
                semantic_role="armhole_matching_notch",
                xy_cm=xy,
                source_name=item["snippet_name"],
                domain="freesewing_named_output",
                evidence="observed_source_snippet",
                provenance={"notch_type": item["notch_type"], "anchor_refs": item["anchor"].get("point_refs", [])},
            )
        )
    return tuple(output)


def _line_from_annotation(item: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    operations = item["path"]["operations"]
    lines = [operation for operation in operations if operation["type"] == "line"]
    if not lines:
        raise ValueError("annotation path has no line")
    # Combined cut-on-fold marks are a three-segment bracket; the longest
    # segment is the actual lengthwise direction.
    selected = max(lines, key=lambda value: math.dist(_cm(value["from"]), _cm(value["to"])))
    return _cm(selected["from"]), _cm(selected["to"])


def _grainlines(raw: Mapping[str, Any], panels: Mapping[str, TracedPanel]) -> tuple[Grainline, ...]:
    output: list[Grainline] = []
    for kind in ("grainlines", "cut_on_fold"):
        for index, item in enumerate(raw["production_semantics"][kind]["items"]):
            start, end = _line_from_annotation(item)
            output.append(
                Grainline(
                    id=f"{item['part']}.{kind}_{index:02d}",
                    panel_id=item["part"],
                    start_cm=start,
                    end_cm=end,
                    semantic_role="grainline" if kind == "grainlines" else "cut_on_fold_and_grainline",
                    source_name=item["path_name"],
                    domain="freesewing_named_output",
                    evidence="observed_source_annotation_path",
                    provenance={"annotation_kind": kind},
                )
            )
    return tuple(output)


def _seam_allowances(raw: Mapping[str, Any], panels: Mapping[str, TracedPanel]) -> tuple[SeamAllowance, ...]:
    requested = float(raw["production_semantics"]["seam_allowance"]["requested_mm"]) / 10.0
    output = []
    for panel in panels.values():
        widths = {}
        for edge in panel.edges:
            if edge.semantic_role in {"hemline", "sleeve_hem"}:
                widths[edge.id] = 3.0 * requested
            elif edge.semantic_role in {"center_front", "center_back"}:
                widths[edge.id] = 0.0
            else:
                widths[edge.id] = requested
        output.append(
            SeamAllowance(
                id=f"{panel.id}.seam_allowance",
                panel_id=panel.id,
                edge_ids=tuple(edge.id for edge in panel.edges),
                width_by_edge_cm=widths,
                source_name="sa",
                domain="freesewing_named_output",
                evidence="observed_source_sa_path_and_policy",
                provenance={"requested_width_cm": requested},
            )
        )
    return tuple(output)


def teagan_json_to_trace(raw: Mapping[str, Any], *, sample_id: str, split: str = "unseen_source") -> TShirtTraceRecord:
    """Convert one deterministic extractor document to a validated record."""

    canonical = _canonical_coordinates(raw)
    panel_values = tuple(_visible_part(raw, name, canonical) for name in ("teagan.front", "teagan.back", "teagan.sleeve"))
    panels = {panel.id: panel for panel in panel_values}
    formula_parameters, formula_references = freesewing_formula_context(raw)
    drafting_formula_targets = build_drafting_formula_targets(
        panel_values,
        source_kind="freesewing_named_output",
        formula_parameters=formula_parameters,
        formula_references=formula_references,
    )
    drafting_seam_relations = build_sleeve_armhole_relation(
        drafting_formula_targets, source_kind="freesewing_named_output"
    )
    operations = tuple(
        ConstructionOperation(
            id=f"freesewing.named_path.{panel.id}.seam",
            order=index,
            operation="extract_author_named_seam_path",
            inputs=(f"{panel.id}.points", f"{panel.id}.paths.seam"),
            outputs=tuple(edge.id for edge in panel.edges),
            parameters={"creation_time_trace_available": False},
            source_reference="@freesewing/teagan@4.10.1/src",
            domain="freesewing_named_output",
            evidence="observed_source_path",
            training_eligible=False,
        )
        for index, panel in enumerate(panel_values)
    )
    named_paths = tuple(
        NamedPath(
            id=f"{panel.id}.named_seam",
            panel_id=panel.id,
            source_name="seam",
            semantic_role="stitch_line_boundary",
            edge_ids=tuple(edge.id for edge in panel.edges),
            closed=True,
            domain="freesewing_named_output",
            evidence="observed_source_path",
        )
        for panel in panel_values
    )
    darts = tuple(
        DartTrace(
            id=f"{panel.id}.dart_not_applicable",
            panel_id=panel.id,
            kind="none",
            applicable=False,
            applicability_reason="FreeSewing Teagan basic T-shirt is dartless",
            domain="freesewing_named_output",
            evidence="absent_by_design",
            provenance={"source_status": raw["canonical_semantics"]["darts"]["status"]},
        )
        for panel in panel_values
        if panel.semantic_role in {"front", "back"}
    )
    record = TShirtTraceRecord(
        sample_id=sample_id,
        split=split,
        source={
            "name": "FreeSewing Teagan",
            "package": raw["source"]["design_package"],
            "version": raw["source"]["design_version"],
            "repository": raw["source"]["repository"],
            "license": raw["source"]["source_code_license_spdx"],
            "license_scope": "source package; generated-output rights not legally determined here",
        },
        body={name: float(value) / 10.0 for name, value in raw["input"]["resolved_measurements_mm"].items()},
        design={name: value for name, value in raw["input"]["resolved_options"].items() if isinstance(value, (int, float, bool))},
        provenance={
            "extractor_schema": raw["schema_version"],
            "model": raw["input"]["model"],
            "annotation_policy": "author-named completed output; no creation-time DAG claim",
            "garmentcode_training_source": False,
        },
        panels=panel_values,
        operations=operations,
        reference_lines=_reference_lines(raw, panels),
        darts=darts,
        named_paths=named_paths,
        notches=_notches(raw, panels),
        grainlines=_grainlines(raw, panels),
        seam_allowances=_seam_allowances(raw, panels),
        drafting_formula_targets=drafting_formula_targets,
        drafting_seam_relations=drafting_seam_relations,
        metadata={
            "canonical_BP_status": "NOT_DEFINED_BY_RECIPE",
            "BL_status": "APPROXIMATE_PROXY_EXCLUDED_FROM_TRAINING",
            "creation_time_operation_DAG": "NOT_AVAILABLE_FROM_PACKAGE_OUTPUT",
            "cross_source_zero_shot": True,
            "dart_applicability": "NOT_APPLICABLE",
            "drafting_formula_target_policy": (
                "author-named completed paths plus explicitly cited public source formulas; "
                "not creation-time operation evidence"
            ),
        },
    )
    record.validate()
    return record


def read_teagan_extractor_json(path: Path, *, sample_id: str | None = None, split: str = "unseen_source") -> TShirtTraceRecord:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return teagan_json_to_trace(raw, sample_id=sample_id or Path(path).stem, split=split)


__all__ = ["teagan_json_to_trace", "read_teagan_extractor_json"]
