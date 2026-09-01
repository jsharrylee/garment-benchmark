"""Deterministic vector compiler for verified Pattern DSL programs.

This module is deliberately separate from the Matplotlib review renderer.  It
emits the analytic Pattern DSL commands as native SVG ``M/L/Q/C/A/Z`` path
operations; no raster image, contour tracing, or polyline approximation is
part of the exported boundary.  Panels are placed in deterministic grid cells
using only translation and a y-axis reflection, so their centimetre scale and
curve parameters remain intact.

The SVG can be used in two modes:

* clean interchange geometry (the default); or
* the same geometry plus optional semantic/proof overlays for human review.

In both modes a machine-readable ``<metadata>`` block retains the verified
relations, derived landmarks, layout transforms, and the source DSL program.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape, quoteattr

from benchmark.gcdv2_exact.pattern_dsl import (
    CurveCommand,
    LandmarkCommand,
    MoveCommand,
    NextCommand,
    PanelCommand,
    PatternDSLError,
    PatternProgram,
    RoleCommand,
    SewnToCommand,
    SharedEndpointCommand,
    parse_pattern_dsl,
)
from benchmark.pattern_pipeline.schema import PatternDocument


SVG_SCHEMA_VERSION = "gcd-pattern-svg/v1"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
EDGE_COLORS = {
    "L": "#28a9b7",
    "Q": "#ec9b27",
    "C": "#e64b92",
    "A": "#64ad55",
}
PANEL_COLORS = (
    "#c8d7f0",
    "#dfc4e9",
    "#bfe1d0",
    "#f1d2b8",
    "#d8d0ee",
    "#c6e1e8",
    "#ead0d0",
    "#dce0b8",
)


@dataclass(frozen=True)
class SvgExportOptions:
    """Layout and review options for :func:`compile_pattern_svg`.

    All distances are SVG user units representing centimetres.  The compiler
    never scales an individual panel, which makes path measurements directly
    comparable to the materialized ``PatternDocument``.
    """

    gap_cm: float = 3.0
    padding_cm: float = 1.5
    max_columns: int = 4
    decimals: int = 6
    include_overlays: bool = False
    include_metadata: bool = True
    include_semantic_facts: bool = False
    include_provenance: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.gap_cm)) or self.gap_cm < 0.0:
            raise ValueError("gap_cm must be finite and non-negative")
        if not math.isfinite(float(self.padding_cm)) or self.padding_cm < 0.0:
            raise ValueError("padding_cm must be finite and non-negative")
        if self.max_columns <= 0:
            raise ValueError("max_columns must be positive")
        if not 0 <= self.decimals <= 12:
            raise ValueError("decimals must be between 0 and 12")


@dataclass(frozen=True)
class _EdgeGeometry:
    command: CurveCommand
    start: tuple[float, float]
    end: tuple[float, float]
    controls: tuple[tuple[float, float], ...]
    radius: float | None


@dataclass(frozen=True)
class _PanelGeometry:
    panel: PanelCommand
    edges: tuple[_EdgeGeometry, ...]
    bbox: tuple[float, float, float, float]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class _Placement:
    x: float
    y: float
    width: float
    height: float


def _format_number(value: float, decimals: int) -> str:
    number = round(float(value), decimals)
    zero_threshold = 0.5 * 10.0 ** (-decimals) if decimals else 0.5
    if abs(number) < zero_threshold:
        number = 0.0
    encoded = f"{number:.{decimals}f}" if decimals else f"{number:.0f}"
    encoded = encoded.rstrip("0").rstrip(".") if "." in encoded else encoded
    return encoded or "0"


def _command_payload(command: Any) -> dict[str, Any]:
    payload = asdict(command)
    payload["op"] = command.op
    return payload


def _coerce_program(source: PatternProgram | PatternDocument) -> PatternProgram:
    if isinstance(source, PatternProgram):
        return source
    if not isinstance(source, PatternDocument):
        raise TypeError("source must be a PatternProgram or PatternDocument")
    encoded = source.annotations.get("pattern_dsl")
    if not isinstance(encoded, str) or not encoded.strip():
        raise PatternDSLError(
            "PatternDocument lacks annotations.pattern_dsl; sampled polylines "
            "cannot be promoted to analytic SVG commands without provenance"
        )
    program = parse_pattern_dsl(encoded)
    if program.pattern_id != source.pattern_id:
        raise PatternDSLError(
            f"PatternDocument/DSL pattern id mismatch: {source.pattern_id!r} != "
            f"{program.pattern_id!r}"
        )
    return program


def _program_maps(program: PatternProgram):
    panels: dict[str, PanelCommand] = {}
    curves: dict[str, dict[str, CurveCommand]] = {}
    moves: dict[str, MoveCommand] = {}
    next_edges: dict[tuple[str, str], str] = {}
    roles: dict[tuple[str, str], str] = {}
    landmarks: dict[str, list[LandmarkCommand]] = {}
    seams: list[SewnToCommand] = []
    for command in program.commands:
        if isinstance(command, PanelCommand):
            panels[command.panel_id] = command
            curves.setdefault(command.panel_id, {})
        elif isinstance(command, CurveCommand):
            curves.setdefault(command.panel_id, {})[command.edge_id] = command
        elif isinstance(command, MoveCommand):
            moves[command.panel_id] = command
        elif isinstance(command, NextCommand):
            next_edges[(command.panel_id, command.first_edge_id)] = command.second_edge_id
        elif isinstance(command, RoleCommand):
            roles[(command.panel_id, command.edge_id)] = command.role
        elif isinstance(command, LandmarkCommand):
            landmarks.setdefault(command.panel_id, []).append(command)
        elif isinstance(command, SewnToCommand):
            seams.append(command)
    return panels, curves, moves, next_edges, roles, landmarks, seams


def _ordered_curves(
    panel_id: str,
    curves: Mapping[str, CurveCommand],
    move: MoveCommand,
    next_edges: Mapping[tuple[str, str], str],
) -> tuple[CurveCommand, ...]:
    candidates = [value for value in curves.values() if value.start_point_id == move.point_id]
    if len(candidates) != 1:
        raise PatternDSLError(f"{panel_id}: cannot resolve one SVG cycle start")
    output: list[CurveCommand] = []
    current = candidates[0]
    seen: set[str] = set()
    while current.edge_id not in seen:
        output.append(current)
        seen.add(current.edge_id)
        try:
            current = curves[next_edges[(panel_id, current.edge_id)]]
        except KeyError as error:
            raise PatternDSLError(f"{panel_id}: incomplete NEXT chain") from error
    if current.edge_id != output[0].edge_id or len(output) != len(curves):
        raise PatternDSLError(f"{panel_id}: NEXT chain is not one closed cycle")
    return tuple(output)


def _materialized_panels(program: PatternProgram) -> tuple[_PanelGeometry, ...]:
    document = program.to_pattern_document(samples_per_curve=33)
    analytic = document.annotations.get("analytic_edge_geometry", {})
    document_panels = {value.id: value for value in document.panels}
    panels, curves, moves, next_edges, _roles, _landmarks, _seams = _program_maps(program)
    output: list[_PanelGeometry] = []
    for panel_id, panel in panels.items():
        ordered = _ordered_curves(panel_id, curves[panel_id], moves[panel_id], next_edges)
        sampled_edges = {value.id: value for value in document_panels[panel_id].edges}
        geometry: list[_EdgeGeometry] = []
        bounds_points: list[tuple[float, float]] = []
        for command in ordered:
            sampled = sampled_edges[command.edge_id]
            start = tuple(float(value) for value in sampled.points[0])
            end = tuple(float(value) for value in sampled.points[-1])
            edge_payload = analytic[f"{panel_id}/{command.edge_id}"]
            curve = edge_payload["curve"]
            controls = tuple(
                (float(value[0]), float(value[1]))
                for value in curve.get("controls_cm", ())
            )
            radius = None
            if command.op == "A":
                radius = float(curve["arc"]["radius_cm"])
            geometry.append(_EdgeGeometry(command, start, end, controls, radius))
            # Bezier curves lie within the convex hull of endpoints/controls.
            # Arc bounds are obtained analytically from the same vector
            # primitive used by PatternDocument materialization.  No raster or
            # sampled contour participates in panel packing.
            bounds_points.extend((start, end, *controls))
            if command.op == "A":
                from svgpathtools import Arc

                arc = Arc(
                    complex(*start),
                    complex(float(radius), float(radius)),
                    rotation=0.0,
                    large_arc=bool(command.large_arc),
                    sweep=bool(command.sweep_y_up),
                    end=complex(*end),
                )
                minimum_x, maximum_x, minimum_y, maximum_y = arc.bbox()
                bounds_points.extend(
                    (
                        (float(minimum_x), float(minimum_y)),
                        (float(maximum_x), float(maximum_y)),
                    )
                )
        if not bounds_points:
            raise PatternDSLError(f"{panel_id}: no geometry to export")
        xs = [value[0] for value in bounds_points]
        ys = [value[1] for value in bounds_points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        if bbox[2] - bbox[0] <= 0.0 or bbox[3] - bbox[1] <= 0.0:
            raise PatternDSLError(f"{panel_id}: degenerate SVG bounding box")
        output.append(_PanelGeometry(panel, tuple(geometry), bbox))
    # Source panel order often follows generator roles.  Canonicalize packing
    # using geometry alone so the default SVG cannot leak front/back/part via
    # panel position or color.  Exact geometric duplicates are interchangeable.
    def geometry_key(value: _PanelGeometry) -> tuple[Any, ...]:
        edge_key = tuple(
            (
                edge.command.op,
                round(edge.command.length_ratio, 12),
                round(edge.command.chord_ratio, 12),
                round(edge.command.turn_sin, 12),
                round(edge.command.turn_cos, 12),
                tuple(
                    (round(control[0], 12), round(control[1], 12))
                    for control in edge.command.controls_chord_frame
                ),
                (
                    round(float(edge.command.arc_radius_over_chord), 12)
                    if edge.command.arc_radius_over_chord is not None
                    else None
                ),
                edge.command.large_arc,
                edge.command.sweep_y_up,
            )
            for edge in value.edges
        )
        return (
            round(value.panel.panel_scale_cm, 12),
            len(value.edges),
            round(value.width, 12),
            round(value.height, 12),
            edge_key,
        )

    return tuple(sorted(output, key=geometry_key))


def _layout_panels(
    panels: Sequence[_PanelGeometry], options: SvgExportOptions
) -> tuple[dict[str, _Placement], float, float]:
    columns = min(options.max_columns, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    column_widths = [0.0] * columns
    row_heights = [0.0] * rows
    for index, panel in enumerate(panels):
        column = index % columns
        row = index // columns
        column_widths[column] = max(column_widths[column], panel.width)
        row_heights[row] = max(row_heights[row], panel.height)
    column_starts: list[float] = []
    cursor = options.padding_cm
    for width in column_widths:
        column_starts.append(cursor)
        cursor += width + options.gap_cm
    canvas_width = cursor - options.gap_cm + options.padding_cm
    row_starts: list[float] = []
    cursor = options.padding_cm
    for height in row_heights:
        row_starts.append(cursor)
        cursor += height + options.gap_cm
    canvas_height = cursor - options.gap_cm + options.padding_cm
    placements = {
        panel.panel.panel_id: _Placement(
            x=column_starts[index % columns],
            y=row_starts[index // columns],
            width=panel.width,
            height=panel.height,
        )
        for index, panel in enumerate(panels)
    }
    return placements, canvas_width, canvas_height


def _transform(
    point: Sequence[float], panel: _PanelGeometry, placement: _Placement
) -> tuple[float, float]:
    # Source coordinates are y-up; SVG canvas coordinates are y-down.
    return (
        placement.x + float(point[0]) - panel.bbox[0],
        placement.y + panel.bbox[3] - float(point[1]),
    )


def _edge_path(
    edge: _EdgeGeometry,
    panel: _PanelGeometry,
    placement: _Placement,
    decimals: int,
    *,
    include_move: bool,
) -> str:
    start = _transform(edge.start, panel, placement)
    end = _transform(edge.end, panel, placement)
    fmt = lambda value: _format_number(value, decimals)
    values: list[str] = []
    if include_move:
        values.append(f"M {fmt(start[0])} {fmt(start[1])}")
    if edge.command.op == "L":
        values.append(f"L {fmt(end[0])} {fmt(end[1])}")
    elif edge.command.op == "Q":
        control = _transform(edge.controls[0], panel, placement)
        values.append(
            f"Q {fmt(control[0])} {fmt(control[1])} {fmt(end[0])} {fmt(end[1])}"
        )
    elif edge.command.op == "C":
        first = _transform(edge.controls[0], panel, placement)
        second = _transform(edge.controls[1], panel, placement)
        values.append(
            "C "
            f"{fmt(first[0])} {fmt(first[1])} "
            f"{fmt(second[0])} {fmt(second[1])} "
            f"{fmt(end[0])} {fmt(end[1])}"
        )
    elif edge.command.op == "A":
        assert edge.radius is not None
        # Reflecting y changes the SVG sweep bit.  Radius and large-arc remain
        # unchanged because panel packing contains no scale or shear.
        sweep_svg = 0 if bool(edge.command.sweep_y_up) else 1
        values.append(
            "A "
            f"{fmt(edge.radius)} {fmt(edge.radius)} 0 "
            f"{int(bool(edge.command.large_arc))} {sweep_svg} "
            f"{fmt(end[0])} {fmt(end[1])}"
        )
    else:  # pragma: no cover - CurveCommand validates this invariant.
        raise PatternDSLError(f"unsupported SVG opcode {edge.command.op!r}")
    return " ".join(values)


def _panel_path(
    panel: _PanelGeometry, placement: _Placement, decimals: int
) -> str:
    values = [
        _edge_path(edge, panel, placement, decimals, include_move=index == 0)
        for index, edge in enumerate(panel.edges)
    ]
    values.append("Z")
    return " ".join(values)


def _edge_midpoint(edge: _EdgeGeometry) -> tuple[float, float]:
    if edge.command.op == "L":
        return ((edge.start[0] + edge.end[0]) * 0.5, (edge.start[1] + edge.end[1]) * 0.5)
    if edge.command.op == "Q":
        control = edge.controls[0]
        return (
            0.25 * edge.start[0] + 0.5 * control[0] + 0.25 * edge.end[0],
            0.25 * edge.start[1] + 0.5 * control[1] + 0.25 * edge.end[1],
        )
    if edge.command.op == "C":
        first, second = edge.controls
        return (
            0.125 * edge.start[0]
            + 0.375 * first[0]
            + 0.375 * second[0]
            + 0.125 * edge.end[0],
            0.125 * edge.start[1]
            + 0.375 * first[1]
            + 0.375 * second[1]
            + 0.125 * edge.end[1],
        )
    # The exact arc midpoint is already available in the analytic vector
    # sampler retained by PatternDocument materialization.  The chord midpoint
    # is sufficient for label placement and does not affect exported geometry.
    return ((edge.start[0] + edge.end[0]) * 0.5, (edge.start[1] + edge.end[1]) * 0.5)


def _metadata_payload(
    program: PatternProgram,
    panels: Sequence[_PanelGeometry],
    placements: Mapping[str, _Placement],
    width: float,
    height: float,
    options: SvgExportOptions,
) -> dict[str, Any]:
    report = program.verify()
    facts = [
        _command_payload(command)
        for command in program.commands
        if isinstance(
            command,
            (
                RoleCommand,
                NextCommand,
                SharedEndpointCommand,
                SewnToCommand,
                LandmarkCommand,
            ),
        )
    ]
    dsl = program.serialize()
    payload: dict[str, Any] = {
        "schema_version": SVG_SCHEMA_VERSION,
        "geometry_contract": {
            "boundary_representation": "native SVG M/L/Q/C/A/Z",
            "raster_or_contour_dependency": False,
            "panel_scale_preserved": True,
            "layout_transform": "translation plus y-axis reflection only",
            "units": "cm",
            "default_export_is_label_free": not (
                options.include_semantic_facts or options.include_provenance
            ),
        },
        "svg_command_counts": {
            opcode: sum(
                edge.command.op == opcode for panel in panels for edge in panel.edges
            )
            for opcode in ("L", "Q", "C", "A")
        }
        | {"M": len(panels), "Z": len(panels)},
        "layout": {
            "canvas_cm": [width, height],
            "gap_cm": options.gap_cm,
            "padding_cm": options.padding_cm,
            "panels": [
                {
                    "panel_id": f"panel_{index:03d}",
                    "svg_bbox_cm_y_down": [
                        placements[panel.panel.panel_id].x,
                        placements[panel.panel.panel_id].y,
                        placements[panel.panel.panel_id].x + panel.width,
                        placements[panel.panel.panel_id].y + panel.height,
                    ],
                }
                for index, panel in enumerate(panels)
            ],
        },
        "verification": {
            "valid": report.valid,
            "error_count": int(report.metrics["error_count"]),
            "closed_cycle_count": int(report.metrics["closed_cycle_count"]),
            "degree_two_panel_count": int(report.metrics["degree_two_panel_count"]),
        },
    }
    if options.include_semantic_facts:
        payload["proof_facts"] = facts
        payload["semantic_verification"] = report.to_dict()
    if options.include_provenance:
        payload["provenance"] = {
            "pattern_id": program.pattern_id,
            "category": program.category,
            "source_dsl_schema": program.schema_version,
            "source_dsl_sha256": hashlib.sha256(dsl.encode("utf-8")).hexdigest(),
            "source_dsl": dsl,
            "panels": [
                {
                    "neutral_panel_id": f"panel_{index:03d}",
                    "source_panel_id": panel.panel.panel_id,
                    "source_bbox_cm_y_up": list(panel.bbox),
                    "weak_role": {
                        "part": panel.panel.part,
                        "surface": panel.panel.surface,
                        "side": panel.panel.side,
                    },
                }
                for index, panel in enumerate(panels)
            ],
        }
    return payload


def compile_pattern_svg(
    source: PatternProgram | PatternDocument,
    *,
    options: SvgExportOptions | None = None,
) -> str:
    """Compile a verified DSL/PatternDocument to deterministic analytic SVG.

    A ``PatternDocument`` is accepted only when it retains the source Pattern
    DSL in ``annotations.pattern_dsl``.  This intentionally rejects attempts
    to infer Bezier/arc primitives from sampled boundary points.
    """

    options = options or SvgExportOptions()
    program = _coerce_program(source)
    report = program.verify()
    if not report.valid:
        codes = ", ".join(
            value.code for value in report.issues if value.severity == "error"
        )
        raise PatternDSLError(f"cannot export symbolically invalid Pattern DSL: {codes}")
    panels = _materialized_panels(program)
    placements, width, height = _layout_panels(panels, options)
    fmt = lambda value: _format_number(value, options.decimals)
    metadata = _metadata_payload(program, panels, placements, width, height, options)
    _panels, _curves, _moves, _next, roles, landmarks, seams = _program_maps(program)
    # Derived landmarks are proof results even when a caller omitted explicit
    # LANDMARK commands from an otherwise semantically labelled program.
    for landmark in report.derived_landmarks:
        current = landmarks.setdefault(landmark.panel_id, [])
        key = (landmark.name, landmark.point_id)
        if key not in {(value.name, value.point_id) for value in current}:
            current.append(landmark)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns={quoteattr(SVG_NAMESPACE)} version="1.1" '
            f'viewBox="0 0 {fmt(width)} {fmt(height)}" '
            f'width="{fmt(width * 10.0)}mm" height="{fmt(height * 10.0)}mm" '
            f'data-schema={quoteattr(SVG_SCHEMA_VERSION)}>'
        ),
        "  <title>Verified analytic Pattern DSL geometry</title>",
        "  <desc>Verified analytic sewing-pattern geometry; no raster tracing.</desc>",
    ]
    if options.include_metadata:
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        lines.append('  <metadata id="pattern-dsl-metadata">' + escape(encoded) + "</metadata>")
    lines.append(
            "  <style>"
            ".panel-boundary{stroke:#20242b;stroke-width:.08;stroke-linejoin:round;fill-opacity:.58;}"
            ".panel-review-label{font:700 .9px sans-serif;fill:#20242b;paint-order:stroke;stroke:#fff;stroke-width:.045;}"
            ".semantic-edge{fill:none;stroke-width:.13;stroke-linecap:round;}"
            ".semantic-label{font:600 .72px sans-serif;fill:#111;paint-order:stroke;stroke:#fff;stroke-width:.04;}"
            ".landmark{fill:#ffe066;stroke:#151515;stroke-width:.05;}"
            ".landmark-label{font:700 .76px sans-serif;fill:#704800;paint-order:stroke;stroke:#fff;stroke-width:.04;}"
            ".seam-proof{fill:none;stroke:#7450a8;stroke-width:.07;stroke-dasharray:.22 .14;}"
            ".seam-label{font:600 .68px sans-serif;fill:#5d3a8e;paint-order:stroke;stroke:#fff;stroke-width:.04;}"
            "</style>"
    )
    panel_lookup = {value.panel.panel_id: value for value in panels}
    edge_lookup = {
        (panel.panel.panel_id, edge.command.edge_id): edge
        for panel in panels
        for edge in panel.edges
    }
    point_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for panel_index, panel in enumerate(panels):
        panel_id = panel.panel.panel_id
        placement = placements[panel_id]
        neutral_id = f"panel_{panel_index:03d}"
        safe = f"panel-{panel_index:03d}"
        path_data = _panel_path(panel, placement, options.decimals)
        lines.append(
            f'  <g id={quoteattr(safe)} class="pattern-panel" '
            f'data-panel-id={quoteattr(neutral_id)}>'
        )
        fill = (
            PANEL_COLORS[panel_index % len(PANEL_COLORS)]
            if options.include_overlays
            else "#d9dee8"
        )
        lines.append(
            f'    <path class="panel-boundary" fill={quoteattr(fill)} '
            f'd={quoteattr(path_data)}/>'
        )
        for edge in panel.edges:
            point_lookup[(panel_id, edge.command.start_point_id)] = _transform(
                edge.start, panel, placement
            )
            point_lookup[(panel_id, edge.command.end_point_id)] = _transform(
                edge.end, panel, placement
            )
        lines.append("  </g>")

    if options.include_overlays:
        lines.append('  <g id="semantic-overlays" aria-label="semantic and proof overlays">')
        for panel in panels:
            panel_id = panel.panel.panel_id
            placement = placements[panel_id]
            lines.append(
                f'    <text class="panel-review-label" x="{fmt(placement.x)}" '
                f'y="{fmt(max(0.75, placement.y - 0.38))}">'
                + escape(
                    f"{panel_id} · {panel.panel.part}/{panel.panel.surface}/{panel.panel.side}"
                )
                + "</text>"
            )
            for edge in panel.edges:
                role = roles.get((panel_id, edge.command.edge_id))
                if role is None:
                    continue
                edge_path = _edge_path(
                    edge, panel, placement, options.decimals, include_move=True
                )
                lines.append(
                    f'    <path class="semantic-edge" data-panel-id={quoteattr(panel_id)} '
                    f'data-edge-id={quoteattr(edge.command.edge_id)} data-role={quoteattr(role)} '
                    f'stroke={quoteattr(EDGE_COLORS[edge.command.op])} d={quoteattr(edge_path)}/>'
                )
                midpoint = _transform(_edge_midpoint(edge), panel, placement)
                lines.append(
                    f'    <text class="semantic-label" x="{fmt(midpoint[0] + 0.08)}" '
                    f'y="{fmt(midpoint[1] - 0.08)}">'
                    + escape(f"{edge.command.edge_id}: {role}")
                    + "</text>"
                )
            for landmark in sorted(
                landmarks.get(panel_id, ()), key=lambda value: (value.name, value.point_id)
            ):
                point = point_lookup.get((panel_id, landmark.point_id))
                if point is None:
                    continue
                lines.append(
                    f'    <circle class="landmark" data-landmark={quoteattr(landmark.name)} '
                    f'data-point-id={quoteattr(landmark.point_id)} cx="{fmt(point[0])}" '
                    f'cy="{fmt(point[1])}" r=".25"/>'
                )
                lines.append(
                    f'    <text class="landmark-label" x="{fmt(point[0] + 0.2)}" '
                    f'y="{fmt(point[1] - 0.2)}">{escape(landmark.name)}</text>'
                )
        for seam in sorted(seams, key=lambda value: value.seam_id):
            first_edge = edge_lookup[(seam.first_panel_id, seam.first_edge_id)]
            second_edge = edge_lookup[(seam.second_panel_id, seam.second_edge_id)]
            first_panel = panel_lookup[seam.first_panel_id]
            second_panel = panel_lookup[seam.second_panel_id]
            first = _transform(
                _edge_midpoint(first_edge), first_panel, placements[seam.first_panel_id]
            )
            second = _transform(
                _edge_midpoint(second_edge), second_panel, placements[seam.second_panel_id]
            )
            lines.append(
                f'    <line class="seam-proof" data-seam-id={quoteattr(seam.seam_id)} '
                f'x1="{fmt(first[0])}" y1="{fmt(first[1])}" '
                f'x2="{fmt(second[0])}" y2="{fmt(second[1])}"/>'
            )
            lines.append(
                f'    <text class="seam-label" x="{fmt((first[0] + second[0]) * 0.5)}" '
                f'y="{fmt((first[1] + second[1]) * 0.5 - 0.12)}">'
                + escape(f"SEWN_TO {seam.seam_id}")
                + "</text>"
            )
        lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_pattern_svg(
    source: PatternProgram | PatternDocument,
    destination: str | Path,
    *,
    options: SvgExportOptions | None = None,
) -> Path:
    """Compile and atomically write one analytic SVG file."""

    destination = Path(destination)
    if destination.suffix.lower() != ".svg":
        raise ValueError("destination must end in .svg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = compile_pattern_svg(source, options=options)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


__all__ = [
    "SVG_SCHEMA_VERSION",
    "SvgExportOptions",
    "compile_pattern_svg",
    "write_pattern_svg",
]
