from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from benchmark.gcdv2_exact.geometry import CURVE_COLORS, sample_curve


PANEL_SCHEMA_VERSION = "gcdv2-exact-single-panel-1.1"
DEFAULT_CANVAS_SIZE = 1024
DEFAULT_PIXELS_PER_CM = 3.0
DEFAULT_MINIMUM_MARGIN_PX = 64.0


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def panel_slug(panel_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(panel_id)).strip("._")
    return value or "panel"


def infer_panel_role(panel_id: str, source_label: str | None) -> dict[str, Any]:
    """Conservative lexical normalization of author-provided panel identifiers.

    These labels are useful weak supervision, not expert drafting semantics.
    The unmodified source identifier is always retained beside them.
    """

    lowered = str(panel_id).lower()
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", lowered) if token)
    token_set = set(tokens)

    if "ftorso" in lowered or "btorso" in lowered or "torso" in token_set:
        part = "bodice"
    elif "collar" in token_set or "collar" in lowered:
        part = "collar"
    elif "hood" in token_set or "hood" in lowered:
        part = "hood"
    elif "cuff" in token_set and ("pant" in token_set or "pant" in lowered):
        part = "pants_cuff"
    elif "cuff" in token_set:
        part = "sleeve_cuff"
    elif "sleeve" in token_set or "sleeve" in lowered:
        part = "sleeve"
    elif lowered.startswith("wb_") or "waistband" in token_set:
        part = "waistband"
    elif lowered.startswith("pant_") or "pants" in token_set:
        part = "pants_leg"
    elif lowered.startswith("ins_skirt"):
        part = "skirt_insert"
    elif "skirt" in token_set or "skirt" in lowered:
        part = "skirt_panel"
    else:
        part = {
            "body": "body_panel",
            "arm": "arm_panel",
            "leg": "leg_panel",
        }.get(str(source_label).lower(), "other")

    front = (
        "front" in token_set
        or "ftorso" in lowered
        or (part in {"sleeve", "sleeve_cuff", "pants_cuff", "pants_leg"} and "f" in token_set)
    )
    back = (
        "back" in token_set
        or "btorso" in lowered
        or (part in {"sleeve", "sleeve_cuff", "pants_cuff", "pants_leg"} and "b" in token_set)
    )
    surface = "front" if front and not back else ("back" if back and not front else "unspecified")

    left = "left" in token_set or (part in {"pants_leg", "pants_cuff"} and "l" in token_set)
    right = "right" in token_set or (part in {"pants_leg", "pants_cuff"} and "r" in token_set)
    side = "left" if left and not right else ("right" if right and not left else "unspecified")

    numeric = [int(token) for token in tokens if token.isdigit()]
    return {
        "part": part,
        "surface": surface,
        "side": side,
        "instance_index": numeric[-1] if numeric else None,
        "source_panel_id": str(panel_id),
        "source_label": source_label,
        "provenance": "LEXICAL_DERIVATION_FROM_SOURCE_PANEL_ID",
        "expert_verified": False,
    }


def _signed_area(vertices: Sequence[Sequence[float]], order: Sequence[int]) -> float:
    return 0.5 * sum(
        float(vertices[first][0]) * float(vertices[second][1])
        - float(vertices[second][0]) * float(vertices[first][1])
        for first, second in zip(order, (*order[1:], order[0]))
    )


def canonical_boundary(panel: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
    """Return one CCW closed cycle and its directed source edges.

    GCDv2 panels are expected to be connected degree-two boundary graphs.  The
    function fails closed if that contract is violated.
    """

    vertices = panel["vertices_cm"]
    edges = panel["edges"]
    count = len(vertices)
    if count < 3 or len(edges) != count:
        raise ValueError(f"panel {panel['panel_id']} is not a single boundary candidate")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for edge_index, edge in enumerate(edges):
        first, second = (int(value) for value in edge["endpoints"])
        if first == second or not (0 <= first < count and 0 <= second < count):
            raise ValueError(f"invalid edge endpoints in {panel['panel_id']}: {edge['endpoints']}")
        adjacency[first].append((second, edge_index))
        adjacency[second].append((first, edge_index))
    if any(len(neighbors) != 2 for neighbors in adjacency):
        raise ValueError(f"panel {panel['panel_id']} is not degree two")

    start = min(
        range(count),
        key=lambda index: (
            round(float(vertices[index][1]), 8),
            round(float(vertices[index][0]), 8),
            index,
        ),
    )

    def walk(first_neighbor: int) -> tuple[int, ...]:
        order = [start]
        previous, current = start, first_neighbor
        while current != start:
            if current in order:
                raise ValueError(f"premature cycle in {panel['panel_id']}")
            order.append(current)
            candidates = [neighbor for neighbor, _ in adjacency[current] if neighbor != previous]
            if len(candidates) != 1:
                raise ValueError(f"ambiguous cycle in {panel['panel_id']}")
            previous, current = current, candidates[0]
        if len(order) != count:
            raise ValueError(f"disconnected cycles in {panel['panel_id']}")
        return tuple(order)

    first, second = (neighbor for neighbor, _ in adjacency[start])
    candidates = (walk(first), walk(second))
    positive = [order for order in candidates if _signed_area(vertices, order) > 0.0]
    if len(positive) == 1:
        order = positive[0]
    else:
        order = min(
            candidates,
            key=lambda current: tuple(
                (round(float(vertices[index][0]), 8), round(float(vertices[index][1]), 8))
                for index in current
            ),
        )

    edge_by_pair = {
        frozenset((int(edge["endpoints"][0]), int(edge["endpoints"][1]))): edge
        for edge in edges
    }
    directed = []
    for first_vertex, second_vertex in zip(order, (*order[1:], order[0])):
        edge = edge_by_pair.get(frozenset((first_vertex, second_vertex)))
        if edge is None:
            raise ValueError(f"missing cycle edge in {panel['panel_id']}")
        source_endpoints = tuple(int(value) for value in edge["endpoints"])
        directed.append(
            {
                "edge": edge,
                "start_source_vertex": first_vertex,
                "end_source_vertex": second_vertex,
                "source_direction_reversed": source_endpoints != (first_vertex, second_vertex),
            }
        )
    return order, tuple(directed)


def _relative_control(
    start: Sequence[float], end: Sequence[float], control: Sequence[float]
) -> list[float]:
    dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        raise ValueError("zero-length chord cannot encode a relative control")
    vx, vy = float(control[0]) - float(start[0]), float(control[1]) - float(start[1])
    return [(vx * dx + vy * dy) / denominator, (vx * -dy + vy * dx) / denominator]


def _angle(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _directed_curve(edge: Mapping[str, Any], reversed_direction: bool) -> dict[str, Any]:
    curve = copy.deepcopy(edge["curve"])
    if not reversed_direction:
        return curve
    if curve["type"] == "cubic_bezier":
        curve["controls_cm"] = list(reversed(curve["controls_cm"]))
    elif curve["type"] == "circular_arc":
        curve["arc"]["sweep_y_up"] = not bool(curve["arc"]["sweep_y_up"])
        curve["arc"]["right"] = not bool(curve["arc"]["right"])
    return curve


def panel_target(
    sample_label: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    pixels_per_cm: float = DEFAULT_PIXELS_PER_CM,
) -> dict[str, Any]:
    order, directed_edges = canonical_boundary(panel)
    bbox = tuple(float(value) for value in panel["local_curve_bbox_cm"])
    center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width * pixels_per_cm > canvas_size - 2 * DEFAULT_MINIMUM_MARGIN_PX:
        raise ValueError(f"panel {panel['panel_id']} exceeds fixed metric canvas width")
    if height * pixels_per_cm > canvas_size - 2 * DEFAULT_MINIMUM_MARGIN_PX:
        raise ValueError(f"panel {panel['panel_id']} exceeds fixed metric canvas height")
    usable_span = canvas_size - 2 * DEFAULT_MINIMUM_MARGIN_PX
    normalized_pixels_per_cm = min(
        usable_span / max(width, 1e-9), usable_span / max(height, 1e-9)
    )

    source_vertices = panel["vertices_cm"]
    vertices = []
    for canonical_index, source_index in enumerate(order):
        source = [float(value) for value in source_vertices[source_index]]
        centered = [source[0] - center[0], source[1] - center[1]]
        vertices.append(
            {
                "vertex_index": canonical_index,
                "source_vertex_index": int(source_index),
                "source_xy_cm": source,
                "centered_xy_cm": centered,
                "image_xy_px": [
                    canvas_size * 0.5 + centered[0] * normalized_pixels_per_cm,
                    canvas_size * 0.5 - centered[1] * normalized_pixels_per_cm,
                ],
                "metric_image_xy_px": [
                    canvas_size * 0.5 + centered[0] * pixels_per_cm,
                    canvas_size * 0.5 - centered[1] * pixels_per_cm,
                ],
            }
        )

    canonical_edges = []
    for index, directed in enumerate(directed_edges):
        edge = directed["edge"]
        source_start = source_vertices[directed["start_source_vertex"]]
        source_end = source_vertices[directed["end_source_vertex"]]
        curve = _directed_curve(edge, bool(directed["source_direction_reversed"]))
        controls = [
            [float(value[0]) - center[0], float(value[1]) - center[1]]
            for value in curve.get("controls_cm", [])
        ]
        start_centered = [float(source_start[0]) - center[0], float(source_start[1]) - center[1]]
        end_centered = [float(source_end[0]) - center[0], float(source_end[1]) - center[1]]
        parameter_payload: dict[str, Any]
        if curve["type"] in {"quadratic_bezier", "cubic_bezier"}:
            absolute_source_controls = [
                [float(value[0]), float(value[1])] for value in curve["controls_cm"]
            ]
            parameter_payload = {
                "relative_controls_chord_frame": [
                    _relative_control(source_start, source_end, control)
                    for control in absolute_source_controls
                ]
            }
        elif curve["type"] == "circular_arc":
            parameter_payload = {
                "radius_cm": float(curve["arc"]["radius_cm"]),
                "large_arc": bool(curve["arc"]["large_arc"]),
                "sweep_y_up": bool(curve["arc"]["sweep_y_up"]),
            }
        else:
            parameter_payload = {}

        if directed["source_direction_reversed"]:
            start_tangent = _angle(float(edge["end_tangent_deg"]) + 180.0)
            end_tangent = _angle(float(edge["start_tangent_deg"]) + 180.0)
        else:
            start_tangent = float(edge["start_tangent_deg"])
            end_tangent = float(edge["end_tangent_deg"])
        chord_direction = math.degrees(
            math.atan2(end_centered[1] - start_centered[1], end_centered[0] - start_centered[0])
        )
        canonical_edges.append(
            {
                "edge_index": index,
                "source_edge_id": str(edge["edge_id"]),
                "source_edge_index": int(edge["edge_index"]),
                "source_direction_reversed": bool(directed["source_direction_reversed"]),
                "start_vertex_index": index,
                "end_vertex_index": (index + 1) % len(order),
                "curve_type": str(curve["type"]),
                "curve_parameters": parameter_payload,
                "centered_controls_cm": controls,
                "length_cm": float(edge["length_cm"]),
                "chord_direction_deg_y_up": chord_direction,
                "start_tangent_deg_y_up": start_tangent,
                "end_tangent_deg_y_up": end_tangent,
            }
        )

    role = infer_panel_role(str(panel["panel_id"]), panel.get("source_label"))
    return {
        "schema_version": PANEL_SCHEMA_VERSION,
        "panel_uid": f"{sample_label['sample_id']}:{panel['panel_id']}",
        "sample_id": str(sample_label["sample_id"]),
        "garment_category": str(sample_label["category"]),
        "source": {
            "dataset": str(sample_label["source_dataset"]),
            "license": str(sample_label["source_license"]),
            "specification_sha256": str(sample_label["source_specification_sha256"]),
            "panel_id": str(panel["panel_id"]),
            "panel_order_index": int(panel["source_order_index"]),
            "source_label": panel.get("source_label"),
        },
        "role_labels": role,
        "input_contract": {
            "one_panel_only": True,
            "uniform_fill_no_role_color": True,
            "canvas_size_px": [canvas_size, canvas_size],
            "origin_px": [canvas_size * 0.5, canvas_size * 0.5],
            "x_axis": "right",
            "y_axis": "up",
            "centering": "source local curve bbox center",
            "normalized_panel_image": {
                "path_key": "panel_image_path",
                "pixels_per_cm": normalized_pixels_per_cm,
                "cm_per_pixel": 1.0 / normalized_pixels_per_cm,
                "minimum_margin_px": DEFAULT_MINIMUM_MARGIN_PX,
                "absolute_length_requires_scale_token": True,
            },
            "metric_validation_image": {
                "path_key": "metric_panel_image_path",
                "pixels_per_cm": pixels_per_cm,
                "physical_span_cm": canvas_size / pixels_per_cm,
                "absolute_length_is_observable_from_fixed_global_pixel_scale": True,
            },
        },
        "geometry": {
            "topology": "single_connected_closed_cycle",
            "source_curve_bbox_cm": list(bbox),
            "source_curve_bbox_center_cm": list(center),
            "width_cm": width,
            "height_cm": height,
            "boundary_vertex_count": len(vertices),
            "boundary_edge_count": len(canonical_edges),
            "boundary_sequence": list(range(len(vertices))),
            "vertices": vertices,
            "edges": canonical_edges,
        },
        "paired_views": copy.deepcopy(sample_label.get("views", [])),
        "claim_boundary": (
            "Geometry is exact GCDv2 source-derived truth. role_labels are lexical weak labels "
            "from source panel IDs and are not expert drafting semantics."
        ),
    }


def _target_curve(target: Mapping[str, Any], edge: Mapping[str, Any]) -> dict[str, Any]:
    curve_type = str(edge["curve_type"])
    curve: dict[str, Any] = {
        "type": curve_type,
        "controls_cm": [list(value) for value in edge["centered_controls_cm"]],
    }
    if curve_type == "circular_arc":
        parameters = edge["curve_parameters"]
        curve["arc"] = {
            "radius_cm": float(parameters["radius_cm"]),
            "large_arc": bool(parameters["large_arc"]),
            "sweep_y_up": bool(parameters["sweep_y_up"]),
            "right": bool(parameters["sweep_y_up"]),
        }
    return curve


def _pixel(
    target: Mapping[str, Any],
    point: Sequence[float],
    scale: int = 1,
    *,
    metric: bool = False,
) -> tuple[float, float]:
    contract = target["input_contract"]
    canvas = float(contract["canvas_size_px"][0])
    mode = "metric_validation_image" if metric else "normalized_panel_image"
    pixels_per_cm = float(contract[mode]["pixels_per_cm"])
    return (
        (canvas * 0.5 + float(point[0]) * pixels_per_cm) * scale,
        (canvas * 0.5 - float(point[1]) * pixels_per_cm) * scale,
    )


def render_panel_input(
    target: Mapping[str, Any],
    destination: Path,
    *,
    supersample: int = 2,
    metric: bool = False,
) -> Path:
    canvas = int(target["input_contract"]["canvas_size_px"][0])
    vertices = [value["centered_xy_cm"] for value in target["geometry"]["vertices"]]
    boundary: list[tuple[float, float]] = []
    for edge in target["geometry"]["edges"]:
        start = vertices[int(edge["start_vertex_index"])]
        end = vertices[int(edge["end_vertex_index"])]
        points = sample_curve(start, end, _target_curve(target, edge), samples=65)
        boundary.extend(
            _pixel(target, point, supersample, metric=metric) for point in points[:-1]
        )
    image = Image.new("L", (canvas * supersample, canvas * supersample), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(boundary, fill=224, outline=255, width=2 * supersample)
    image = image.resize((canvas, canvas), Image.Resampling.LANCZOS).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)
    return destination


def render_panel_overlay(target: Mapping[str, Any], destination: Path) -> Path:
    canvas = int(target["input_contract"]["canvas_size_px"][0])
    image = Image.new("RGB", (canvas, canvas), "#0b0c0f")
    draw = ImageDraw.Draw(image)
    vertices = [value["centered_xy_cm"] for value in target["geometry"]["vertices"]]
    for edge in target["geometry"]["edges"]:
        start = vertices[int(edge["start_vertex_index"])]
        end = vertices[int(edge["end_vertex_index"])]
        points = sample_curve(start, end, _target_curve(target, edge), samples=65)
        pixels = [_pixel(target, point) for point in points]
        draw.line(pixels, fill=CURVE_COLORS[str(edge["curve_type"])], width=5, joint="curve")
        midpoint = pixels[len(pixels) // 2]
        draw.text(
            (midpoint[0] + 4, midpoint[1] + 4),
            f"e{edge['edge_index']} {edge['curve_type']} {edge['length_cm']:.1f}cm",
            font=_font(14),
            fill="white",
        )
    for vertex in target["geometry"]["vertices"]:
        x, y = _pixel(target, vertex["centered_xy_cm"])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#ffffff", outline="#0b0c0f")
        draw.text((x + 7, y - 8), f"v{vertex['vertex_index']}", font=_font(16, bold=True), fill="#ffffff")
    role = target["role_labels"]
    draw.rectangle((12, 12, 680, 78), fill="#15171d")
    draw.text(
        (24, 24),
        f"{target['garment_category']} · {role['part']} · {role['surface']} · {role['side']}",
        font=_font(22, bold=True),
        fill="white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)
    return destination


__all__ = [
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_PIXELS_PER_CM",
    "PANEL_SCHEMA_VERSION",
    "canonical_boundary",
    "infer_panel_role",
    "panel_slug",
    "panel_target",
    "render_panel_input",
    "render_panel_overlay",
]
