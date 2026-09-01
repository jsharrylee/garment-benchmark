from __future__ import annotations

import colorsys
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from svgpathtools import Arc


SCHEMA_VERSION = "gcdv2-exact-pair-1.0"
CURVE_TYPES = ("line", "quadratic_bezier", "cubic_bezier", "circular_arc")
CURVE_COLORS = {
    "line": "#39c6d6",
    "quadratic_bezier": "#f1a340",
    "cubic_bezier": "#ef5da8",
    "circular_arc": "#77c66e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point(value: Sequence[float]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _relative_control(
    start: Sequence[float], end: Sequence[float], relative: Sequence[float]
) -> tuple[float, float]:
    start_x, start_y = _point(start)
    end_x, end_y = _point(end)
    along, normal = _point(relative)
    dx, dy = end_x - start_x, end_y - start_y
    return start_x + along * dx - normal * dy, start_y + along * dy + normal * dx


def _curve_payload(vertices: Sequence[Sequence[float]], edge: Mapping[str, Any]) -> dict[str, Any]:
    start_index, end_index = (int(value) for value in edge["endpoints"])
    start, end = _point(vertices[start_index]), _point(vertices[end_index])
    curvature = edge.get("curvature")
    if not curvature:
        return {"type": "line", "source_type": "line", "source_params": None, "controls_cm": []}
    if isinstance(curvature, list):
        source_type, params = "quadratic", curvature
    else:
        source_type = str(curvature.get("type", "line"))
        params = curvature.get("params", [])
    if source_type == "quadratic":
        if len(params) != 1:
            raise ValueError(f"quadratic edge must have one relative control, got {params!r}")
        return {
            "type": "quadratic_bezier",
            "source_type": source_type,
            "source_params": params,
            "controls_cm": [list(_relative_control(start, end, params[0]))],
        }
    if source_type == "cubic":
        if len(params) != 2:
            raise ValueError(f"cubic edge must have two relative controls, got {params!r}")
        return {
            "type": "cubic_bezier",
            "source_type": source_type,
            "source_params": params,
            "controls_cm": [
                list(_relative_control(start, end, params[0])),
                list(_relative_control(start, end, params[1])),
            ],
        }
    if source_type == "circle":
        if len(params) != 3:
            raise ValueError(f"circle edge must have radius/large-arc/right, got {params!r}")
        radius, large_arc, right = float(params[0]), bool(params[1]), bool(params[2])
        return {
            "type": "circular_arc",
            "source_type": source_type,
            "source_params": params,
            "controls_cm": [],
            "arc": {
                "radius_cm": abs(radius),
                "large_arc": large_arc,
                "right": right,
                # GCD stores the y-up SVG sweep flag directly.  Only the
                # legacy PNG renderer negates it after flipping y for SVG.
                "sweep_y_up": right,
            },
        }
    raise ValueError(f"unsupported GarmentCode curve type: {source_type!r}")


def _arc(start: Sequence[float], end: Sequence[float], curve: Mapping[str, Any]) -> Arc:
    arc = curve["arc"]
    radius = float(arc["radius_cm"])
    return Arc(
        complex(*_point(start)),
        complex(radius, radius),
        rotation=0,
        large_arc=bool(arc["large_arc"]),
        sweep=bool(arc["sweep_y_up"]),
        end=complex(*_point(end)),
    )


def sample_curve(
    start: Sequence[float],
    end: Sequence[float],
    curve: Mapping[str, Any],
    *,
    samples: int = 65,
) -> tuple[tuple[float, float], ...]:
    start, end = _point(start), _point(end)
    values = np.linspace(0.0, 1.0, max(2, int(samples)))
    kind = str(curve["type"])
    if kind == "line":
        return (start, end)
    if kind == "quadratic_bezier":
        control = _point(curve["controls_cm"][0])
        return tuple(
            (
                (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0],
                (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1],
            )
            for t in values
        )
    if kind == "cubic_bezier":
        first, second = (_point(value) for value in curve["controls_cm"])
        return tuple(
            (
                (1 - t) ** 3 * start[0]
                + 3 * (1 - t) ** 2 * t * first[0]
                + 3 * (1 - t) * t**2 * second[0]
                + t**3 * end[0],
                (1 - t) ** 3 * start[1]
                + 3 * (1 - t) ** 2 * t * first[1]
                + 3 * (1 - t) * t**2 * second[1]
                + t**3 * end[1],
            )
            for t in values
        )
    if kind == "circular_arc":
        arc = _arc(start, end, curve)
        return tuple((float(arc.point(float(t)).real), float(arc.point(float(t)).imag)) for t in values)
    raise ValueError(f"unsupported curve type: {kind!r}")


def _tangent_degrees(start: Sequence[float], end: Sequence[float], curve: Mapping[str, Any], at: float) -> float:
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    kind = str(curve["type"])
    if kind == "line":
        tangent = end - start
    elif kind == "quadratic_bezier":
        control = np.asarray(curve["controls_cm"][0], dtype=float)
        tangent = 2.0 * ((1.0 - at) * (control - start) + at * (end - control))
    elif kind == "cubic_bezier":
        first, second = (np.asarray(value, dtype=float) for value in curve["controls_cm"])
        tangent = (
            3.0 * (1.0 - at) ** 2 * (first - start)
            + 6.0 * (1.0 - at) * at * (second - first)
            + 3.0 * at**2 * (end - second)
        )
    else:
        derivative = _arc(start, end, curve).derivative(at)
        tangent = np.asarray((derivative.real, derivative.imag), dtype=float)
    return float(math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))))


def _curve_length(start: Sequence[float], end: Sequence[float], curve: Mapping[str, Any]) -> float:
    if curve["type"] == "line":
        return float(math.dist(start, end))
    if curve["type"] == "circular_arc":
        return float(_arc(start, end, curve).length(error=1e-10))
    points = sample_curve(start, end, curve, samples=257)
    return float(sum(math.dist(first, second) for first, second in zip(points, points[1:])))


def _panel_color(index: int, total: int) -> tuple[int, int, int]:
    hue = (0.67 + 0.61803398875 * index) % 1.0
    saturation = 0.48 + 0.12 * ((index % 3) / 2.0)
    lightness = 0.48 + 0.10 * ((index % 2))
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return tuple(int(round(value * 255)) for value in rgb)


@dataclass(frozen=True)
class _PackItem:
    panel_id: str
    source_index: int
    bbox: tuple[float, float, float, float]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def _pack_shelves(items: Sequence[_PackItem], gap: float) -> tuple[dict[str, tuple[float, float]], float, float]:
    ordered = sorted(items, key=lambda item: (-item.height, -item.width, item.source_index, item.panel_id))
    area = sum((item.width + gap) * (item.height + gap) for item in ordered)
    largest = max(item.width for item in ordered)
    base = max(largest, math.sqrt(max(area, 1e-6)))
    candidates = sorted({max(largest, base * factor) for factor in (0.7, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0)})
    best: tuple[float, dict[str, tuple[float, float]], float, float] | None = None
    for limit in candidates:
        positions: dict[str, tuple[float, float]] = {}
        x = y = row_height = max_width = 0.0
        for item in ordered:
            if x > 0.0 and x + item.width > limit:
                y += row_height + gap
                x = 0.0
                row_height = 0.0
            positions[item.panel_id] = (x - item.bbox[0], y - item.bbox[1])
            x += item.width + gap
            row_height = max(row_height, item.height)
            max_width = max(max_width, x - gap)
        total_height = y + row_height
        score = max(max_width, total_height) + 0.18 * abs(max_width - total_height)
        candidate = (score, positions, max_width, total_height)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _transform_point(point: Sequence[float], transform: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(point[0]) + float(transform["translation_cm"][0])) * float(transform["scale_px_per_cm"])
        + float(transform["canvas_offset_px"][0]),
        float(transform["canvas_size_px"][1])
        - (
            (float(point[1]) + float(transform["translation_cm"][1])) * float(transform["scale_px_per_cm"])
            + float(transform["canvas_offset_px"][1])
        ),
    )


def _render_label(
    label: Mapping[str, Any],
    destination: Path,
    *,
    overlay: bool,
    size: int = 1024,
) -> Path:
    supersample = 2
    image = Image.new("RGB", (size * supersample, size * supersample), "#0b0c0f")
    draw = ImageDraw.Draw(image)
    for panel in label["panels"]:
        transformed_edges = []
        for edge in panel["edges"]:
            points = sample_curve(edge["start_cm"], edge["end_cm"], edge["curve"], samples=65)
            transformed_edges.append([tuple(value * supersample for value in _transform_point(point, label["packing"][panel["panel_id"]])) for point in points])
        boundary = [point for edge in transformed_edges for point in edge[:-1]]
        if boundary:
            draw.polygon(boundary, fill=tuple(panel["render_color_rgb"]), outline="#050507")
        for edge, points in zip(panel["edges"], transformed_edges):
            color = CURVE_COLORS[edge["curve"]["type"]] if overlay else "#09090b"
            draw.line(points, fill=color, width=(4 if overlay else 3) * supersample, joint="curve")
            if overlay:
                start, end = points[0], points[-1]
                radius = 3 * supersample
                draw.ellipse((start[0]-radius, start[1]-radius, start[0]+radius, start[1]+radius), fill="#ffffff")
                mid = points[len(points) // 2]
                message = f"e{edge['edge_index']} {edge['curve']['type'].replace('_bezier','').replace('circular_arc','arc')}\n{edge['length_cm']:.1f}cm {edge['chord_direction_deg']:.0f}°"
                draw.text((mid[0] + 4 * supersample, mid[1] - 16 * supersample), message, font=_font(10 * supersample, bold=True), fill="#ffffff", stroke_width=2 * supersample, stroke_fill="#000000")
        if overlay:
            bbox = panel["packed_bbox_px"]
            x = (bbox[0] + bbox[2]) * 0.5 * supersample
            y = (bbox[1] + 6) * supersample
            draw.text((x, y), panel["panel_id"], anchor="ma", font=_font(15 * supersample, bold=True), fill="#ffffff", stroke_width=3 * supersample, stroke_fill="#000000")
            for index, point in enumerate(panel["vertices_cm"]):
                px = tuple(value * supersample for value in _transform_point(point, label["packing"][panel["panel_id"]]))
                draw.text((px[0] + 4 * supersample, px[1] + 3 * supersample), f"v{index}", font=_font(9 * supersample, bold=True), fill="#ffffff", stroke_width=2 * supersample, stroke_fill="#000000")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)
    return destination


def _packed_uv(point_px: Sequence[float], size: int) -> list[float]:
    return [float(point_px[0]) / size, float(point_px[1]) / size]


def build_exact_sample(
    specification_path: Path,
    *,
    category: str,
    view_paths: Sequence[Path],
    view_labels: Sequence[str] = ("front", "back", "left", "right"),
    output_dir: Path,
    image_size: int = 1024,
    gap_fraction: float = 0.055,
    render_overlay: bool = False,
    allow_missing_views: bool = False,
) -> dict[str, Any]:
    specification_path = Path(specification_path)
    if len(view_paths) != len(view_labels):
        raise ValueError("view_paths and view_labels must have equal length")
    raw = json.loads(specification_path.read_text(encoding="utf-8"))
    source = raw["pattern"]
    sample_id = specification_path.stem.removesuffix("_specification")
    panel_order = [str(value) for value in source.get("panel_order", tuple(source["panels"]))]
    interim: list[dict[str, Any]] = []
    pack_items: list[_PackItem] = []
    all_spans = []
    for panel_index, panel_id in enumerate(panel_order):
        panel = source["panels"][panel_id]
        vertices = [_point(value) for value in panel["vertices"]]
        edge_rows = []
        dense = []
        for edge_index, edge in enumerate(panel["edges"]):
            start_index, end_index = (int(value) for value in edge["endpoints"])
            curve = _curve_payload(vertices, edge)
            sampled = sample_curve(vertices[start_index], vertices[end_index], curve, samples=65)
            dense.extend(sampled)
            start_tangent = _tangent_degrees(vertices[start_index], vertices[end_index], curve, 0.0)
            end_tangent = _tangent_degrees(vertices[start_index], vertices[end_index], curve, 1.0)
            chord = math.degrees(math.atan2(vertices[end_index][1]-vertices[start_index][1], vertices[end_index][0]-vertices[start_index][0]))
            edge_rows.append(
                {
                    "edge_id": f"{panel_id}.edge_{edge_index}",
                    "edge_index": edge_index,
                    "endpoints": [start_index, end_index],
                    "start_cm": list(vertices[start_index]),
                    "end_cm": list(vertices[end_index]),
                    "label": edge.get("label"),
                    "curve": curve,
                    "length_cm": _curve_length(vertices[start_index], vertices[end_index], curve),
                    "chord_direction_deg": float(chord),
                    "start_tangent_deg": start_tangent,
                    "end_tangent_deg": end_tangent,
                }
            )
        values = np.asarray(dense, dtype=float)
        minimum, maximum = values.min(axis=0), values.max(axis=0)
        bbox = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
        pack_items.append(_PackItem(panel_id, panel_index, bbox))
        all_spans.extend((bbox[2]-bbox[0], bbox[3]-bbox[1]))
        matrix = Rotation.from_euler("XYZ", panel.get("rotation", [0.0, 0.0, 0.0]), degrees=True).as_matrix()
        interim.append(
            {
                "panel_id": panel_id,
                "source_order_index": panel_index,
                "source_label": panel.get("label"),
                "vertices_cm": [list(value) for value in vertices],
                "edges": edge_rows,
                "local_curve_bbox_cm": list(bbox),
                "initial_3d_placement": {
                    "translation_cm": [float(value) for value in panel.get("translation", [0.0, 0.0, 0.0])],
                    "rotation_euler_xyz_deg": [float(value) for value in panel.get("rotation", [0.0, 0.0, 0.0])],
                    "x_axis": [float(value) for value in matrix @ np.asarray([1.0, 0.0, 0.0])],
                    "y_axis": [float(value) for value in matrix @ np.asarray([0.0, 1.0, 0.0])],
                    "normal": [float(value) for value in matrix @ np.asarray([0.0, 0.0, 1.0])],
                },
            }
        )
    gap_cm = max(all_spans) * float(gap_fraction)
    placements, packed_width, packed_height = _pack_shelves(pack_items, gap_cm)
    border_px = max(24, round(image_size * 0.035))
    scale = (image_size - 2 * border_px) / max(packed_width, packed_height)
    x_offset = border_px + 0.5 * (image_size - 2 * border_px - packed_width * scale)
    y_offset = border_px + 0.5 * (image_size - 2 * border_px - packed_height * scale)
    packing = {
        panel_id: {
            "translation_cm": [float(position[0]), float(position[1])],
            "scale_px_per_cm": float(scale),
            "canvas_offset_px": [float(x_offset), float(y_offset)],
            "canvas_size_px": [image_size, image_size],
            "rotation_deg": 0.0,
        }
        for panel_id, position in placements.items()
    }
    panels = []
    for panel in interim:
        transform = packing[panel["panel_id"]]
        for edge in panel["edges"]:
            edge["packed_start_px"] = list(_transform_point(edge["start_cm"], transform))
            edge["packed_end_px"] = list(_transform_point(edge["end_cm"], transform))
            edge["packed_start_uv"] = _packed_uv(edge["packed_start_px"], image_size)
            edge["packed_end_uv"] = _packed_uv(edge["packed_end_px"], image_size)
            edge["packed_controls_px"] = [list(_transform_point(point, transform)) for point in edge["curve"].get("controls_cm", [])]
        bbox = panel["local_curve_bbox_cm"]
        corners = [_transform_point((bbox[0], bbox[1]), transform), _transform_point((bbox[2], bbox[3]), transform)]
        panel["packed_bbox_px"] = [min(corners[0][0], corners[1][0]), min(corners[0][1], corners[1][1]), max(corners[0][0], corners[1][0]), max(corners[0][1], corners[1][1])]
        panel["render_color_rgb"] = list(_panel_color(panel["source_order_index"], len(interim)))
        panels.append(panel)
    output_dir = Path(output_dir)
    label_path = output_dir / "labels.json"
    pattern_path = output_dir / "pattern.png"
    overlay_path = output_dir / "pattern_overlay.png"
    label = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "category": category,
        "source_dataset": "GarmentCodeData v2",
        "source_license": "CC BY 4.0",
        "source_specification_sha256": _sha256(specification_path),
        "coordinate_contract": {
            "source_units": "cm",
            "panel_coordinates": "panel_local_x_right_y_up",
            "packed_coordinates": "image_x_right_y_down",
            "packing_is_display_only": True,
            "lossless_truth": "vertices, edge endpoints, native curvature type/params, stitches, and initial 3D placement come from source specification; lengths/directions/tangents are deterministic geometry derivatives",
        },
        "panels": panels,
        "stitches": source.get("stitches", []),
        "packing": packing,
        "pattern_image": str(pattern_path.as_posix()),
        "overlay_image": str(overlay_path.as_posix()) if render_overlay else None,
        "views": [
            {
                "view_index": index,
                "view_label": str(view_labels[index]),
                "path": str(Path(path).as_posix()),
                "sha256": (
                    _sha256(Path(path))
                    if Path(path).is_file()
                    else None
                ),
                "available": Path(path).is_file(),
            }
            for index, path in enumerate(view_paths)
        ],
    }
    if not allow_missing_views and not all(value["available"] for value in label["views"]):
        missing = [value["path"] for value in label["views"] if not value["available"]]
        raise FileNotFoundError(f"four-view bundle is incomplete: {missing}")
    validation = validate_exact_label(label)
    label["validation"] = validation
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _render_label(label, pattern_path, overlay=False, size=image_size)
    if render_overlay:
        _render_label(label, overlay_path, overlay=True, size=image_size)
    return {
        "sample_id": sample_id,
        "category": category,
        "label_path": str(label_path.as_posix()),
        "pattern_path": str(pattern_path.as_posix()),
        "overlay_path": str(overlay_path.as_posix()) if render_overlay else None,
        "view_paths": [str(Path(value).as_posix()) for value in view_paths],
        "panel_count": len(panels),
        "edge_count": sum(len(panel["edges"]) for panel in panels),
        "curve_type_counts": {
            kind: sum(edge["curve"]["type"] == kind for panel in panels for edge in panel["edges"])
            for kind in CURVE_TYPES
        },
        "validation": validation,
    }


def _boxes_overlap(first: Sequence[float], second: Sequence[float], tolerance: float = 0.5) -> bool:
    return not (
        first[2] <= second[0] + tolerance
        or second[2] <= first[0] + tolerance
        or first[3] <= second[1] + tolerance
        or second[3] <= first[1] + tolerance
    )


def validate_exact_label(label: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    seen_panels = set()
    edge_count = 0
    for panel in label["panels"]:
        panel_id = panel["panel_id"]
        if panel_id in seen_panels:
            failures.append(f"duplicate_panel:{panel_id}")
        seen_panels.add(panel_id)
        vertices = panel["vertices_cm"]
        seen_edges = set()
        for edge in panel["edges"]:
            edge_count += 1
            if edge["edge_id"] in seen_edges:
                failures.append(f"duplicate_edge:{edge['edge_id']}")
            seen_edges.add(edge["edge_id"])
            first, second = edge["endpoints"]
            if edge["start_cm"] != vertices[first] or edge["end_cm"] != vertices[second]:
                failures.append(f"endpoint_mismatch:{edge['edge_id']}")
            sampled = sample_curve(edge["start_cm"], edge["end_cm"], edge["curve"], samples=9)
            if math.dist(sampled[0], edge["start_cm"]) > 1e-6 or math.dist(sampled[-1], edge["end_cm"]) > 1e-6:
                failures.append(f"curve_endpoint_mismatch:{edge['edge_id']}")
            if not math.isfinite(float(edge["length_cm"])) or float(edge["length_cm"]) <= 0.0:
                failures.append(f"invalid_length:{edge['edge_id']}")
    panels = list(label["panels"])
    for index, panel in enumerate(panels):
        for other in panels[index + 1 :]:
            if _boxes_overlap(panel["packed_bbox_px"], other["packed_bbox_px"]):
                failures.append(f"packed_overlap:{panel['panel_id']}:{other['panel_id']}")
    view_failures = [value["path"] for value in label["views"] if not Path(value["path"]).is_file()]
    failures.extend(f"missing_view:{value}" for value in view_failures)
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "panel_count": len(panels),
        "edge_count": edge_count,
        "packed_non_overlap": not any(value.startswith("packed_overlap:") for value in failures),
        "all_views_present": not view_failures,
    }


def load_exact_label(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported exact-label schema: {value.get('schema_version')!r}")
    return value


def render_exact_overlay(label_path: Path, destination: Path, *, size: int = 1600) -> Path:
    label = load_exact_label(label_path)
    return _render_label(label, Path(destination), overlay=True, size=size)
