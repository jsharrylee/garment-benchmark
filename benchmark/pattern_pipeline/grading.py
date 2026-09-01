from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .schema import Edge, Panel, PatternDocument, Placement


MEASUREMENT_KEYS = {
    "top": ("bust", "height"),
    "pants": ("hips", "leg_length"),
    "shorts": ("hips", "leg_length"),
    "skirt": ("hips", "leg_length"),
    "dress": ("bust", "height"),
    "jumpsuit": ("hips", "height"),
}


def _ratio(target: Mapping[str, float], source: Mapping[str, float], key: str, *, fallback: float = 1.0) -> float:
    numerator = float(target.get(key, target.get(f"_{key}", 0.0)) or 0.0)
    denominator = float(source.get(key, source.get(f"_{key}", 0.0)) or 0.0)
    return numerator / denominator if numerator > 0.0 and denominator > 0.0 else fallback


def grading_scales(
    category: str,
    source: Mapping[str, float],
    target: Mapping[str, float],
    *,
    minimum: float = 0.72,
    maximum: float = 1.38,
) -> tuple[float, float, float]:
    """Return width, vertical and depth scales for body-aware pattern grading.

    Circumference drives the horizontal pattern direction.  Height or leg
    length drives the vertical direction.  Depth changes more conservatively
    because GarmentCode placement depth is an initialization offset rather than
    a flat-pattern dimension.
    """

    width_key, length_key = MEASUREMENT_KEYS.get(category, ("bust", "height"))
    width = _ratio(target, source, width_key)
    if category in {"pants", "shorts", "skirt", "jumpsuit"}:
        waist = _ratio(target, source, "waist", fallback=width)
        width = 0.7 * width + 0.3 * waist
    length = _ratio(target, source, length_key)
    if category == "shorts":
        length = 0.6 * length + 0.4
    width = max(minimum, min(maximum, width))
    length = max(minimum, min(maximum, length))
    depth = max(minimum, min(maximum, width**0.65))
    return width, length, depth


def grade_pattern(
    document: PatternDocument,
    *,
    category: str,
    source_measurements: Mapping[str, float],
    target_measurements: Mapping[str, float],
    panel_mesh_spacing_cm: float | None = None,
) -> PatternDocument:
    """Grade a retrieved pattern while preserving topology and stitch IDs."""

    width, length, depth = grading_scales(category, source_measurements, target_measurements)
    panels = []
    for panel in document.panels:
        edges = tuple(
            replace(
                edge,
                points=tuple((float(point[0]) * width, float(point[1]) * length) for point in edge.points),
            )
            for edge in panel.edges
        )
        placement = panel.placement
        if placement is not None:
            placement = Placement(
                origin=(
                    float(placement.origin[0]) * width,
                    float(placement.origin[1]) * length,
                    float(placement.origin[2]) * depth,
                ),
                x_axis=placement.x_axis,
                y_axis=placement.y_axis,
                normal=placement.normal,
                method="garmentcode_body_measurement_grading",
            )
        panels.append(replace(panel, edges=edges, placement=placement))
    annotations = {
        **document.annotations,
        "refinement_status": "body_measurement_graded_anchor",
        "pin_strategy": "edge_midpoints",
        "body_grading": {
            "category": category,
            "width_scale": width,
            "length_scale": length,
            "placement_depth_scale": depth,
            "source_measurements_cm": dict(source_measurements),
            "target_measurements_cm": dict(target_measurements),
            "topology_preserved": True,
            "stitch_graph_preserved": True,
        },
    }
    if panel_mesh_spacing_cm is not None:
        annotations["panel_mesh_spacing_cm"] = float(panel_mesh_spacing_cm)
    return replace(
        document,
        pattern_id=f"{document.pattern_id}_graded",
        generator=f"{document.generator} + MPFB body measurement grading",
        panels=tuple(panels),
        annotations=annotations,
    )
