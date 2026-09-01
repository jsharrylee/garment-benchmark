from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from .schema import PatternDocument


@dataclass(frozen=True)
class SewingMeshPlan:
    vertices: tuple[tuple[float, float, float], ...]
    panel_loops: tuple[tuple[int, ...], ...]
    panel_faces: tuple[tuple[int, int, int], ...]
    sewing_edges: tuple[tuple[int, int], ...]
    edge_vertices: dict[tuple[str, str], tuple[int, ...]]
    pinned_vertices: tuple[int, ...] = ()


def _sample_open_edge(points: tuple[tuple[float, float], ...], maximum: int) -> tuple[tuple[float, float], ...]:
    if len(points) <= maximum + 1:
        return points[:-1]
    indices = np.unique(np.linspace(0, len(points) - 2, maximum, dtype=int))
    return tuple(points[int(index)] for index in indices)


def _schema_to_blender(point: np.ndarray) -> tuple[float, float, float]:
    # The pretrained garment coordinate convention is X horizontal, Y up, Z depth.
    # Blender is X horizontal, Y depth, Z up.
    return float(point[0] * 0.01), float(point[2] * 0.01), float(point[1] * 0.01)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray, *, tolerance: float = 1e-7) -> bool:
    """Return True inside or on the boundary of a simple polygon."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        segment = end - start
        length_squared = float(segment @ segment)
        if length_squared:
            fraction = max(0.0, min(1.0, float((point - start) @ segment) / length_squared))
            if float(np.linalg.norm(point - (start + fraction * segment))) <= tolerance:
                return True
        if (start[1] > y) != (end[1] > y):
            crossing_x = float((end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0])
            if x < crossing_x:
                inside = not inside
    return inside


def _dense_panel_faces(
    boundary: list[tuple[float, float]],
    boundary_indices: list[int],
    spacing: float,
    placement: tuple[np.ndarray, np.ndarray, np.ndarray],
    vertices: list[tuple[float, float, float]],
) -> list[tuple[int, int, int]]:
    polygon = np.asarray(boundary, dtype=float)
    local_points = [tuple(point) for point in polygon]
    global_indices = list(boundary_indices)
    minimum, maximum = polygon.min(axis=0), polygon.max(axis=0)
    if spacing > 0:
        for x in np.arange(minimum[0] + spacing, maximum[0], spacing):
            for y in np.arange(minimum[1] + spacing, maximum[1], spacing):
                point = np.asarray((x, y), dtype=float)
                if _point_in_polygon(point, polygon):
                    origin, x_axis, y_axis = placement
                    vertices.append(_schema_to_blender(origin + x * x_axis + y * y_axis))
                    local_points.append((float(x), float(y)))
                    global_indices.append(len(vertices) - 1)
    if len(local_points) < 3:
        return []
    triangulation = Delaunay(np.asarray(local_points, dtype=float))
    faces = []
    for simplex in triangulation.simplices:
        triangle = np.asarray([local_points[int(index)] for index in simplex])
        probes = [triangle.mean(axis=0), *(0.5 * (triangle[index] + triangle[(index + 1) % 3]) for index in range(3))]
        if all(_point_in_polygon(probe, polygon, tolerance=1e-5) for probe in probes):
            faces.append(tuple(global_indices[int(index)] for index in simplex))
    return faces


def build_sewing_mesh_plan(document: PatternDocument, *, samples_per_edge: int = 8) -> SewingMeshPlan:
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least two")
    vertices: list[tuple[float, float, float]] = []
    loops: list[tuple[int, ...]] = []
    faces: list[tuple[int, int, int]] = []
    edge_vertices: dict[tuple[str, str], tuple[int, ...]] = {}

    for panel in document.panels:
        panel_indices: list[int] = []
        starts: list[int] = []
        sampled_edges: list[tuple[tuple[float, float], ...]] = []
        placement = panel.placement
        origin = np.asarray(placement.origin if placement else (0.0, 0.0, 0.0), dtype=float)
        x_axis = np.asarray(placement.x_axis if placement else (1.0, 0.0, 0.0), dtype=float)
        y_axis = np.asarray(placement.y_axis if placement else (0.0, 1.0, 0.0), dtype=float)
        for edge in panel.edges:
            sampled = _sample_open_edge(edge.points, samples_per_edge)
            sampled_edges.append(sampled)
            starts.append(len(vertices))
            for u, v in sampled:
                vertices.append(_schema_to_blender(origin + u * x_axis + v * y_axis))
                panel_indices.append(len(vertices) - 1)
        loops.append(tuple(panel_indices))
        spacing = float(document.annotations.get("panel_mesh_spacing_cm", 0.0) or 0.0)
        if spacing > 0:
            boundary = [point for sampled in sampled_edges for point in sampled]
            faces.extend(_dense_panel_faces(boundary, panel_indices, spacing, (origin, x_axis, y_axis), vertices))
        for edge_index, edge in enumerate(panel.edges):
            start = starts[edge_index]
            count = len(sampled_edges[edge_index])
            following = starts[(edge_index + 1) % len(starts)]
            edge_vertices[(panel.id, edge.id)] = tuple(range(start, start + count)) + (following,)

    sewing_edges: list[tuple[int, int]] = []
    for stitch in document.stitches:
        side_a = edge_vertices[(stitch.side_a.panel_id, stitch.side_a.edge_id)]
        side_b = edge_vertices[(stitch.side_b.panel_id, stitch.side_b.edge_id)]
        if stitch.side_a.reversed:
            side_a = tuple(reversed(side_a))
        if stitch.side_b.reversed:
            side_b = tuple(reversed(side_b))
        count = max(len(side_a), len(side_b))
        indices_a = np.rint(np.linspace(0, len(side_a) - 1, count)).astype(int)
        indices_b = np.rint(np.linspace(0, len(side_b) - 1, count)).astype(int)
        sewing_edges.extend((side_a[int(a)], side_b[int(b)]) for a, b in zip(indices_a, indices_b, strict=True))

    edge_labels = document.annotations.get("edge_labels", {})
    pin_semantics = tuple(str(value).lower() for value in document.annotations.get("pin_semantics", []))
    pin_strategy = str(document.annotations.get("pin_strategy", "edge_all"))
    pinned: set[int] = set()
    for key, label in edge_labels.items():
        if any(semantic in str(label).lower() for semantic in pin_semantics):
            panel_id, edge_id = key.split("/", 1)
            candidates = edge_vertices.get((panel_id, edge_id), ())
            if pin_strategy == "edge_midpoints" and candidates:
                pinned.add(candidates[len(candidates) // 2])
            else:
                pinned.update(candidates)

    preserve_absolute_placement = bool(document.annotations.get("preserve_absolute_placement", False))
    if vertices and not preserve_absolute_placement:
        center = np.mean(np.asarray(vertices), axis=0)
        vertices = [tuple(np.asarray(vertex) - center) for vertex in vertices]
    return SewingMeshPlan(tuple(vertices), tuple(loops), tuple(faces), tuple(sewing_edges), edge_vertices, tuple(sorted(pinned)))


def mesh_plan_to_dict(plan: SewingMeshPlan) -> dict[str, Any]:
    return {
        "vertices": [list(value) for value in plan.vertices],
        "panel_loops": [list(value) for value in plan.panel_loops],
        "panel_faces": [list(value) for value in plan.panel_faces],
        "sewing_edges": [list(value) for value in plan.sewing_edges],
        "edge_vertices": {f"{panel}/{edge}": list(value) for (panel, edge), value in plan.edge_vertices.items()},
        "pinned_vertices": list(plan.pinned_vertices),
    }
