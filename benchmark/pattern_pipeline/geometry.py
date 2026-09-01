from __future__ import annotations

import math
from collections.abc import Iterable

from .schema import Edge, Panel, Point2


def distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(points: Iterable[Point2]) -> float:
    points = tuple(points)
    return sum(distance(a, b) for a, b in zip(points, points[1:]))


def boundary_points(panel: Panel) -> tuple[Point2, ...]:
    points: list[Point2] = []
    for edge in panel.edges:
        points.extend(edge.points if not points else edge.points[1:])
    return tuple(points)


def signed_area(points: Iterable[Point2]) -> float:
    points = tuple(points)
    if len(points) < 3:
        return 0.0
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def panel_diagonal(panel: Panel) -> float:
    points = boundary_points(panel)
    if not points:
        return 0.0
    xs, ys = zip(*points)
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def edge_by_id(panel: Panel, edge_id: str) -> Edge | None:
    return next((edge for edge in panel.edges if edge.id == edge_id), None)


def _orientation(a: Point2, b: Point2, c: Point2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _strict_intersection(a: Point2, b: Point2, c: Point2, d: Point2, epsilon: float = 1e-9) -> bool:
    return _orientation(a, b, c) * _orientation(a, b, d) < -epsilon and _orientation(c, d, a) * _orientation(c, d, b) < -epsilon


def self_intersections(panel: Panel) -> int:
    points = boundary_points(panel)
    segments = list(zip(points, points[1:]))
    if len(points) > 2 and distance(points[-1], points[0]) > 1e-9:
        segments.append((points[-1], points[0]))
    count = 0
    for i, first in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if abs(i - j) <= 1 or {i, j} == {0, len(segments) - 1}:
                continue
            count += int(_strict_intersection(*first, *segments[j]))
    return count
