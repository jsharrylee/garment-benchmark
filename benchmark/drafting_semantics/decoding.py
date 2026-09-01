from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Integral

from .schema import EDGE_ROLES, PanelAnnotation


def _shared_vertex(panel: PanelAnnotation, first_role: str, second_role: str, roles: Sequence[str]) -> int | None:
    first = [edge for edge, role in zip(panel.edges, roles, strict=True) if role == first_role]
    second = [edge for edge, role in zip(panel.edges, roles, strict=True) if role == second_role]
    shared = {
        vertex
        for edge in first
        for vertex in edge.endpoints
        if any(vertex in other.endpoints for other in second)
    }
    return min(shared) if shared else None


def _normalize_roles(panel: PanelAnnotation, edge_roles: Sequence[str | int]) -> tuple[str, ...]:
    if len(edge_roles) != len(panel.edges):
        raise ValueError("edge_roles must have one item per panel edge")
    roles = tuple(EDGE_ROLES[int(value)] if isinstance(value, Integral) else value for value in edge_roles)
    unknown = set(roles) - set(EDGE_ROLES)
    if unknown:
        raise ValueError(f"unknown edge roles: {sorted(unknown)}")
    return roles


def decode_named_landmarks(panel: PanelAnnotation, edge_roles: Sequence[str | int]) -> dict[str, tuple[float, float]]:
    """Decode textbook bodice points from semantic boundary-edge roles.

    The decoder deliberately returns no guess when the required role junction is
    absent.  This keeps the learned model's errors visible instead of silently
    repairing them with source-specific heuristics.
    """

    roles = _normalize_roles(panel, edge_roles)
    if panel.role not in {"front_bodice", "back_bodice"}:
        return {}

    center_role = "center_front" if panel.role == "front_bodice" else "center_back"
    center_name = "FNP" if panel.role == "front_bodice" else "BNP"
    requests = (
        (center_name, "neckline", center_role),
        ("SNP", "neckline", "shoulder"),
        ("SP", "shoulder", "armhole"),
    )
    output: dict[str, tuple[float, float]] = {}
    for name, first_role, second_role in requests:
        vertex = _shared_vertex(panel, first_role, second_role, roles)
        if vertex is not None:
            output[name] = panel.vertices_cm[vertex]
    return output


def decode_darts(panel: PanelAnnotation, edge_roles: Sequence[str | int]) -> tuple[dict, ...]:
    """Decode adjacent predicted dart-leg pairs and their measurable geometry."""

    roles = _normalize_roles(panel, edge_roles)
    candidates = [index for index, role in enumerate(roles) if role == "dart_leg"]
    used: set[int] = set()
    output = []
    for first in candidates:
        if first in used:
            continue
        first_vertices = set(panel.edges[first].endpoints)
        partners = [
            second
            for second in candidates
            if second > first and second not in used and first_vertices & set(panel.edges[second].endpoints)
        ]
        if not partners:
            continue
        second = min(partners, key=lambda value: abs(value - first))
        shared = first_vertices & set(panel.edges[second].endpoints)
        apex_index = min(shared)
        first_mouth = next(value for value in panel.edges[first].endpoints if value != apex_index)
        second_mouth = next(value for value in panel.edges[second].endpoints if value != apex_index)
        apex = panel.vertices_cm[apex_index]
        base = panel.vertices_cm[first_mouth], panel.vertices_cm[second_mouth]
        vx, vy = base[1][0] - base[0][0], base[1][1] - base[0][1]
        denominator = max(math.hypot(vx, vy), 1e-9)
        depth = abs(vy * apex[0] - vx * apex[1] + base[1][0] * base[0][1] - base[1][1] * base[0][0]) / denominator
        output.append(
            {
                "leg_edge_indices": [first, second],
                "apex_vertex_index": apex_index,
                "apex_cm": list(apex),
                "base_cm": [list(base[0]), list(base[1])],
                "intake_cm": math.dist(*base),
                "depth_cm": depth,
                "kind": "waist_dart" if abs(vx) >= abs(vy) else "side_dart",
            }
        )
        used.update((first, second))
    return tuple(output)


def decode_path_measurements(panel: PanelAnnotation, edge_roles: Sequence[str | int]) -> dict[str, float]:
    """Measure selected semantic paths using source curve arc lengths."""

    roles = _normalize_roles(panel, edge_roles)
    output: dict[str, float] = {}
    shoulder = [edge for edge, role in zip(panel.edges, roles, strict=True) if role == "shoulder"]
    neckline = [edge for edge, role in zip(panel.edges, roles, strict=True) if role == "neckline"]
    if shoulder:
        output["shoulder_path_length_cm"] = sum(edge.length_cm for edge in shoulder)
        first, last = shoulder[0].start_cm, shoulder[-1].end_cm
        output["shoulder_chord_angle_deg"] = math.degrees(math.atan2(last[1] - first[1], last[0] - first[0]))
    if neckline:
        output["neckline_arc_length_cm"] = sum(edge.length_cm for edge in neckline)
    return output


def landmark_error_summary(
    panel: PanelAnnotation,
    predicted_edge_roles: Sequence[str | int],
    *,
    names: frozenset[str] = frozenset({"FNP", "BNP", "SNP", "SP"}),
) -> dict[str, float | int]:
    """Measure decoded point availability and normalized localization error."""

    target = {item.name: item.xy_cm for item in panel.landmarks if item.name in names and item.training_eligible}
    predicted = decode_named_landmarks(panel, predicted_edge_roles)
    available = set(target) & set(predicted)
    xs = [point[0] for point in panel.vertices_cm]
    ys = [point[1] for point in panel.vertices_cm]
    diagonal = max(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 1e-9)
    distances = [
        ((predicted[name][0] - target[name][0]) ** 2 + (predicted[name][1] - target[name][1]) ** 2) ** 0.5 / diagonal
        for name in available
    ]
    exact = sum(distance <= 1e-9 for distance in distances)
    return {
        "target_count": len(target),
        "decoded_count": len(available),
        "exact_count": exact,
        "normalized_distance_sum": float(sum(distances)),
    }
