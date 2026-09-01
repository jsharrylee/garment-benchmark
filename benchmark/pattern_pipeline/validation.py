from __future__ import annotations

from dataclasses import asdict, dataclass

from .geometry import distance, edge_by_id, panel_diagonal, polyline_length, self_intersections, signed_area, boundary_points
from .schema import PatternDocument


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    subject: str
    value: float | int | str | None = None


@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    issues: tuple[Issue, ...]
    metrics: dict[str, float | int]

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "issues": [asdict(issue) for issue in self.issues], "metrics": self.metrics}


def validate_pattern(
    document: PatternDocument,
    *,
    closure_ratio_limit: float = 0.02,
    seam_length_ratio_warning: float = 0.25,
) -> ValidationReport:
    issues: list[Issue] = []
    panel_map = {panel.id: panel for panel in document.panels}
    edge_keys: set[tuple[str, str]] = set()
    closure_gaps: list[float] = []
    used_sides: set[tuple[str, str]] = set()

    if not document.panels:
        issues.append(Issue("error", "NO_PANELS", document.pattern_id))
    if len(panel_map) != len(document.panels):
        issues.append(Issue("error", "DUPLICATE_PANEL_ID", document.pattern_id))

    for panel in document.panels:
        diagonal = panel_diagonal(panel)
        if len(panel.edges) < 3:
            issues.append(Issue("error", "TOO_FEW_EDGES", panel.id, len(panel.edges)))
        for edge in panel.edges:
            key = (panel.id, edge.id)
            if key in edge_keys:
                issues.append(Issue("error", "DUPLICATE_EDGE_ID", panel.id, edge.id))
            edge_keys.add(key)
            if len(edge.points) < 2 or polyline_length(edge.points) <= 1e-8:
                issues.append(Issue("error", "DEGENERATE_EDGE", edge.id))
            if any(not (-1e100 < coordinate < 1e100) for point in edge.points for coordinate in point):
                issues.append(Issue("error", "NONFINITE_COORDINATE", edge.id))
        if panel.edges:
            for edge, following in zip(panel.edges, panel.edges[1:] + panel.edges[:1]):
                gap = distance(edge.points[-1], following.points[0])
                closure_gaps.append(gap)
                if diagonal <= 0 or gap > closure_ratio_limit * diagonal:
                    issues.append(Issue("error", "BOUNDARY_GAP", f"{edge.id}->{following.id}", gap))
        area = abs(signed_area(boundary_points(panel)))
        if area <= max(1e-8, diagonal * diagonal * 1e-4):
            issues.append(Issue("error", "DEGENERATE_PANEL_AREA", panel.id, area))
        intersections = self_intersections(panel)
        if intersections:
            issues.append(Issue("error", "SELF_INTERSECTION", panel.id, intersections))

    seam_mismatch: list[float] = []
    for stitch in document.stitches:
        sides = (stitch.side_a, stitch.side_b)
        resolved = []
        for side in sides:
            panel = panel_map.get(side.panel_id)
            edge = edge_by_id(panel, side.edge_id) if panel else None
            if edge is None:
                issues.append(Issue("error", "INVALID_STITCH_REFERENCE", stitch.id, f"{side.panel_id}/{side.edge_id}"))
            else:
                resolved.append(edge)
            key = (side.panel_id, side.edge_id)
            if key in used_sides:
                issues.append(Issue("error", "EDGE_STITCHED_MORE_THAN_ONCE", stitch.id, "/".join(key)))
            used_sides.add(key)
        if len(resolved) == 2:
            lengths = [polyline_length(edge.points) for edge in resolved]
            ratio = abs(lengths[0] - lengths[1]) / max(lengths)
            seam_mismatch.append(ratio)
            if ratio > seam_length_ratio_warning:
                issues.append(Issue("warning", "SEAM_LENGTH_MISMATCH", stitch.id, ratio))

    errors = sum(issue.severity == "error" for issue in issues)
    return ValidationReport(
        accepted=errors == 0,
        issues=tuple(issues),
        metrics={
            "panel_count": len(document.panels),
            "edge_count": sum(len(panel.edges) for panel in document.panels),
            "stitch_count": len(document.stitches),
            "error_count": errors,
            "warning_count": sum(issue.severity == "warning" for issue in issues),
            "mean_closure_gap_cm": sum(closure_gaps) / len(closure_gaps) if closure_gaps else 0.0,
            "max_closure_gap_cm": max(closure_gaps, default=0.0),
            "mean_seam_length_mismatch": sum(seam_mismatch) / len(seam_mismatch) if seam_mismatch else 0.0,
        },
    )
