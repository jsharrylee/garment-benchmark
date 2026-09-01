from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .geometry import distance, panel_diagonal
from .schema import Edge, Panel, PatternDocument
from .validation import ValidationReport, validate_pattern


@dataclass(frozen=True)
class RepairReceipt:
    hypothesis: str
    accepted: bool
    changed_junctions: int
    before: dict
    after: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _error_score(report: ValidationReport) -> tuple[int, float]:
    return int(report.metrics["error_count"]), float(report.metrics["mean_closure_gap_cm"])


def snap_boundary_junctions(
    document: PatternDocument,
    *,
    max_gap_ratio: float = 0.05,
) -> tuple[PatternDocument, RepairReceipt]:
    """One bounded repair hypothesis: snap only adjacent generated edge endpoints."""
    before = validate_pattern(document)
    changed = 0
    new_panels: list[Panel] = []
    for panel in document.panels:
        if not panel.edges:
            new_panels.append(panel)
            continue
        diagonal = panel_diagonal(panel)
        mutable = [[list(point) for point in edge.points] for edge in panel.edges]
        for index in range(len(panel.edges)):
            next_index = (index + 1) % len(panel.edges)
            end = tuple(mutable[index][-1])
            start = tuple(mutable[next_index][0])
            gap = distance(end, start)
            if 1e-9 < gap <= max_gap_ratio * diagonal:
                midpoint = [(end[0] + start[0]) / 2.0, (end[1] + start[1]) / 2.0]
                mutable[index][-1] = midpoint
                mutable[next_index][0] = midpoint
                changed += 1
        edges = tuple(replace(edge, points=tuple(tuple(float(v) for v in point) for point in points)) for edge, points in zip(panel.edges, mutable))
        new_panels.append(replace(panel, edges=edges))
    candidate = replace(document, panels=tuple(new_panels))
    after = validate_pattern(candidate)
    accepted = changed > 0 and _error_score(after) < _error_score(before)
    result = candidate if accepted else document
    receipt = RepairReceipt(
        hypothesis="snap adjacent generated edge endpoints within 5% of panel diagonal",
        accepted=accepted,
        changed_junctions=changed,
        before=before.to_dict(),
        after=after.to_dict(),
    )
    return result, receipt
