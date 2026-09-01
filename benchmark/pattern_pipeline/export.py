from __future__ import annotations

import json
from pathlib import Path

from .geometry import boundary_points, polyline_length
from .schema import PatternDocument


def _layouts(document: PatternDocument, gap: float = 5.0) -> dict[str, tuple[float, float]]:
    offsets: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for panel in document.panels:
        points = boundary_points(panel)
        if not points:
            offsets[panel.id] = (cursor, 0.0)
            continue
        xs, ys = zip(*points)
        offsets[panel.id] = (cursor - min(xs), -min(ys))
        cursor += max(xs) - min(xs) + gap
    return offsets


def write_stitch_manifest(document: PatternDocument, path: Path) -> None:
    panel_map = {panel.id: panel for panel in document.panels}
    edge_map = {(panel.id, edge.id): edge for panel in document.panels for edge in panel.edges}
    payload = {
        "pattern_id": document.pattern_id,
        "units": document.units,
        "stitches": [
            {
                **stitch.__dict__,
                "side_a": stitch.side_a.__dict__,
                "side_b": stitch.side_b.__dict__,
                "side_a_length": polyline_length(edge_map[(stitch.side_a.panel_id, stitch.side_a.edge_id)].points),
                "side_b_length": polyline_length(edge_map[(stitch.side_b.panel_id, stitch.side_b.edge_id)].points),
            }
            for stitch in document.stitches
        ],
        "panel_ids": sorted(panel_map),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_svg(document: PatternDocument, path: Path) -> None:
    layouts = _layouts(document)
    laid_out = []
    for panel in document.panels:
        ox, oy = layouts[panel.id]
        laid_out.append((panel, [(x + ox, y + oy) for x, y in boundary_points(panel)]))
    all_points = [point for _, points in laid_out for point in points]
    width = max((x for x, _ in all_points), default=1.0) + 2.0
    height = max((y for _, y in all_points), default=1.0) + 2.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -1 {width:.6f} {height:.6f}">', '<g fill="none" stroke="black" stroke-width="0.15">']
    for panel, points in laid_out:
        encoded = " ".join(f"{x:.6f},{height-y:.6f}" for x, y in points)
        parts.append(f'<polygon id="{panel.id}" points="{encoded}"/>')
    parts.extend(["</g>", "</svg>"])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_dxf_r12(document: PatternDocument, path: Path) -> None:
    """Write generic R12 closed polylines; stitch semantics remain in stitches.json."""
    layouts = _layouts(document)
    lines = ["0", "SECTION", "2", "HEADER", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for panel in document.panels:
        ox, oy = layouts[panel.id]
        lines.extend(["0", "POLYLINE", "8", panel.id[:31], "66", "1", "70", "1"])
        for x, y in boundary_points(panel):
            lines.extend(["0", "VERTEX", "8", panel.id[:31], "10", f"{x + ox:.9f}", "20", f"{y + oy:.9f}", "30", "0.0"])
        lines.extend(["0", "SEQEND"])
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def export_bundle(document: PatternDocument, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "canonical_json": directory / "pattern.json",
        "stitch_manifest": directory / "stitches.json",
        "svg": directory / "pattern.svg",
        "dxf_outline": directory / "pattern_outline_r12.dxf",
    }
    document.write_json(paths["canonical_json"])
    write_stitch_manifest(document, paths["stitch_manifest"])
    write_svg(document, paths["svg"])
    write_dxf_r12(document, paths["dxf_outline"])
    return paths
