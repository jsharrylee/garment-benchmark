from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark.drafting_semantics.tshirt_learning import edge_length_cm
from benchmark.drafting_semantics.tshirt_schema import TShirtTraceRecord


COLORS = {
    "neckline": "#9c27b0",
    "shoulder": "#00897b",
    "armhole": "#e91e63",
    "side_seam": "#ef6c00",
    "center_front": "#3949ab",
    "center_back": "#3949ab",
    "hemline": "#546e7a",
    "sleeve_head": "#e91e63",
    "sleeve_underarm": "#ef6c00",
    "sleeve_hem": "#546e7a",
    "other": "#757575",
}


def _pick(records, count: int):
    wanted = ("train", "iid_test", "test_body", "test_design", "test_double", "unseen_body", "unseen_design")
    groups = {split: [record for record in records if record.split == split] for split in wanted}
    output = []
    cursor = 0
    while len(output) < count and any(groups.values()):
        split = wanted[cursor % len(wanted)]
        if groups[split]:
            index = (len(output) * 17 + cursor * 7) % len(groups[split])
            output.append(groups[split].pop(index))
        cursor += 1
    return tuple(output)


def _json_values(path: Path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _read_selected_records(path: Path, count: int):
    """Select in two streaming passes without retaining all trace DAGs."""

    metadata = tuple(
        SimpleNamespace(sample_id=str(value["sample_id"]), split=str(value["split"]))
        for value in _json_values(path)
    )
    selection = _pick(metadata, count)
    order = {item.sample_id: index for index, item in enumerate(selection)}
    selected = []
    for value in _json_values(path):
        if value.get("sample_id") in order:
            record = TShirtTraceRecord.from_dict(value)
            record.validate()
            selected.append(record)
    return tuple(sorted(selected, key=lambda item: order[item.sample_id]))


def _path(edge, transform):
    geometry = edge.geometry
    start = transform(geometry.start_cm)
    end = transform(geometry.end_cm)
    if geometry.kind in {"quadratic_bezier", "bezier"} and len(geometry.control_points_cm) == 1:
        control = transform(geometry.control_points_cm[0])
        return f"M {start[0]:.2f},{start[1]:.2f} Q {control[0]:.2f},{control[1]:.2f} {end[0]:.2f},{end[1]:.2f}"
    if geometry.kind in {"cubic_bezier", "bezier"} and len(geometry.control_points_cm) >= 2:
        first = transform(geometry.control_points_cm[0])
        second = transform(geometry.control_points_cm[1])
        return (
            f"M {start[0]:.2f},{start[1]:.2f} C {first[0]:.2f},{first[1]:.2f} "
            f"{second[0]:.2f},{second[1]:.2f} {end[0]:.2f},{end[1]:.2f}"
        )
    # Arc display is sampled only for the review board.  The JSON truth keeps
    # its exact source parameters.
    if geometry.kind == "arc" and geometry.control_points_cm:
        control = transform(geometry.control_points_cm[0])
        return f"M {start[0]:.2f},{start[1]:.2f} Q {control[0]:.2f},{control[1]:.2f} {end[0]:.2f},{end[1]:.2f}"
    return f"M {start[0]:.2f},{start[1]:.2f} L {end[0]:.2f},{end[1]:.2f}"


def _svg(record) -> str:
    panels = list(record.panels)
    cell_width, cell_height = 430, 390
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{columns * cell_width}" height="{rows * cell_height + 80}" viewBox="0 0 {columns * cell_width} {rows * cell_height + 80}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="17" font-weight="bold">{html.escape(record.sample_id)}</text>',
        '<text x="20" y="52" font-family="sans-serif" font-size="12" fill="#555">Synthetic generator proposal — expert review required; BP/darts are N/A.</text>',
    ]
    references = {}
    for line in record.reference_lines:
        references.setdefault(line.panel_id, []).append(line)
    for index, panel in enumerate(panels):
        column, row = index % columns, index // columns
        left, top = column * cell_width + 24, row * cell_height + 90
        coordinates = []
        for edge in panel.edges:
            coordinates.extend((edge.geometry.start_cm, edge.geometry.end_cm, *edge.geometry.control_points_cm))
        minimum_x = min(point[0] for point in coordinates)
        maximum_x = max(point[0] for point in coordinates)
        minimum_y = min(point[1] for point in coordinates)
        maximum_y = max(point[1] for point in coordinates)
        scale = min(350 / max(maximum_x - minimum_x, 1e-6), 290 / max(maximum_y - minimum_y, 1e-6))

        def transform(point):
            return left + (point[0] - minimum_x) * scale, top + 305 - (point[1] - minimum_y) * scale

        body.append(
            f'<text x="{left}" y="{top - 12}" font-family="sans-serif" font-size="13" font-weight="bold">{html.escape(panel.id)} [{panel.semantic_role}]</text>'
        )
        for line in references.get(panel.id, []):
            start = transform(line.geometry.start_cm)
            end = transform(line.geometry.end_cm)
            opacity = "0.55" if line.training_eligible else "0.25"
            body.append(
                f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#1565c0" stroke-width="1" stroke-dasharray="5 4" opacity="{opacity}"/>'
            )
            body.append(
                f'<text x="{end[0] + 3:.2f}" y="{end[1] - 3:.2f}" font-family="sans-serif" font-size="9" fill="#1565c0">{line.canonical_name}</text>'
            )
        for edge in panel.edges:
            color = COLORS.get(edge.semantic_role, COLORS["other"])
            body.append(
                f'<path d="{_path(edge, transform)}" fill="none" stroke="{color}" stroke-width="3" data-edge="{html.escape(edge.id)}"/>'
            )
            midpoint = transform(
                ((edge.geometry.start_cm[0] + edge.geometry.end_cm[0]) / 2, (edge.geometry.start_cm[1] + edge.geometry.end_cm[1]) / 2)
            )
            body.append(
                f'<text x="{midpoint[0] + 3:.2f}" y="{midpoint[1] - 3:.2f}" font-family="sans-serif" font-size="8" fill="{color}">{html.escape(edge.semantic_role)}</text>'
            )
        for point in panel.points:
            if not point.canonical_name:
                continue
            x, y = transform(point.xy_cm)
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#111"/>')
            body.append(
                f'<text x="{x + 6:.2f}" y="{y - 6:.2f}" font-family="sans-serif" font-size="11" font-weight="bold">{point.canonical_name}</text>'
            )
    body.append("</svg>")
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ignored basic-T-shirt expert-review packet.")
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/tshirt_traces.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/expert_review"))
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    selected = _read_selected_records(args.records, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    edge_rows = []
    point_rows = []
    for record in selected:
        (args.output / f"{record.sample_id}.svg").write_text(_svg(record), encoding="utf-8")
        for panel in record.panels:
            for edge in panel.edges:
                edge_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "split": record.split,
                        "panel_id": panel.id,
                        "panel_role": panel.semantic_role,
                        "edge_id": edge.id,
                        "proposed_role": edge.semantic_role,
                        "length_cm": round(edge_length_cm(edge), 4),
                        "decision": "",
                        "corrected_role": "",
                        "reviewer_note": "",
                    }
                )
            for point in panel.points:
                if not point.canonical_name:
                    continue
                point_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "split": record.split,
                        "panel_id": panel.id,
                        "point_id": point.id,
                        "x_cm": round(point.xy_cm[0], 4),
                        "y_cm": round(point.xy_cm[1], 4),
                        "proposed_name": point.canonical_name,
                        "decision": "",
                        "corrected_name": "",
                        "reviewer_note": "",
                    }
                )
    for file_name, rows in (("edge_review.csv", edge_rows), ("point_review.csv", point_rows)):
        with (args.output / file_name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "schema_version": "tshirt-expert-review-1.0",
        "record_count": len(selected),
        "edge_review_count": len(edge_rows),
        "point_review_count": len(point_rows),
        "sample_ids": [record.sample_id for record in selected],
        "source_status": "SYNTHETIC_GENERATOR_PROPOSALS_NOT_EXPERT_GROUND_TRUTH",
        "review_schema": "benchmark/configs/tshirt_expert_review_schema.json",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "sample_ids"}))


if __name__ == "__main__":
    main()
