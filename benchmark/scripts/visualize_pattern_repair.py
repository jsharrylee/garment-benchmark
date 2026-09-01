from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.visualization.contact_sheet import create_contact_sheet


def draw_comparison(original_path: Path, repaired_path: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    documents = [PatternDocument.read_json(original_path), PatternDocument.read_json(repaired_path)]
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, document, title in zip(axes, documents, ("Before learned repair", "After learned repair"), strict=True):
        report = validate_pattern(document)
        invalid = {issue.subject for issue in report.issues if issue.severity == "error"}
        bounds = []
        for panel in document.panels:
            values = np.asarray([point for edge in panel.edges for point in edge.points], dtype=float)
            bounds.append((values.min(axis=0), values.max(axis=0)))
        max_width = max((maximum[0] - minimum[0] for minimum, maximum in bounds), default=1.0)
        max_height = max((maximum[1] - minimum[1] for minimum, maximum in bounds), default=1.0)
        columns = 4
        for index, (panel, (minimum, maximum)) in enumerate(zip(document.panels, bounds, strict=True)):
            column, row = index % columns, index // columns
            center = (minimum + maximum) * 0.5
            offset = np.array([column * max_width * 1.25, -row * max_height * 1.25]) - center
            color = "#d62728" if panel.id in invalid else "#2ca02c"
            for edge in panel.edges:
                points = np.asarray(edge.points) + offset
                axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5)
            axis.text(*(offset + center), panel.id.replace("panel_", ""), fontsize=7, ha="center", va="center")
        axis.set_title(f"{title}\nerrors={report.metrics['error_count']}, warnings={report.metrics['warning_count']}")
        axis.set_aspect("equal")
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ignored before/after review board for learned pattern repair")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True)
    parser.add_argument("--simulation-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = args.output.with_name(args.output.stem + "_patterns.png")
    draw_comparison(args.original, args.repaired, comparison)
    create_contact_sheet(
        [args.condition, comparison, args.target_mask, args.simulation_mask],
        args.output,
        ["Four-view target layer", "PatternRepairNet before / after", "Target front mask", "Simulated repaired front"],
        cell=(640, 480),
        columns=2,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
