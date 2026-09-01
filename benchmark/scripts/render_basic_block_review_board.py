from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.drafting_semantics import basic_blocks as blocks


ROLE_COLORS = {
    "neckline": "#d81b60",
    "shoulder": "#00897b",
    "armhole": "#7b1fa2",
    "sleeve_head": "#7b1fa2",
    "side_seam": "#1e88e5",
    "outseam": "#1e88e5",
    "inseam": "#43a047",
    "crotch_curve": "#fb8c00",
    "waistline": "#6d4c41",
    "dart_leg": "#7cb342",
    "hemline": "#3949ab",
    "slit": "#f4511e",
    "center_front": "#ef6c00",
    "center_back": "#ef6c00",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_panel(axis, panel: blocks.Panel) -> None:
    for path in panel.paths:
        points = blocks._vector_path_points(panel, path, curve_samples=48)
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=ROLE_COLORS.get(path.role, "#607d8b"),
            linewidth=2.0,
        )
    for line in panel.reference_lines:
        axis.plot(
            [line.start_cm[0], line.end_cm[0]],
            [line.start_cm[1], line.end_cm[1]],
            color="#90a4ae",
            linewidth=0.8,
            linestyle="--",
        )
        midpoint = (
            (line.start_cm[0] + line.end_cm[0]) / 2.0,
            (line.start_cm[1] + line.end_cm[1]) / 2.0,
        )
        axis.text(*midpoint, line.name, color="#546e7a", fontsize=7)
    for landmark in panel.landmarks:
        axis.scatter(*landmark.xy_cm, s=9, color="#263238", zorder=3)
        axis.annotate(
            landmark.name,
            landmark.xy_cm,
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color="#263238",
        )
    axis.set_title(f"{panel.id}  [{panel.role}]", fontsize=10, weight="bold")
    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis()
    axis.grid(alpha=0.12)
    axis.set_xlabel("cm")
    axis.set_ylabel("cm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render expert-reviewable provisional T-shirt, pants, and skirt basic blocks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/basic_blocks/default_basic_blocks_review.png"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/drafting_semantics/basic_blocks/default_basic_blocks_review.json"),
    )
    args = parser.parse_args()

    records = [blocks.build_basic_block(category) for category in blocks.SUPPORTED_CATEGORIES]
    figure, axes = plt.subplots(3, 3, figsize=(15, 16), constrained_layout=True)
    for row, record in enumerate(records):
        for column in range(3):
            axis = axes[row, column]
            if column >= len(record.panels):
                axis.axis("off")
                continue
            _draw_panel(axis, record.panels[column])
        axes[row, 0].text(
            0.0,
            1.13,
            f"{record.category.upper()} · {record.provenance.status} · expert review {record.provenance.expert_review}",
            transform=axes[row, 0].transAxes,
            fontsize=13,
            weight="bold",
        )
    figure.suptitle(
        "Provisional common-garment basic blocks · named drafting landmarks and paths",
        fontsize=17,
        weight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    payload = {
        "schema_version": "basic-block-review-board/v1",
        "status": blocks.PROVENANCE_STATUS,
        "expert_review": "PENDING",
        "sample_ids": [record.sample_id for record in records],
        "contains_source_dataset_images": False,
        "output": args.output.as_posix(),
        "output_sha256": _sha256(args.output),
        "claim_boundary": "Diagram visualizes provisional formulas; it is not an expert approval or industrial pattern.",
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
