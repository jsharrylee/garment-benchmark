from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.visualization.contact_sheet import create_contact_sheet


CAMERAS = ("CAM000.png", "CAM001.png", "CAM002.png", "CAM003.png")
LABELS = ("front", "back", "left", "right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare target, retrieved source and simulated rerank winner.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--target-masks", type=Path, required=True)
    parser.add_argument("--simulation-masks", type=Path, required=True)
    parser.add_argument("--source-views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    labels = []
    for root, row_label in (
        (args.target_masks, "MPFB target"),
        (args.source_views / args.sample_id, f"retrieved GCDv2 {args.sample_id}"),
        (args.simulation_masks, "graded + Blender cloth"),
    ):
        for camera, label in zip(CAMERAS, LABELS, strict=True):
            path = root / camera
            if not path.is_file():
                raise SystemExit(f"missing comparison image: {path}")
            paths.append(path)
            labels.append(f"{row_label} · {label}")
    create_contact_sheet(paths, args.output, labels, cell=(360, 360), columns=4)
    print(args.output)


if __name__ == "__main__":
    main()
