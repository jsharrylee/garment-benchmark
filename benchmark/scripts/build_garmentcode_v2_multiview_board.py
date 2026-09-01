from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.visualization.contact_sheet import create_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local review board for official and rerendered GCDv2 views.")
    parser.add_argument("sample_ids", nargs="+")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--views", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/review_boards/garmentcode_v2_multiview.jpg"))
    args = parser.parse_args()

    paths = []
    labels = []
    for sample_id in args.sample_ids:
        source = args.dataset / sample_id
        rerendered = args.views / sample_id
        columns = (
            (source / f"{sample_id}_render_front.png", "official front + body"),
            (rerendered / "CAM000.png", "garment-only front"),
            (rerendered / "CAM002.png", "rerendered left"),
            (rerendered / "CAM003.png", "rerendered right"),
        )
        for path, label in columns:
            if not path.is_file():
                raise SystemExit(f"missing review image: {path}")
            paths.append(path)
            labels.append(f"{sample_id} · {label}")
    create_contact_sheet(paths, args.output, labels, cell=(320, 320), columns=4)
    print(args.output)


if __name__ == "__main__":
    main()
