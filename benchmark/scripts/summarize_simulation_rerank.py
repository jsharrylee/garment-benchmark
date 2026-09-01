from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the highest-IoU candidate after real cloth simulation.")
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mean-iou", type=float, default=0.45)
    args = parser.parse_args()
    rows = []
    for folder in args.outputs:
        receipt = json.loads((folder / "simulation_receipt.json").read_text(encoding="utf-8"))
        metadata = json.loads((folder / "simulation_metadata.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": folder.name,
                "pattern_id": receipt["pattern_id"],
                "panel_count": receipt["structural_validation"]["metrics"]["panel_count"],
                "stitch_count": receipt["structural_validation"]["metrics"]["stitch_count"],
                "mean_iou": receipt["four_view_comparison"]["mean_iou"],
                "per_view_iou": receipt["four_view_comparison"]["per_view_iou"],
                "drape_extent_ratio_blender_xyz": metadata["drape_extent_ratio_blender_xyz"],
            }
        )
    rows.sort(key=lambda row: (-row["mean_iou"], row["panel_count"], row["pattern_id"]))
    payload = {
        "mode": "retrieval_body_grade_real_cloth_simulation_rerank",
        "candidate_count": len(rows),
        "rows": rows,
        "selected": rows[0],
        "minimum_mean_iou": args.minimum_mean_iou,
        "status": "PASS" if rows[0]["mean_iou"] >= args.minimum_mean_iou else "FAILED_VISUAL_FIDELITY",
        "selection_is_not_license_clearance": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
