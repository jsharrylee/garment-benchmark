from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Blender cloth drape ratios with GarmentCode reference-simulator priors.")
    parser.add_argument("simulation_outputs", type=Path, nargs="+")
    parser.add_argument("--reference-priors", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    priors = json.loads(args.reference_priors.read_text(encoding="utf-8"))
    source_xyz = priors["by_category"][args.category]["extent_ratio_xyz"]["median"]
    # GarmentCode X/Y/Z = horizontal/vertical/depth; Blender X/Y/Z = horizontal/depth/vertical.
    target = [float(source_xyz[0]), float(source_xyz[2]), float(source_xyz[1])]
    rows = []
    for output in args.simulation_outputs:
        metadata = json.loads((output / "simulation_metadata.json").read_text(encoding="utf-8"))
        receipt = json.loads((output / "simulation_receipt.json").read_text(encoding="utf-8"))
        observed = [float(value) for value in metadata["drape_extent_ratio_blender_xyz"]]
        log_error = sum(abs(math.log(max(value, 1e-8) / max(goal, 1e-8))) for value, goal in zip(observed, target, strict=True)) / 3.0
        rows.append(
            {
                "run": output.name,
                "observed_ratio_blender_xyz": observed,
                "target_reference_ratio_blender_xyz": target,
                "mean_absolute_log_ratio_error": log_error,
                "mean_four_view_iou": receipt.get("four_view_comparison", {}).get("mean_iou"),
                "calibration": metadata.get("calibration", {}),
                "pinned_vertex_count": metadata.get("pinned_vertex_count"),
            }
        )
    rows.sort(key=lambda row: (row["mean_absolute_log_ratio_error"], -(row["mean_four_view_iou"] or 0.0)))
    best = rows[0]
    current_pre = best["calibration"].get("precompensation_scale_blender_xyz", [1.0, 1.0, 1.0])
    correction = [
        max(0.7, min(1.5, float(current_pre[index]) * target[index] / max(best["observed_ratio_blender_xyz"][index], 1e-8)))
        for index in range(3)
    ]
    measured = next((row for row in rows if row["calibration"].get("status") == "MEASURED_TRANSFER_CALIBRATION"), None)
    correction_validation = "NOT_RUN"
    if measured is not None:
        correction_validation = (
            "PASS" if measured["mean_absolute_log_ratio_error"] < best["mean_absolute_log_ratio_error"] else "REJECTED_NONLINEAR_RESPONSE"
        )
    payload = {
        "schema_version": "1.0",
        "category": args.category,
        "reference_sample_count": priors["by_category"][args.category]["samples"],
        "axis_mapping": "GarmentCode XYZ -> Blender XZY",
        "runs": rows,
        "selected_run": best["run"],
        "recommended_precompensation_scale_blender_xyz": correction,
        "precompensation_validation": correction_validation,
        "recommended_action": "DO_NOT_AUTO_APPLY" if correction_validation == "REJECTED_NONLINEAR_RESPONSE" else "VALIDATE_ON_HELD_OUT_PATTERN",
        "adopted_calibration": best["calibration"],
        "selection_scope": "extent-transfer calibration; final acceptance still requires four-view target IoU",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
