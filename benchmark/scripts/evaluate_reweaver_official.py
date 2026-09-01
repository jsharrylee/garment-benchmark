from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReWeaver predictions with the authors' released metric code.")
    parser.add_argument("--repo", type=Path, default=Path("external/ReWeaver-Code"))
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metric_dir = (args.repo / "metric").resolve()
    sys.path.insert(0, str(metric_dir))
    from metric import Metric

    records = []
    for sample in args.samples:
        metric = Metric(args.ground_truth_root, args.prediction_root, sample)
        records.append(
            {
                "sample_id": sample,
                "panel_accuracy": float(metric.cal_panel_acc()),
                "edge_count_accuracy": float(metric.cal_edge_acc()),
                "edge_chamfer_distance": float(metric.cal_edge_cd()),
                "panel_iou": float(metric.cal_panel_iou()),
                "patch_chamfer_distance": float(metric.cal_patch_cd()),
                "patch_chamfer_distance_scaled": float(metric.cal_patch_cd_scaled()),
                "curve_chamfer_distance": float(metric.cal_curve_cd()),
                "panel_scale_l2": float(np.asarray(metric.cal_scale_l2()).mean()),
            }
        )
    result = {
        "metric_implementation": "external/ReWeaver-Code/metric/metric.py",
        "samples": records,
        "mean": {
            key: sum(record[key] for record in records) / len(records)
            for key in records[0]
            if key != "sample_id"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
