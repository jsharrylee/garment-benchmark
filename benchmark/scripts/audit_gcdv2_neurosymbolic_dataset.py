from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.neurosymbolic_dataset import CONTOUR_SAMPLES, VISUAL_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GCDv2 visual/formal/constraint-separated data.")
    parser.add_argument("--root", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1"))
    args = parser.parse_args()
    panels = [json.loads(line) for line in (args.root / "panel_index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    garments = [json.loads(line) for line in (args.root / "garment_index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    panel_uids = {value["panel_uid"] for value in panels}
    failures = []
    observability = Counter()
    primitives = Counter()
    point_count = curve_count = stitch_count = 0
    for index, row in enumerate(panels, 1):
        try:
            graph = json.loads(Path(row["formal_graph_path"]).read_text(encoding="utf-8"))
            points, curves = graph["points"], graph["curves"]
            if len(points) != len(curves) or len(points) < 3:
                raise ValueError("panel is not a one-cycle point/curve graph")
            for edge_index, curve in enumerate(curves):
                if curve["start_point_id"] != f"p{edge_index}" or curve["end_point_id"] != f"p{(edge_index+1)%len(points)}":
                    raise ValueError("curve incidence mismatch")
                primitives[str(curve["primitive"])] += 1
            for point in points:
                observability[str(point["observability"])] += 1
            with np.load(row["visual_truth_path"]) as visual:
                if visual["mask_u8"].shape != (VISUAL_SIZE, VISUAL_SIZE):
                    raise ValueError("mask shape mismatch")
                if visual["sdf_cm_f16"].shape != (VISUAL_SIZE, VISUAL_SIZE):
                    raise ValueError("SDF shape mismatch")
                if visual["visible_junction_heatmap_f16"].shape != (VISUAL_SIZE, VISUAL_SIZE):
                    raise ValueError("junction heatmap shape mismatch")
                if visual["dense_contour_uv_f32"].shape != (CONTOUR_SAMPLES, 2):
                    raise ValueError("dense contour shape mismatch")
                if not np.isfinite(visual["sdf_cm_f16"]).all() or not np.isfinite(visual["dense_contour_uv_f32"]).all():
                    raise ValueError("non-finite visual truth")
            point_count += len(points)
            curve_count += len(curves)
        except Exception as error:
            failures.append({"panel_uid": row.get("panel_uid"), "error": f"{type(error).__name__}: {error}"})
        if index % 2500 == 0 or index == len(panels):
            print(json.dumps({"panels": index, "total": len(panels), "failures": len(failures)}), flush=True)
    seen_garment_panels = set()
    split_ids = {name: set() for name in ("train", "validation", "test")}
    for row in garments:
        garment = json.loads(Path(row["garment_record_path"]).read_text(encoding="utf-8"))
        split_ids[garment["split"]].add(garment["sample_id"])
        for panel in garment["panels"]:
            seen_garment_panels.add(panel["panel_uid"])
        for stitch in garment["stitch_constraints"]:
            if len(stitch["sides"]) != 2 or any(side["panel_uid"] not in panel_uids for side in stitch["sides"]):
                failures.append({"sample_id": garment["sample_id"], "error": "invalid stitch reference"})
            stitch_count += 1
    if seen_garment_panels != panel_uids:
        failures.append({"dataset": "garment-panel-index", "error": "panel coverage mismatch"})
    disjoint = not (split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"] or split_ids["validation"] & split_ids["test"])
    if not disjoint:
        failures.append({"dataset": "split", "error": "garment leakage"})
    result = {
        "status": "PASS" if not failures else "FAIL",
        "garment_count": len(garments),
        "panel_count": len(panels),
        "point_count": point_count,
        "curve_count": curve_count,
        "stitch_constraint_count": stitch_count,
        "point_observability_counts": dict(sorted(observability.items())),
        "primitive_counts": dict(sorted(primitives.items())),
        "garment_disjoint_split": disjoint,
        "failures": failures,
    }
    (args.root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
