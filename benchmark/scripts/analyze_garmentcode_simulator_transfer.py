from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


def mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    bounds = np.asarray(mesh.bounds, dtype=float)
    return bounds[0], bounds[1]


def summarize(rows: list[dict]) -> dict:
    values = np.asarray([row["extent_ratio_xyz"] for row in rows], dtype=float)
    shifts = np.asarray([row["center_shift_xyz_cm"] for row in rows], dtype=float)
    return {
        "samples": len(rows),
        "extent_ratio_xyz": {
            "median": np.median(values, axis=0).tolist(),
            "p10": np.quantile(values, 0.10, axis=0).tolist(),
            "p90": np.quantile(values, 0.90, axis=0).tolist(),
        },
        "center_shift_xyz_cm": {
            "median": np.median(shifts, axis=0).tolist(),
            "p10": np.quantile(shifts, 0.10, axis=0).tolist(),
            "p90": np.quantile(shifts, 0.90, axis=0).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GarmentCode reference-simulator drape deltas from boxmesh to sim PLY.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_reference_simulator_priors.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    records = {row["sample_id"]: row for row in catalog["records"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    failures = []
    for index, sample_id in enumerate(sorted(records), start=1):
        record = records[sample_id]
        if record.get("render_quality") != "PASS":
            continue
        folder = args.dataset / sample_id
        initial_path = folder / f"{sample_id}_boxmesh.ply"
        final_path = folder / f"{sample_id}_sim.ply"
        try:
            initial_min, initial_max = mesh_bounds(initial_path)
            final_min, final_max = mesh_bounds(final_path)
            initial_extent = initial_max - initial_min
            final_extent = final_max - final_min
            ratio = final_extent / np.maximum(initial_extent, 1e-6)
            if not np.all(np.isfinite(ratio)) or np.any(ratio < 0.05) or np.any(ratio > 8.0):
                raise ValueError(f"implausible extent ratio {ratio.tolist()}")
            row = {
                "sample_id": sample_id,
                "extent_ratio_xyz": ratio.tolist(),
                "center_shift_xyz_cm": (((final_min + final_max) - (initial_min + initial_max)) * 0.5).tolist(),
            }
            grouped[record["category"]].append(row)
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})
        if args.limit and sum(len(rows) for rows in grouped.values()) >= args.limit:
            break
        if index % 500 == 0:
            print(json.dumps({"inspected": index, "accepted": sum(len(rows) for rows in grouped.values()), "failures": len(failures)}), flush=True)

    all_rows = [row for rows in grouped.values() for row in rows]
    payload = {
        "schema_version": "1.0",
        "dataset": "GarmentCodeData v2 garments_5000_0/default_body",
        "comparison": "boxmesh initial placement to official reference simulator output",
        "coordinate_axes": ["horizontal_x", "vertical_y", "depth_z"],
        "overall": summarize(all_rows),
        "by_category": {category: summarize(rows) for category, rows in sorted(grouped.items())},
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
