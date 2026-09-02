from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.gcdv2_exact.garment_panel_set_learning import garment_disjoint_split, read_garments
from benchmark.gcdv2_exact.neurosymbolic_dataset import (
    CONTOUR_SAMPLES,
    SCHEMA_VERSION,
    VISIBLE_CORNER_THRESHOLD_DEG,
    VISUAL_SIZE,
    build_visual_truth,
    formal_graph,
    stitch_constraints,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual/formal/constraint-separated GCDv2 panel data.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gcdv2_neurosymbolic_v1.json"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line]
    by_sample: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)
    garments, _ = read_garments(args.index)
    assignments, split_audit = garment_disjoint_split(garments, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    def build_sample(sample_id: str, sample_rows: list[dict[str, Any]]):
        sample_rows.sort(key=lambda value: int(value["source_panel_order_index"]))
        sample_label = json.loads(Path(sample_rows[0]["source_exact_label_path"]).read_text(encoding="utf-8"))
        targets = {
            str(row["source_panel_id"]): json.loads(Path(row["target_path"]).read_text(encoding="utf-8"))
            for row in sample_rows
        }
        panel_records = []
        for row in sample_rows:
            panel_id = str(row["source_panel_id"])
            target = targets[panel_id]
            panel_dir = args.output / "panels" / str(row["garment_category"]) / sample_id / f"{int(row['source_panel_order_index']):02d}_{panel_id}"
            graph_path = panel_dir / "formal_graph.json"
            visual_path = panel_dir / "visual_truth.npz"
            if args.force or not graph_path.is_file():
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_text(
                    json.dumps(formal_graph(target), sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            if args.force or not visual_path.is_file():
                visual_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_visual_path = visual_path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary_visual_path,
                    **build_visual_truth(target, Path(row["panel_image_path"])),
                )
                temporary_visual_path.replace(visual_path)
            panel_records.append({
                "panel_uid": row["panel_uid"],
                "sample_id": sample_id,
                "garment_category": row["garment_category"],
                "split": assignments[sample_id],
                "source_panel_id": panel_id,
                "weak_role": {"part": row["role_part"], "surface": row["role_surface"], "side": row["role_side"], "expert_verified": False},
                "input_panel_image": row["panel_image_path"],
                "input_scale_cm_per_pixel": row["panel_image_cm_per_pixel"],
                "visual_truth_path": visual_path.as_posix(),
                "formal_graph_path": graph_path.as_posix(),
                "paired_four_view_available": row["paired_four_view_available"],
            })
        constraints = stitch_constraints(sample_label, targets)
        garment_dir = args.output / "garments" / str(sample_rows[0]["garment_category"])
        garment_path = garment_dir / f"{sample_id}.json"
        garment = {
            "schema_version": "gcdv2-neurosymbolic-garment-1.0",
            "sample_id": sample_id,
            "garment_category": sample_rows[0]["garment_category"],
            "split": assignments[sample_id],
            "input_contract": "unordered set of panel image plus cm_per_pixel records",
            "panels": panel_records,
            "stitch_constraints": constraints,
            "paired_views": sample_rows[0]["paired_view_paths"] if sample_rows[0]["paired_four_view_available"] else [],
            "supervision_order": ["visual_contour", "visible_junctions", "formal_graph", "garment_stitch_constraints"],
        }
        garment_path.parent.mkdir(parents=True, exist_ok=True)
        garment_path.write_text(
            json.dumps(garment, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return panel_records, {
            "sample_id": sample_id,
            "garment_category": sample_rows[0]["garment_category"],
            "split": assignments[sample_id],
            "panel_count": len(panel_records),
            "stitch_count": len(constraints),
            "garment_record_path": garment_path.as_posix(),
        }

    panel_records, garment_records, failures = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(build_sample, sample_id, values): sample_id for sample_id, values in by_sample.items()}
        for index, future in enumerate(as_completed(futures), 1):
            sample_id = futures[future]
            try:
                panels, garment = future.result()
                panel_records.extend(panels)
                garment_records.append(garment)
            except Exception as error:
                failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})
                print(json.dumps({"failed_sample": sample_id, "error": failures[-1]["error"]}), flush=True)
            if index == 1 or index % 100 == 0 or index == len(futures):
                print(json.dumps({"garments": index, "total": len(futures), "panels": len(panel_records), "failures": len(failures)}), flush=True)
    panel_records.sort(key=lambda value: value["panel_uid"])
    garment_records.sort(key=lambda value: value["sample_id"])
    panel_index = args.output / "panel_index.jsonl"
    garment_index = args.output / "garment_index.jsonl"
    panel_index.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in panel_records),
        encoding="utf-8",
        newline="\n",
    )
    garment_index.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in garment_records),
        encoding="utf-8",
        newline="\n",
    )
    split_counts = Counter((value["split"], value["garment_category"]) for value in garment_records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "PARTIAL",
        "source_panel_index_sha256": _sha256(args.index),
        "garment_count": len(garment_records),
        "panel_count": len(panel_records),
        "stitch_constraint_count": sum(value["stitch_count"] for value in garment_records),
        "visual_contract": {"size": VISUAL_SIZE, "dense_contour_samples": CONTOUR_SAMPLES, "visible_corner_threshold_deg": VISIBLE_CORNER_THRESHOLD_DEG, "targets": ["mask", "signed_distance_cm", "dense_contour_uv", "visible_junction_heatmap"]},
        "formal_contract": {"objects": ["point", "line", "quadratic_bezier", "cubic_bezier", "circular_arc"], "relations": ["NEXT", "SHARED_ENDPOINT", "CLOSED_CYCLE", "DEGREE_EQUALS", "SEWN_TO"], "cyclic_loss_required": True},
        "split_audit": split_audit,
        "split_category_counts": {f"{split}:{category}": count for (split, category), count in sorted(split_counts.items())},
        "panel_index": panel_index.as_posix(),
        "panel_index_sha256": _sha256(panel_index),
        "garment_index": garment_index.as_posix(),
        "garment_index_sha256": _sha256(garment_index),
        "failures": failures,
        "claim_boundary": "Visual targets are raster-observable. Latent smooth source subdivisions, exact primitives, and stitch relations are separately marked source-formal supervision.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
