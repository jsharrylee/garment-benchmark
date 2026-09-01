from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.gcdv2_exact.panel_dataset import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_PIXELS_PER_CM,
    PANEL_SCHEMA_VERSION,
    panel_slug,
    panel_target,
    render_panel_input,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows.sort(key=lambda row: (str(row["sample_id"]), str(row["category"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split exact GCDv2 samples into one-panel image/vector training pairs."
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        type=Path,
        default=[
            Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"),
            Path("artifacts/gcdv2_exact_pairs_v1/quarantine_missing_four_view.jsonl"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/gcdv2_exact_panels_v1.json")
    )
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--pixels-per-cm", type=float, default=DEFAULT_PIXELS_PER_CM)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sample_rows = _index_rows(list(args.indexes))
    if args.limit_samples:
        sample_rows = sample_rows[: args.limit_samples]
    args.output.mkdir(parents=True, exist_ok=True)

    def build_sample(row: dict[str, Any]) -> list[dict[str, Any]]:
        label_path = Path(row["label_path"])
        sample = json.loads(label_path.read_text(encoding="utf-8"))
        records = []
        for panel in sorted(sample["panels"], key=lambda value: int(value["source_order_index"])):
            target = panel_target(
                sample,
                panel,
                canvas_size=args.canvas_size,
                pixels_per_cm=args.pixels_per_cm,
            )
            directory = (
                args.output
                / str(sample["category"])
                / str(sample["sample_id"])
                / f"{int(panel['source_order_index']):02d}_{panel_slug(str(panel['panel_id']))}"
            )
            image_path = directory / "panel.png"
            metric_image_path = directory / "panel_metric.png"
            target_path = directory / "target.json"
            if args.force or not image_path.is_file():
                render_panel_input(target, image_path)
            if args.force or not metric_image_path.is_file():
                render_panel_input(target, metric_image_path, metric=True)
            if args.force or not target_path.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(
                    json.dumps(target, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
            role = target["role_labels"]
            records.append(
                {
                    "panel_uid": target["panel_uid"],
                    "sample_id": target["sample_id"],
                    "garment_category": target["garment_category"],
                    "source_panel_id": target["source"]["panel_id"],
                    "source_panel_order_index": target["source"]["panel_order_index"],
                    "role_part": role["part"],
                    "role_surface": role["surface"],
                    "role_side": role["side"],
                    "role_expert_verified": False,
                    "vertex_count": target["geometry"]["boundary_vertex_count"],
                    "edge_count": target["geometry"]["boundary_edge_count"],
                    "width_cm": target["geometry"]["width_cm"],
                    "height_cm": target["geometry"]["height_cm"],
                    "panel_image_path": image_path.as_posix(),
                    "metric_panel_image_path": metric_image_path.as_posix(),
                    "panel_image_cm_per_pixel": target["input_contract"]["normalized_panel_image"]["cm_per_pixel"],
                    "target_path": target_path.as_posix(),
                    "source_exact_label_path": label_path.as_posix(),
                    "sample_pattern_path": str(row["pattern_path"]),
                    "paired_four_view_available": bool(row["validation"]["all_views_present"]),
                    "paired_view_paths": list(row["view_paths"]),
                }
            )
        return records

    records: list[dict[str, Any]] = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(build_sample, row): row["sample_id"] for row in sample_rows}
        for index, future in enumerate(as_completed(futures), 1):
            sample_id = futures[future]
            try:
                records.extend(future.result())
            except Exception as error:
                failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})
            if index == 1 or index % 100 == 0 or index == len(futures):
                print(json.dumps({"samples": index, "total": len(futures), "panels": len(records), "failures": len(failures)}), flush=True)
    records.sort(key=lambda row: (row["sample_id"], int(row["source_panel_order_index"])))
    index_path = args.output / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )

    categories = Counter(row["garment_category"] for row in records)
    parts = Counter(row["role_part"] for row in records)
    surfaces = Counter(row["role_surface"] for row in records)
    sides = Counter(row["role_side"] for row in records)
    manifest = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": "PASS" if not failures else "PARTIAL",
        "source_dataset": "GarmentCodeData v2",
        "source_license": "CC BY 4.0",
        "sample_count": len(sample_rows),
        "panel_count": len(records),
        "four_view_paired_panel_count": sum(row["paired_four_view_available"] for row in records),
        "two_dimensional_only_panel_count": sum(not row["paired_four_view_available"] for row in records),
        "category_panel_counts": dict(sorted(categories.items())),
        "weak_role_part_counts": dict(sorted(parts.items())),
        "weak_role_surface_counts": dict(sorted(surfaces.items())),
        "weak_role_side_counts": dict(sorted(sides.items())),
        "maximum_vertices_per_panel": max((row["vertex_count"] for row in records), default=0),
        "maximum_edges_per_panel": max((row["edge_count"] for row in records), default=0),
        "input_contract": {
            "one_panel_per_image": True,
            "uniform_fill_no_category_or_role_color": True,
            "canvas_size_px": [args.canvas_size, args.canvas_size],
            "centering": "source local curve bbox center",
            "primary_panel_image": "tight normalized silhouette plus cm_per_pixel scale token",
            "metric_validation_image_pixels_per_cm": args.pixels_per_cm,
            "metric_validation_physical_span_cm": args.canvas_size / args.pixels_per_cm,
        },
        "target_contract": [
            "garment_category",
            "source_panel_id",
            "weak_lexical_part_surface_side",
            "ordered_closed_boundary_vertices_in_source_and_centered_cm",
            "edge_start_and_end_vertex_indices",
            "line_quadratic_bezier_cubic_bezier_circular_arc_type",
            "directed_curve_parameters",
            "length_and_direction_and_tangents",
        ],
        "index_artifact": index_path.as_posix(),
        "index_sha256": _sha256(index_path),
        "failures": failures,
        "claim_boundary": (
            "Vector geometry is exact source-derived truth. Normalized panel roles are lexical weak labels, "
            "not expert-verified garment drafting semantics. The normalized image must be paired with its "
            "cm_per_pixel scale token for absolute lengths; the fixed-scale image is retained for validation."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
