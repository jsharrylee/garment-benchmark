from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from benchmark.gcdv2_exact.geometry import CURVE_TYPES, SCHEMA_VERSION, build_exact_sample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact GCDv2 geometry labels, non-overlap pattern renders, and paired four-view records.")
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_catalog.json"))
    parser.add_argument("--input", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--views", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gcdv2_exact_pairs_v1.json"))
    parser.add_argument("--categories", nargs="+", default=["top", "skirt", "pants"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--render-failures-only",
        action="store_true",
        help="Build exact 2D labels for catalog rows rejected before four-view rendering; records remain quarantined.",
    )
    parser.add_argument("--index-name", default="index.jsonl")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))["records"]
    if args.render_failures_only:
        wanted = [
            row for row in catalog
            if row["category"] in set(args.categories) and row.get("render_quality") != "PASS"
        ]
    else:
        wanted = [
            row for row in catalog
            if row["category"] in set(args.categories) and row.get("render_quality") == "PASS"
        ]
    if args.limit:
        wanted = wanted[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    def build(row: dict) -> dict:
        sample_id = row["sample_id"]
        sample_source = args.input / sample_id
        specification = sample_source / f"{sample_id}_specification.json"
        # Source front panels sit at +source-Z, transformed to +Blender-Y.
        # The +Y camera is CAM001, so legacy CAM000/CAM001 filenames must be
        # mapped to semantic back/front rather than trusted by their receipt.
        view_paths = [
            args.views / sample_id / "CAM001.png",  # semantic front
            args.views / sample_id / "CAM000.png",  # semantic back
            args.views / sample_id / "CAM002.png",  # semantic left
            args.views / sample_id / "CAM003.png",  # semantic right
        ]
        destination = args.output / row["category"] / sample_id
        existing = destination / "labels.json"
        if existing.is_file() and (destination / "pattern.png").is_file() and not args.force:
            label = json.loads(existing.read_text(encoding="utf-8"))
            return {
                "sample_id": sample_id,
                "category": row["category"],
                "label_path": existing.as_posix(),
                "pattern_path": (destination / "pattern.png").as_posix(),
                "overlay_path": None,
                "view_paths": [value.as_posix() for value in view_paths],
                "panel_count": len(label["panels"]),
                "edge_count": sum(len(panel["edges"]) for panel in label["panels"]),
                "curve_type_counts": {
                    kind: sum(edge["curve"]["type"] == kind for panel in label["panels"] for edge in panel["edges"])
                    for kind in CURVE_TYPES
                },
                "validation": label["validation"],
            }
        return build_exact_sample(
            specification,
            category=row["category"],
            view_paths=view_paths,
            view_labels=("front", "back", "left", "right"),
            output_dir=destination,
            image_size=args.image_size,
            allow_missing_views=args.render_failures_only,
        )

    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(build, row): row["sample_id"] for row in wanted}
        for index, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})
            if index == 1 or index % 100 == 0 or index == len(futures):
                print(json.dumps({"processed": index, "total": len(futures), "failures": len(failures)}), flush=True)
    records.sort(key=lambda value: value["sample_id"])
    index_path = args.output / args.index_name
    index_path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in records), encoding="utf-8")
    category_counts = Counter(value["category"] for value in records)
    curve_counts = Counter()
    for record in records:
        curve_counts.update(record["curve_type_counts"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "GarmentCodeData v2",
        "license": "CC BY 4.0",
        "requested_categories": list(args.categories),
        "scope": "render_failures_quarantine" if args.render_failures_only else "complete_four_view_pairs",
        "eligible_record_count": len(wanted),
        "built_record_count": len(records),
        "failed_record_count": len(failures),
        "category_counts": dict(sorted(category_counts.items())),
        "curve_type_counts": dict(sorted(curve_counts.items())),
        "validation_pass_count": sum(row["validation"]["status"] == "PASS" for row in records),
        "non_overlap_pass_count": sum(row["validation"]["packed_non_overlap"] for row in records),
        "all_views_present_count": sum(row["validation"]["all_views_present"] for row in records),
        "image_size": args.image_size,
        "index_artifact": index_path.as_posix(),
        "index_sha256": _sha256(index_path),
        "failures": failures,
        "claim_boundary": "Geometry is source-exact or deterministic geometry derived from GCDv2 specification.json. Packed panel positions are display-only and never geometric truth. Semantic drafting names are outside this schema.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
