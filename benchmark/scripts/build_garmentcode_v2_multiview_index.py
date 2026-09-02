from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark.retrieval.corpus import PatternRecord, SEMANTIC_TOKENS, sha256
from benchmark.retrieval.features import multiview_descriptor
from benchmark.retrieval.index import PatternIndex


CAMERAS = ("CAM000.png", "CAM001.png", "CAM002.png", "CAM003.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full GCDv2 four-view structured-pattern retrieval index.")
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_catalog.json"))
    parser.add_argument("--canonical", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_full_canonical"))
    parser.add_argument("--views", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/garmentcode_v2_batch_0_multiview_index.json"))
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    records = []
    skipped = Counter()
    for index, row in enumerate(catalog["records"], start=1):
        sample_id = row["sample_id"]
        canonical = args.canonical / row["category"] / f"{sample_id}.json"
        views = tuple(args.views / sample_id / name for name in CAMERAS)
        if row.get("render_quality") != "PASS":
            skipped["render_quality"] += 1
            continue
        if not canonical.is_file():
            skipped["canonical_validation"] += 1
            continue
        if not all(path.is_file() for path in views):
            skipped["missing_views"] += 1
            continue
        panel_names = tuple(row["panel_names"])
        semantic_counts = {
            token: sum(token in name.lower() or (token == "waistband" and name.lower().startswith("wb_")) for name in panel_names)
            for token in SEMANTIC_TOKENS
        }
        records.append(
            PatternRecord(
                sample_id=sample_id,
                category=row["category"],
                panel_names=panel_names,
                panel_count=int(row["panel_count"]),
                edge_count=int(row["edge_count"]),
                mean_edges_per_panel=float(row["edge_count"] / max(1, row["panel_count"])),
                semantic_counts=semantic_counts,
                visual_descriptor=multiview_descriptor(views),
                source_pattern_sha256=row["specification_sha256"],
                view_sha256=tuple(sha256(path) for path in views),
                source_pattern=str(canonical),
                source_views=tuple(str(path) for path in views),
                source_dataset="GarmentCodeData v2",
                source_license="CC BY 4.0",
            )
        )
        if index % 500 == 0:
            print(json.dumps({"scanned": index, "indexed": len(records), "skipped": dict(skipped)}), flush=True)
    if not records:
        raise SystemExit("no complete four-view canonical records")
    PatternIndex(records).write_json(args.output)
    manifest = {
        "schema_version": "1.0",
        "dataset": "GarmentCodeData v2",
        "batch": "garments_5000_0/default_body",
        "license": "CC BY 4.0",
        "view_generation": "official reference sim PLY rerendered front/back/left/right in Blender orthographic cameras",
        "record_count": len(records),
        "category_counts": dict(sorted(Counter(record.category for record in records).items())),
        "skipped": dict(skipped),
        "index_sha256": sha256(args.output),
        "local_paths_omitted": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
