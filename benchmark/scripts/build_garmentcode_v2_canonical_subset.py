from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.retrieval.corpus import infer_garment_category
from benchmark.retrieval.garmentcode import convert_garmentcode_specification


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an extracted GarmentCodeData v2 subset to canonical validated patterns.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_balanced_subset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_canonical"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/garmentcode_v2_batch_0_canonical_subset.json"))
    parser.add_argument("--accepted-only", action="store_true", help="Do not materialize rejected canonical patterns.")
    parser.add_argument("--compact-manifest", action="store_true", help="Record aggregate counts and rejected IDs instead of every row.")
    args = parser.parse_args()

    paths = sorted(args.input.rglob("*_specification.json"))
    if not paths:
        raise SystemExit(f"no specifications below {args.input}")
    records = []
    accepted_panel_count = 0
    for path_index, path in enumerate(paths, start=1):
        raw = json.loads(path.read_text(encoding="utf-8"))
        sample_id = next((part for part in path.parts if part.startswith("rand_")), path.stem.replace("_specification", ""))
        category = infer_garment_category(tuple(raw["pattern"]["panels"]))
        document = convert_garmentcode_specification(path, anchor_id=sample_id, source_license="CC BY 4.0")
        validation = validate_pattern(document)
        output = args.output / category / f"{sample_id}.json"
        if validation.accepted or not args.accepted_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            document.write_json(output)
        if validation.accepted:
            accepted_panel_count += len(document.panels)
        records.append(
            {
                "sample_id": sample_id,
                "category": category,
                "panel_count": len(document.panels),
                "edge_count": sum(len(panel.edges) for panel in document.panels),
                "stitch_count": len(document.stitches),
                "structural_validation": validation.to_dict(),
                "source_artifact_sha256": document.provenance["source_artifact_sha256"],
            }
        )
        if path_index % 250 == 0:
            print(json.dumps({"converted": path_index, "total": len(paths), "accepted": sum(row["structural_validation"]["accepted"] for row in records)}), flush=True)
    accepted = sum(record["structural_validation"]["accepted"] for record in records)
    manifest = {
        "schema_version": "1.0",
        "dataset": "GarmentCodeData v2",
        "doi": "https://doi.org/10.3929/ethz-b-000690432",
        "license": "CC BY 4.0",
        "batch": "garments_5000_0/default_body",
        "record_count": len(records),
        "accepted_count": accepted,
        "rejected_count": len(records) - accepted,
        "accepted_clean_panel_count": accepted_panel_count,
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "accepted_category_counts": dict(sorted(Counter(record["category"] for record in records if record["structural_validation"]["accepted"]).items())),
        "rejected_records": [record for record in records if not record["structural_validation"]["accepted"]],
    }
    if not args.compact_manifest:
        manifest["records"] = records
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "accepted": accepted, "rejected": len(records) - accepted, "accepted_clean_panels": accepted_panel_count}))


if __name__ == "__main__":
    main()
