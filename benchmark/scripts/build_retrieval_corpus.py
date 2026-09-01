from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.retrieval.corpus import build_gcd_ts_record
from benchmark.retrieval.index import PatternIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small official GCD-TS pattern retrieval bank.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/reweaver_gcd_ts/test"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/gcd_ts_8_index.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/retrieval_anchored_v2_corpus.json"))
    args = parser.parse_args()

    records = []
    for pattern_path in sorted(args.root.glob("*/*_2d_panel.json")):
        alpha_dir = pattern_path.parent / "render_output" / "alpha"
        views = sorted(alpha_dir.glob("*.png"))
        records.append(build_gcd_ts_record(pattern_path, views))
    if not records:
        raise SystemExit(f"no GCD-TS records found below {args.root}")
    index = PatternIndex(records)
    index.write_json(args.output)

    manifest = {
        "schema_version": "2.0",
        "mode": "retrieval_anchored_v2",
        "source_dataset": "SII-LiMing/ReWeaver-GCD-TS",
        "source_license": "CC BY-NC 4.0",
        "record_count": len(records),
        "categories": {category: sum(record.category == category for record in records) for category in sorted({record.category for record in records})},
        "records": [record.to_dict(include_local_paths=False) for record in records],
        "large_download_performed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
