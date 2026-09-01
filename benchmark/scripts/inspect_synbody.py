from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.adapters.synbody import discover_bundles, validate_bundle
from benchmark.preprocessing.grouping import rank_candidate_bundles
from benchmark.visualization.contact_sheet import create_contact_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect extracted SynBody RGB data without modifying it")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synbody_discovery"))
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    bundles = discover_bundles(args.root)
    candidates = rank_candidate_bundles(bundles)[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, bundle in enumerate(candidates):
        validation = validate_bundle(bundle)
        board = args.output / f"candidate_{index:03d}.jpg"
        create_contact_sheet(list(bundle.views), board, ["CAM000", "CAM001", "CAM002", "CAM003"])
        records.append({
            "candidate": index,
            "scene": bundle.scene,
            "sequence": bundle.sequence,
            "frame": bundle.frame,
            "views": [str(path) for path in bundle.views],
            "board": str(board),
            "validation": validation,
        })
    (args.output / "candidates.json").write_text(json.dumps({"bundle_count": len(bundles), "candidates": records}, indent=2), encoding="utf-8")
    print(json.dumps({"bundle_count": len(bundles), "candidate_count": len(candidates), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
