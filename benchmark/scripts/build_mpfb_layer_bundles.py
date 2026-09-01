from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.adapters.blender_character import prepare_layer_bundles


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build upper/lower emphasized four-view conditions from existing MPFB renders")
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark" / "configs" / "mpfb_samples.json")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "processed" / "mpfb")
    parser.add_argument("--samples", nargs="*")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = set(args.samples or [sample["sample_id"] for sample in config["samples"]])
    results = {}
    for sample in config["samples"]:
        if sample["sample_id"] not in selected:
            continue
        results[sample["sample_id"]] = prepare_layer_bundles(
            args.root / sample["sample_id"], split_ratio=float(sample["layer_split_ratio"])
        )
    print(json.dumps({"status": "PASS", "samples": results}, indent=2))


if __name__ == "__main__":
    main()
