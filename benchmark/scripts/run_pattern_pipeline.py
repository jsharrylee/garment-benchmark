from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.pattern_pipeline.runner import run_reweaver_pattern_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize and validate a generated ReWeaver sewing pattern")
    parser.add_argument("source", type=Path, help="ReWeaver generated NPZ")
    parser.add_argument("output", type=Path, help="Ignored artifact output directory")
    args = parser.parse_args()
    receipt = run_reweaver_pattern_pipeline(args.source, args.output)
    print(json.dumps({"pattern_id": receipt["pattern_id"], "structural_export": receipt["structural_export"], "full_simulation_benchmark": receipt["full_simulation_benchmark"]}))


if __name__ == "__main__":
    main()
