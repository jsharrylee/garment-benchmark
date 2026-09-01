from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.evaluation.binding import compare_garment_particles_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Garment Particles output changes when its image condition changes.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare_garment_particles_outputs(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
