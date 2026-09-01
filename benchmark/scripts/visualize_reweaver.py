from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from benchmark.visualization.geometry_render import render_reweaver_geometry
from benchmark.visualization.pattern_render import render_reweaver_patterns


def validate_render(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 10_000:
        return {"valid": False, "failure": "VISUAL_REVIEW_FAILURE"}
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {"valid": image.width >= 640 and image.height >= 480, "dimensions": [image.width, image.height], "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/reweaver"))
    parser.add_argument("--samples", nargs="+", default=["synbody_cyan_jacket", "synbody_patterned_shirt"])
    args = parser.parse_args()
    records = []
    for sample_id in args.samples:
        sample_root = args.artifact_root / sample_id
        prediction = sample_root / f"{sample_id}.npz"
        pattern = sample_root / "pattern.png"
        geometry = sample_root / "geometry.png"
        record = {
            "sample_id": sample_id,
            "pattern": render_reweaver_patterns(prediction, pattern, title=f"ReWeaver 2D panels · {sample_id}"),
            "geometry": render_reweaver_geometry(prediction, geometry, title=f"ReWeaver 3D patches and curves · {sample_id}"),
            "pattern_validation": validate_render(pattern),
            "geometry_validation": validate_render(geometry),
        }
        records.append(record)
    valid = all(item["pattern_validation"]["valid"] and item["geometry_validation"]["valid"] for item in records)
    print(json.dumps({"status": "PASS" if valid else "FAILED_VALIDATION", "samples": records}))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
