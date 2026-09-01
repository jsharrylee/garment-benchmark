from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from benchmark.visualization.garment_particles_render import (
    render_garment_particles_geometry,
    render_garment_particles_pattern,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ignored local Garment Particles review images.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/garment_particles"))
    args = parser.parse_args()
    directory = args.root / args.sample_id
    prediction = directory / "prediction.npz"
    pattern_path = directory / "pattern.png"
    geometry_path = directory / "geometry.png"
    result = {
        "pattern": render_garment_particles_pattern(prediction, pattern_path, title=f"Garment Particles · {args.sample_id}"),
        "geometry": render_garment_particles_geometry(prediction, geometry_path, title=f"Garment Particles · {args.sample_id}"),
    }
    for path in (pattern_path, geometry_path):
        with Image.open(path) as image:
            if image.width < 600 or image.height < 400 or path.stat().st_size < 10_000:
                raise ValueError(f"VISUAL_REVIEW artifact is invalid: {path}")
    (directory / "visualization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", **result}))


if __name__ == "__main__":
    main()
