from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from benchmark.adapters.reweaver import VIEW_ORDER, sha256, validate_input_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic flipped ReWeaver binding-test bundle.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for camera in VIEW_ORDER:
        source = args.input / f"{camera}.png"
        destination = args.output / source.name
        with Image.open(source) as image:
            image.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(destination)
        records.append({"camera": camera, "source_sha256": sha256(source), "flipped_sha256": sha256(destination)})
    validation = validate_input_directory(args.output)
    if not validation["valid"]:
        raise ValueError(validation)
    print(json.dumps({"status": "PASS", "transformation": "horizontal_flip_all_views", "views": records}))


if __name__ == "__main__":
    main()
