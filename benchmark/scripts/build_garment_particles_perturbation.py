from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic horizontal-flip binding perturbation.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.source) as image:
        ImageOps.mirror(image.convert("RGB")).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
