from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = ROOT / "external" / "blender-4.5.12-windows-x64" / "blender.exe"
BLENDER_SCRIPT = ROOT / "benchmark" / "blender_scripts" / "measure_body.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a body mesh in a Blender source bundle.")
    parser.add_argument("blend", type=Path)
    parser.add_argument("--object", default="Human")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(args.blender.resolve()), "--background", "--python", str(BLENDER_SCRIPT), "--", "--blend", str(args.blend.resolve()), "--object", args.object, "--output", str(args.output.resolve())],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
