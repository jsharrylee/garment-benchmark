from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmark.adapters.blender_character import prepare_inference_bundle, validate_blender_character_bundle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = ROOT / "external" / "blender-4.5.12-windows-x64" / "blender.exe"
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "blender_studio_samples.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "blender_studio"
BLENDER_SCRIPT = ROOT / "benchmark" / "blender_scripts" / "blender_studio_render.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render official Blender Studio characters for visual-to-pattern evaluation")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = {}
    for sample in config["samples"]:
        blend_file = ROOT / sample["blend_file"]
        if not blend_file.is_file():
            raise SystemExit(f"official source .blend not found: {blend_file}")
        sample_dir = args.output / sample["sample_id"]
        command = [
            str(args.blender),
            "--background",
            "--disable-autoexec",
            str(blend_file),
            "--python",
            str(BLENDER_SCRIPT),
            "--",
            "--config",
            str(args.config.resolve()),
            "--sample",
            sample["sample_id"],
            "--output",
            str(sample_dir.resolve()),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        prepared = prepare_inference_bundle(sample_dir)
        results[sample["sample_id"]] = {
            **validate_blender_character_bundle(sample_dir),
            "reweaver_input_validation": prepared,
        }
    receipt = {
        "generator": "Blender Studio source adapter",
        "source": config["source"],
        "samples": results,
        "all_valid": all(item["valid"] for item in results.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "generation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not receipt["all_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
