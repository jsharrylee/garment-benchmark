from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmark.adapters.blender_character import prepare_inference_bundle, prepare_layer_bundles, validate_blender_character_bundle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = ROOT / "external" / "blender-4.5.12-windows-x64" / "blender.exe"
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "mpfb_samples.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "mpfb"
BLENDER_SCRIPT = ROOT / "benchmark" / "blender_scripts" / "mpfb_generate_render.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and render deterministic MPFB benchmark characters")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.blender.is_file():
        raise SystemExit(f"portable Blender not found: {args.blender}")
    command = [
        str(args.blender),
        "--background",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--config",
        str(args.config.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = {}
    for sample in config["samples"]:
        sample_dir = args.output / sample["sample_id"]
        prepared = prepare_inference_bundle(sample_dir)
        layers = prepare_layer_bundles(sample_dir, split_ratio=float(sample["layer_split_ratio"]))
        results[sample["sample_id"]] = {
            **validate_blender_character_bundle(sample_dir),
            "reweaver_input_validation": prepared,
            "layer_inputs": {name: value["input_validation"] for name, value in layers["layers"].items()},
        }
    receipt = {"generator": "MPFB", "config": args.config.name, "samples": results, "all_valid": all(item["valid"] for item in results.values())}
    receipt_path = args.output / "generation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not receipt["all_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
