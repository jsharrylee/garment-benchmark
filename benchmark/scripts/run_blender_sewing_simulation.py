from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.pattern_pipeline.evaluation import compare_orthogonal_masks
from benchmark.pattern_pipeline.refinement import load_reference_masks
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.sewing_mesh import build_sewing_mesh_plan, mesh_plan_to_dict
from benchmark.pattern_pipeline.validation import validate_pattern


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = ROOT / "external" / "blender-4.5.12-windows-x64" / "blender.exe"
BLENDER_SCRIPT = ROOT / "benchmark" / "blender_scripts" / "simulate_canonical_pattern.py"
CAMERA_TO_VIEW = {"CAM000": "front", "CAM001": "back", "CAM002": "left", "CAM003": "right"}


def load_simulated_masks(directory: Path) -> dict[str, np.ndarray]:
    result = {}
    for camera, view in CAMERA_TO_VIEW.items():
        path = directory / f"{camera}.png"
        with Image.open(path) as image:
            alpha = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
            result[view] = np.asarray(alpha) > 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical-pattern Blender cloth sewing and compare four silhouettes")
    parser.add_argument("pattern", type=Path)
    parser.add_argument("--target-masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--min-mean-iou", type=float, default=0.45)
    parser.add_argument("--body-blend", type=Path)
    parser.add_argument("--body-object", default="Human")
    parser.add_argument("--camera-metadata", type=Path)
    parser.add_argument("--subdivision-levels", type=int, default=0)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    args.pattern = args.pattern.resolve()
    args.target_masks = args.target_masks.resolve()
    args.output = args.output.resolve()
    args.blender = args.blender.resolve()
    args.body_blend = args.body_blend.resolve() if args.body_blend else None
    args.camera_metadata = args.camera_metadata.resolve() if args.camera_metadata else None
    args.calibration = args.calibration.resolve() if args.calibration else None
    args.output.mkdir(parents=True, exist_ok=True)

    document = PatternDocument.read_json(args.pattern)
    validation = validate_pattern(document)
    result = {
        "pattern_id": document.pattern_id,
        "structural_validation": validation.to_dict(),
        "backend": "blender_sewing_springs",
        "mock_or_proxy_used": False,
        "template_retrieval": bool(document.annotations.get("template_retrieval", False)),
    }
    if not validation.accepted:
        result["status"] = "BLOCKED_STRUCTURAL_VALIDATION"
    elif not args.blender.is_file():
        result.update({"status": "BLOCKED_BACKEND_MISSING", "blender": str(args.blender)})
    else:
        plan = build_sewing_mesh_plan(document)
        plan_path = args.output / "sewing_mesh_plan.json"
        plan_path.write_text(json.dumps(mesh_plan_to_dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
                str(args.blender),
                "--background",
                "--python",
                str(BLENDER_SCRIPT),
                "--",
                "--mesh-plan",
                str(plan_path),
                "--output",
                str(args.output),
                "--frames",
                str(args.frames),
                "--subdivision-levels",
                str(args.subdivision_levels),
            ]
        if args.body_blend:
            command.extend(["--body-blend", str(args.body_blend), "--body-object", args.body_object])
        if args.camera_metadata:
            command.extend(["--camera-metadata", str(args.camera_metadata)])
        if args.calibration:
            command.extend(["--calibration", str(args.calibration)])
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )
        metadata_path = args.output / "simulation_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError("Blender did not produce simulation_metadata.json; inspect its traceback above")
        comparison = compare_orthogonal_masks(load_reference_masks(args.target_masks), load_simulated_masks(args.output / "masks"))
        result["four_view_comparison"] = comparison
        result["minimum_mean_iou"] = args.min_mean_iou
        result["status"] = "PASS" if comparison["mean_iou"] >= args.min_mean_iou else "FAILED_VISUAL_FIDELITY"
    (args.output / "simulation_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
