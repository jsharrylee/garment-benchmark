from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark.adapters.blender_character import prepare_layer_bundles


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end layered, four-view, generative pattern and sewing benchmark")
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--split-ratio", type=float, required=True)
    parser.add_argument("--layers", nargs="+", choices=("upper", "lower"), default=["upper", "lower"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260825, 20260826])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--reuse-refinement", action="store_true")
    parser.add_argument(
        "--repair-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "pattern_repair" / "pattern_repair_net.pt",
    )
    parser.add_argument("--repair-maximum-passes", type=int, default=3)
    parser.add_argument("--skip-pattern-repair", action="store_true")
    args = parser.parse_args()

    sample_root = args.sample_root.resolve()
    layer_manifest = prepare_layer_bundles(sample_root, split_ratio=args.split_ratio)
    records = []
    for layer in args.layers:
        layer_id = f"{args.sample_id}_{layer}"
        layer_root = sample_root / "layers" / layer
        refinement_root = ROOT / "artifacts" / "garment_particles_refinement" / layer_id
        refinement_path = refinement_root / "refinement.json"
        if not args.reuse_refinement or not refinement_path.is_file():
            run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.scripts.refine_garment_particles",
                    "--input-dir",
                    str(layer_root / "garment_particles" / "views"),
                    "--target-masks",
                    str(layer_root / "masks"),
                    "--sample-id",
                    layer_id,
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                    "--steps",
                    str(args.steps),
                ]
            )
        refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
        selected_id = refinement["selection"]["selected_candidate_id"]
        selected = next(candidate for candidate in refinement["candidates"] if candidate["candidate_id"] == selected_id)
        canonical = ROOT / selected["canonical_pattern"]
        repair = None
        if not args.skip_pattern_repair:
            if not args.repair_checkpoint.is_file():
                raise FileNotFoundError(
                    f"pattern repair checkpoint missing: {args.repair_checkpoint}; "
                    "run benchmark.scripts.train_pattern_repair_model"
                )
            repair_output = ROOT / "artifacts" / "pattern_repair" / layer_id
            run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.scripts.apply_pattern_repair_model",
                    str(canonical),
                    "--checkpoint",
                    str(args.repair_checkpoint.resolve()),
                    "--output",
                    str(repair_output),
                    "--maximum-passes",
                    str(args.repair_maximum_passes),
                ]
            )
            repair = json.loads((repair_output / "repair_receipt.json").read_text(encoding="utf-8"))
            canonical = repair_output / "pattern.json"
        simulation_output = ROOT / "artifacts" / "blender_sewing" / layer_id
        run(
            [
                sys.executable,
                "-m",
                "benchmark.scripts.run_blender_sewing_simulation",
                str(canonical),
                "--target-masks",
                str(layer_root / "masks"),
                "--output",
                str(simulation_output),
                "--frames",
                str(args.frames),
            ]
        )
        simulation = json.loads((simulation_output / "simulation_receipt.json").read_text(encoding="utf-8"))
        records.append(
            {
                "layer": layer,
                "layer_id": layer_id,
                "refinement": refinement["selection"],
                "selected_validation": selected["validation"],
                "particle_silhouette_proxy": selected["particle_silhouette_proxy"],
                "learned_pattern_repair": repair,
                "sewing_simulation": simulation,
            }
        )

    output = ROOT / "artifacts" / "layered_generative_pipeline" / args.sample_id
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "sample_id": args.sample_id,
        "layer_split": {
            "method": layer_manifest["method"],
            "split_ratio": layer_manifest["split_ratio"],
            "source_pattern_inferred_from_uv": False,
        },
        "generation_contract": {
            "variable_topology": True,
            "template_retrieval": False,
            "nearest_garmentcode_pattern_selection": False,
            "four_view_conditioning": True,
            "exact_source_pattern_recovery_required": False,
            "learned_topology_preserving_repair": not args.skip_pattern_repair,
        },
        "layers": records,
        "structural_pass": all(record["sewing_simulation"]["structural_validation"]["accepted"] for record in records),
        "simulation_executed": all(
            record["sewing_simulation"]["status"] not in {"BLOCKED_STRUCTURAL_VALIDATION", "BLOCKED_BACKEND_MISSING"}
            for record in records
        ),
        "visual_pass": all(record["sewing_simulation"]["status"] == "PASS" for record in records),
        "technical_pass": all(record["sewing_simulation"]["status"] == "PASS" for record in records),
        "manufacturing_ready": False,
    }
    (output / "pipeline_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
