from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark.pattern_pipeline.refinement import (
    final_validation,
    load_reference_masks,
    particle_silhouette_proxy,
    select_generated_candidate,
)
from benchmark.pattern_pipeline.runner import run_garment_particles_pattern_pipeline


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate and validate variable-topology Garment Particles candidates")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-masks", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260825, 20260826])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "garment_particles_refinement")
    parser.add_argument("--generation-output-root", type=Path, default=ROOT / "artifacts" / "garment_particles_layered")
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.target_masks = args.target_masks.resolve()
    args.output_root = args.output_root.resolve()
    args.generation_output_root = args.generation_output_root.resolve()

    references = load_reference_masks(args.target_masks)
    refinement_dir = args.output_root / args.sample_id
    refinement_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        candidate_id = f"{args.sample_id}_seed_{seed}"
        command = [
            sys.executable,
            "-m",
            "benchmark.scripts.run_garment_particles",
            "--input-dir",
            str(args.input_dir),
            "--sample-id",
            candidate_id,
            "--output-root",
            str(args.generation_output_root),
            "--steps",
            str(args.steps),
            "--seed",
            str(seed),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        generated_dir = args.generation_output_root / candidate_id
        prediction = generated_dir / "prediction.npz"
        canonical_dir = refinement_dir / "candidates" / candidate_id / "canonical"
        receipt = run_garment_particles_pattern_pipeline(prediction, canonical_dir)
        record = {
            "candidate_id": candidate_id,
            "seed": seed,
            "prediction": str(prediction.relative_to(ROOT)),
            "canonical_pattern": str((canonical_dir / "pattern.json").relative_to(ROOT)),
            "validation": final_validation(receipt),
            "particle_silhouette_proxy": particle_silhouette_proxy(prediction, references),
        }
        records.append(record)
        print(json.dumps({"completed_candidate": candidate_id, "validation": record["validation"], "proxy": record["particle_silhouette_proxy"]}), flush=True)

    selection = select_generated_candidate(records)
    result = {
        "sample_id": args.sample_id,
        "condition_mode": "four_view_mean_image_token_fusion",
        "selection": selection,
        "candidates": records,
        "contract": {
            "target": "visually_similar_structured_pattern",
            "exact_source_pattern_recovery_required": False,
            "variable_topology": True,
            "template_retrieval": False,
            "garmentcode_nearest_template_selection": False,
            "particle_silhouette_score_is_simulation": False,
        },
    }
    (refinement_dir / "refinement.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
