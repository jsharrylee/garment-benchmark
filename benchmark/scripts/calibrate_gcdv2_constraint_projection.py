from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.residual_learning import (
    build_retrieved_residual_model,
    build_visual_retrieval_pairs,
    geometry_metrics,
    load_crossmodal_embedding_bank,
    materialize_prediction,
    project_anchor_direction_constraints,
    read_exact_geometry_records,
)
from benchmark.scripts.calibrate_gcdv2_residual_damping import _blend, _predict


def _project(pairs, vertices, parameters, *, weight):
    projected_vertices, projected_parameters = [], []
    for pair, raw_vertices, raw_parameters in zip(pairs, vertices, parameters):
        raw_vertices = np.asarray(raw_vertices, dtype=np.float32)
        value = project_anchor_direction_constraints(
            pair.anchor.vertices_cm,
            raw_vertices,
            pair.target.edges,
            weight=weight,
        )
        curve = np.asarray(raw_parameters, dtype=np.float32).copy()
        displacement = value - raw_vertices
        # Bezier controls are absolute panel coordinates. Move them with the
        # mean projected endpoint displacement; arc radii remain scalar.
        for edge_index, (start, end) in enumerate(pair.target.edges):
            shift = 0.5 * (displacement[int(start)] + displacement[int(end)])
            curve_type = int(pair.target.curve_types[edge_index])
            if curve_type == 1:  # quadratic
                curve[edge_index, 0:2] += shift
            elif curve_type == 2:  # cubic
                curve[edge_index, 0:2] += shift
                curve[edge_index, 2:4] += shift
        projected_vertices.append(value)
        projected_parameters.append(curve)
    return projected_vertices, projected_parameters


def _objective(metrics):
    # Prefer useful geometry correction, but reject direction degradation
    # strongly enough that a tiny vertex gain cannot hide it.
    regret = max(
        0.0,
        float(metrics["edge_direction"]["edited_mae_deg"])
        - float(metrics["edge_direction"]["baseline_anchor_mae_deg"]),
    )
    return (
        float(metrics["vertex"]["edited_rmse_cm"])
        + 0.20 * float(metrics["curve_parameters"]["edited_rmse_cm"] or 0.0)
        + 5.0 * regret
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-calibrate direction-preserving least-squares projection.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"))
    parser.add_argument("--retrieval-embeddings", type=Path, default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/embeddings.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_exact/retrieved_residual.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_models/retrieved_residual/constraint_calibration.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_exact_models/retrieved_residual/constrained_heldout_predictions.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    records = read_exact_geometry_records(args.index, args.features, strict_features=True)
    splits, image, pattern = load_crossmodal_embedding_bank(args.retrieval_embeddings)
    pairs, audit = build_visual_retrieval_pairs(
        records,
        {record.sample_id: splits[record.sample_id] for record in records},
        crossmodal_image_embeddings=image,
        crossmodal_pattern_embeddings=pattern,
    )
    validation_pairs = [pair for pair in pairs if pair.split == "validation"]
    test_pairs = [pair for pair in pairs if pair.split == "test"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_retrieved_residual_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    maximum_vertices = int(checkpoint["config"]["maximum_vertices"])
    maximum_edges = int(checkpoint["config"]["maximum_edges"])
    validation_raw = _predict(
        model,
        validation_pairs,
        batch_size=args.batch_size,
        maximum_vertices=maximum_vertices,
        maximum_edges=maximum_edges,
        device=device,
    )
    candidates = []
    for alpha in (0.50, 0.65, 0.80, 0.90, 1.00):
        blended = _blend(validation_pairs, *validation_raw, alpha)
        for weight in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            projected = _project(validation_pairs, *blended, weight=weight)
            metrics = geometry_metrics(validation_pairs, *projected)
            candidates.append(
                {
                    "alpha": alpha,
                    "constraint_weight": weight,
                    "objective": _objective(metrics),
                    "metrics": metrics,
                }
            )
    best = min(candidates, key=lambda row: (row["objective"], row["constraint_weight"], row["alpha"]))
    test_raw = _predict(
        model,
        test_pairs,
        batch_size=args.batch_size,
        maximum_vertices=maximum_vertices,
        maximum_edges=maximum_edges,
        device=device,
    )
    test_blended = _blend(test_pairs, *test_raw, best["alpha"])
    test_projected = _project(test_pairs, *test_blended, weight=best["constraint_weight"])
    test_metrics = geometry_metrics(test_pairs, *test_projected)
    payload = {
        "schema_version": "gcdv2-residual-direction-constraint-calibration-1.0",
        "selection_data": "validation only",
        "test_targets_used_for_selection": False,
        "checkpoint": args.checkpoint.as_posix(),
        "candidates": candidates,
        "selected_alpha": best["alpha"],
        "selected_constraint_weight": best["constraint_weight"],
        "selected_validation_objective": best["objective"],
        "validation": best["metrics"],
        "test": test_metrics,
        "retrieval": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.predictions.open("w", encoding="utf-8") as stream:
        for pair, vertices, parameters in zip(test_pairs, *test_projected):
            stream.write(json.dumps(materialize_prediction(pair, vertices, parameters), sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_alpha": best["alpha"],
                "selected_constraint_weight": best["constraint_weight"],
                "validation": best["metrics"],
                "test": test_metrics,
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
