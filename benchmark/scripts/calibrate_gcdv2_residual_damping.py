from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.residual_learning import (
    COORDINATE_SCALE_CM,
    batch_residual_pairs,
    build_retrieved_residual_model,
    build_visual_retrieval_pairs,
    geometry_metrics,
    load_crossmodal_embedding_bank,
    materialize_prediction,
    read_exact_geometry_records,
)
from benchmark.scripts.train_gcdv2_retrieved_residual import (
    _project_native_curve_parameters,
    _tensor_batch,
)


def _predict(model, pairs, *, batch_size, maximum_vertices, maximum_edges, device):
    import torch

    vertices, parameters = [], []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(pairs), batch_size):
            current = pairs[offset : offset + batch_size]
            raw = batch_residual_pairs(
                current,
                maximum_vertices=maximum_vertices,
                maximum_edges=maximum_edges,
            )
            batch = _tensor_batch(raw, device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    visual_features=batch["visual_features"],
                    anchor_vertices=batch["anchor_vertices"],
                    vertex_mask=batch["vertex_mask"],
                    vertex_panel_indices=batch["vertex_panel_indices"],
                    vertex_local_indices=batch["vertex_local_indices"],
                    anchor_curve_parameters=batch["anchor_curve_parameters"],
                    edge_mask=batch["edge_mask"],
                    edge_vertices=batch["edge_vertices"],
                    edge_panel_indices=batch["edge_panel_indices"],
                    edge_local_indices=batch["edge_local_indices"],
                    curve_types=batch["curve_types"],
                    category=batch["category"],
                )
            raw_vertices = output["predicted_vertices"].float().cpu().numpy() * COORDINATE_SCALE_CM
            raw_parameters = output["predicted_curve_parameters"].float().cpu().numpy() * COORDINATE_SCALE_CM
            for index, pair in enumerate(current):
                vertices.append(raw_vertices[index, : len(pair.target.vertices_cm)])
                parameters.append(
                    _project_native_curve_parameters(
                        pair, raw_parameters[index, : len(pair.target.edges)]
                    )
                )
    return vertices, parameters


def _blend(pairs, vertices, parameters, alpha):
    blended_vertices, blended_parameters = [], []
    for pair, predicted_vertices, predicted_parameters in zip(pairs, vertices, parameters):
        value_vertices = pair.anchor.vertices_cm + alpha * (
            np.asarray(predicted_vertices) - pair.anchor.vertices_cm
        )
        value_parameters = pair.anchor.curve_parameters_cm + alpha * (
            np.asarray(predicted_parameters) - pair.anchor.curve_parameters_cm
        )
        blended_vertices.append(value_vertices)
        blended_parameters.append(_project_native_curve_parameters(pair, value_parameters))
    return blended_vertices, blended_parameters


def _objective(metrics, direction_regret_weight):
    direction_regret = max(
        0.0,
        float(metrics["edge_direction"]["edited_mae_deg"])
        - float(metrics["edge_direction"]["baseline_anchor_mae_deg"]),
    )
    return (
        float(metrics["vertex"]["edited_rmse_cm"])
        + 0.20 * float(metrics["curve_parameters"]["edited_rmse_cm"] or 0.0)
        + direction_regret_weight * direction_regret
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-calibrate residual damping without touching held-out targets.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"))
    parser.add_argument("--retrieval-embeddings", type=Path, default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/embeddings.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_exact/retrieved_residual.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_models/retrieved_residual/damping_calibration.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_exact_models/retrieved_residual/damped_heldout_predictions.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--direction-regret-weight", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    records = read_exact_geometry_records(args.index, args.features, strict_features=True)
    splits, image_embeddings, pattern_embeddings = load_crossmodal_embedding_bank(
        args.retrieval_embeddings
    )
    split_by_id = {record.sample_id: splits[record.sample_id] for record in records}
    pairs, audit = build_visual_retrieval_pairs(
        records,
        split_by_id,
        crossmodal_image_embeddings=image_embeddings,
        crossmodal_pattern_embeddings=pattern_embeddings,
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
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = _blend(validation_pairs, *validation_raw, float(alpha))
        metrics = geometry_metrics(validation_pairs, *blended)
        candidates.append(
            {
                "alpha": float(alpha),
                "objective": _objective(metrics, args.direction_regret_weight),
                "metrics": metrics,
            }
        )
    best = min(candidates, key=lambda value: (value["objective"], value["alpha"]))
    test_raw = _predict(
        model,
        test_pairs,
        batch_size=args.batch_size,
        maximum_vertices=maximum_vertices,
        maximum_edges=maximum_edges,
        device=device,
    )
    test_blended = _blend(test_pairs, *test_raw, float(best["alpha"]))
    test_metrics = geometry_metrics(test_pairs, *test_blended)
    payload = {
        "schema_version": "gcdv2-residual-damping-calibration-1.0",
        "checkpoint": args.checkpoint.as_posix(),
        "selection_data": "validation only",
        "test_targets_used_for_alpha_selection": False,
        "candidate_alphas": candidates,
        "selected_alpha": best["alpha"],
        "selected_validation_objective": best["objective"],
        "validation": best["metrics"],
        "test": test_metrics,
        "retrieval": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as stream:
        for pair, vertices, parameters in zip(test_pairs, *test_blended):
            stream.write(json.dumps(materialize_prediction(pair, vertices, parameters), sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_alpha": best["alpha"],
                "validation": best["metrics"],
                "test": test_metrics,
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
