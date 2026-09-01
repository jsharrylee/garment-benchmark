from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.residual_learning import (
    COORDINATE_SCALE_CM,
    ExactGeometryRecord,
    RetrievedResidualPair,
    batch_residual_pairs,
    build_retrieved_residual_model,
    build_visual_retrieval_pairs,
    deterministic_topology_split,
    geometry_metrics,
    load_crossmodal_embedding_bank,
    materialize_prediction,
    read_exact_geometry_records,
    residual_loss,
)


def _smoke_subset(
    records: Sequence[ExactGeometryRecord], maximum_records: int
) -> tuple[ExactGeometryRecord, ...]:
    """Select complete, repeatable topology groups for a meaningful smoke run."""

    groups: dict[str, list[ExactGeometryRecord]] = defaultdict(list)
    for record in records:
        groups[record.topology_hash].append(record)
    selected: list[ExactGeometryRecord] = []
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (-len(group), group[0].category, group[0].topology_hash),
    )
    # At least four records are needed to produce two train-bank items plus
    # validation and test targets from a topology.
    for group in ordered_groups:
        if len(group) < 4:
            continue
        remaining = maximum_records - len(selected)
        if remaining < 4:
            break
        selected.extend(sorted(group, key=lambda item: item.sample_id)[:remaining])
        if len(selected) >= maximum_records:
            break
    if len(selected) < 4:
        raise ValueError("smoke subset has no topology group with at least four records")
    return tuple(selected)


def _tensor_batch(raw: Mapping[str, Any], device):
    import torch

    boolean = {"vertex_mask", "curve_parameter_mask", "edge_mask"}
    integer = {
        "vertex_panel_indices",
        "vertex_local_indices",
        "edge_vertices",
        "edge_panel_indices",
        "edge_local_indices",
        "curve_types",
        "category",
    }
    result = {}
    for key, value in raw.items():
        if not isinstance(value, np.ndarray):
            continue
        tensor = torch.from_numpy(value)
        if key in boolean:
            tensor = tensor.bool()
        elif key in integer:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        result[key] = tensor.to(device)
    return result


def _project_native_curve_parameters(
    pair: RetrievedResidualPair, values: np.ndarray
) -> np.ndarray:
    projected = np.asarray(values, dtype=np.float32).copy()
    for index, curve_type in enumerate(pair.target.curve_types):
        # circular_arc is the fourth type in the exact schema.  Arc branch
        # flags remain frozen; only a positive radius is admissible.
        if int(curve_type) == 3:
            projected[index, 4] = max(abs(float(projected[index, 4])), 1e-4)
    return projected


def _evaluate(
    model,
    pairs: Sequence[RetrievedResidualPair],
    *,
    batch_size: int,
    maximum_vertices: int,
    maximum_edges: int,
    device,
    include_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    if not pairs:
        return {"pair_count": 0, "status": "NO_COMPATIBLE_PAIRS"}, []
    predicted_vertices = []
    predicted_parameters = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            current = pairs[start : start + batch_size]
            raw = batch_residual_pairs(
                current,
                maximum_vertices=maximum_vertices,
                maximum_edges=maximum_edges,
            )
            batch = _tensor_batch(raw, device)
            with torch.amp.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
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
            vertices = (
                output["predicted_vertices"].float().cpu().numpy()
                * COORDINATE_SCALE_CM
            )
            parameters = (
                output["predicted_curve_parameters"].float().cpu().numpy()
                * COORDINATE_SCALE_CM
            )
            for index, pair in enumerate(current):
                vertex_count = len(pair.target.vertices_cm)
                edge_count = len(pair.target.edges)
                predicted_vertices.append(vertices[index, :vertex_count])
                predicted_parameters.append(
                    _project_native_curve_parameters(pair, parameters[index, :edge_count])
                )
    metrics = geometry_metrics(pairs, predicted_vertices, predicted_parameters)
    predictions = []
    if include_predictions:
        predictions = [
            materialize_prediction(pair, vertices, parameters)
            for pair, vertices, parameters in zip(
                pairs, predicted_vertices, predicted_parameters
            )
        ]
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train topology-preserving exact-geometry corrections from semantic four-view FPN "
            "tokens and a non-self visually retrieved training anchor."
        )
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_exact_models/retrieved_residual"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gcdv2_exact/retrieved_residual.pt"),
    )
    parser.add_argument(
        "--retrieval-embeddings",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/embeddings.npz"),
        help="Stage-2 image/pattern embeddings and their leakage-safe split assignments.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--vertex-weight", type=float, default=1.0)
    parser.add_argument("--curve-weight", type=float, default=0.50)
    parser.add_argument("--chord-length-weight", type=float, default=0.40)
    parser.add_argument("--direction-weight", type=float, default=1.0)
    parser.add_argument(
        "--direction-regret-weight",
        type=float,
        default=0.50,
        help="Validation penalty per degree by which editing is worse than the retrieved anchor.",
    )
    parser.add_argument(
        "--maximum-records",
        type=int,
        help="deterministic complete-topology subset for smoke/debug runs",
    )
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    records = read_exact_geometry_records(args.index, args.features)
    exact_index_rows = [
        json.loads(line)
        for line in args.index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feature_record_ids = {record.sample_id for record in records}
    missing_feature_ids = [
        str(row["sample_id"])
        for row in exact_index_rows
        if str(row["sample_id"]) not in feature_record_ids
    ]
    feature_coverage = {
        "exact_pair_records": len(exact_index_rows),
        "records_with_existing_fpn_features": len(records),
        "records_missing_existing_fpn_features": len(missing_feature_ids),
        "coverage": len(records) / max(len(exact_index_rows), 1),
        "missing_sample_ids": missing_feature_ids,
        "policy": "missing features excluded, never replaced with zeros",
    }
    print(json.dumps({"feature_coverage": feature_coverage["coverage"], "missing": len(missing_feature_ids)}), flush=True)
    if args.maximum_records is not None:
        records = _smoke_subset(records, int(args.maximum_records))
    if args.retrieval_embeddings.is_file():
        stage2_splits, stage2_images, stage2_patterns = load_crossmodal_embedding_bank(
            args.retrieval_embeddings
        )
        missing_stage2 = [record.sample_id for record in records if record.sample_id not in stage2_splits]
        if missing_stage2:
            raise KeyError(f"Stage-2 embeddings are incomplete; first missing ID: {missing_stage2[0]}")
        split_by_id = {record.sample_id: stage2_splits[record.sample_id] for record in records}
        pairs, retrieval_audit = build_visual_retrieval_pairs(
            records,
            split_by_id,
            crossmodal_image_embeddings=stage2_images,
            crossmodal_pattern_embeddings=stage2_patterns,
        )
        split_method = "reuse Stage-2 stratified split"
    else:
        split_by_id = deterministic_topology_split(records, seed=args.seed)
        pairs, retrieval_audit = build_visual_retrieval_pairs(records, split_by_id)
        split_method = "standalone topology-stratified fallback"
    train_pairs = [pair for pair in pairs if pair.split == "train"]
    validation_pairs = [pair for pair in pairs if pair.split == "validation"]
    test_pairs = [pair for pair in pairs if pair.split == "test"]
    if not train_pairs or not validation_pairs or not test_pairs:
        raise ValueError(
            "training requires pairable train/validation/test lanes; got "
            f"{len(train_pairs)}/{len(validation_pairs)}/{len(test_pairs)}"
        )
    maximum_vertices = max(len(record.vertices_cm) for record in records)
    maximum_edges = max(len(record.edges) for record in records)
    maximum_panels = max(len(record.panel_ids) for record in records)
    maximum_local_vertices = max(
        int(record.vertex_local_indices.max()) + 1 for record in records
    )
    maximum_local_edges = max(int(record.edge_local_indices.max()) + 1 for record in records)
    visual_shape = records[0].spatial_features.shape
    config = {
        "seed": args.seed,
        "width": args.width,
        "heads": args.heads,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "visual_feature_dimension": int(visual_shape[-1]),
        "maximum_visual_tokens_per_view": int(visual_shape[-2]),
        "maximum_vertices": maximum_vertices,
        "maximum_edges": maximum_edges,
        "maximum_panels": maximum_panels,
        "maximum_local_vertices": maximum_local_vertices,
        "maximum_local_edges": maximum_local_edges,
        "coordinate_scale_cm": COORDINATE_SCALE_CM,
        "view_contract": {
            "cache_order": ["CAM000", "CAM001", "CAM002", "CAM003"],
            "model_order": ["front/CAM001", "back/CAM000", "left/CAM002", "right/CAM003"],
            "reorder_indices": [1, 0, 2, 3],
        },
        "frozen_topology": [
            "panel inventory/order/identity",
            "edge-to-vertex incidence",
            "native curve type",
            "circular arc branch flags",
            "stitch pairs",
        ],
        "learned_geometry": [
            "one xy residual per shared panel vertex",
            "quadratic/cubic Bezier control residuals",
            "circular arc radius residual",
        ],
        "loss_weights": {
            "vertex": args.vertex_weight,
            "curve": args.curve_weight,
            "chord_length": args.chord_length_weight,
            "direction": args.direction_weight,
            "validation_direction_regret": args.direction_regret_weight,
        },
    }
    model = build_retrieved_residual_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_pairs))
        totals: Counter[str] = Counter()
        batches = 0
        for start in range(0, len(order), args.batch_size):
            current = [train_pairs[index] for index in order[start : start + args.batch_size]]
            raw = batch_residual_pairs(
                current,
                maximum_vertices=maximum_vertices,
                maximum_edges=maximum_edges,
            )
            batch = _tensor_batch(raw, device)
            with torch.amp.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
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
                losses = residual_loss(
                    output,
                    batch,
                    vertex_weight=args.vertex_weight,
                    curve_weight=args.curve_weight,
                    chord_length_weight=args.chord_length_weight,
                    direction_weight=args.direction_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu())
            batches += 1
        validation_metrics, _ = _evaluate(
            model,
            validation_pairs,
            batch_size=args.batch_size,
            maximum_vertices=maximum_vertices,
            maximum_edges=maximum_edges,
            device=device,
        )
        direction_regret = max(
            0.0,
            float(validation_metrics["edge_direction"]["edited_mae_deg"])
            - float(validation_metrics["edge_direction"]["baseline_anchor_mae_deg"]),
        )
        score = (
            float(validation_metrics["vertex"]["edited_rmse_cm"])
            + 0.20 * float(validation_metrics["curve_parameters"]["edited_rmse_cm"] or 0.0)
            + args.direction_regret_weight * direction_regret
        )
        row = {
            "epoch": epoch,
            "training": {key: value / max(batches, 1) for key, value in totals.items()},
            "validation_score": score,
            "validation_vertex_rmse_cm": validation_metrics["vertex"]["edited_rmse_cm"],
            "validation_anchor_vertex_rmse_cm": validation_metrics["vertex"]["baseline_anchor_rmse_cm"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics, _ = _evaluate(
        model,
        train_pairs,
        batch_size=args.batch_size,
        maximum_vertices=maximum_vertices,
        maximum_edges=maximum_edges,
        device=device,
    )
    validation_metrics, validation_predictions = _evaluate(
        model,
        validation_pairs,
        batch_size=args.batch_size,
        maximum_vertices=maximum_vertices,
        maximum_edges=maximum_edges,
        device=device,
        include_predictions=True,
    )
    test_metrics, test_predictions = _evaluate(
        model,
        test_pairs,
        batch_size=args.batch_size,
        maximum_vertices=maximum_vertices,
        maximum_edges=maximum_edges,
        device=device,
        include_predictions=True,
    )
    elapsed = time.perf_counter() - started
    split_counts = Counter(split_by_id.values())
    pair_manifest = [
        {
            "target_id": pair.target.sample_id,
            "anchor_id": pair.anchor.sample_id,
            "split": pair.split,
            "category": pair.target.category,
            "topology_hash": pair.target.topology_hash,
            "visual_cosine_similarity": pair.visual_cosine_similarity,
            "anchor_is_training_bank": split_by_id[pair.anchor.sample_id] == "train",
            "target_self_excluded": pair.target.sample_id != pair.anchor.sample_id,
        }
        for pair in pairs
    ]
    payload = {
        "schema_version": "gcdv2-retrieved-residual-training-1.0",
        "status": "COMPLETE",
        "architecture": {
            "input": "semantic-order four-view frozen ResNet50-FPN tokens plus exact compatible anchor geometry",
            "model": "two cross-attention Transformer decoders: shared-vertex residuals and native curve-parameter residuals",
            "trainable_parameters": sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
            "curve_type_head": None,
            "curve_type_policy": "fixed by compatible topology instead of riskily reclassified",
            **config,
        },
        "split_contract": {
            "method": split_method,
            "seed": args.seed,
            "record_counts": dict(split_counts),
            "pair_counts": {
                "train": len(train_pairs),
                "validation": len(validation_pairs),
                "test": len(test_pairs),
            },
            "all_anchors_from_train_bank": all(
                split_by_id[pair.anchor.sample_id] == "train" for pair in pairs
            ),
            "all_targets_different_from_anchor": all(
                pair.target.sample_id != pair.anchor.sample_id for pair in pairs
            ),
            "all_topologies_equal": all(
                pair.target.topology_hash == pair.anchor.topology_hash for pair in pairs
            ),
            "held_out_target_geometry_used_for_retrieval": False,
            "held_out_target_topology_used_as_compatibility_gate": True,
        },
        "retrieval_coverage": retrieval_audit,
        "stage2_integration": {
            "embedding_artifact": str(args.retrieval_embeddings.as_posix()),
            "embedding_artifact_used": args.retrieval_embeddings.is_file(),
            "split_contract": split_method,
        },
        "existing_fpn_feature_coverage": feature_coverage,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "training_seconds": elapsed,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "history": history,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "interpretation_gate": {
            "technical_success_requires_test_vertex_improvement": (
                test_metrics["vertex"]["relative_improvement"] is not None
                and test_metrics["vertex"]["relative_improvement"] > 0.0
            ),
            "uncovered_topologies_are_not_silently_edited": True,
            "prediction_scope": "geometry correction only; no topology generation",
        },
        "limitations": [
            "Exact-topology compatibility leaves singleton and rare generator structures uncovered.",
            "The split is sample-disjoint but not generator-family-disjoint; it measures the accepted same-generator regime.",
            "Anchor selection uses the trained Stage-2 four-view-to-2D embedding when available; exact topology remains an oracle compatibility gate.",
            "The exact target topology is used as an oracle compatibility gate; deployment needs a preceding topology classifier/retrieval gate.",
            "No new panels, edges, stitches, or curve types can be generated in this stage.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "retrieval_pairs.jsonl").open("w", encoding="utf-8") as stream:
        for row in pair_manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output / "heldout_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in (*validation_predictions, *test_predictions):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": payload["schema_version"],
            "config": config,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "split_by_id": split_by_id,
            "training_bank_ids": sorted(
                sample_id for sample_id, split in split_by_id.items() if split == "train"
            ),
        },
        args.checkpoint,
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "stopped_epoch": history[-1]["epoch"],
                "training_seconds": elapsed,
                "retrieval_coverage": retrieval_audit["coverage"],
                "test_vertex": test_metrics["vertex"],
                "test_edge_length": test_metrics["edge_length"],
                "test_edge_direction": test_metrics["edge_direction"],
                "metrics": str((args.output / "training_metrics.json").as_posix()),
                "predictions": str((args.output / "heldout_predictions.jsonl").as_posix()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
