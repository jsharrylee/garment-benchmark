from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.pattern_learning import (
    CATEGORIES,
    IMAGE_SIZE,
    MAXIMUM_EDGES,
    MAXIMUM_PANELS,
    PRIMITIVE_TYPES,
    PatternExample,
    build_pattern_parser_model,
    classification_metrics,
    family_disjoint_split,
    hungarian_matches,
    padded_pattern_batch,
    ordered_pattern_parser_loss,
    pattern_parser_loss,
    read_pattern_examples,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_batch(batch: Mapping[str, Any], device):
    import torch

    return {
        key: torch.from_numpy(value).to(device)
        for key, value in batch.items()
        if isinstance(value, np.ndarray)
    }


def _binary_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    predicted, expected = predicted.astype(bool), expected.astype(bool)
    tp = int(np.sum(predicted & expected))
    fp = int(np.sum(predicted & ~expected))
    fn = int(np.sum(~predicted & expected))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def _evaluate(
    model,
    examples: Sequence[PatternExample],
    *,
    batch_size: int,
    device,
    category_weights=None,
    primitive_weights=None,
) -> dict[str, Any]:
    import torch

    model.eval()
    losses = []
    category_predictions, category_targets = [], []
    primitive_predictions, primitive_targets = [], []
    edge_presence_predictions, edge_presence_targets = [], []
    panel_presence_predictions, panel_presence_targets = [], []
    endpoint_errors, length_errors, direction_errors, panel_errors = [], [], [], []
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            current = examples[offset : offset + batch_size]
            raw = padded_pattern_batch(current)
            batch = _tensor_batch(raw, device)
            output = model(batch["spatial_features"])
            result = pattern_parser_loss(
                output,
                batch,
                category_weights=category_weights,
                primitive_weights=primitive_weights,
            )
            losses.extend([float(result["loss"].item())] * len(current))
            category_predictions.extend(output["category_logits"].argmax(dim=-1).cpu().tolist())
            category_targets.extend(batch["categories"].cpu().tolist())
            for row, (edge_match, panel_match) in enumerate(
                zip(result["edge_matches"], result["panel_matches"])
            ):
                edge_query, edge_target = edge_match
                edge_expected_presence = torch.zeros(MAXIMUM_EDGES, device=device, dtype=torch.bool)
                edge_expected_presence[edge_query] = True
                edge_presence_predictions.extend(
                    (output["edge_presence_logits"][row].sigmoid() >= 0.5).cpu().tolist()
                )
                edge_presence_targets.extend(edge_expected_presence.cpu().tolist())
                if len(edge_query):
                    prediction = output["edge_geometry"][row, edge_query]
                    expected = batch["edge_geometry"][row, edge_target]
                    primitive_predictions.extend(
                        output["edge_type_logits"][row, edge_query].argmax(dim=-1).cpu().tolist()
                    )
                    primitive_targets.extend(batch["edge_types"][row, edge_target].cpu().tolist())
                    endpoint_errors.extend(
                        torch.abs(prediction[:, :4] - expected[:, :4]).mean(dim=-1).cpu().tolist()
                    )
                    length_errors.extend(torch.abs(prediction[:, 4] - expected[:, 4]).cpu().tolist())
                    cosine = (prediction[:, 5:7] * expected[:, 5:7]).sum(dim=-1).clamp(-1.0, 1.0)
                    direction_errors.extend(torch.rad2deg(torch.acos(cosine)).cpu().tolist())
                panel_query, panel_target = panel_match
                panel_expected_presence = torch.zeros(MAXIMUM_PANELS, device=device, dtype=torch.bool)
                panel_expected_presence[panel_query] = True
                panel_presence_predictions.extend(
                    (output["panel_presence_logits"][row].sigmoid() >= 0.5).cpu().tolist()
                )
                panel_presence_targets.extend(panel_expected_presence.cpu().tolist())
                if len(panel_query):
                    panel_errors.extend(
                        torch.abs(
                            output["panel_boxes"][row, panel_query]
                            - batch["panel_boxes"][row, panel_target]
                        )
                        .mean(dim=-1)
                        .cpu()
                        .tolist()
                    )
    category = classification_metrics(
        np.asarray(category_predictions), np.asarray(category_targets), CATEGORIES
    )
    primitive = classification_metrics(
        np.asarray(primitive_predictions), np.asarray(primitive_targets), PRIMITIVE_TYPES
    )
    return {
        "loss": float(np.mean(losses)),
        "selection_score": -float(np.mean(losses)),
        "sample_count": len(examples),
        "category": category,
        "primitive_type_on_matched_edges": primitive,
        "edge_presence": _binary_metrics(
            np.asarray(edge_presence_predictions), np.asarray(edge_presence_targets)
        ),
        "panel_presence": _binary_metrics(
            np.asarray(panel_presence_predictions), np.asarray(panel_presence_targets)
        ),
        "packed_endpoint_mae_canvas_fraction": float(np.mean(endpoint_errors)),
        "packed_endpoint_mae_pixels_at_1024": float(np.mean(endpoint_errors)) * IMAGE_SIZE,
        "rendered_length_mae_canvas_fraction": float(np.mean(length_errors)),
        "rendered_length_mae_pixels_at_1024": float(np.mean(length_errors)) * IMAGE_SIZE,
        "direction_mean_angular_error_deg": float(np.mean(direction_errors)),
        "panel_bbox_mae_canvas_fraction": float(np.mean(panel_errors)),
    }


def _prediction_rows(model, examples: Sequence[PatternExample], device) -> list[dict[str, Any]]:
    import torch

    output_rows = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            raw = padded_pattern_batch([example])
            batch = _tensor_batch(raw, device)
            output = model(batch["spatial_features"])
            edge_matches, panel_matches = hungarian_matches(output, batch)
            scores = output["edge_presence_logits"][0].sigmoid()
            types = output["edge_type_logits"][0].softmax(dim=-1)
            geometry = output["edge_geometry"][0]
            matched = []
            for query, target in zip(edge_matches[0][0].cpu().tolist(), edge_matches[0][1].cpu().tolist()):
                prediction = geometry[query].cpu().numpy()
                truth = example.edge_geometry[target]
                predicted_type = int(types[query].argmax().item())
                length_fraction = float(prediction[4])
                matched.append(
                    {
                        "query_index": query,
                        "target_set_index": target,
                        "target_ref": example.edge_refs[target],
                        "presence_probability": float(scores[query]),
                        "predicted_primitive_type": PRIMITIVE_TYPES[predicted_type],
                        "target_primitive_type": PRIMITIVE_TYPES[int(example.edge_types[target])],
                        "predicted_packed_start_uv": prediction[0:2].tolist(),
                        "predicted_packed_end_uv": prediction[2:4].tolist(),
                        "target_packed_start_uv": truth[0:2].tolist(),
                        "target_packed_end_uv": truth[2:4].tolist(),
                        "predicted_rendered_length_canvas_fraction": length_fraction,
                        "target_rendered_length_canvas_fraction": float(truth[4]),
                        "predicted_direction_sin_cos_image_xy": prediction[5:7].tolist(),
                        "target_direction_sin_cos_image_xy": truth[5:7].tolist(),
                        # This is an evaluation-only conversion using sidecar
                        # metadata.  It was not available to the image model.
                        "predicted_length_cm_posthoc_from_sidecar_scale": length_fraction
                        * example.canvas_size_px
                        / example.packing_scale_px_per_cm,
                    }
                )
            panels = []
            panel_scores = output["panel_presence_logits"][0].sigmoid()
            for query, target in zip(panel_matches[0][0].cpu().tolist(), panel_matches[0][1].cpu().tolist()):
                panels.append(
                    {
                        "query_index": query,
                        "target_set_index": target,
                        "target_ref": example.panel_refs[target],
                        "presence_probability": float(panel_scores[query]),
                        "predicted_packed_bbox_uv": output["panel_boxes"][0, query].cpu().tolist(),
                        "target_packed_bbox_uv": example.panel_boxes[target].tolist(),
                    }
                )
            output_rows.append(
                {
                    "schema_version": "gcdv2-pattern-parser-prediction-1.0",
                    "sample_id": example.sample_id,
                    "split": "test",
                    "input_pattern_path": str(example.pattern_path.as_posix()),
                    "exact_truth_sidecar": str(example.label_path.as_posix()),
                    "input_contract": "clean pattern.png only",
                    "scale_contract": {
                        "network_outputs": "packed UV endpoints, rendered length/canvas, image-space direction",
                        "absolute_cm_regressed_from_image": False,
                        "packing_scale_px_per_cm_evaluation_metadata_only": example.packing_scale_px_per_cm,
                    },
                    "target_category": example.category,
                    "predicted_category": CATEGORIES[int(output["category_logits"][0].argmax())],
                    "target_panel_count": len(example.panel_boxes),
                    "predicted_panel_count_at_0_5": int((panel_scores >= 0.5).sum()),
                    "target_edge_count": len(example.edge_geometry),
                    "predicted_edge_count_at_0_5": int((scores >= 0.5).sum()),
                    "matched_panels": panels,
                    "matched_edges": matched,
                }
            )
    return output_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_nonempty_splits(examples, assignments, seed):
    """Small-cache smoke fallback; the full corpus remains family-disjoint."""

    counts = Counter(assignments.values())
    if all(counts[name] for name in ("train", "validation", "test")):
        return assignments, None
    ordered = sorted(
        examples,
        key=lambda item: hashlib.sha256(f"fallback:{seed}:{item.sample_id}".encode()).hexdigest(),
    )
    if len(ordered) < 3:
        raise ValueError("at least three cached examples are required")
    fallback = {item.sample_id: "train" for item in ordered}
    fallback[ordered[0].sample_id] = "test"
    fallback[ordered[1].sample_id] = "validation"
    return fallback, "sample_disjoint_fallback_for_incomplete_smoke_cache"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an image-only DETR-style exact-pattern set parser."
    )
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/pattern_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gcdv2_exact/pattern_set_parser.pt"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/pattern_parser_metrics.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/pattern_parser_test_predictions.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--training-matching",
        choices=("hungarian", "canonical"),
        default="hungarian",
        help="Hungarian is permutation-invariant; canonical is the faster ablation.",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=4,
        help="Run the slower Hungarian set evaluation every N epochs.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
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
    examples = read_pattern_examples(args.index, feature_path=args.features, limit=args.limit)
    assignments, split_audit = family_disjoint_split(examples, seed=args.seed)
    assignments, fallback = _ensure_nonempty_splits(examples, assignments, args.seed)
    if fallback:
        split_audit["fallback"] = fallback
        split_audit["family_disjoint"] = False
    split = {
        name: tuple(item for item in examples if assignments[item.sample_id] == name)
        for name in ("train", "validation", "test")
    }
    config = {
        "feature_dim": 256,
        "width": args.width,
        "heads": args.heads,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "feedforward_multiplier": 4,
        "dropout": 0.1,
    }
    model = build_pattern_parser_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    category_counts = np.ones(len(CATEGORIES), dtype=np.float64)
    primitive_counts = np.ones(len(PRIMITIVE_TYPES), dtype=np.float64)
    for example in split["train"]:
        category_counts[CATEGORIES.index(example.category)] += 1
        primitive_counts += np.bincount(example.edge_types, minlength=len(PRIMITIVE_TYPES))
    category_weights_np = 1.0 / np.sqrt(category_counts)
    category_weights_np /= category_weights_np.mean()
    primitive_weights_np = 1.0 / np.sqrt(primitive_counts)
    primitive_weights_np /= primitive_weights_np.mean()
    category_weights = torch.as_tensor(category_weights_np, dtype=torch.float32, device=device)
    primitive_weights = torch.as_tensor(primitive_weights_np, dtype=torch.float32, device=device)
    generator = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    started = time.perf_counter()
    train_rows = list(split["train"])
    for epoch in range(1, args.epochs + 1):
        generator.shuffle(train_rows)
        model.train()
        totals: dict[str, list[float]] = defaultdict(list)
        for offset in range(0, len(train_rows), args.batch_size):
            raw = padded_pattern_batch(train_rows[offset : offset + args.batch_size])
            batch = _tensor_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["spatial_features"])
                loss_function = (
                    pattern_parser_loss
                    if args.training_matching == "hungarian"
                    else ordered_pattern_parser_loss
                )
                losses = loss_function(
                    output,
                    batch,
                    category_weights=category_weights,
                    primitive_weights=primitive_weights,
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                if key.endswith("loss") and hasattr(value, "item"):
                    totals[key].append(float(value.item()))
        train_summary = {key: float(np.mean(value)) for key, value in totals.items()}
        should_validate = epoch == 1 or epoch % max(1, args.validation_interval) == 0 or epoch == args.epochs
        row = {"epoch": epoch, "train": train_summary, "validation": None}
        if should_validate:
            validation = _evaluate(
                model,
                split["validation"],
                batch_size=args.batch_size,
                device=device,
                category_weights=category_weights,
                primitive_weights=primitive_weights,
            )
            row["validation"] = validation
            print(
                f"epoch {epoch:03d} train={train_summary['loss']:.5f} "
                f"val={validation['loss']:.5f} category={validation['category']['accuracy']:.3f} "
                f"primitive={validation['primitive_type_on_matched_edges']['accuracy']:.3f}"
            )
            if validation["loss"] < best_loss - 1e-5:
                best_loss = validation["loss"]
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        elif epoch % 2 == 0:
            print(f"epoch {epoch:03d} train={train_summary['loss']:.5f}")
        history.append(row)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    validation = _evaluate(
        model,
        split["validation"],
        batch_size=args.batch_size,
        device=device,
        category_weights=category_weights,
        primitive_weights=primitive_weights,
    )
    test = _evaluate(
        model,
        split["test"],
        batch_size=args.batch_size,
        device=device,
        category_weights=category_weights,
        primitive_weights=primitive_weights,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "gcdv2-pattern-parser-checkpoint-1.0",
            "model_state_dict": best_state,
            "model_config": config,
            "categories": CATEGORIES,
            "primitive_types": PRIMITIVE_TYPES,
            "maximum_panels": MAXIMUM_PANELS,
            "maximum_edges": MAXIMUM_EDGES,
            "best_epoch": best_epoch,
            "split_assignments": assignments,
            "input_contract": "clean pattern.png only via frozen local ResNet50-FPN features",
            "output_scale_contract": "packed UV and canvas-normalized rendered length; no image-only absolute cm regression",
        },
        args.checkpoint,
    )
    predictions = _prediction_rows(model, split["test"], device)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    metrics = {
        "schema_version": "gcdv2-pattern-parser-evaluation-1.0",
        "status": "PASS",
        "input_contract": {
            "network_input": "clean pattern.png only",
            "cached_representation": "frozen local Mask R-CNN ResNet50-FPN tokens",
            "labels_or_scale_used_as_input": False,
        },
        "output_contract": {
            "category": list(CATEGORIES),
            "edge_set": [
                "presence",
                "primitive_type",
                "packed_start_uv",
                "packed_end_uv",
                "rendered_length_canvas_fraction",
                "direction_sin_cos_in_image_coordinates",
            ],
            "panel_set": ["presence", "packed_bbox_uv"],
            "absolute_cm_regression": False,
            "exact_cm_truth": "retained in each labels.json sidecar",
        },
        "dataset": {
            "total": len(examples),
            "splits": {name: len(rows) for name, rows in split.items()},
            "category_counts": dict(sorted(Counter(item.category for item in examples).items())),
            "maximum_panels_observed": max(len(item.panel_boxes) for item in examples),
            "maximum_edges_observed": max(len(item.edge_geometry) for item in examples),
            "index": str(args.index.as_posix()),
            "index_sha256": _sha256(args.index),
            "features": str(args.features.as_posix()),
            "features_sha256": _sha256(args.features),
            "split_audit": split_audit,
        },
        "model": {
            **config,
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "training_matching": (
                "Hungarian set matching"
                if args.training_matching == "hungarian"
                else "canonical packed-geometry order independent of source JSON serialization"
            ),
            "evaluation_matching": "Hungarian set matching",
            "category_class_weights": category_weights_np.tolist(),
            "primitive_class_weights": primitive_weights_np.tolist(),
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
            "history": history,
        },
        "validation": validation,
        "test": test,
        "checkpoint": str(args.checkpoint.as_posix()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "predictions": str(args.predictions.as_posix()),
        "predictions_sha256": _sha256(args.predictions),
    }
    _write_json(args.metrics, metrics)
    print(json.dumps({"best_epoch": best_epoch, "validation": validation, "test": test}, indent=2))


if __name__ == "__main__":
    main()
