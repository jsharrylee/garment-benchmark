from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.gcdv2_exact.garment_panel_set_learning import (
    CATEGORIES,
    CURVE_TYPES,
    GarmentPanelDataset,
    MAXIMUM_EDGES,
    PARTS,
    SIDES,
    SURFACES,
    build_model,
    collate_garments,
    garment_disjoint_split,
    model_loss,
    read_garments,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(batch: Mapping[str, Any], device) -> dict[str, Any]:
    import torch

    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _classification(predicted: Sequence[int], expected: Sequence[int], labels: Sequence[str]) -> dict[str, Any]:
    per_class = {}
    f1_values = []
    for index, label in enumerate(labels):
        tp = sum(p == index and e == index for p, e in zip(predicted, expected))
        fp = sum(p == index and e != index for p, e in zip(predicted, expected))
        fn = sum(p != index and e == index for p, e in zip(predicted, expected))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(e == index for e in expected)}
        if per_class[label]["support"]:
            f1_values.append(f1)
    return {
        "accuracy": sum(p == e for p, e in zip(predicted, expected)) / max(len(expected), 1),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def evaluate(model, loader, device, source_ids: Sequence[str], *, export_predictions: bool = False):
    import torch

    model.eval()
    losses = []
    class_predictions: dict[str, list[int]] = {key: [] for key in ("category", "source", "part", "surface", "side", "curve")}
    class_targets = {key: [] for key in class_predictions}
    count_errors, count_exact = [], []
    vertex_pixel_errors, vertex_cm_errors, vertex_hits = [], [], []
    length_errors, direction_errors, tangent_errors = [], [], []
    control_errors, arc_radius_errors, arc_flag_correct = [], [], []
    graph_tp = graph_fp = graph_fn = 0
    prediction_rows = []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            output = model(batch["images"], batch["scales"], batch["panel_mask"])
            losses.append(float(model_loss(output, batch)["loss"]))
            class_predictions["category"].extend(output["category_logits"].argmax(-1).cpu().tolist())
            class_targets["category"].extend(batch["category"].cpu().tolist())
            batch_size = batch["images"].shape[0]
            for b in range(batch_size):
                garment_prediction = {
                    "sample_id": raw["sample_ids"][b],
                    "target_category": CATEGORIES[int(batch["category"][b])],
                    "predicted_category": CATEGORIES[int(output["category_logits"][b].argmax())],
                    "panels": [],
                }
                for p, panel_uid in enumerate(raw["panel_uids"][b]):
                    if not bool(batch["panel_mask"][b, p]):
                        continue
                    for key, output_key, target_key in (
                        ("source", "source_logits", "source_id"),
                        ("part", "part_logits", "part"),
                        ("surface", "surface_logits", "surface"),
                        ("side", "side_logits", "side"),
                    ):
                        class_predictions[key].append(int(output[output_key][b, p].argmax()))
                        class_targets[key].append(int(batch[target_key][b, p]))
                    true_count = int(batch["count"][b, p])
                    predicted_count = int(output["count_logits"][b, p].argmax())
                    predicted_count = min(MAXIMUM_EDGES, max(3, predicted_count))
                    count_errors.append(abs(predicted_count - true_count))
                    count_exact.append(predicted_count == true_count)
                    cm_per_pixel = math.exp(float(batch["scales"][b, p, 0]))
                    maximum = max(true_count, predicted_count)
                    panel_vertex_errors = []
                    for edge_index in range(true_count):
                        pixel_error = float(
                            torch.linalg.vector_norm(
                                output["vertices"][b, p, edge_index] - batch["vertices"][b, p, edge_index]
                            )
                            * 1024.0
                        )
                        vertex_pixel_errors.append(pixel_error)
                        vertex_cm_errors.append(pixel_error * cm_per_pixel)
                        vertex_hits.append(pixel_error <= 8.0)
                        panel_vertex_errors.append(pixel_error)
                        predicted_type = int(output["edge_type_logits"][b, p, edge_index].argmax())
                        true_type = int(batch["edge_types"][b, p, edge_index])
                        class_predictions["curve"].append(predicted_type)
                        class_targets["curve"].append(true_type)
                        length_errors.append(abs(float(output["lengths"][b, p, edge_index] - batch["lengths"][b, p, edge_index])) * 100.0)
                        cosine = float((output["directions"][b, p, edge_index] * batch["directions"][b, p, edge_index]).sum().clamp(-1, 1))
                        direction_errors.append(math.degrees(math.acos(cosine)))
                        for offset in (0, 2):
                            cosine = float((output["tangents"][b, p, edge_index, offset:offset+2] * batch["tangents"][b, p, edge_index, offset:offset+2]).sum().clamp(-1, 1))
                            tangent_errors.append(math.degrees(math.acos(cosine)))
                        mask = batch["control_masks"][b, p, edge_index].bool()
                        if bool(mask.any()):
                            control_errors.extend((output["controls"][b, p, edge_index][mask] - batch["controls"][b, p, edge_index][mask]).abs().cpu().tolist())
                        if bool(batch["arc_mask"][b, p, edge_index]):
                            arc_radius_errors.append(abs(float(output["arc_radius"][b, p, edge_index] - batch["arc_radius"][b, p, edge_index])) * 100.0)
                            arc_flag_correct.extend(((output["arc_flag_logits"][b, p, edge_index].sigmoid() >= 0.5) == batch["arc_flags"][b, p, edge_index].bool()).cpu().tolist())
                    common = min(true_count, predicted_count)
                    for edge_index in range(common):
                        next_index = (edge_index + 1) % true_count
                        start_ok = panel_vertex_errors[edge_index] <= 8.0
                        end_error = float(torch.linalg.vector_norm(output["vertices"][b, p, (edge_index + 1) % predicted_count] - batch["vertices"][b, p, next_index]) * 1024.0)
                        type_ok = int(output["edge_type_logits"][b, p, edge_index].argmax()) == int(batch["edge_types"][b, p, edge_index])
                        if start_ok and end_error <= 8.0 and type_ok:
                            graph_tp += 1
                    graph_fp += predicted_count
                    graph_fn += true_count
                    if export_predictions:
                        garment_prediction["panels"].append({
                            "panel_uid": panel_uid,
                            "target_source_panel_id": source_ids[int(batch["source_id"][b, p])],
                            "predicted_source_panel_id": source_ids[int(output["source_logits"][b, p].argmax())],
                            "target_part": PARTS[int(batch["part"][b, p])],
                            "predicted_part": PARTS[int(output["part_logits"][b, p].argmax())],
                            "target_surface": SURFACES[int(batch["surface"][b, p])],
                            "predicted_surface": SURFACES[int(output["surface_logits"][b, p].argmax())],
                            "target_side": SIDES[int(batch["side"][b, p])],
                            "predicted_side": SIDES[int(output["side_logits"][b, p].argmax())],
                            "target_count": true_count,
                            "predicted_count": predicted_count,
                            "predicted_vertices_uv": output["vertices"][b, p, :predicted_count].cpu().tolist(),
                            "target_vertices_uv": batch["vertices"][b, p, :true_count].cpu().tolist(),
                            "predicted_curve_types": [CURVE_TYPES[int(value)] for value in output["edge_type_logits"][b, p, :predicted_count].argmax(-1).cpu().tolist()],
                            "target_curve_types": [CURVE_TYPES[int(value)] for value in batch["edge_types"][b, p, :true_count].cpu().tolist()],
                            "predicted_lengths_cm": (output["lengths"][b, p, :predicted_count] * 100.0).cpu().tolist(),
                            "target_lengths_cm": (batch["lengths"][b, p, :true_count] * 100.0).cpu().tolist(),
                            "predicted_directions_sin_cos": output["directions"][b, p, :predicted_count].cpu().tolist(),
                            "predicted_tangents_sin_cos": output["tangents"][b, p, :predicted_count].cpu().tolist(),
                            "predicted_relative_controls": output["controls"][b, p, :predicted_count].cpu().tolist(),
                            "predicted_arc_radius_cm": (output["arc_radius"][b, p, :predicted_count] * 100.0).cpu().tolist(),
                            "predicted_arc_flags": (output["arc_flag_logits"][b, p, :predicted_count].sigmoid() >= 0.5).int().cpu().tolist(),
                        })
                prediction_rows.append(garment_prediction)
    graph_precision = graph_tp / max(graph_fp, 1)
    graph_recall = graph_tp / max(graph_fn, 1)
    metrics = {
        "loss": float(np.mean(losses)),
        "garment_category": _classification(class_predictions["category"], class_targets["category"], CATEGORIES),
        "source_panel_id": _classification(class_predictions["source"], class_targets["source"], source_ids),
        "weak_part": _classification(class_predictions["part"], class_targets["part"], PARTS),
        "weak_surface": _classification(class_predictions["surface"], class_targets["surface"], SURFACES),
        "weak_side": _classification(class_predictions["side"], class_targets["side"], SIDES),
        "curve_type": _classification(class_predictions["curve"], class_targets["curve"], CURVE_TYPES),
        "vertex_count_exact_accuracy": float(np.mean(count_exact)),
        "vertex_count_mae": float(np.mean(count_errors)),
        "vertex_coordinate_mae_px": float(np.mean(vertex_pixel_errors)),
        "vertex_coordinate_mae_cm": float(np.mean(vertex_cm_errors)),
        "vertex_within_8px": float(np.mean(vertex_hits)),
        "edge_length_mae_cm": float(np.mean(length_errors)),
        "chord_direction_mae_deg": float(np.mean(direction_errors)),
        "endpoint_tangent_mae_deg": float(np.mean(tangent_errors)),
        "relative_control_component_mae": float(np.mean(control_errors)) if control_errors else None,
        "arc_radius_mae_cm": float(np.mean(arc_radius_errors)) if arc_radius_errors else None,
        "arc_flag_accuracy": float(np.mean(arc_flag_correct)) if arc_flag_correct else None,
        "exact_graph_edge_at_8px_and_type": {
            "precision": graph_precision,
            "recall": graph_recall,
            "f1": 2 * graph_precision * graph_recall / max(graph_precision + graph_recall, 1e-12),
        },
    }
    return metrics, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an unordered garment-set to ordered single-panel graph model.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/gcdv2_garment_panel_set.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_exact/garment_panel_set.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/metrics.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/test_predictions.jsonl"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    garments, source_ids = read_garments(args.index)
    assignments, split_audit = garment_disjoint_split(garments, seed=args.seed)
    splits = {name: [value for value in garments if assignments[value.sample_id] == name] for name in ("train", "validation", "test")}
    datasets = {
        name: GarmentPanelDataset(values, source_ids, shuffle_panels=name == "train")
        for name, values in splits.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_garments,
        )
        for name, dataset in datasets.items()
    }
    model = build_model(len(source_ids), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 0
    history = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_losses = []
        for raw in loaders["train"]:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["images"], batch["scales"], batch["panel_mask"])
                loss = model_loss(output, batch)["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(float(loss.detach()))
        validation, _ = evaluate(model, loaders["validation"], device, source_ids)
        row = {"epoch": epoch, "training_loss": float(np.mean(training_losses)), "validation_loss": validation["loss"], "validation_graph_f1": validation["exact_graph_edge_at_8px_and_type"]["f1"], "validation_vertex_mae_px": validation["vertex_coordinate_mae_px"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "source_ids": source_ids, "split_assignments": assignments, "epoch": epoch, "validation": validation}, args.checkpoint)
        elif epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation, _ = evaluate(model, loaders["validation"], device, source_ids)
    test, predictions = evaluate(model, loaders["test"], device, source_ids, export_predictions=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    result = {
        "status": "PASS",
        "claim": "sample-ID-disjoint unseen garments from the same GCDv2 generator and render domain",
        "device": str(device),
        "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_audit": split_audit,
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation,
        "test": test,
        "artifacts": {"checkpoint": args.checkpoint.as_posix(), "predictions": args.predictions.as_posix()},
        "hashes": {"panel_index_sha256": _sha256(args.index), "checkpoint_sha256": _sha256(args.checkpoint), "predictions_sha256": _sha256(args.predictions)},
        "limitations": [
            "source panel ID and left/right are not identifiable when two unordered input silhouettes are exactly symmetric",
            "weak roles are lexical source-ID labels rather than expert drafting semantics",
            "the test is garment-ID-disjoint but not generator-family-disjoint",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test}, indent=2))


if __name__ == "__main__":
    main()
