from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import (
    GARMENT_ROLES,
    MULTIGARMENT_EDGE_ROLES,
    MULTIGARMENT_PANEL_ROLES,
    MultiGarmentExample,
    build_multigarment_model,
    padded_garment_batch,
    randomize_boundary_serialization,
    read_gcd_multigarment_examples,
    read_teagan_multigarment_examples,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_weights(examples: Sequence[MultiGarmentExample], kind: str, count: int) -> np.ndarray:
    counts = np.ones(count, dtype=np.float64)
    for example in examples:
        if kind == "garment":
            counts[example.garment_target] += 1
        elif kind == "panel":
            counts += np.bincount([panel.panel_target for panel in example.panels], minlength=count)
        elif kind == "edge":
            for panel in example.panels:
                targets = panel.edge_targets[panel.edge_targets >= 0]
                counts += np.bincount(targets, minlength=count)
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return weights.astype(np.float32)


def _classification(predictions: np.ndarray, targets: np.ndarray, names: Sequence[str], *, ignore: int = -100) -> dict:
    valid = targets != ignore
    predictions = predictions[valid]
    targets = targets[valid]
    per_role = {}
    f1_values = []
    for index, name in enumerate(names):
        tp = int(np.sum((predictions == index) & (targets == index)))
        fp = int(np.sum((predictions == index) & (targets != index)))
        fn = int(np.sum((predictions != index) & (targets == index)))
        support = int(np.sum(targets == index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_role[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support and name != "other":
            f1_values.append(f1)
    return {
        "accuracy": float(np.mean(predictions == targets)) if len(targets) else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "support": int(len(targets)),
        "per_role": per_role,
    }


def _evaluate(model, examples, config, device) -> dict:
    import torch

    model.eval()
    edge_predictions, edge_targets = [], []
    panel_predictions, panel_targets = [], []
    garment_predictions, garment_targets = [], []
    same_predictions, same_targets = [], []
    seam_errors = []
    by_source: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    batch_size = int(config["batch_size"])
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            current = examples[start : start + batch_size]
            batch = padded_garment_batch(
                current,
                maximum_panels=int(config["maximum_panels"]),
                maximum_edges=int(config["maximum_edges"]),
            )
            features = torch.from_numpy(batch["features"]).to(device)
            edge_valid = torch.from_numpy(batch["edge_valid"]).to(device)
            panel_valid = torch.from_numpy(batch["panel_valid"]).to(device)
            output = model(features, edge_valid, panel_valid)
            edge_pred = output["edge_logits"].argmax(dim=-1).cpu().numpy()
            panel_pred = output["panel_logits"].argmax(dim=-1).cpu().numpy()
            garment_pred = output["garment_logits"].argmax(dim=-1).cpu().numpy()
            same_pred = (output["same_path_logits"].sigmoid() >= 0.5).cpu().numpy().astype(np.int64)
            ratio = output["seam_ratio"].cpu().numpy()
            valid_edge_targets = batch["edge_targets"]
            edge_predictions.append(edge_pred[valid_edge_targets >= 0])
            edge_targets.append(valid_edge_targets[valid_edge_targets >= 0])
            panel_predictions.append(panel_pred[batch["panel_valid"]])
            panel_targets.append(batch["panel_targets"][batch["panel_valid"]])
            garment_predictions.append(garment_pred)
            garment_targets.append(batch["garment_targets"])
            same_predictions.append(same_pred[batch["same_path_mask"]])
            same_targets.append(batch["same_path_targets"][batch["same_path_mask"]].astype(np.int64))
            if np.any(batch["seam_ratio_mask"]):
                seam_errors.extend(
                    np.abs(ratio[batch["seam_ratio_mask"]] - batch["seam_ratio_targets"][batch["seam_ratio_mask"]]).tolist()
                )
            for row, source in enumerate(batch["sources"]):
                edge_mask = valid_edge_targets[row] >= 0
                panel_mask = batch["panel_valid"][row]
                by_source[source]["edge_predictions"].append(edge_pred[row][edge_mask])
                by_source[source]["edge_targets"].append(valid_edge_targets[row][edge_mask])
                by_source[source]["panel_predictions"].append(panel_pred[row][panel_mask])
                by_source[source]["panel_targets"].append(batch["panel_targets"][row][panel_mask])
                by_source[source]["garment_predictions"].append(np.asarray([garment_pred[row]]))
                by_source[source]["garment_targets"].append(np.asarray([batch["garment_targets"][row]]))

    concatenate = lambda values: np.concatenate(values) if values else np.asarray([], dtype=np.int64)
    result = {
        "edge": _classification(concatenate(edge_predictions), concatenate(edge_targets), MULTIGARMENT_EDGE_ROLES),
        "panel": _classification(concatenate(panel_predictions), concatenate(panel_targets), MULTIGARMENT_PANEL_ROLES),
        "garment": _classification(concatenate(garment_predictions), concatenate(garment_targets), GARMENT_ROLES),
        "same_path": _classification(concatenate(same_predictions), concatenate(same_targets), ("different", "same"), ignore=-1),
        "sleeve_head_to_armhole_ratio_mae": float(np.mean(seam_errors)) if seam_errors else None,
        "sleeve_head_to_armhole_ratio_count": len(seam_errors),
        "sources": {},
    }
    for source, values in sorted(by_source.items()):
        result["sources"][source] = {
            "edge": _classification(concatenate(values["edge_predictions"]), concatenate(values["edge_targets"]), MULTIGARMENT_EDGE_ROLES),
            "panel": _classification(concatenate(values["panel_predictions"]), concatenate(values["panel_targets"]), MULTIGARMENT_PANEL_ROLES),
            "garment": _classification(concatenate(values["garment_predictions"]), concatenate(values["garment_targets"]), GARMENT_ROLES),
        }
    result["selection_score"] = (
        0.5 * result["edge"]["macro_f1"]
        + 0.3 * result["panel"]["macro_f1"]
        + 0.2 * result["garment"]["macro_f1"]
    )
    return result


def _compact(metrics: dict) -> dict:
    return {
        "selection_score": metrics["selection_score"],
        "edge_macro_f1": metrics["edge"]["macro_f1"],
        "panel_macro_f1": metrics["panel"]["macro_f1"],
        "garment_macro_f1": metrics["garment"]["macro_f1"],
        "same_path_f1": metrics["same_path"]["per_role"]["same"]["f1"],
        "seam_ratio_mae": metrics["sleeve_head_to_armhole_ratio_mae"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hierarchical multi-garment semantic Transformer.")
    parser.add_argument("--gcd-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--teagan-records", type=Path, default=Path("artifacts/drafting_semantics/teagan_diverse.jsonl.gz"))
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/multigarment_semantics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multigarment_graph_transformer.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/drafting_semantics/multigarment/training_metrics.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    loaded = (*read_gcd_multigarment_examples(args.gcd_records), *read_teagan_multigarment_examples(args.teagan_records))
    train = [item for item in loaded if item.split == "train"]
    validation = tuple(item for item in loaded if item.split == "validation")
    test = tuple(item for item in loaded if item.split == "test")
    auxiliary = tuple(item for item in loaded if item.split == "auxiliary")
    if not train or not validation or not test:
        raise SystemExit("train, validation, and test splits must all be non-empty")
    dataset_stats = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "auxiliary_not_used": len(auxiliary),
        "source_counts": dict(sorted(Counter(item.source for item in loaded).items())),
        "garment_counts": dict(sorted(Counter(GARMENT_ROLES[item.garment_target] for item in loaded).items())),
        "maximum_panels_observed": max(len(item.panels) for item in loaded),
        "maximum_edges_per_panel_observed": max(len(panel.features) for item in loaded for panel in item.panels),
    }

    model = build_multigarment_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    edge_weights = torch.from_numpy(_class_weights(train, "edge", len(MULTIGARMENT_EDGE_ROLES))).to(device)
    panel_weights = torch.from_numpy(_class_weights(train, "panel", len(MULTIGARMENT_PANEL_ROLES))).to(device)
    garment_weights = torch.from_numpy(_class_weights(train, "garment", len(GARMENT_ROLES))).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(seed)
    weights = config["loss_weights"]
    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []
    patience = int(config["early_stopping_patience"])
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        generator.shuffle(train)
        model.train()
        losses = []
        component_totals: dict[str, list[float]] = defaultdict(list)
        for start in range(0, len(train), int(config["batch_size"])):
            current = [randomize_boundary_serialization(item, generator) for item in train[start : start + int(config["batch_size"])]]
            batch = padded_garment_batch(current, maximum_panels=int(config["maximum_panels"]), maximum_edges=int(config["maximum_edges"]))
            tensors = {key: torch.from_numpy(batch[key]).to(device) for key in (
                "features", "edge_targets", "edge_valid", "panel_targets", "panel_valid", "garment_targets",
                "same_path_targets", "same_path_mask", "panel_presence_targets", "seam_ratio_targets", "seam_ratio_mask"
            )}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(tensors["features"], tensors["edge_valid"], tensors["panel_valid"])
                edge_loss = torch.nn.functional.cross_entropy(
                    output["edge_logits"].reshape(-1, len(MULTIGARMENT_EDGE_ROLES)),
                    tensors["edge_targets"].reshape(-1), weight=edge_weights, ignore_index=-100,
                )
                panel_loss = torch.nn.functional.cross_entropy(
                    output["panel_logits"].reshape(-1, len(MULTIGARMENT_PANEL_ROLES)),
                    tensors["panel_targets"].reshape(-1), weight=panel_weights, ignore_index=-100,
                )
                garment_loss = torch.nn.functional.cross_entropy(output["garment_logits"], tensors["garment_targets"], weight=garment_weights)
                same_values = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["same_path_logits"], tensors["same_path_targets"], reduction="none"
                )
                same_loss = same_values[tensors["same_path_mask"]].mean()
                presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["panel_presence_logits"], tensors["panel_presence_targets"]
                )
                if bool(tensors["seam_ratio_mask"].any()):
                    seam_loss = torch.nn.functional.smooth_l1_loss(
                        output["seam_ratio"][tensors["seam_ratio_mask"]],
                        tensors["seam_ratio_targets"][tensors["seam_ratio_mask"]],
                    )
                else:
                    seam_loss = output["seam_ratio"].sum() * 0.0
                loss = (
                    float(weights["edge"]) * edge_loss
                    + float(weights["panel"]) * panel_loss
                    + float(weights["garment"]) * garment_loss
                    + float(weights["same_path"]) * same_loss
                    + float(weights["element_presence"]) * presence_loss
                    + 0.25 * seam_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            for name, value in (("edge", edge_loss), ("panel", panel_loss), ("garment", garment_loss), ("same_path", same_loss), ("presence", presence_loss), ("seam", seam_loss)):
                component_totals[name].append(float(value.detach().cpu()))

        validation_metrics = _evaluate(model, validation, config, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_components": {key: float(np.mean(value)) for key, value in component_totals.items()},
            "validation": _compact(validation_metrics),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["selection_score"] > best_score + 1e-5:
            best_score = validation_metrics["selection_score"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = _evaluate(model, validation, config, device)
    test_metrics = _evaluate(model, test, config, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "edge_roles": MULTIGARMENT_EDGE_ROLES,
            "panel_roles": MULTIGARMENT_PANEL_ROLES,
            "garment_roles": GARMENT_ROLES,
            "gcd_records_sha256": _sha256(args.gcd_records),
            "teagan_records_sha256": _sha256(args.teagan_records),
            "best_epoch": best_epoch,
        },
        args.checkpoint,
    )
    result = {
        "status": "PASS_HIERARCHICAL_MULTIGARMENT_TRANSFORMER_TRAINED",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "dataset": dataset_stats,
        "config": config,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "claim_boundary": [
            "FreeSewing test is unseen body/options inside the Teagan/Brian implementation family, not unseen recipe",
            "GarmentCode labels are generator/topology-derived and have not yet been independently expert-audited",
            "unlabeled edges are masked instead of treated as semantic other",
            "attention weights are diagnostic only and require deletion/occlusion checks",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "best_epoch": best_epoch, "parameter_count": result["parameter_count"],
        "training_seconds": training_seconds, "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "validation": _compact(validation_metrics), "test": _compact(test_metrics),
    }, indent=2))


if __name__ == "__main__":
    main()
