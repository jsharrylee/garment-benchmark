from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import GARMENT_ROLES
from benchmark.drafting_semantics.multiview_element_geometry import (
    GEOMETRY_TARGET_NAMES,
    GEOMETRY_PANEL_ROLES,
    GEOMETRY_PATH_ROLES,
    PANEL_GEOMETRY_COMPONENTS,
    PATH_GEOMETRY_COMPONENTS,
    PRESENCE_TARGET_NAMES,
    MaskedTargetStandardizer,
    build_multiview_geometry_model,
    multiview_geometry_batch,
    read_multiview_geometry_examples,
)
from benchmark.drafting_semantics.multiview_pattern_semantics import (
    PANEL_COUNT_NAMES,
    SEMANTIC_COUNT_NAMES,
    VIEW_NAMES,
)


PANEL_DIMENSIONS = len(GEOMETRY_PANEL_ROLES) * len(PANEL_GEOMETRY_COMPONENTS)
PATH_DIMENSIONS = len(GEOMETRY_PATH_ROLES) * len(PATH_GEOMETRY_COMPONENTS)


def _category_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    values = []
    per_role = {}
    for index, role in enumerate(GARMENT_ROLES):
        tp = int(np.sum((predictions == index) & (targets == index)))
        fp = int(np.sum((predictions == index) & (targets != index)))
        fn = int(np.sum((predictions != index) & (targets == index)))
        support = int(np.sum(targets == index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_role[role] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            values.append(f1)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(values)),
        "per_role": per_role,
    }


def _presence_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    expected = targets >= 0.5
    values = []
    per_role = {}
    for index, role in enumerate(PRESENCE_TARGET_NAMES):
        tp = int(np.sum(predictions[:, index] & expected[:, index]))
        fp = int(np.sum(predictions[:, index] & ~expected[:, index]))
        fn = int(np.sum(~predictions[:, index] & expected[:, index]))
        support = int(expected[:, index].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_role[role] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            values.append(f1)
    return {
        "exact_inventory_rate": float(np.mean(np.all(predictions == expected, axis=1))),
        "macro_f1": float(np.mean(values)),
        "per_role": per_role,
    }


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return None
    left_ranks = np.argsort(np.argsort(left)).astype(np.float64)
    right_ranks = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _geometry_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    standardizer: MaskedTargetStandardizer,
) -> dict:
    deviations = np.asarray(standardizer.standard_deviations, dtype=np.float32)
    per_target = {}
    normalized_values = []
    group_values: dict[str, list[float]] = {"panel": [], "path": [], "seam": []}
    for index, name in enumerate(GEOMETRY_TARGET_NAMES):
        valid = mask[:, index]
        support = int(valid.sum())
        if not support:
            per_target[name] = {"support": 0}
            continue
        truth = target[valid, index]
        estimate = predicted[valid, index]
        error = np.abs(estimate - truth)
        normalized = float(error.mean() / deviations[index])
        residual = float(np.square(estimate - truth).sum())
        centered = float(np.square(truth - truth.mean()).sum())
        r2 = 1.0 - residual / max(centered, 1e-9)
        correlation = _rank_correlation(estimate, truth)
        per_target[name] = {
            "support": support,
            "mae": float(error.mean()),
            "normalized_mae": normalized,
            "r2": r2,
            "rank_correlation": correlation,
        }
        normalized_values.append(normalized)
        group = "panel" if index < PANEL_DIMENSIONS else ("path" if index < PANEL_DIMENSIONS + PATH_DIMENSIONS else "seam")
        group_values[group].append(normalized)
    return {
        "mean_dimension_balanced_normalized_mae": float(np.mean(normalized_values)),
        "group_normalized_mae": {
            name: float(np.mean(values)) if values else None for name, values in group_values.items()
        },
        "per_target": per_target,
    }


def _category_mean_predictions(train, target_examples) -> np.ndarray:
    train_values = np.stack([item.geometry_target for item in train])
    train_masks = np.stack([item.geometry_mask for item in train])
    train_categories = np.asarray([item.category_target for item in train])
    global_mean = np.zeros(train_values.shape[1], dtype=np.float32)
    for index in range(train_values.shape[1]):
        valid = train_masks[:, index]
        if valid.any():
            global_mean[index] = float(train_values[valid, index].mean())
    means = np.repeat(global_mean[None, :], len(GARMENT_ROLES), axis=0)
    for category in range(len(GARMENT_ROLES)):
        for index in range(train_values.shape[1]):
            valid = (train_categories == category) & train_masks[:, index]
            if valid.any():
                means[category, index] = float(train_values[valid, index].mean())
    return np.stack([means[item.category_target] for item in target_examples])


def _evaluate(model, examples, train, standardizer, config, device, *, missing_view: int | None = None) -> dict:
    import torch

    categories, category_targets = [], []
    predicted_geometry, raw_targets, masks, presence, presence_targets = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), int(config["batch_size"])):
            current = examples[start : start + int(config["batch_size"])]
            batch = multiview_geometry_batch(current, standardizer)
            views = torch.from_numpy(batch["view_features"]).to(device)
            view_valid = torch.ones(views.shape[:2], dtype=torch.bool, device=device)
            if missing_view is not None:
                views[:, missing_view] = 0
                view_valid[:, missing_view] = False
            output = model(views, view_valid=view_valid)
            categories.append(output["category_logits"].argmax(dim=-1).cpu().numpy())
            category_targets.append(batch["category_targets"])
            predicted_geometry.append(
                standardizer.decode(output["geometry_prediction"].cpu().float().numpy())
            )
            raw_targets.append(batch["raw_geometry_targets"])
            masks.append(batch["geometry_mask"])
            presence.append(output["presence_logits"].sigmoid().cpu().float().numpy())
            presence_targets.append(batch["presence_targets"])
    category_prediction = np.concatenate(categories)
    category_target = np.concatenate(category_targets)
    predicted = np.concatenate(predicted_geometry)
    target = np.concatenate(raw_targets)
    mask = np.concatenate(masks)
    presence_probability = np.concatenate(presence)
    presence_target = np.concatenate(presence_targets)
    geometry = _geometry_metrics(predicted, target, mask, standardizer)
    category_mean = _category_mean_predictions(train, examples)
    baseline = _geometry_metrics(category_mean, target, mask, standardizer)
    return {
        "sample_count": len(examples),
        "category": _category_metrics(category_prediction, category_target),
        "presence": _presence_metrics(presence_probability, presence_target),
        "geometry": geometry,
        "oracle_category_mean_baseline": baseline,
        "normalized_mae_gain_over_oracle_category_mean": (
            baseline["mean_dimension_balanced_normalized_mae"]
            - geometry["mean_dimension_balanced_normalized_mae"]
        ),
    }


def _selection(metrics: dict) -> float:
    return (
        metrics["category"]["macro_f1"]
        + 0.15 * metrics["presence"]["macro_f1"]
        - 0.25 * metrics["geometry"]["mean_dimension_balanced_normalized_mae"]
    )


def _balanced_geometry_loss(values, targets, mask):
    import torch

    raw = torch.nn.functional.smooth_l1_loss(values, targets, reduction="none")
    groups = ((0, PANEL_DIMENSIONS), (PANEL_DIMENSIONS, PANEL_DIMENSIONS + PATH_DIMENSIONS), (PANEL_DIMENSIONS + PATH_DIMENSIONS, len(GEOMETRY_TARGET_NAMES)))
    losses = []
    for start, end in groups:
        group_raw = raw[:, start:end]
        group_mask = mask[:, start:end]
        counts = group_mask.sum(dim=0)
        valid_dimensions = counts > 0
        if valid_dimensions.any():
            dimension_mean = (group_raw * group_mask).sum(dim=0) / counts.clamp_min(1)
            losses.append(dimension_mean[valid_dimensions].mean())
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train target-specific four-view semantic geometry regression.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--semantic-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"))
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/multiview_element_geometry_resnet50.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multiview_element_geometry_resnet50.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/drafting_semantics/multiview_element_geometry/training_metrics.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    examples = read_multiview_geometry_examples(args.index, args.split, args.semantic_records, args.features)
    train = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    test = tuple(item for item in examples if item.split == "test")
    auxiliary = tuple(item for item in examples if item.split == "auxiliary")
    standardizer = MaskedTargetStandardizer.fit(train)
    model = build_multiview_geometry_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        order = generator.permutation(len(train))
        model.train()
        losses = []
        for start in range(0, len(train), int(config["batch_size"])):
            current = [train[int(index)] for index in order[start : start + int(config["batch_size"])]]
            batch = multiview_geometry_batch(current, standardizer)
            views = torch.from_numpy(batch["view_features"]).to(device)
            view_valid = torch.ones(views.shape[:2], dtype=torch.bool, device=device)
            if generator.random() < float(config["view_dropout_probability"]):
                missing = int(generator.integers(len(VIEW_NAMES)))
                views[:, missing] = 0
                view_valid[:, missing] = False
            category_target = torch.from_numpy(batch["category_targets"]).to(device)
            geometry_target = torch.from_numpy(batch["geometry_targets"]).to(device)
            geometry_mask = torch.from_numpy(batch["geometry_mask"]).to(device)
            presence_target = torch.from_numpy(batch["presence_targets"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(views, view_valid=view_valid)
                category_loss = torch.nn.functional.cross_entropy(output["category_logits"], category_target)
                geometry_loss = _balanced_geometry_loss(output["geometry_prediction"], geometry_target, geometry_mask)
                presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["presence_logits"], presence_target
                )
                weights = config["loss_weights"]
                loss = (
                    float(weights["category"]) * category_loss
                    + float(weights["geometry"]) * geometry_loss
                    + float(weights["presence"]) * presence_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_metrics = _evaluate(model, validation, train, standardizer, config, device)
        score = _selection(validation_metrics)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": {
                "selection_score": score,
                "category_macro_f1": validation_metrics["category"]["macro_f1"],
                "presence_macro_f1": validation_metrics["presence"]["macro_f1"],
                "geometry_normalized_mae": validation_metrics["geometry"]["mean_dimension_balanced_normalized_mae"],
                "gain_over_oracle_category_mean": validation_metrics["normalized_mae_gain_over_oracle_category_mean"],
            },
        })
        print(json.dumps(history[-1]), flush=True)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                break
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    validation_metrics = _evaluate(model, validation, train, standardizer, config, device)
    test_metrics = _evaluate(model, test, train, standardizer, config, device)
    ablations = {
        VIEW_NAMES[index]: _evaluate(model, test, train, standardizer, config, device, missing_view=index)
        for index in range(len(VIEW_NAMES))
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "target_standardizer": {
                "means": standardizer.means,
                "standard_deviations": standardizer.standard_deviations,
            },
            "geometry_target_names": GEOMETRY_TARGET_NAMES,
            "presence_target_names": PRESENCE_TARGET_NAMES,
            "best_epoch": best_epoch,
        },
        args.checkpoint,
    )
    result = {
        "status": "PASS_SAME_DOMAIN_CONTINUOUS_SEMANTIC_GEOMETRY_BASELINE",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "split_counts": {
            "train": len(train), "validation": len(validation), "test": len(test), "auxiliary_not_used": len(auxiliary)
        },
        "target_count": len(GEOMETRY_TARGET_NAMES),
        "presence_target_count": len(PRESENCE_TARGET_NAMES),
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_leave_one_view_out": ablations,
        "claim_boundary": [
            "targets are normalized aggregate semantic measurements, not panel vertices, splines, landmarks, or stitch graphs",
            "all inputs and targets are from one GCDv2 generator, renderer, material style, and neutral body",
            "per-sample camera framing removes absolute scale; only relative 2D geometry is targeted",
            "role-query attention is target-specific across four global views but has no image-region localization",
            "observational prediction is not a causal estimate of how changing one pattern parameter changes 3D shape",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "best_epoch": best_epoch, "stopped_epoch": history[-1]["epoch"],
        "parameter_count": result["parameter_count"], "training_seconds": training_seconds,
        "test_category_macro_f1": test_metrics["category"]["macro_f1"],
        "test_presence_macro_f1": test_metrics["presence"]["macro_f1"],
        "test_geometry_normalized_mae": test_metrics["geometry"]["mean_dimension_balanced_normalized_mae"],
        "gain_over_oracle_category_mean": test_metrics["normalized_mae_gain_over_oracle_category_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
