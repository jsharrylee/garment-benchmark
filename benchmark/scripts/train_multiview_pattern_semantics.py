from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import GARMENT_ROLES
from benchmark.drafting_semantics.multiview_pattern_semantics import (
    PATTERN_TARGET_NAMES,
    VIEW_NAMES,
    TargetStandardizer,
    build_multiview_pattern_model,
    multiview_batch,
    read_multiview_pattern_examples,
)


def _category_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    per_role = {}
    f1 = []
    for index, role in enumerate(GARMENT_ROLES):
        tp = int(np.sum((predictions == index) & (targets == index)))
        fp = int(np.sum((predictions == index) & (targets != index)))
        fn = int(np.sum((predictions != index) & (targets == index)))
        support = int(np.sum(targets == index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        value = 2 * precision * recall / max(precision + recall, 1e-12)
        per_role[role] = {"precision": precision, "recall": recall, "f1": value, "support": support}
        if support:
            f1.append(value)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(f1)),
        "per_role": per_role,
    }


def _evaluate(model, examples, standardizer, config, device, *, missing_view: int | None = None) -> dict:
    import torch

    model.eval()
    categories, category_targets = [], []
    predicted_patterns, target_patterns = [], []
    batch_size = int(config["batch_size"])
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = multiview_batch(examples[start : start + batch_size], standardizer)
            view_features = torch.from_numpy(batch["view_features"]).to(device)
            view_valid = torch.ones(view_features.shape[:2], dtype=torch.bool, device=device)
            if missing_view is not None:
                view_features[:, missing_view] = 0
                view_valid[:, missing_view] = False
            output = model(view_features, view_valid=view_valid)
            categories.append(output["category_logits"].argmax(dim=-1).cpu().numpy())
            category_targets.append(batch["category_targets"])
            predicted_patterns.append(standardizer.decode(output["pattern_prediction"].cpu().float().numpy()))
            target_patterns.append(batch["raw_pattern_targets"])
    category_prediction = np.concatenate(categories)
    category_target = np.concatenate(category_targets)
    predicted = np.concatenate(predicted_patterns)
    target = np.concatenate(target_patterns)
    errors = np.abs(predicted - target)
    standard_deviations = np.asarray(standardizer.standard_deviations, dtype=np.float32)
    normalized_mae = errors.mean(axis=0) / standard_deviations
    residual = ((predicted - target) ** 2).sum(axis=0)
    centered = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - residual / np.maximum(centered, 1e-9)
    result = {
        "category": _category_metrics(category_prediction, category_target),
        "pattern": {
            name: {"mae": float(errors[:, index].mean()), "normalized_mae": float(normalized_mae[index]), "r2": float(r2[index])}
            for index, name in enumerate(PATTERN_TARGET_NAMES)
        },
        "mean_normalized_pattern_mae": float(normalized_mae.mean()),
        "sample_count": int(len(target)),
    }
    result["selection_score"] = result["category"]["macro_f1"] - 0.15 * result["mean_normalized_pattern_mae"]
    return result


def _compact(value: dict) -> dict:
    return {
        "selection_score": value["selection_score"],
        "category_macro_f1": value["category"]["macro_f1"],
        "mean_normalized_pattern_mae": value["mean_normalized_pattern_mae"],
        "panel_count_mae": value["pattern"]["panel_count"]["mae"],
        "edge_count_mae": value["pattern"]["edge_count"]["mae"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a four-view to 2D pattern semantic baseline.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--semantic-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--precomputed-features", type=Path)
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/multiview_pattern_semantics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multiview_pattern_semantics.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/training_metrics.json"))
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
    examples = read_multiview_pattern_examples(
        args.index, args.split, args.semantic_records, args.precomputed_features
    )
    train = [item for item in examples if item.split == "train"]
    validation = tuple(item for item in examples if item.split == "validation")
    test = tuple(item for item in examples if item.split == "test")
    auxiliary = tuple(item for item in examples if item.split == "auxiliary")
    standardizer = TargetStandardizer.fit(train)
    model = build_multiview_pattern_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(seed)
    category_counts = np.ones(len(GARMENT_ROLES), dtype=np.float64)
    category_counts += np.bincount([item.category_target for item in train], minlength=len(GARMENT_ROLES))
    category_weights = 1.0 / np.sqrt(category_counts)
    category_weights /= category_weights.mean()
    category_weights_t = torch.from_numpy(category_weights.astype(np.float32)).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_weights = config["loss_weights"]
    best_score, best_epoch, best_state = -1e9, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        generator.shuffle(train)
        model.train()
        losses = []
        for start in range(0, len(train), int(config["batch_size"])):
            batch = multiview_batch(train[start : start + int(config["batch_size"])], standardizer)
            views = torch.from_numpy(batch["view_features"]).to(device)
            categories = torch.from_numpy(batch["category_targets"]).to(device)
            patterns = torch.from_numpy(batch["pattern_targets"]).to(device)
            view_valid = torch.ones(views.shape[:2], dtype=torch.bool, device=device)
            # Random leave-one-view-out is a label-preserving augmentation and
            # later supports a causal view-ablation audit.
            if generator.random() < 0.35:
                missing = int(generator.integers(len(VIEW_NAMES)))
                views[:, missing] = 0
                view_valid[:, missing] = False
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(views, pattern_targets=patterns, view_valid=view_valid)
                category_loss = torch.nn.functional.cross_entropy(output["category_logits"], categories, weight=category_weights_t)
                pattern_loss = torch.nn.functional.smooth_l1_loss(output["pattern_prediction"], patterns)
                logits = output["image_embedding"] @ output["pattern_embedding"].T / float(config["temperature"])
                labels = torch.arange(len(views), device=device)
                contrastive_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                )
                loss = (
                    float(loss_weights["category"]) * category_loss
                    + float(loss_weights["pattern"]) * pattern_loss
                    + float(loss_weights["contrastive"]) * contrastive_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_metrics = _evaluate(model, validation, standardizer, config, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": _compact(validation_metrics)}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        if validation_metrics["selection_score"] > best_score + 1e-5:
            best_score = validation_metrics["selection_score"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= int(config["early_stopping_patience"]):
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = _evaluate(model, validation, standardizer, config, device)
    test_metrics = _evaluate(model, test, standardizer, config, device)
    ablations = {
        VIEW_NAMES[index]: _evaluate(model, test, standardizer, config, device, missing_view=index)
        for index in range(len(VIEW_NAMES))
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "target_standardizer": {"means": standardizer.means, "standard_deviations": standardizer.standard_deviations},
            "garment_roles": GARMENT_ROLES,
            "pattern_target_names": PATTERN_TARGET_NAMES,
            "best_epoch": best_epoch,
        },
        args.checkpoint,
    )
    result = {
        "status": "PASS_FOUR_VIEW_TO_PATTERN_STRUCTURE_BASELINE",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test), "auxiliary_not_used": len(auxiliary)},
        "input_features": (
            "precomputed_resnet50_from_four_raw_images"
            if args.precomputed_features is not None
            else "four_21d_silhouette_descriptors"
        ),
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_leave_one_view_out": ablations,
        "claim_boundary": [
            (
                "input is a frozen ResNet-50 embedding extracted from each of four real orthographic renders; "
                "the Transformer is trained on embeddings rather than end-to-end pixels"
                if args.precomputed_features is not None
                else "input is a deterministic 21D descriptor per real orthographic render, not raw pixels"
            ),
            "output is garment category and 2D pattern structural inventory, not spline/control-point generation",
            "all samples share one neutral body, so body generalization is untested",
            "view ablation supports feature importance but attention alone is not a causal explanation",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "best_epoch": best_epoch, "parameter_count": result["parameter_count"],
        "training_seconds": training_seconds, "test": _compact(test_metrics),
        "leave_one_view_out": {name: _compact(value) for name, value in ablations.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
