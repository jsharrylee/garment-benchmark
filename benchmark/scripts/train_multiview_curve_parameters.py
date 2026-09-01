from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multiview_curve_parameters import (
    CURVE_PARAMETER_NAMES,
    CURVE_QUERY_NAMES,
    CurveParameterStandardizer,
    build_spatial_curve_model,
    curve_formula_loss,
    curve_reconstruction_metrics,
    multiview_curve_batch,
    read_multiview_curve_examples,
)
from benchmark.drafting_semantics.multiview_pattern_semantics import VIEW_NAMES


def _presence_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    predicted = probabilities >= 0.5
    expected = targets.astype(bool)
    per_query = {}
    f1_values = []
    for index, name in enumerate(CURVE_QUERY_NAMES):
        tp = int(np.sum(predicted[:, index] & expected[:, index]))
        fp = int(np.sum(predicted[:, index] & ~expected[:, index]))
        fn = int(np.sum(~predicted[:, index] & expected[:, index]))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        support = int(expected[:, index].sum())
        per_query[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)
    return {
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "exact_inventory_rate": float(np.mean(np.all(predicted == expected, axis=1))),
        "per_query": per_query,
    }


def _parameter_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    standardizer: CurveParameterStandardizer,
) -> dict:
    deviations = np.asarray(standardizer.standard_deviations, dtype=np.float32)
    per_query = {}
    macro = []
    for query, query_name in enumerate(CURVE_QUERY_NAMES):
        valid = mask[:, query]
        current = {}
        for parameter, parameter_name in enumerate(CURVE_PARAMETER_NAMES):
            support = int(valid.sum())
            if not support:
                current[parameter_name] = {"support": 0}
                continue
            truth = target[valid, query, parameter]
            estimate = predicted[valid, query, parameter]
            error = np.abs(estimate - truth)
            normalized = float(error.mean() / deviations[query, parameter])
            residual = float(np.square(estimate - truth).sum())
            centered = float(np.square(truth - truth.mean()).sum())
            r2 = 1.0 - residual / max(centered, 1e-9)
            current[parameter_name] = {
                "support": support,
                "mae": float(error.mean()),
                "normalized_mae": normalized,
                "r2": r2,
            }
            macro.append(normalized)
        per_query[query_name] = current
    return {
        "macro_normalized_mae": float(np.mean(macro)) if macro else None,
        "per_query": per_query,
    }


def _evaluate(
    model,
    examples,
    standardizer,
    config,
    device,
    *,
    removed_view: int | None = None,
    capture_attention: bool = False,
):
    import torch

    predicted = []
    expected = []
    masks = []
    probabilities = []
    sample_ids = []
    view_attention = []
    level_attention = []
    model.eval()
    batch_size = int(config["precomputed_feature_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            current = examples[start : start + batch_size]
            batch = multiview_curve_batch(current, standardizer)
            features = torch.from_numpy(batch["spatial_features"]).to(device).float()
            valid = torch.ones((len(current), len(VIEW_NAMES)), dtype=torch.bool, device=device)
            if removed_view is not None:
                valid[:, removed_view] = False
            output = model(
                spatial_features=features,
                view_valid=valid,
                capture_attention=capture_attention,
            )
            raw = standardizer.decode(output["curve_prediction"].cpu().float().numpy())
            predicted.append(raw)
            expected.append(batch["raw_curve_targets"])
            masks.append(batch["role_mask"])
            probabilities.append(torch.sigmoid(output["presence_logits"]).cpu().float().numpy())
            sample_ids.extend(batch["sample_ids"])
            if capture_attention:
                weights = output["spatial_attention"][-1].cpu().float().numpy()
                # [batch, head, query, view, patch].  View mass is comparable
                # because cross-attention is normalized over all view-patches.
                view_attention.append(weights.sum(axis=-1).mean(axis=1))
                offsets = np.cumsum((0, *(int(value) ** 2 for value in config["pyramid_grid_sizes"])))
                level_attention.append(
                    np.stack(
                        [weights[..., offsets[i] : offsets[i + 1]].sum(axis=-1).mean(axis=1) for i in range(len(offsets) - 1)],
                        axis=-1,
                    )
                )
    predicted_array = np.concatenate(predicted)
    target_array = np.concatenate(expected)
    mask_array = np.concatenate(masks)
    probability_array = np.concatenate(probabilities)
    reconstruction = curve_reconstruction_metrics(predicted_array, target_array, mask_array)
    means = np.asarray(standardizer.means, dtype=np.float32)
    mean_baseline = np.broadcast_to(means, target_array.shape)
    baseline_reconstruction = curve_reconstruction_metrics(mean_baseline, target_array, mask_array)
    result = {
        "presence": _presence_metrics(probability_array, mask_array),
        "parameters": _parameter_metrics(
            predicted_array, target_array, mask_array, standardizer
        ),
        "reconstruction": reconstruction,
        "train_mean_formula_baseline": baseline_reconstruction,
        "reconstruction_rmse_gain_over_train_mean": (
            float(baseline_reconstruction["macro_pointwise_rmse_over_chord"])
            - float(reconstruction["macro_pointwise_rmse_over_chord"])
        ),
    }
    if capture_attention:
        result["attention"] = {
            "contract": "last decoder layer cross-attention mass; descriptive, not causal importance",
            "mean_query_to_view_mass": np.concatenate(view_attention).mean(axis=0).tolist(),
            "mean_query_view_to_level_mass": np.concatenate(level_attention).mean(axis=0).tolist(),
            "query_names": list(CURVE_QUERY_NAMES),
            "view_names": list(VIEW_NAMES),
            "pyramid_levels": list(config["pyramid_levels"]),
        }
    arrays = {
        "sample_ids": np.asarray(sample_ids),
        "predicted_curve_parameters": predicted_array.astype(np.float32),
        "target_curve_parameters": target_array.astype(np.float32),
        "target_role_mask": mask_array,
        "predicted_presence_probability": probability_array.astype(np.float32),
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the spatial role-query Transformer for named curve formulas."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark/configs/multiview_curve_parameters_fpn.json"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"),
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json")
    )
    parser.add_argument(
        "--semantic-records",
        type=Path,
        default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/resnet50_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drafting_semantics/multiview_curve_parameters_fpn.pt"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/training_metrics.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/test_predictions.npz"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--maximum-examples", type=int)
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    examples = list(
        read_multiview_curve_examples(
            args.index, args.split, args.semantic_records, args.features
        )
    )
    if args.maximum_examples is not None:
        # Preserve all three official lanes in a deterministic smoke subset.
        selected = []
        for split_name in ("train", "validation", "test"):
            lane = [
                item for item in examples if item.split == split_name and item.role_mask.any()
            ]
            selected.extend(lane[: max(2, args.maximum_examples // 3)])
        examples = selected
    train = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    test = [item for item in examples if item.split == "test"]
    if not train or not validation or not test:
        raise ValueError(
            f"official train/validation/test examples required; got {len(train)}/{len(validation)}/{len(test)}"
        )
    standardizer = CurveParameterStandardizer.fit(train)
    model = build_spatial_curve_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    weights = config["loss_weights"]
    batch_size = int(config["precomputed_feature_batch_size"])
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        order = np.random.permutation(len(train))
        accumulated = Counter()
        batches = 0
        for start in range(0, len(order), batch_size):
            current = [train[index] for index in order[start : start + batch_size]]
            batch = multiview_curve_batch(current, standardizer)
            features = torch.from_numpy(batch["spatial_features"]).to(device).float()
            targets = torch.from_numpy(batch["curve_targets"]).to(device)
            role_mask = torch.from_numpy(batch["role_mask"]).to(device)
            view_valid = torch.rand((len(current), len(VIEW_NAMES)), device=device) >= float(
                config["view_dropout_probability"]
            )
            empty = ~view_valid.any(dim=1)
            if empty.any():
                view_valid[empty, torch.randint(0, len(VIEW_NAMES), (int(empty.sum()),), device=device)] = True
            output = model(spatial_features=features, view_valid=view_valid)
            losses = curve_formula_loss(
                output,
                targets,
                role_mask,
                standardizer,
                parameter_weight=float(weights["parameter"]),
                sampled_curve_weight=float(weights["sampled_curve"]),
                presence_weight=float(weights["presence"]),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in losses.items():
                accumulated[key] += float(value.detach().cpu())
            batches += 1
        validation_metrics, _ = _evaluate(
            model, validation, standardizer, config, device
        )
        score = float(validation_metrics["reconstruction"]["macro_pointwise_rmse_over_chord"]) + 0.2 * (
            1.0 - float(validation_metrics["presence"]["macro_f1"])
        )
        row = {
            "epoch": epoch,
            "training": {key: value / max(batches, 1) for key, value in accumulated.items()},
            "validation_score": score,
            "validation_reconstruction_rmse": validation_metrics["reconstruction"]["macro_pointwise_rmse_over_chord"],
            "validation_presence_f1": validation_metrics["presence"]["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics, _ = _evaluate(model, train, standardizer, config, device)
    validation_metrics, _ = _evaluate(model, validation, standardizer, config, device)
    test_metrics, prediction_arrays = _evaluate(
        model, test, standardizer, config, device, capture_attention=True
    )
    ablation = {}
    for index, view in enumerate(VIEW_NAMES):
        current, _ = _evaluate(
            model, test, standardizer, config, device, removed_view=index
        )
        ablation[view] = {
            "reconstruction_rmse": current["reconstruction"]["macro_pointwise_rmse_over_chord"],
            "increase": float(current["reconstruction"]["macro_pointwise_rmse_over_chord"])
            - float(test_metrics["reconstruction"]["macro_pointwise_rmse_over_chord"]),
        }
    elapsed = time.perf_counter() - started
    provenance = Counter(
        value
        for item in examples
        for value in item.target_provenance
        if value != "ABSENT"
    )
    payload = {
        "schema_version": "multiview-spatial-curve-parameter-training-1.0",
        "architecture": {
            "backbone": config.get(
                "feature_encoder_description",
                "frozen local Mask R-CNN v2 ResNet-50-FPN pooled to 85 spatial tokens/view",
            ),
            "trainable": config.get(
                "trainable_description",
                "spatial memory Transformer + five target-specific role queries + landmark/metric/two-cubic heads",
            ),
            "tokens_per_view": sum(
                int(value) ** 2 for value in config["pyramid_grid_sizes"]
            ),
            "input_feature_dimension": int(config["spatial_feature_dim"]),
            "model_width": int(config["width"]),
            "attention_heads": int(config["heads"]),
            "memory_layers": int(config.get("memory_layers", 1)),
            "decoder_layers": int(config.get("decoder_layers", 3)),
            "query_names": list(CURVE_QUERY_NAMES),
            "parameter_names": list(CURVE_PARAMETER_NAMES),
            "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        },
        "ablation_contract": config.get("ablation_contract"),
        "truth_provenance_counts": dict(provenance),
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "training_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "history": history,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_leave_one_view_out": ablation,
        "limitations": [
            "Dense GCD targets are fitted approximations, not original generator Bezier controls.",
            "The decoder predicts named curve formulas, not a complete panel/stitch/notch CAD pattern.",
            "Attention is reported descriptively and is not treated as causal feature importance."
        ],
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": payload["schema_version"],
            "config": config,
            "state_dict": model.state_dict(),
            "standardizer": {
                "means": standardizer.means,
                "standard_deviations": standardizer.standard_deviations,
            },
            "best_epoch": best_epoch,
            "query_names": CURVE_QUERY_NAMES,
            "parameter_names": CURVE_PARAMETER_NAMES,
        },
        args.checkpoint,
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.predictions, **prediction_arrays)
    print(json.dumps({key: payload[key] for key in ("best_epoch", "stopped_epoch", "training_seconds", "split_counts")}, indent=2))


if __name__ == "__main__":
    main()
