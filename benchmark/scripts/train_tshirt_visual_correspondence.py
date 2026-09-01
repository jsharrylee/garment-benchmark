"""Train the first observational T-shirt 4-view semantic correspondence proof."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np

from benchmark.drafting_semantics.gcdv2_surface_correspondence import (
    ELEMENT_NAMES,
    PARAMETER_TO_ELEMENTS,
    TSHIRT_PARAMETER_NAMES,
    build_tshirt_visual_correspondence_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/drafting_semantics/tshirt_visual_causality/surface_correspondence.npz"),
    )
    parser.add_argument(
        "--fpn-cache",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/tshirt_visual_causality/correspondence_model"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fold(sample_id: str, folds: int) -> int:
    value = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(value[:8], "little") % int(folds)


def _association_target(device) -> Any:
    import torch

    target = torch.zeros((len(TSHIRT_PARAMETER_NAMES), len(ELEMENT_NAMES)), device=device)
    for parameter_index, parameter in enumerate(TSHIRT_PARAMETER_NAMES):
        for element in PARAMETER_TO_ELEMENTS[parameter]:
            target[parameter_index, ELEMENT_NAMES.index(element)] = 1.0
    return target / target.sum(-1, keepdim=True).clamp_min(1.0)


def _losses(
    output: dict[str, Any],
    heatmaps,
    element_valid,
    parameters,
    parameter_valid,
    association_target,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    # Do not normalize all pyramid levels together.  A single 1x1 token is a
    # trivial location answer and previously dominated the metric.  Supervise
    # the 8x8, 4x4 and 2x2 levels independently; the 1x1 global token is kept
    # in memory as context but receives no localization score.
    location_terms = []
    offset = 0
    for count in (64, 16, 4):
        target_level = heatmaps[..., offset : offset + count]
        logits_level = output["element_location_logits"][..., offset : offset + count]
        target_mass = target_level.sum(-1, keepdim=True)
        view_valid = (target_mass[..., 0] > 1e-6) & element_valid[..., None]
        distribution = target_level / target_mass.clamp_min(1e-6)
        per = -(distribution * F.log_softmax(logits_level, dim=-1)).sum(-1)
        location_terms.append((per * view_valid).sum() / view_valid.sum().clamp_min(1))
        offset += count
    location = torch.stack(location_terms).mean()

    log_variance = output["parameter_log_variance"]
    squared = torch.square(output["parameter_mean"] - parameters)
    parameter_per = 0.5 * (torch.exp(-log_variance) * squared + log_variance)
    parameter = (parameter_per * parameter_valid).sum() / parameter_valid.sum().clamp_min(1)

    attention = output["parameter_element_attention"].mean(1).clamp_min(1e-8)
    association = -(
        association_target[None] * torch.log(attention)
    ).sum(-1).mean()
    total = location + 2.0 * parameter + 0.2 * association
    return {
        "total": total,
        "location": location,
        "parameter": parameter,
        "association": association,
    }


def _batch_indices(indices: np.ndarray, batch_size: int, *, shuffle: bool, seed: int):
    values = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _evaluate_loss(
    model,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    feature_cache,
    parameter_mean: np.ndarray,
    parameter_std: np.ndarray,
    identifiable: np.ndarray,
    association_target,
    device,
    batch_size: int,
) -> float:
    import torch

    model.eval()
    totals = []
    with torch.inference_mode():
        for current in _batch_indices(indices, batch_size, shuffle=False, seed=0):
            features = torch.from_numpy(
                np.asarray(feature_cache["features"][arrays["feature_indices"][current]], dtype=np.float32)
                [:, [1, 0, 2, 3]]
            ).to(device)
            heatmaps = torch.from_numpy(arrays["element_heatmaps"][current].astype(np.float32)).to(device)
            element_valid = torch.from_numpy(arrays["element_valid"][current]).to(device)
            target = (arrays["parameter_values"][current] - parameter_mean) / parameter_std
            parameters = torch.from_numpy(target.astype(np.float32)).to(device)
            valid = arrays["parameter_valid"][current] & identifiable[None]
            parameter_valid = torch.from_numpy(valid.astype(np.float32)).to(device)
            output = model(features)
            totals.append(
                float(
                    _losses(
                        output,
                        heatmaps,
                        element_valid,
                        parameters,
                        parameter_valid,
                        association_target,
                    )["total"].cpu()
                )
            )
    return float(np.mean(totals))


def _r_squared(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    if denominator <= 1e-10:
        return None
    return float(1.0 - np.sum(np.square(truth - prediction)) / denominator)


def _json_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    import torch

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = np.load(args.corpus, allow_pickle=False)
    fpn = np.load(args.fpn_cache, allow_pickle=False, mmap_mode="r")
    arrays = {name: corpus[name] for name in corpus.files}
    sample_ids = arrays["sample_ids"].astype(str)
    folds = np.asarray([_fold(value, args.folds) for value in sample_ids], dtype=np.int64)
    association_target = _association_target(device)
    args.output.mkdir(parents=True, exist_ok=True)

    all_predictions: list[dict[str, Any]] = []
    fold_receipts: list[dict[str, Any]] = []
    location_values: defaultdict[str, list[float]] = defaultdict(list)
    baseline_location_values: defaultdict[str, list[float]] = defaultdict(list)
    parameter_truth: defaultdict[int, list[float]] = defaultdict(list)
    parameter_prediction: defaultdict[int, list[float]] = defaultdict(list)
    parameter_prior_prediction: defaultdict[int, list[float]] = defaultdict(list)

    for fold_index in range(args.folds):
        test_indices = np.flatnonzero(folds == fold_index)
        validation_indices = np.flatnonzero(folds == ((fold_index + 1) % args.folds))
        train_indices = np.flatnonzero((folds != fold_index) & (folds != ((fold_index + 1) % args.folds)))
        train_parameters = arrays["parameter_values"][train_indices].astype(np.float64)
        parameter_mean = np.mean(train_parameters, axis=0)
        raw_std = np.std(train_parameters, axis=0)
        identifiable = raw_std > 1e-3
        parameter_std = np.where(identifiable, raw_std, 1.0)
        train_mean_heatmap = np.mean(
            arrays["element_heatmaps"][train_indices].astype(np.float32), axis=0
        )

        _seed_everything(args.seed + fold_index)
        model = build_tshirt_visual_correspondence_model(
            width=args.width,
            layers=args.layers,
            heads=args.heads,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        best_validation = math.inf
        best_state = None
        best_epoch = -1
        remaining = args.patience
        history = []
        for epoch in range(args.epochs):
            model.train()
            epoch_losses: defaultdict[str, list[float]] = defaultdict(list)
            for current in _batch_indices(
                train_indices,
                args.batch_size,
                shuffle=True,
                seed=args.seed + fold_index * 10000 + epoch,
            ):
                features = torch.from_numpy(
                    np.asarray(fpn["features"][arrays["feature_indices"][current]], dtype=np.float32)
                    [:, [1, 0, 2, 3]]
                ).to(device)
                heatmaps = torch.from_numpy(arrays["element_heatmaps"][current].astype(np.float32)).to(device)
                element_valid = torch.from_numpy(arrays["element_valid"][current]).to(device)
                normalized = (arrays["parameter_values"][current] - parameter_mean) / parameter_std
                parameters = torch.from_numpy(normalized.astype(np.float32)).to(device)
                valid = arrays["parameter_valid"][current] & identifiable[None]
                parameter_valid = torch.from_numpy(valid.astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(features)
                losses = _losses(
                    output,
                    heatmaps,
                    element_valid,
                    parameters,
                    parameter_valid,
                    association_target,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for name, value in losses.items():
                    epoch_losses[name].append(float(value.detach().cpu()))
            validation_loss = _evaluate_loss(
                model,
                validation_indices,
                arrays,
                fpn,
                parameter_mean,
                parameter_std,
                identifiable,
                association_target,
                device,
                args.batch_size,
            )
            row = {
                "epoch": epoch,
                **{name: float(np.mean(values)) for name, values in epoch_losses.items()},
                "validation_total": validation_loss,
            }
            history.append(row)
            if validation_loss < best_validation - 1e-5:
                best_validation = validation_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                remaining = args.patience
            else:
                remaining -= 1
            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "fold": fold_index,
                            "epoch": epoch + 1,
                            "train": row["total"],
                            "validation": validation_loss,
                            "remaining_patience": remaining,
                        }
                    ),
                    flush=True,
                )
            if remaining <= 0:
                break
        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        torch.save(
            {
                "schema_version": "tshirt-visual-correspondence-model/v2",
                "model_state": best_state,
                "parameter_mean": parameter_mean,
                "parameter_std": parameter_std,
                "identifiable": identifiable,
                "parameter_names": TSHIRT_PARAMETER_NAMES,
                "element_names": ELEMENT_NAMES,
                "width": args.width,
                "layers": args.layers,
                "heads": args.heads,
                "fold": fold_index,
                "claim_boundary": "observational same-generator cross-validation; no causal visual claim",
            },
            args.output / f"fold_{fold_index:02d}.pt",
        )
        (args.output / f"fold_{fold_index:02d}_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

        with torch.inference_mode():
            for current in _batch_indices(test_indices, args.batch_size, shuffle=False, seed=0):
                features = torch.from_numpy(
                    np.asarray(fpn["features"][arrays["feature_indices"][current]], dtype=np.float32)
                    [:, [1, 0, 2, 3]]
                ).to(device)
                output = model(features)
                location = output["element_location_logits"].float().cpu().numpy()
                normalized_prediction = output["parameter_mean"].float().cpu().numpy()
                prediction = normalized_prediction * parameter_std + parameter_mean
                target_heatmaps = arrays["element_heatmaps"][current].astype(np.float32)
                for batch_index, source_index in enumerate(current):
                    sample_row = {
                        "sample_id": str(sample_ids[source_index]),
                        "fold": fold_index,
                        "parameters": {},
                        "element_location": {},
                    }
                    for parameter_index, name in enumerate(TSHIRT_PARAMETER_NAMES):
                        truth = float(arrays["parameter_values"][source_index, parameter_index])
                        estimate = float(prediction[batch_index, parameter_index])
                        status = "EVALUATED" if identifiable[parameter_index] else "NOT_IDENTIFIABLE_CONSTANT_TARGET"
                        sample_row["parameters"][name] = {
                            "truth": truth,
                            "prediction": estimate,
                            "status": status,
                        }
                        if identifiable[parameter_index]:
                            parameter_truth[parameter_index].append(truth)
                            parameter_prediction[parameter_index].append(estimate)
                            parameter_prior_prediction[parameter_index].append(
                                float(parameter_mean[parameter_index])
                            )
                    for element_index, element in enumerate(ELEMENT_NAMES):
                        per_view = {}
                        for view in range(4):
                            target = target_heatmaps[batch_index, element_index, view]
                            if float(target.sum()) <= 1e-6:
                                continue
                            # The correspondence metric uses only the finest
                            # 8x8 FPN grid.  Coarser/global tokens remain model
                            # context but cannot earn a trivial localization hit.
                            fine_target = target[:64]
                            ranking = np.argsort(
                                location[batch_index, element_index, view, :64]
                            )[::-1]
                            baseline_ranking = np.argsort(
                                train_mean_heatmap[element_index, view, :64]
                            )[::-1]
                            top1_score = float(fine_target[ranking[0]])
                            top3_hit = float(np.max(fine_target[ranking[:3]]) >= 0.5)
                            baseline_top1_score = float(fine_target[baseline_ranking[0]])
                            baseline_top3_hit = float(
                                np.max(fine_target[baseline_ranking[:3]]) >= 0.5
                            )
                            location_values["top1_target_score"].append(top1_score)
                            location_values["top3_hit"].append(top3_hit)
                            baseline_location_values["top1_target_score"].append(
                                baseline_top1_score
                            )
                            baseline_location_values["top3_hit"].append(baseline_top3_hit)
                            per_view[str(view)] = {
                                "top1_target_score": top1_score,
                                "top3_hit": bool(top3_hit),
                                "mean_pattern_prior_top1_target_score": baseline_top1_score,
                                "mean_pattern_prior_top3_hit": bool(baseline_top3_hit),
                            }
                        sample_row["element_location"][element] = per_view
                    all_predictions.append(sample_row)
        fold_receipts.append(
            {
                "fold": fold_index,
                "train_count": int(len(train_indices)),
                "validation_count": int(len(validation_indices)),
                "test_count": int(len(test_indices)),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_validation),
                "identifiable_parameters": [
                    name for name, valid in zip(TSHIRT_PARAMETER_NAMES, identifiable) if valid
                ],
                "constant_parameters": [
                    name for name, valid in zip(TSHIRT_PARAMETER_NAMES, identifiable) if not valid
                ],
            }
        )

    parameter_metrics = {}
    parameter_prior_metrics = {}
    for parameter_index, name in enumerate(TSHIRT_PARAMETER_NAMES):
        truth = np.asarray(parameter_truth.get(parameter_index, ()), dtype=np.float64)
        prediction = np.asarray(parameter_prediction.get(parameter_index, ()), dtype=np.float64)
        if not len(truth):
            parameter_metrics[name] = {
                "status": "NOT_IDENTIFIABLE_CONSTANT_TARGET",
                "count": 0,
            }
            parameter_prior_metrics[name] = {
                "status": "NOT_IDENTIFIABLE_CONSTANT_TARGET",
                "count": 0,
            }
            continue
        error = prediction - truth
        parameter_metrics[name] = {
            "status": "EVALUATED_OBSERVATIONAL",
            "count": int(len(truth)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "r_squared": _json_float(_r_squared(truth, prediction)),
            "truth_standard_deviation": float(np.std(truth)),
        }
        prior = np.asarray(parameter_prior_prediction[parameter_index], dtype=np.float64)
        prior_error = prior - truth
        parameter_prior_metrics[name] = {
            "status": "EVALUATED_FOLD_TRAIN_MEAN_PRIOR",
            "count": int(len(truth)),
            "mae": float(np.mean(np.abs(prior_error))),
            "rmse": float(np.sqrt(np.mean(np.square(prior_error)))),
            "r_squared": _json_float(_r_squared(truth, prior)),
        }
    location_metrics = {
        key: float(np.mean(values)) for key, values in location_values.items()
    }
    prior_location_metrics = {
        key: float(np.mean(values)) for key, values in baseline_location_values.items()
    }
    improved_parameter_count = sum(
        1
        for name, value in parameter_metrics.items()
        if value.get("count", 0)
        and value["mae"] < parameter_prior_metrics[name]["mae"]
    )
    location_beats_prior = (
        location_metrics["top1_target_score"]
        > prior_location_metrics["top1_target_score"]
    )
    model_status = (
        "PASS_OBSERVATIONAL_MODEL_BEATS_PRIORS"
        if location_beats_prior and improved_parameter_count >= 4
        else "PARTIAL_OBSERVATIONAL_MODEL_BELOW_PRIOR"
    )
    metrics = {
        "schema_version": "tshirt-visual-correspondence-training-result/v2",
        "status": model_status,
        "device": str(device),
        "sample_count": int(len(sample_ids)),
        "evaluation": {
            "protocol": f"deterministic {args.folds}-fold sample-id cross-validation",
            "same_generator": True,
            "fixed_neutral_body": True,
            "family_disjoint": False,
            "counterfactual": False,
        },
        "location_metrics": location_metrics,
        "mean_pattern_prior_location_metrics": prior_location_metrics,
        "parameter_metrics": parameter_metrics,
        "fold_train_mean_parameter_metrics": parameter_prior_metrics,
        "prior_comparison": {
            "location_top1_beats_mean_pattern_prior": location_beats_prior,
            "parameters_with_lower_mae_than_fold_train_mean": improved_parameter_count,
            "evaluated_parameter_count": int(
                sum(value.get("count", 0) > 0 for value in parameter_metrics.values())
            ),
        },
        "folds": fold_receipts,
        "claim_boundary": (
            "The model learned where exact 2D semantic elements project in existing GCDv2 "
            "four-view renders and whether seven varying 2D quantities are visually recoverable. "
            "It did not observe one-parameter edited render pairs, so influence and causal inverse "
            "claims remain disabled. Shoulder slope is constant and is not scored."
        ),
        "causal_visual_metrics_enabled": False,
        "true_counterfactual_four_view_pairs": 0,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "cross_validation_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(all_predictions, key=lambda value: value["sample_id"]):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
