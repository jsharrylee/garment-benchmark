from __future__ import annotations

"""Audit target-specific view attention in the continuous geometry model.

This script intentionally reports two different quantities side by side:

* decoder cross-attention: an association between a semantic role query and
  four *global* encoded view tokens; and
* leave-one-view-out error change: model sensitivity when one global view is
  removed.

Neither quantity localizes an image region, and neither establishes a causal
relationship between a garment edit and a 3D appearance change.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import GARMENT_ROLES
from benchmark.drafting_semantics.multiview_element_geometry import (
    GEOMETRY_PANEL_ROLES,
    GEOMETRY_PATH_ROLES,
    GEOMETRY_TARGET_NAMES,
    PANEL_GEOMETRY_COMPONENTS,
    PATH_GEOMETRY_COMPONENTS,
    PRESENCE_TARGET_NAMES,
    MaskedTargetStandardizer,
    build_multiview_geometry_model,
    multiview_geometry_batch,
    read_multiview_geometry_examples,
)
from benchmark.drafting_semantics.multiview_pattern_semantics import VIEW_NAMES


def _rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including ties."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, rank: bool = False) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    left_array = left_array[valid]
    right_array = right_array[valid]
    if len(left_array) < 3:
        return None
    if rank:
        left_array = _rank(left_array)
        right_array = _rank(right_array)
    if np.std(left_array) < 1e-12 or np.std(right_array) < 1e-12:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _role_geometry_targets() -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for role in GEOMETRY_PANEL_ROLES:
        count = len(PANEL_GEOMETRY_COMPONENTS)
        output[f"panel:{role}"] = tuple(GEOMETRY_TARGET_NAMES[cursor : cursor + count])
        cursor += count
    for role in GEOMETRY_PATH_ROLES:
        count = len(PATH_GEOMETRY_COMPONENTS)
        output[f"path:{role}"] = tuple(GEOMETRY_TARGET_NAMES[cursor : cursor + count])
        cursor += count
    output["seam:sleeve_head_to_armhole"] = (GEOMETRY_TARGET_NAMES[cursor],)
    if cursor + 1 != len(GEOMETRY_TARGET_NAMES):
        raise AssertionError("geometry target layout no longer matches semantic role queries")
    if tuple(output) != tuple(PRESENCE_TARGET_NAMES):
        raise AssertionError("geometry role order no longer matches presence query order")
    return output


def _capture_attention(model, examples, standardizer, config: Mapping[str, Any], device):
    """Return presence-conditioned decoder attention statistics.

    Raw layer tensors have shape [batch, head, role-query, global-view-token].
    Each role is averaged only over samples where its ground-truth presence is
    true.  This prevents absent sleeve/crotch queries from dominating means.
    """

    import torch

    layer_sums = None
    layer_squares = None
    role_counts = np.zeros(len(PRESENCE_TARGET_NAMES), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), int(config["batch_size"])):
            current = examples[start : start + int(config["batch_size"])]
            batch = multiview_geometry_batch(current, standardizer)
            output = model(
                torch.from_numpy(batch["view_features"]).to(device),
                capture_attention=True,
            )
            layers = np.stack(
                [weights.detach().cpu().float().numpy() for weights in output["role_attention"]],
                axis=0,
            )
            # Normalize defensively. MultiheadAttention already normalizes over
            # keys, but this also protects the audit if implementation changes.
            layers = layers / np.maximum(layers.sum(axis=-1, keepdims=True), 1e-12)
            if layer_sums is None:
                layer_sums = np.zeros(
                    (layers.shape[0], layers.shape[2], layers.shape[3], layers.shape[4]),
                    dtype=np.float64,
                )
                layer_squares = np.zeros_like(layer_sums)
            present = np.asarray(batch["presence_targets"]) >= 0.5
            for role_index in range(len(PRESENCE_TARGET_NAMES)):
                selected = present[:, role_index]
                if not selected.any():
                    continue
                # Slice the query dimension before boolean indexing.  NumPy
                # otherwise moves the advanced-indexed batch axis in front of
                # the decoder-layer axis.
                values = layers[:, :, :, role_index, :][:, selected, :, :]
                layer_sums[:, :, role_index, :] += values.sum(axis=1)
                layer_squares[:, :, role_index, :] += np.square(values).sum(axis=1)
                role_counts[role_index] += int(selected.sum())
    if layer_sums is None or layer_squares is None:
        raise RuntimeError("no test attention was captured")
    denominator = np.maximum(role_counts[None, None, :, None], 1)
    means = layer_sums / denominator
    variances = np.maximum(layer_squares / denominator - np.square(means), 0.0)
    return means, np.sqrt(variances), role_counts


def _metric_value(metrics: Mapping[str, Any], target: str, key: str) -> float | None:
    record = metrics["geometry"]["per_target"].get(target, {})
    value = record.get(key)
    return None if value is None else float(value)


def _build_ablation(metrics: Mapping[str, Any]) -> tuple[dict, np.ndarray]:
    baseline = metrics["test"]
    ablations = metrics["test_leave_one_view_out"]
    role_targets = _role_geometry_targets()
    role_delta = np.full((len(PRESENCE_TARGET_NAMES), len(VIEW_NAMES)), np.nan, dtype=np.float64)
    target_payload: dict[str, Any] = {}

    for target in GEOMETRY_TARGET_NAMES:
        baseline_error = _metric_value(baseline, target, "normalized_mae")
        per_view = {}
        for view in VIEW_NAMES:
            ablated_error = _metric_value(ablations[view], target, "normalized_mae")
            per_view[view] = {
                "baseline_normalized_mae": baseline_error,
                "ablated_normalized_mae": ablated_error,
                "normalized_mae_delta": (
                    None
                    if baseline_error is None or ablated_error is None
                    else ablated_error - baseline_error
                ),
            }
        target_payload[target] = per_view

    role_payload = {}
    for role_index, role in enumerate(PRESENCE_TARGET_NAMES):
        role_view_payload = {}
        baseline_presence = baseline["presence"]["per_role"][role]
        targets = role_targets[role]
        for view_index, view in enumerate(VIEW_NAMES):
            deltas = [
                target_payload[target][view]["normalized_mae_delta"] for target in targets
            ]
            valid_deltas = [float(value) for value in deltas if value is not None]
            mean_delta = float(np.mean(valid_deltas)) if valid_deltas else None
            if mean_delta is not None:
                role_delta[role_index, view_index] = mean_delta
            ablated_presence = ablations[view]["presence"]["per_role"][role]
            f1_drop = float(baseline_presence["f1"] - ablated_presence["f1"])
            role_view_payload[view] = {
                "mean_geometry_normalized_mae_delta": mean_delta,
                "presence_f1_drop": f1_drop,
                "target_count": len(valid_deltas),
            }
        role_payload[role] = {
            "geometry_targets": list(targets),
            "baseline_presence_support": int(baseline_presence["support"]),
            "views": role_view_payload,
        }

    global_payload = {}
    baseline_geometry = float(baseline["geometry"]["mean_dimension_balanced_normalized_mae"])
    baseline_category = float(baseline["category"]["macro_f1"])
    baseline_presence = float(baseline["presence"]["macro_f1"])
    for view in VIEW_NAMES:
        current = ablations[view]
        global_payload[view] = {
            "geometry_normalized_mae_delta": (
                float(current["geometry"]["mean_dimension_balanced_normalized_mae"])
                - baseline_geometry
            ),
            "category_macro_f1_drop": baseline_category - float(current["category"]["macro_f1"]),
            "presence_macro_f1_drop": baseline_presence - float(current["presence"]["macro_f1"]),
        }

    return {
        "global": global_payload,
        "by_role": role_payload,
        "by_target": target_payload,
    }, role_delta


def _attention_payload(
    means: np.ndarray,
    standard_deviations: np.ndarray,
    role_counts: np.ndarray,
) -> dict:
    layers = []
    for layer_index in range(means.shape[0]):
        roles = {}
        for role_index, role in enumerate(PRESENCE_TARGET_NAMES):
            heads = {}
            for head_index in range(means.shape[1]):
                heads[f"head_{head_index}"] = {
                    view: {
                        "mean_attention": float(means[layer_index, head_index, role_index, view_index]),
                        "sample_standard_deviation": float(
                            standard_deviations[layer_index, head_index, role_index, view_index]
                        ),
                    }
                    for view_index, view in enumerate(VIEW_NAMES)
                }
            roles[role] = {
                "present_test_samples": int(role_counts[role_index]),
                "head_mean_view_distribution": {
                    view: float(means[layer_index, :, role_index, view_index].mean())
                    for view_index, view in enumerate(VIEW_NAMES)
                },
                "heads": heads,
            }
        layers.append({"decoder_layer": layer_index, "roles": roles})
    return {
        "tensor_order": "decoder_layer x attention_head x semantic_role_query x global_view_token",
        "tensor_shape": list(means.shape),
        "conditioning": "mean over test samples where the ground-truth semantic role is present",
        "layers": layers,
    }


def _alignment_payload(final_attention: np.ndarray, role_delta: np.ndarray) -> dict:
    mean_attention = final_attention.mean(axis=0)
    valid = np.isfinite(role_delta)
    overall = {
        "pearson_signed_delta": _correlation(mean_attention[valid], role_delta[valid]),
        "spearman_signed_delta": _correlation(mean_attention[valid], role_delta[valid], rank=True),
        "pearson_positive_only_delta": _correlation(
            mean_attention[valid], np.maximum(role_delta[valid], 0.0)
        ),
        "spearman_positive_only_delta": _correlation(
            mean_attention[valid], np.maximum(role_delta[valid], 0.0), rank=True
        ),
    }
    per_head = []
    for head_index in range(final_attention.shape[0]):
        values = final_attention[head_index]
        per_head.append(
            {
                "head": head_index,
                "pearson_signed_delta": _correlation(values[valid], role_delta[valid]),
                "spearman_signed_delta": _correlation(values[valid], role_delta[valid], rank=True),
            }
        )
    matches = []
    for role_index, role in enumerate(PRESENCE_TARGET_NAMES):
        if not np.isfinite(role_delta[role_index]).any():
            continue
        attention_view = int(np.nanargmax(mean_attention[role_index]))
        ablation_view = int(np.nanargmax(role_delta[role_index]))
        matches.append(
            {
                "role": role,
                "top_attention_view": VIEW_NAMES[attention_view],
                "largest_error_increase_view": VIEW_NAMES[ablation_view],
                "match": attention_view == ablation_view,
            }
        )
    overall["top_view_match_rate"] = float(np.mean([item["match"] for item in matches]))
    overall["top_view_match_count"] = int(sum(item["match"] for item in matches))
    overall["role_count"] = len(matches)
    return {"overall": overall, "per_head": per_head, "role_top_view_comparison": matches}


def _predict_geometry(model, examples, standardizer, config: Mapping[str, Any], device) -> dict[str, np.ndarray]:
    import torch

    predictions: dict[str, np.ndarray] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), int(config["batch_size"])):
            current = examples[start : start + int(config["batch_size"])]
            batch = multiview_geometry_batch(current, standardizer)
            output = model(torch.from_numpy(batch["view_features"]).to(device))
            decoded = standardizer.decode(
                output["geometry_prediction"].detach().cpu().float().numpy()
            )
            for row, sample_id in enumerate(batch["sample_ids"]):
                predictions[str(sample_id)] = decoded[row]
    return predictions


def _pair_delta_audit(examples, predictions: Mapping[str, np.ndarray], standardizer):
    """Compare descriptor differences for fixed, same-category test pairs.

    Pair formation uses only category and a seeded hash of sample ID.  It does
    not inspect targets, predictions, or images, so examples are not selected
    for favorable results.  All pairs are non-overlapping within category.
    """

    seed = "same-category-pair-audit-20260828"
    by_category: dict[int, list[Any]] = {}
    for example in examples:
        by_category.setdefault(int(example.category_target), []).append(example)
    pairs = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda item: hashlib.sha256(f"{seed}|{item.sample_id}".encode("utf-8")).hexdigest(),
        )
        pairs.extend((category, ordered[index], ordered[index + 1]) for index in range(0, len(ordered) - 1, 2))

    deviations = np.asarray(standardizer.standard_deviations, dtype=np.float64)
    material_threshold = 0.10
    target_true: list[list[float]] = [[] for _ in GEOMETRY_TARGET_NAMES]
    target_predicted: list[list[float]] = [[] for _ in GEOMETRY_TARGET_NAMES]
    pair_records = []
    scatter_true = []
    scatter_predicted = []
    for category, left, right in pairs:
        jointly_present = np.asarray(left.geometry_mask) & np.asarray(right.geometry_mask)
        true_normalized = (
            np.asarray(right.geometry_target, dtype=np.float64)
            - np.asarray(left.geometry_target, dtype=np.float64)
        ) / deviations
        predicted_normalized = (
            np.asarray(predictions[right.sample_id], dtype=np.float64)
            - np.asarray(predictions[left.sample_id], dtype=np.float64)
        ) / deviations
        valid_indices = np.flatnonzero(jointly_present)
        for index in valid_indices:
            target_true[index].append(float(true_normalized[index]))
            target_predicted[index].append(float(predicted_normalized[index]))
        scatter_true.extend(float(true_normalized[index]) for index in valid_indices)
        scatter_predicted.extend(float(predicted_normalized[index]) for index in valid_indices)
        absolute_true = np.abs(true_normalized[valid_indices])
        absolute_predicted = np.abs(predicted_normalized[valid_indices])
        top_k = min(5, len(valid_indices))
        true_order = valid_indices[np.argsort(-absolute_true, kind="mergesort")[:top_k]]
        predicted_order = valid_indices[np.argsort(-absolute_predicted, kind="mergesort")[:top_k]]
        material = valid_indices[absolute_true >= material_threshold]
        pair_records.append(
            {
                "category_target": category,
                "category": GARMENT_ROLES[category],
                "left_sample_id": left.sample_id,
                "right_sample_id": right.sample_id,
                "joint_target_count": int(len(valid_indices)),
                "material_delta_count": int(len(material)),
                "absolute_delta_rank_spearman": _correlation(
                    absolute_true, absolute_predicted, rank=True
                ),
                "top_5_changed_target_overlap": (
                    float(len(set(true_order) & set(predicted_order)) / max(top_k, 1))
                ),
                "true_top_changed_targets": [GEOMETRY_TARGET_NAMES[index] for index in true_order],
                "predicted_top_changed_targets": [
                    GEOMETRY_TARGET_NAMES[index] for index in predicted_order
                ],
            }
        )

    per_target = {}
    model_errors = []
    zero_errors = []
    sign_scores = []
    role_targets = _role_geometry_targets()
    for index, name in enumerate(GEOMETRY_TARGET_NAMES):
        truth = np.asarray(target_true[index], dtype=np.float64)
        estimate = np.asarray(target_predicted[index], dtype=np.float64)
        if not len(truth):
            per_target[name] = {"joint_pair_support": 0}
            continue
        model_error = float(np.mean(np.abs(estimate - truth)))
        zero_error = float(np.mean(np.abs(truth)))
        material = np.abs(truth) >= material_threshold
        sign_accuracy = (
            float(np.mean(np.sign(estimate[material]) == np.sign(truth[material])))
            if material.any()
            else None
        )
        rank_correlation = _correlation(estimate, truth, rank=True)
        per_target[name] = {
            "joint_pair_support": int(len(truth)),
            "material_delta_support": int(material.sum()),
            "normalized_delta_mae": model_error,
            "zero_delta_baseline_normalized_mae": zero_error,
            "normalized_mae_gain_over_zero_delta": zero_error - model_error,
            "material_delta_sign_accuracy": sign_accuracy,
            "signed_delta_rank_spearman": rank_correlation,
        }
        model_errors.append(model_error)
        zero_errors.append(zero_error)
        if sign_accuracy is not None:
            sign_scores.append(sign_accuracy)

    role_payload = {}
    role_model_error = []
    role_zero_error = []
    for role, targets in role_targets.items():
        rows = [per_target[target] for target in targets if per_target[target].get("joint_pair_support", 0)]
        model_error = float(np.mean([row["normalized_delta_mae"] for row in rows])) if rows else None
        zero_error = float(np.mean([row["zero_delta_baseline_normalized_mae"] for row in rows])) if rows else None
        role_payload[role] = {
            "target_count": len(rows),
            "mean_normalized_delta_mae": model_error,
            "mean_zero_delta_baseline_normalized_mae": zero_error,
            "mean_gain_over_zero_delta": (
                None if model_error is None or zero_error is None else zero_error - model_error
            ),
        }
        role_model_error.append(model_error if model_error is not None else np.nan)
        role_zero_error.append(zero_error if zero_error is not None else np.nan)

    pair_rank = [
        record["absolute_delta_rank_spearman"]
        for record in pair_records
        if record["absolute_delta_rank_spearman"] is not None
    ]
    pair_top_overlap = [record["top_5_changed_target_overlap"] for record in pair_records]
    aggregate = {
        "pair_count": len(pair_records),
        "dimension_balanced_normalized_delta_mae": float(np.mean(model_errors)),
        "zero_delta_baseline_dimension_balanced_normalized_mae": float(np.mean(zero_errors)),
        "normalized_mae_gain_over_same_category_zero_delta": float(
            np.mean(zero_errors) - np.mean(model_errors)
        ),
        "dimension_balanced_material_delta_sign_accuracy": float(np.mean(sign_scores)),
        "mean_within_pair_absolute_delta_rank_spearman": float(np.mean(pair_rank)),
        "median_within_pair_absolute_delta_rank_spearman": float(np.median(pair_rank)),
        "mean_top_5_changed_target_overlap": float(np.mean(pair_top_overlap)),
        "material_delta_threshold_in_train_standard_deviations": material_threshold,
    }
    payload = {
        "pairing_protocol": (
            "non-overlapping same-category official-test pairs ordered by SHA-256 of a fixed seed and sample ID; "
            "targets, predictions, and images are not used for pair selection"
        ),
        "aggregate": aggregate,
        "by_role": role_payload,
        "by_target": per_target,
        "pairs": pair_records,
        "interpretation_contract": (
            "observational paired prediction of relative normalized 2D descriptors; each pair can differ in many "
            "latent garment parameters, so this is not a controlled counterfactual or causal edit estimate"
        ),
    }
    return (
        payload,
        np.asarray(scatter_true, dtype=np.float64),
        np.asarray(scatter_predicted, dtype=np.float64),
        np.asarray(role_model_error, dtype=np.float64),
        np.asarray(role_zero_error, dtype=np.float64),
    )


def _plot(
    image_path: Path,
    final_attention: np.ndarray,
    role_delta: np.ndarray,
    role_counts: np.ndarray,
    global_ablation: Mapping[str, Mapping[str, float]],
    alignment: Mapping[str, Any],
    pair_delta: Mapping[str, Any],
    pair_true: np.ndarray,
    pair_predicted: np.ndarray,
    pair_role_error: np.ndarray,
    pair_role_zero_error: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mean_attention = final_attention.mean(axis=0)
    head_view = final_attention.mean(axis=1)
    roles = [name.replace("panel:", "P:").replace("path:", "L:").replace("seam:", "S:") for name in PRESENCE_TARGET_NAMES]
    fig = plt.figure(figsize=(18, 24), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.2, 1.0), height_ratios=(1.25, 1.0, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(mean_attention, vmin=0.0, vmax=max(0.45, float(mean_attention.max())), cmap="viridis", aspect="auto")
    ax.set_title("Final decoder cross-attention · mean over 8 heads")
    ax.set_xticks(range(len(VIEW_NAMES)), VIEW_NAMES)
    ax.set_yticks(range(len(roles)), [f"{role}  (n={role_counts[index]})" for index, role in enumerate(roles)])
    for row in range(len(roles)):
        for column in range(len(VIEW_NAMES)):
            value = mean_attention[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.29 else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, label="attention share among four global view tokens")

    ax = fig.add_subplot(grid[0, 1])
    finite = role_delta[np.isfinite(role_delta)]
    extent = max(float(np.max(np.abs(finite))) if len(finite) else 0.0, 1e-4)
    image = ax.imshow(
        role_delta,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent),
        aspect="auto",
    )
    ax.set_title("Leave-one-view-out · role mean normalized-MAE change")
    ax.set_xticks(range(len(VIEW_NAMES)), VIEW_NAMES)
    ax.set_yticks(range(len(roles)), roles)
    for row in range(len(roles)):
        for column in range(len(VIEW_NAMES)):
            value = role_delta[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=ax, fraction=0.035, label="positive = error worsened after view removal")

    ax = fig.add_subplot(grid[1, 0])
    image = ax.imshow(head_view, vmin=0.0, vmax=max(0.4, float(head_view.max())), cmap="magma", aspect="auto")
    ax.set_title("Final decoder heads · average across present semantic roles")
    ax.set_xticks(range(len(VIEW_NAMES)), VIEW_NAMES)
    ax.set_yticks(range(head_view.shape[0]), [f"H{index}" for index in range(head_view.shape[0])])
    for row in range(head_view.shape[0]):
        for column in range(len(VIEW_NAMES)):
            value = head_view[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value > 0.25 else "black")
    fig.colorbar(image, ax=ax, fraction=0.035, label="attention share")

    ax = fig.add_subplot(grid[1, 1])
    valid = np.isfinite(role_delta)
    x = mean_attention[valid]
    y = role_delta[valid]
    ax.scatter(x, y, s=28, alpha=0.68, color="#3366aa", edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("final attention share")
    ax.set_ylabel("normalized-MAE change after removing that view")
    ax.set_title("Attention–ablation agreement (association check, not causality)")
    overall = alignment["overall"]
    global_lines = [
        f"Pearson(attn, signed Δerror): {overall['pearson_signed_delta']:+.3f}",
        f"Spearman(attn, signed Δerror): {overall['spearman_signed_delta']:+.3f}",
        f"top-view match: {overall['top_view_match_count']}/{overall['role_count']}",
        "",
        "Global leave-one-view-out Δ normalized-MAE:",
    ]
    global_lines.extend(
        f"  {view:>5}: {global_ablation[view]['geometry_normalized_mae_delta']:+.4f}"
        for view in VIEW_NAMES
    )
    global_lines.extend(
        (
            "",
            "Important boundary:",
            "attention keys are four global ResNet/Transformer view tokens;",
            "there is no pixel/garment-region localization and no causal edit claim.",
        )
    )
    ax.text(
        0.03,
        0.97,
        "\n".join(global_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )

    ax = fig.add_subplot(grid[2, 0])
    y = np.arange(len(roles))
    ax.barh(y - 0.19, pair_role_error, height=0.38, label="model predicted descriptor delta", color="#4477aa")
    ax.barh(y + 0.19, pair_role_zero_error, height=0.38, label="same-category zero-delta baseline", color="#cc6677")
    ax.set_yticks(y, roles)
    ax.invert_yaxis()
    ax.set_xlabel("normalized delta MAE (lower is better)")
    ax.set_title("Deterministic same-category test pairs · error by semantic role")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(grid[2, 1])
    # Plot all jointly present descriptor deltas. High alpha would hide the
    # dense center, so use a translucent cloud plus an identity line.
    ax.scatter(pair_true, pair_predicted, s=12, alpha=0.16, color="#228833", edgecolors="none")
    finite_pair = np.concatenate((pair_true[np.isfinite(pair_true)], pair_predicted[np.isfinite(pair_predicted)]))
    pair_extent = max(float(np.quantile(np.abs(finite_pair), 0.995)), 0.5)
    ax.plot([-pair_extent, pair_extent], [-pair_extent, pair_extent], color="black", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="#999999", linewidth=0.6)
    ax.axvline(0.0, color="#999999", linewidth=0.6)
    ax.set_xlim(-pair_extent, pair_extent)
    ax.set_ylim(-pair_extent, pair_extent)
    ax.set_xlabel("true normalized 2D descriptor difference")
    ax.set_ylabel("predicted normalized 2D descriptor difference")
    ax.set_title("Observational pair deltas · not a controlled counterfactual")
    pair_summary = pair_delta["aggregate"]
    ax.text(
        0.03,
        0.97,
        "\n".join(
            (
                f"fixed non-overlapping pairs: {pair_summary['pair_count']}",
                f"delta nMAE model / zero: {pair_summary['dimension_balanced_normalized_delta_mae']:.3f} / "
                f"{pair_summary['zero_delta_baseline_dimension_balanced_normalized_mae']:.3f}",
                f"gain over zero: {pair_summary['normalized_mae_gain_over_same_category_zero_delta']:+.3f}",
                f"material-delta sign accuracy: {pair_summary['dimension_balanced_material_delta_sign_accuracy']:.3f}",
                f"within-pair |delta| rank rho: {pair_summary['mean_within_pair_absolute_delta_rank_spearman']:+.3f}",
                f"top-5 changed-target overlap: {pair_summary['mean_top_5_changed_target_overlap']:.3f}",
            )
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )

    fig.suptitle(
        "Multiview semantic-geometry role-query audit · frozen official test split\n"
        "Cross-attention association, explicit global-view removal, and observational same-category deltas",
        fontsize=16,
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit role-query global-view attention against leave-one-view-out sensitivity."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drafting_semantics/multiview_element_geometry_resnet50.pt"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_element_geometry/training_metrics.json"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/raw/garmentcode_v2/metadata/official_split.json"),
    )
    parser.add_argument(
        "--semantic-records",
        type=Path,
        default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_element_geometry/attention_audit.json"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_element_geometry/attention_audit.png"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if tuple(checkpoint["geometry_target_names"]) != tuple(GEOMETRY_TARGET_NAMES):
        raise RuntimeError("checkpoint geometry target order does not match current code")
    if tuple(checkpoint["presence_target_names"]) != tuple(PRESENCE_TARGET_NAMES):
        raise RuntimeError("checkpoint role-query order does not match current code")
    standardizer = MaskedTargetStandardizer(
        tuple(checkpoint["target_standardizer"]["means"]),
        tuple(checkpoint["target_standardizer"]["standard_deviations"]),
    )
    model = build_multiview_geometry_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    examples = tuple(
        item
        for item in read_multiview_geometry_examples(
            args.index, args.split, args.semantic_records, args.features
        )
        if item.split == "test"
    )
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if int(metrics["best_epoch"]) != int(checkpoint["best_epoch"]):
        raise RuntimeError("metrics and checkpoint best epochs do not match")
    if int(metrics["split_counts"]["test"]) != len(examples):
        raise RuntimeError("metrics and loaded official test split sizes do not match")
    means, standard_deviations, role_counts = _capture_attention(
        model, examples, standardizer, config, device
    )
    ablation, role_delta = _build_ablation(metrics)
    alignment = _alignment_payload(means[-1], role_delta)
    predictions = _predict_geometry(model, examples, standardizer, config, device)
    (
        pair_delta,
        pair_true,
        pair_predicted,
        pair_role_error,
        pair_role_zero_error,
    ) = _pair_delta_audit(examples, predictions, standardizer)
    payload = {
        "schema_version": "multiview-element-geometry-attention-audit-1.0",
        "checkpoint": str(args.checkpoint),
        "test_sample_count": len(examples),
        "best_epoch": int(checkpoint["best_epoch"]),
        "attention": _attention_payload(means, standard_deviations, role_counts),
        "leave_one_view_out": ablation,
        "attention_ablation_alignment": alignment,
        "same_category_observational_pair_delta": pair_delta,
        "interpretation_contract": [
            "Cross-attention is an association between each semantic role query and four global encoded view tokens.",
            "The audit does not provide pixel-level or garment-region localization.",
            "Leave-one-view-out is a model-sensitivity intervention; it does not identify a causal garment-edit effect.",
            "Attention magnitude is not treated as feature importance; agreement with ablation is reported rather than assumed.",
            "Attention means are conditioned on ground-truth role presence in the frozen official test split.",
            "Same-category descriptor-delta pairs are observational and can differ in many latent parameters; they are not counterfactual edits.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(
        args.image,
        means[-1],
        role_delta,
        role_counts,
        ablation["global"],
        alignment,
        pair_delta,
        pair_true,
        pair_predicted,
        pair_role_error,
        pair_role_zero_error,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "image": str(args.image),
                "device": str(device),
                "test_samples": len(examples),
                "global_leave_one_out": ablation["global"],
                "alignment": alignment["overall"],
                "same_category_pair_delta": pair_delta["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
