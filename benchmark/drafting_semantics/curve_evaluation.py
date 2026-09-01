"""Frozen-test metrics for named two-cubic garment curve predictions.

The spatial four-view model predicts a compact formula for five semantic
paths.  This module keeps evaluation independent of the training loop so the
same frozen predictions can be audited without loading a model or GPU.  Curve
geometry is evaluated in the chord-local frame used by
``multiview_curve_parameters``; panel landmarks and length metrics remain in
their original target units.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .multiview_curve_parameters import (
    CONTROL_SLICE,
    CURVE_PARAMETER_NAMES,
    CURVE_QUERY_NAMES,
    METRIC_SLICE,
    sample_two_cubic_formula,
)


_EPSILON = 1e-8


def _validate_prediction_contract(
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    role_mask = np.asarray(role_mask, dtype=bool)
    contract = (len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES))
    if predicted.shape != expected.shape or predicted.ndim != 3:
        raise ValueError("predicted and expected must have the same rank-3 shape")
    if predicted.shape[1:] != contract:
        raise ValueError(
            f"predicted and expected must have trailing shape {contract}; got {predicted.shape[1:]}"
        )
    if role_mask.shape != predicted.shape[:2]:
        raise ValueError("role_mask must have shape [sample, curve_query]")
    observed = np.broadcast_to(role_mask[:, :, None], predicted.shape)
    if not np.isfinite(predicted[observed]).all():
        raise ValueError("observed predictions contain non-finite values")
    if not np.isfinite(expected[observed]).all():
        raise ValueError("observed targets contain non-finite values")
    return predicted, expected, role_mask


def coefficient_of_determination(
    predicted: np.ndarray, expected: np.ndarray
) -> float | None:
    """Return ordinary test-set R2, or ``None`` for a constant target."""

    predicted = np.asarray(predicted, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 1:
        raise ValueError("R2 inputs must be same-shaped one-dimensional arrays")
    if not len(expected):
        return None
    total = float(np.square(expected - expected.mean()).sum())
    if total <= _EPSILON:
        return None
    residual = float(np.square(predicted - expected).sum())
    return 1.0 - residual / total


def parameter_regression_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    train_standard_deviations: np.ndarray,
) -> dict[str, Any]:
    """Raw-parameter MAE/nMAE/R2 for every named curve and parameter.

    nMAE is MAE divided by the corresponding *training-split* standard
    deviation.  This avoids test leakage and keeps quantities with different
    units comparable.  R2 is calculated only where the frozen test target has
    non-zero variance.
    """

    predicted, expected, role_mask = _validate_prediction_contract(
        predicted, expected, role_mask
    )
    deviations = np.asarray(train_standard_deviations, dtype=np.float64)
    contract = (len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES))
    if deviations.shape != contract:
        raise ValueError(f"train_standard_deviations must have shape {contract}")
    if not np.isfinite(deviations).all() or np.any(deviations <= 0.0):
        raise ValueError("training standard deviations must be finite and positive")

    all_normalized_mae: list[float] = []
    all_r2: list[float] = []
    per_query: dict[str, Any] = {}
    for query_index, query_name in enumerate(CURVE_QUERY_NAMES):
        valid = role_mask[:, query_index]
        query_normalized_mae: list[float] = []
        query_r2: list[float] = []
        parameters: dict[str, Any] = {}
        for parameter_index, parameter_name in enumerate(CURVE_PARAMETER_NAMES):
            truth = expected[valid, query_index, parameter_index]
            estimate = predicted[valid, query_index, parameter_index]
            if not len(truth):
                parameters[parameter_name] = {"support": 0}
                continue
            absolute = np.abs(estimate - truth)
            mae = float(absolute.mean())
            normalized_mae = mae / float(deviations[query_index, parameter_index])
            r2 = coefficient_of_determination(estimate, truth)
            parameters[parameter_name] = {
                "support": int(len(truth)),
                "mae": mae,
                "normalized_mae_by_train_std": normalized_mae,
                "r2": r2,
            }
            query_normalized_mae.append(normalized_mae)
            all_normalized_mae.append(normalized_mae)
            if r2 is not None:
                query_r2.append(r2)
                all_r2.append(r2)
        per_query[query_name] = {
            "support": int(valid.sum()),
            "macro_normalized_mae_by_train_std": (
                float(np.mean(query_normalized_mae)) if query_normalized_mae else None
            ),
            "macro_r2_over_nonconstant_parameters": (
                float(np.mean(query_r2)) if query_r2 else None
            ),
            "parameters": parameters,
        }
    return {
        "macro_normalized_mae_by_train_std": (
            float(np.mean(all_normalized_mae)) if all_normalized_mae else None
        ),
        "macro_r2_over_nonconstant_query_parameters": (
            float(np.mean(all_r2)) if all_r2 else None
        ),
        "per_query": per_query,
    }


def _two_cubic_control_polygons(control_parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(control_parameters, dtype=np.float64)
    if values.shape != (CONTROL_SLICE.stop - CONTROL_SLICE.start,):
        raise ValueError(
            f"two-cubic control vector must have shape {(CONTROL_SLICE.stop - CONTROL_SLICE.start,)}"
        )
    knot = values[0:2]
    first = np.stack(
        (np.asarray((0.0, 0.0)), values[2:4], values[4:6], knot), axis=0
    )
    second = np.stack(
        (knot, values[6:8], values[8:10], np.asarray((1.0, 0.0))), axis=0
    )
    return first, second


def endpoint_tangent_vectors(control_parameters: np.ndarray) -> np.ndarray:
    """Return start and end tangent vectors for the connected cubic pair."""

    first, second = _two_cubic_control_polygons(control_parameters)
    return np.stack((first[1] - first[0], second[3] - second[2]), axis=0)


def vector_angle_error_degrees(predicted: np.ndarray, expected: np.ndarray) -> float:
    """Angular difference in [0, 180], penalizing a single zero vector."""

    predicted = np.asarray(predicted, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if predicted.shape != (2,) or expected.shape != (2,):
        raise ValueError("angle inputs must be 2D vectors")
    predicted_norm = float(np.linalg.norm(predicted))
    expected_norm = float(np.linalg.norm(expected))
    if predicted_norm <= _EPSILON and expected_norm <= _EPSILON:
        return 0.0
    if predicted_norm <= _EPSILON or expected_norm <= _EPSILON:
        return 180.0
    cosine = float(
        np.clip(np.dot(predicted, expected) / (predicted_norm * expected_norm), -1.0, 1.0)
    )
    return float(np.degrees(np.arccos(cosine)))


def curve_pair_metrics(
    predicted_parameters: np.ndarray,
    expected_parameters: np.ndarray,
    *,
    samples_per_segment: int = 65,
) -> dict[str, float]:
    """Geometric and drafting-scale errors for one observed curve role."""

    predicted = np.asarray(predicted_parameters, dtype=np.float64)
    expected = np.asarray(expected_parameters, dtype=np.float64)
    expected_shape = (len(CURVE_PARAMETER_NAMES),)
    if predicted.shape != expected_shape or expected.shape != expected_shape:
        raise ValueError(f"curve parameter vectors must have shape {expected_shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("curve parameters must be finite")
    left = sample_two_cubic_formula(
        predicted[CONTROL_SLICE].astype(np.float32), samples_per_segment
    ).astype(np.float64)
    right = sample_two_cubic_formula(
        expected[CONTROL_SLICE].astype(np.float32), samples_per_segment
    ).astype(np.float64)
    pairwise = np.linalg.norm(left[:, None] - right[None, :], axis=-1)
    predicted_to_target = pairwise.min(axis=1)
    target_to_predicted = pairwise.min(axis=0)
    predicted_tangents = endpoint_tangent_vectors(predicted[CONTROL_SLICE])
    expected_tangents = endpoint_tangent_vectors(expected[CONTROL_SLICE])
    start_angle = vector_angle_error_degrees(predicted_tangents[0], expected_tangents[0])
    end_angle = vector_angle_error_degrees(predicted_tangents[1], expected_tangents[1])
    chord_index = METRIC_SLICE.start
    arc_index = METRIC_SLICE.start + 1
    chord_relative_error = abs(predicted[chord_index] - expected[chord_index]) / max(
        abs(expected[chord_index]), _EPSILON
    )
    arc_relative_error = abs(predicted[arc_index] - expected[arc_index]) / max(
        abs(expected[arc_index]), _EPSILON
    )
    return {
        "pointwise_rmse_over_chord": float(
            np.sqrt(np.mean(np.sum(np.square(left - right), axis=1)))
        ),
        "symmetric_chamfer_over_chord": float(
            0.5 * (predicted_to_target.mean() + target_to_predicted.mean())
        ),
        "hausdorff_over_chord": float(
            max(predicted_to_target.max(), target_to_predicted.max())
        ),
        "start_tangent_angle_error_degrees": start_angle,
        "end_tangent_angle_error_degrees": end_angle,
        "endpoint_tangent_angle_error_degrees": 0.5 * (start_angle + end_angle),
        "chord_relative_error": float(chord_relative_error),
        "arc_relative_error": float(arc_relative_error),
    }


def curve_geometry_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    *,
    samples_per_segment: int = 65,
) -> dict[str, Any]:
    """Aggregate sampled-curve, tangent, chord, and arc errors per role."""

    predicted, expected, role_mask = _validate_prediction_contract(
        predicted, expected, role_mask
    )
    metric_names = tuple(
        curve_pair_metrics(predicted[role_mask][0], expected[role_mask][0], samples_per_segment=3)
    ) if role_mask.any() else (
        "pointwise_rmse_over_chord",
        "symmetric_chamfer_over_chord",
        "hausdorff_over_chord",
        "start_tangent_angle_error_degrees",
        "end_tangent_angle_error_degrees",
        "endpoint_tangent_angle_error_degrees",
        "chord_relative_error",
        "arc_relative_error",
    )
    aggregate: dict[str, list[float]] = {name: [] for name in metric_names}
    per_query: dict[str, Any] = {}
    for query_index, query_name in enumerate(CURVE_QUERY_NAMES):
        current: dict[str, list[float]] = {name: [] for name in metric_names}
        for sample_index in np.flatnonzero(role_mask[:, query_index]):
            values = curve_pair_metrics(
                predicted[sample_index, query_index],
                expected[sample_index, query_index],
                samples_per_segment=samples_per_segment,
            )
            for name, value in values.items():
                current[name].append(value)
                aggregate[name].append(value)
        per_query[query_name] = {
            "support": int(role_mask[:, query_index].sum()),
            **{
                name: (float(np.mean(values)) if values else None)
                for name, values in current.items()
            },
        }
    return {
        "observed_curve_count": int(role_mask.sum()),
        **{
            f"mean_{name}": (float(np.mean(values)) if values else None)
            for name, values in aggregate.items()
        },
        "per_query": per_query,
    }


def presence_classification_metrics(
    probabilities: np.ndarray,
    expected_presence: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Presence precision/recall/F1 for the five named curve queries."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(expected_presence, dtype=bool)
    if probabilities.shape != expected.shape or probabilities.ndim != 2:
        raise ValueError("presence probabilities and targets must have the same rank-2 shape")
    if probabilities.shape[1] != len(CURVE_QUERY_NAMES):
        raise ValueError("presence arrays do not match the named curve query contract")
    if not np.isfinite(probabilities).all():
        raise ValueError("presence probabilities contain non-finite values")
    predicted = probabilities >= threshold
    per_query: dict[str, Any] = {}
    f1_values: list[float] = []
    for query_index, query_name in enumerate(CURVE_QUERY_NAMES):
        truth = expected[:, query_index]
        estimate = predicted[:, query_index]
        true_positive = int(np.sum(estimate & truth))
        false_positive = int(np.sum(estimate & ~truth))
        false_negative = int(np.sum(~estimate & truth))
        support = int(truth.sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, _EPSILON)
        per_query[query_name] = {
            "support": support,
            "predicted_present": int(estimate.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1 if support else None,
        }
        if support:
            f1_values.append(f1)
    return {
        "threshold": float(threshold),
        "macro_f1_over_queries_with_positive_support": (
            float(np.mean(f1_values)) if f1_values else None
        ),
        "exact_inventory_rate": float(np.mean(np.all(predicted == expected, axis=1))),
        "per_query": per_query,
    }


def _relative_reduction(model_value: float | None, baseline_value: float | None) -> float | None:
    if model_value is None or baseline_value is None or abs(baseline_value) <= _EPSILON:
        return None
    return float((baseline_value - model_value) / baseline_value)


def evaluate_frozen_curve_predictions(
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    presence_probabilities: np.ndarray,
    train_means: np.ndarray,
    train_standard_deviations: np.ndarray,
    *,
    train_presence_rates: Sequence[float] | None = None,
    samples_per_segment: int = 65,
) -> dict[str, Any]:
    """Evaluate a frozen test archive and its leakage-free train-mean baseline."""

    predicted, expected, role_mask = _validate_prediction_contract(
        predicted, expected, role_mask
    )
    train_means = np.asarray(train_means, dtype=np.float64)
    expected_contract = (len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES))
    if train_means.shape != expected_contract:
        raise ValueError(f"train_means must have shape {expected_contract}")
    baseline = np.broadcast_to(train_means, expected.shape).copy()
    model_parameter = parameter_regression_metrics(
        predicted, expected, role_mask, train_standard_deviations
    )
    baseline_parameter = parameter_regression_metrics(
        baseline, expected, role_mask, train_standard_deviations
    )
    model_geometry = curve_geometry_metrics(
        predicted, expected, role_mask, samples_per_segment=samples_per_segment
    )
    baseline_geometry = curve_geometry_metrics(
        baseline, expected, role_mask, samples_per_segment=samples_per_segment
    )
    model_presence = presence_classification_metrics(
        presence_probabilities, role_mask
    )
    baseline_presence = None
    if train_presence_rates is not None:
        rates = np.asarray(train_presence_rates, dtype=np.float64)
        if rates.shape != (len(CURVE_QUERY_NAMES),):
            raise ValueError(
                f"train_presence_rates must have shape {(len(CURVE_QUERY_NAMES),)}"
            )
        baseline_presence = presence_classification_metrics(
            np.broadcast_to(rates, role_mask.shape), role_mask
        )

    geometry_comparison = {}
    for metric_name in (
        "symmetric_chamfer_over_chord",
        "hausdorff_over_chord",
        "endpoint_tangent_angle_error_degrees",
        "chord_relative_error",
        "arc_relative_error",
    ):
        key = f"mean_{metric_name}"
        geometry_comparison[f"{metric_name}_relative_reduction"] = _relative_reduction(
            model_geometry[key], baseline_geometry[key]
        )
    model_r2 = model_parameter["macro_r2_over_nonconstant_query_parameters"]
    baseline_r2 = baseline_parameter["macro_r2_over_nonconstant_query_parameters"]
    comparison = {
        "parameter_nmae_relative_reduction": _relative_reduction(
            model_parameter["macro_normalized_mae_by_train_std"],
            baseline_parameter["macro_normalized_mae_by_train_std"],
        ),
        "parameter_macro_r2_gain": (
            float(model_r2 - baseline_r2)
            if model_r2 is not None and baseline_r2 is not None
            else None
        ),
        **geometry_comparison,
        "presence_macro_f1_gain": (
            float(
                model_presence["macro_f1_over_queries_with_positive_support"]
                - baseline_presence["macro_f1_over_queries_with_positive_support"]
            )
            if baseline_presence is not None
            and model_presence["macro_f1_over_queries_with_positive_support"] is not None
            and baseline_presence["macro_f1_over_queries_with_positive_support"] is not None
            else None
        ),
    }
    return {
        "schema_version": "frozen-multiview-two-cubic-curve-evaluation-1.0",
        "sample_count": int(predicted.shape[0]),
        "query_names": list(CURVE_QUERY_NAMES),
        "parameter_names": list(CURVE_PARAMETER_NAMES),
        "model": {
            "raw_parameter_regression": model_parameter,
            "sampled_curve_geometry": model_geometry,
            "presence": model_presence,
        },
        "train_mean_baseline": {
            "contract": "query-wise parameter mean and presence rate fitted on training split only",
            "raw_parameter_regression": baseline_parameter,
            "sampled_curve_geometry": baseline_geometry,
            "presence": baseline_presence,
        },
        "model_gain_over_train_mean": comparison,
    }

