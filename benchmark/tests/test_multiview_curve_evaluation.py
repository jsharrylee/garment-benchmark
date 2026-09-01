from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.curve_evaluation import (
    coefficient_of_determination,
    curve_pair_metrics,
    evaluate_frozen_curve_predictions,
    parameter_regression_metrics,
    vector_angle_error_degrees,
)
from benchmark.drafting_semantics.multiview_curve_parameters import (
    CONTROL_SLICE,
    CURVE_PARAMETER_NAMES,
    CURVE_QUERY_NAMES,
)
from benchmark.scripts.evaluate_multiview_curve_parameters import (
    _validate_ablation_alignment,
)


def _curve_parameters() -> np.ndarray:
    values = np.zeros(len(CURVE_PARAMETER_NAMES), dtype=np.float32)
    values[:6] = (0.1, 0.2, 0.8, 0.7, 0.42, 0.55)
    values[CONTROL_SLICE] = (
        0.5,
        0.25,
        0.12,
        0.10,
        0.35,
        0.23,
        0.65,
        0.23,
        0.88,
        0.10,
    )
    return values


class CurveEvaluationTests(unittest.TestCase):
    def test_global_ablation_archive_must_match_frozen_test_order_and_truth(self) -> None:
        curve = _curve_parameters()
        targets = np.stack(
            (np.stack((curve,) * len(CURVE_QUERY_NAMES)),) * 2
        )
        masks = np.ones((2, len(CURVE_QUERY_NAMES)), dtype=bool)

        def archive(ids: tuple[str, ...]) -> dict[str, np.ndarray]:
            return {
                "sample_ids": np.asarray(ids),
                "predicted_curve_parameters": targets.copy(),
                "target_curve_parameters": targets.copy(),
                "target_role_mask": masks.copy(),
                "predicted_presence_probability": masks.astype(np.float32),
            }

        reference = archive(("first", "second"))
        matching = archive(("first", "second"))
        _validate_ablation_alignment(reference, matching)
        reordered = archive(("second", "first"))
        with self.assertRaisesRegex(ValueError, "exact frozen-test order"):
            _validate_ablation_alignment(reference, reordered)
        changed_truth = archive(("first", "second"))
        changed_truth["target_curve_parameters"][0, 0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "curve parameters"):
            _validate_ablation_alignment(reference, changed_truth)

    def test_identical_two_cubic_curve_has_zero_geometric_error(self) -> None:
        curve = _curve_parameters()
        metrics = curve_pair_metrics(curve, curve)
        self.assertAlmostEqual(metrics["symmetric_chamfer_over_chord"], 0.0, places=7)
        self.assertAlmostEqual(metrics["hausdorff_over_chord"], 0.0, places=7)
        self.assertAlmostEqual(metrics["endpoint_tangent_angle_error_degrees"], 0.0, places=7)
        self.assertAlmostEqual(metrics["chord_relative_error"], 0.0, places=7)
        self.assertAlmostEqual(metrics["arc_relative_error"], 0.0, places=7)

    def test_curve_metrics_detect_shape_tangent_and_length_changes(self) -> None:
        truth = _curve_parameters()
        predicted = truth.copy()
        predicted[4] *= 1.2
        predicted[5] *= 0.9
        predicted[CONTROL_SLICE.start + 3] += 0.35
        predicted[CONTROL_SLICE.start + 9] -= 0.25
        metrics = curve_pair_metrics(predicted, truth)
        self.assertGreater(metrics["symmetric_chamfer_over_chord"], 0.01)
        self.assertGreater(metrics["hausdorff_over_chord"], 0.02)
        self.assertGreater(metrics["endpoint_tangent_angle_error_degrees"], 1.0)
        self.assertAlmostEqual(metrics["chord_relative_error"], 0.2, places=5)
        self.assertAlmostEqual(metrics["arc_relative_error"], 0.1, places=5)

    def test_angle_and_r2_helpers_handle_edge_cases(self) -> None:
        self.assertAlmostEqual(
            vector_angle_error_degrees(np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0))),
            90.0,
        )
        self.assertEqual(
            vector_angle_error_degrees(np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0))),
            180.0,
        )
        self.assertIsNone(
            coefficient_of_determination(np.ones(3), np.ones(3))
        )
        self.assertAlmostEqual(
            coefficient_of_determination(
                np.asarray((1.0, 2.0, 3.0)), np.asarray((1.0, 2.0, 3.0))
            ),
            1.0,
        )

    def test_parameter_metrics_and_train_mean_baseline_are_leakage_free(self) -> None:
        base = _curve_parameters()
        expected = np.stack(
            [
                np.stack([base + 0.05 * row for _ in CURVE_QUERY_NAMES])
                for row in range(3)
            ]
        )
        predicted = expected.copy()
        mask = np.ones((3, len(CURVE_QUERY_NAMES)), dtype=bool)
        deviations = np.full(
            (len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES)), 0.5, dtype=np.float32
        )
        metrics = parameter_regression_metrics(predicted, expected, mask, deviations)
        self.assertAlmostEqual(metrics["macro_normalized_mae_by_train_std"], 0.0)
        self.assertAlmostEqual(metrics["macro_r2_over_nonconstant_query_parameters"], 1.0)

        train_means = np.broadcast_to(base, deviations.shape).copy()
        evaluation = evaluate_frozen_curve_predictions(
            predicted,
            expected,
            mask,
            np.full(mask.shape, 0.9, dtype=np.float32),
            train_means,
            deviations,
            train_presence_rates=np.full(len(CURVE_QUERY_NAMES), 0.8),
            samples_per_segment=17,
        )
        self.assertAlmostEqual(
            evaluation["model"]["presence"]["macro_f1_over_queries_with_positive_support"],
            1.0,
        )
        self.assertGreater(
            evaluation["model_gain_over_train_mean"]["parameter_nmae_relative_reduction"],
            0.99,
        )


if __name__ == "__main__":
    unittest.main()
