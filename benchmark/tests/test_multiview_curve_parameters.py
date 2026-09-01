from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.multiview_curve_parameters import (
    CONTROL_SLICE,
    CURVE_PARAMETER_NAMES,
    CURVE_QUERY_NAMES,
    CURVE_TRUTH_DENSE_APPROXIMATION,
    CURVE_TRUTH_GENERATOR_FORMULA,
    CurveParameterStandardizer,
    MultiviewCurveExample,
    build_spatial_curve_model,
    curve_formula_loss,
    curve_formula_targets,
    curve_reconstruction_metrics,
    fit_two_cubic_formula,
    sample_two_cubic_formula,
    spatial_attention_maps,
)
from benchmark.drafting_semantics.schema import (
    DraftingSemanticRecord,
    EdgeAnnotation,
    Landmark,
    PanelAnnotation,
)


def _cubic(points: np.ndarray, count: int = 33) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, count)[:, None]
    inverse = 1.0 - parameter
    return (
        inverse**3 * points[0]
        + 3.0 * inverse**2 * parameter * points[1]
        + 3.0 * inverse * parameter**2 * points[2]
        + parameter**3 * points[3]
    ).astype(np.float32)


class MultiviewCurveParameterTests(unittest.TestCase):
    def test_two_cubic_formula_reconstructs_known_curve(self) -> None:
        first = np.asarray(((0, 0), (0.15, 0.2), (0.35, 0.25), (0.5, 0.3)), dtype=np.float32)
        second = np.asarray(((0.5, 0.3), (0.65, 0.25), (0.85, 0.1), (1, 0)), dtype=np.float32)
        dense = np.concatenate((_cubic(first), _cubic(second)[1:]))
        controls, fit_error = fit_two_cubic_formula(dense)
        reconstructed = sample_two_cubic_formula(controls, 33)
        self.assertLess(fit_error, 0.012)
        # The fitter parameterizes by observed arc length while ``dense`` was
        # sampled in the source Bezier t-domain, so pointwise indices are not
        # exact even when the geometric curves agree closely.
        self.assertLess(float(np.sqrt(np.mean(np.sum((reconstructed - dense) ** 2, axis=1)))), 0.025)

    def test_targets_merge_two_armhole_primitives_and_mark_dense_provenance(self) -> None:
        vertices = ((0.0, 0.0), (0.0, 3.0), (2.0, 5.0), (4.0, 4.0), (5.0, 2.0), (4.0, 0.0))

        def edge(identifier: str, index: int, endpoints: tuple[int, int], role: str, curve: str = "line"):
            return EdgeAnnotation(
                id=identifier,
                index=index,
                endpoints=endpoints,
                start_cm=vertices[endpoints[0]],
                end_cm=vertices[endpoints[1]],
                curvature_type=curve,
                role=role,
                stitched=False,
                self_stitched=False,
                length_cm=float(np.linalg.norm(np.subtract(vertices[endpoints[1]], vertices[endpoints[0]]))),
                evidence="derived_topology",
                confidence=1.0,
            )

        panel = PanelAnnotation(
            id="front",
            role="front_bodice",
            vertices_cm=vertices,
            edges=(
                edge("e0", 0, (0, 1), "center_front"),
                edge("e1", 1, (1, 2), "neckline", "cubic"),
                edge("e2", 2, (2, 3), "shoulder"),
                edge("e3", 3, (3, 4), "armhole", "cubic"),
                edge("e4", 4, (4, 5), "armhole", "cubic"),
                edge("e5", 5, (5, 0), "side_seam"),
            ),
            landmarks=(
                Landmark("FNP", "front", vertices[1], "derived_topology", 1.0, 1),
                Landmark("SNP", "front", vertices[2], "derived_topology", 1.0, 2),
                Landmark("SP", "front", vertices[3], "derived_topology", 1.0, 3),
            ),
        )
        record = DraftingSemanticRecord(
            sample_id="front",
            split="test",
            panels=(panel,),
            darts=(),
            measurements={},
            construction_steps=(),
            body_condition_cm={},
            program={},
            provenance={},
        )
        canonical = {
            "panels": [
                {
                    "id": "front",
                    "edges": [
                        {"id": "e0", "points": [[0, 0], [0, 3]]},
                        {"id": "e1", "points": [[0, 3], [0.5, 4.3], [2, 5]]},
                        {"id": "e2", "points": [[2, 5], [4, 4]]},
                        {"id": "e3", "points": [[4, 4], [4.7, 3.4], [5, 2]]},
                        {"id": "e4", "points": [[5, 2], [5.0, 0.8], [4, 0]]},
                        {"id": "e5", "points": [[4, 0], [0, 0]]},
                    ],
                }
            ]
        }
        target = curve_formula_targets(record, canonical)
        neckline = CURVE_QUERY_NAMES.index("front_neckline")
        armhole = CURVE_QUERY_NAMES.index("front_armhole")
        self.assertTrue(target.role_mask[neckline])
        self.assertTrue(target.role_mask[armhole])
        self.assertEqual(target.observation_count[armhole], 1)
        self.assertEqual(target.provenance[armhole], CURVE_TRUTH_DENSE_APPROXIMATION)
        self.assertGreater(target.values[armhole, 5], target.values[armhole, 4])

        captured = np.linspace(0.0, 1.0, len(CURVE_PARAMETER_NAMES), dtype=np.float32)
        overridden = curve_formula_targets(
            record, canonical, generator_formula_truth={"front_armhole": captured}
        )
        np.testing.assert_allclose(overridden.values[armhole], captured)
        self.assertEqual(overridden.provenance[armhole], CURVE_TRUTH_GENERATOR_FORMULA)
        self.assertEqual(overridden.fit_rmse_over_chord[armhole], 0.0)

    def test_spatial_role_queries_return_headwise_patch_attention(self) -> None:
        import torch

        config = {
            "pyramid_levels": ["0", "1"],
            "pyramid_grid_sizes": [2, 1],
            "spatial_feature_dim": 8,
            "width": 24,
            "heads": 4,
            "memory_layers": 1,
            "decoder_layers": 2,
            "feedforward_multiplier": 2,
            "dropout": 0.0,
        }
        model = build_spatial_curve_model(config)
        features = torch.randn(2, 4, 5, 8)
        view_valid = torch.tensor([[True, True, True, True], [True, False, True, True]])
        output = model(
            spatial_features=features, view_valid=view_valid, capture_attention=True
        )
        self.assertEqual(
            tuple(output["curve_prediction"].shape),
            (2, len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES)),
        )
        self.assertEqual(tuple(output["presence_logits"].shape), (2, len(CURVE_QUERY_NAMES)))
        self.assertEqual(len(output["spatial_attention"]), 2)
        self.assertEqual(
            tuple(output["spatial_attention"][-1].shape),
            (2, 4, len(CURVE_QUERY_NAMES), 4, 5),
        )
        maps = spatial_attention_maps(output["spatial_attention"][-1], (2, 1))
        self.assertEqual(tuple(maps[0].shape), (2, 4, len(CURVE_QUERY_NAMES), 4, 2, 2))
        self.assertEqual(tuple(maps[1].shape), (2, 4, len(CURVE_QUERY_NAMES), 4, 1, 1))

    def test_masked_loss_and_reconstruction_metric(self) -> None:
        import torch

        values = np.zeros((len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES)), dtype=np.float32)
        values[:, CONTROL_SLICE] = np.asarray(
            (0.5, 0.2, 0.15, 0.1, 0.35, 0.2, 0.65, 0.2, 0.85, 0.1), dtype=np.float32
        )
        first_mask = np.asarray((True, False, True, False, True))
        second_mask = np.asarray((True, True, True, True, False))

        def example(identifier: str, mask: np.ndarray) -> MultiviewCurveExample:
            return MultiviewCurveExample(
                sample_id=identifier,
                split="train",
                view_paths=("a", "b", "c", "d"),
                pattern_path="pattern.json",
                curve_target=values.copy(),
                role_mask=mask,
                fit_rmse_over_chord=np.zeros(len(CURVE_QUERY_NAMES), dtype=np.float32),
                target_provenance=tuple(
                    CURVE_TRUTH_DENSE_APPROXIMATION if value else "ABSENT" for value in mask
                ),
            )

        standardizer = CurveParameterStandardizer.fit(
            (example("a", first_mask), example("b", second_mask))
        )
        encoded = standardizer.encode(np.stack((values, values)))
        prediction = torch.tensor(encoded, requires_grad=True)
        output = {
            "curve_prediction": prediction,
            "presence_logits": torch.zeros((2, len(CURVE_QUERY_NAMES)), requires_grad=True),
        }
        losses = curve_formula_loss(
            output, torch.tensor(encoded), torch.tensor(np.stack((first_mask, second_mask))), standardizer
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        metrics = curve_reconstruction_metrics(
            np.stack((values, values)),
            np.stack((values, values)),
            np.stack((first_mask, second_mask)),
        )
        self.assertAlmostEqual(metrics["macro_pointwise_rmse_over_chord"], 0.0)


if __name__ == "__main__":
    unittest.main()
