from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.multiview_element_geometry import (
    GEOMETRY_TARGET_NAMES,
    PRESENCE_TARGET_NAMES,
    MaskedTargetStandardizer,
    MultiviewGeometryExample,
    build_multiview_geometry_model,
    geometry_targets,
)
from benchmark.drafting_semantics.schema import (
    DraftingSemanticRecord,
    EdgeAnnotation,
    PanelAnnotation,
)


class MultiviewElementGeometryTests(unittest.TestCase):
    def test_geometry_target_measures_two_armhole_primitives_as_one_path(self) -> None:
        def edge(identifier: str, index: int, endpoints: tuple[int, int], role: str) -> EdgeAnnotation:
            vertices = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
            return EdgeAnnotation(
                id=identifier,
                index=index,
                endpoints=endpoints,
                start_cm=vertices[endpoints[0]],
                end_cm=vertices[endpoints[1]],
                curvature_type="cubic" if role == "armhole" else "line",
                role=role,
                stitched=False,
                self_stitched=False,
                length_cm=2.0,
                evidence="derived_topology",
                confidence=1.0,
            )

        record = DraftingSemanticRecord(
            sample_id="two-piece-armhole",
            split="test",
            panels=(PanelAnnotation(
                id="front",
                role="front_bodice",
                vertices_cm=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
                edges=(
                    edge("e0", 0, (0, 1), "armhole"),
                    edge("e1", 1, (1, 2), "armhole"),
                    edge("e2", 2, (2, 0), "side_seam"),
                ),
            ),),
            darts=(), measurements={}, construction_steps=(), body_condition_cm={},
            program={}, provenance={},
        )
        canonical = {"panels": [{
            "id": "front",
            "edges": [
                {"id": "e0", "points": [[0, 0], [1, 0], [1, 1]]},
                {"id": "e1", "points": [[1, 1], [2, 1], [2, 2]]},
                {"id": "e2", "points": [[2, 2], [0, 0]]},
            ],
        }]}

        values, mask, presence = geometry_targets(record, canonical)
        length_index = GEOMETRY_TARGET_NAMES.index("path:armhole:mean_length")
        chord_index = GEOMETRY_TARGET_NAMES.index("path:armhole:mean_chord")
        curve_index = GEOMETRY_TARGET_NAMES.index("path:armhole:mean_primitive_curvedness")
        presence_index = PRESENCE_TARGET_NAMES.index("path:armhole")
        self.assertTrue(mask[length_index])
        self.assertAlmostEqual(values[length_index], 2.0)
        self.assertAlmostEqual(values[chord_index], np.sqrt(2.0))
        self.assertAlmostEqual(values[curve_index], 1.0 - np.sqrt(2.0) / 2.0)
        self.assertEqual(presence[presence_index], 1.0)

    def test_masked_standardizer_ignores_absent_zero_values(self) -> None:
        def example(value: float, present: bool) -> MultiviewGeometryExample:
            target = np.zeros(len(GEOMETRY_TARGET_NAMES), dtype=np.float32)
            mask = np.zeros(len(GEOMETRY_TARGET_NAMES), dtype=bool)
            target[0] = value
            mask[0] = present
            return MultiviewGeometryExample(
                sample_id=str(value), split="train", category_target=0,
                view_features=np.zeros((4, 8), dtype=np.float32),
                geometry_target=target, geometry_mask=mask,
                presence_target=np.zeros(len(PRESENCE_TARGET_NAMES), dtype=np.float32),
                view_paths=("a", "b", "c", "d"), pattern_path="pattern.png",
            )

        standardizer = MaskedTargetStandardizer.fit((example(2.0, True), example(4.0, True), example(0.0, False)))
        self.assertAlmostEqual(standardizer.means[0], 3.0)
        self.assertAlmostEqual(standardizer.standard_deviations[0], 1.0)

    def test_model_returns_geometry_presence_and_headwise_attention(self) -> None:
        import torch

        config = {
            "view_feature_dim": 8, "width": 24, "heads": 4, "layers": 2,
            "feedforward_multiplier": 2, "dropout": 0.0, "decoder_layers": 1,
        }
        model = build_multiview_geometry_model(config)
        output = model(torch.zeros(2, 4, 8), capture_attention=True)
        self.assertEqual(tuple(output["geometry_prediction"].shape), (2, len(GEOMETRY_TARGET_NAMES)))
        self.assertEqual(tuple(output["presence_logits"].shape), (2, len(PRESENCE_TARGET_NAMES)))
        self.assertEqual(len(output["attention"]), 2)
        self.assertEqual(tuple(output["attention"][-1].shape), (2, 4, 5, 5))
        self.assertEqual(len(output["role_attention"]), 1)
        self.assertEqual(tuple(output["role_attention"][-1].shape), (2, 4, len(PRESENCE_TARGET_NAMES), 4))


if __name__ == "__main__":
    unittest.main()
