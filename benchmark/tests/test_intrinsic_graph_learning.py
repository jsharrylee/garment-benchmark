import unittest

import numpy as np
import torch

from benchmark.gcdv2_exact.intrinsic_graph_learning import build_corner_model, intrinsic_contour_features, intrinsic_segment_features


class IntrinsicGraphLearningTest(unittest.TestCase):
    def test_contour_features_ignore_translation_rotation_and_scale(self):
        angle = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        points = np.column_stack((np.cos(angle) * 2, np.sin(angle)))
        theta = 0.71
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transformed = points @ rotation.T * 4.3 + np.asarray([21.0, -7.0])
        np.testing.assert_allclose(intrinsic_contour_features(points), intrinsic_contour_features(transformed), atol=4e-5)

    def test_segment_targets_are_chord_frame_invariant(self):
        t = np.linspace(0, 1, 32)
        points = np.column_stack((t, 0.3 * np.sin(np.pi * t)))
        _, first = intrinsic_segment_features(points)
        transformed = points * 8.0 + np.asarray([4.0, -9.0])
        _, second = intrinsic_segment_features(transformed)
        np.testing.assert_allclose(first, second, atol=1e-5)

    def test_corner_model_shape(self):
        model = build_corner_model(width=32, heads=4, layers=1)
        output = model(torch.randn(2, 256, 25))
        self.assertEqual(output["corner_logits"].shape, (2, 256))
        self.assertEqual(output["count_logits"].shape, (2, 37))


if __name__ == "__main__":
    unittest.main()
