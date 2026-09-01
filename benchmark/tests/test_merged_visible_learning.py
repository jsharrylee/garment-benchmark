import unittest

import numpy as np
import torch

from benchmark.drafting_semantics.merged_visible_learning import build_merged_semantic_model, decode_landmarks


class MergedVisibleLearningTest(unittest.TestCase):
    def test_model_shapes(self):
        model = build_merged_semantic_model(width=32, heads=4, segment_layers=1, graph_layers=1)
        output = model(torch.randn(2, 10, 32, 8), torch.ones(2, 10, dtype=torch.bool), torch.tensor([0, 1]))
        self.assertEqual(output.shape, (2, 10, 9))

    def test_landmarks_are_edge_junctions_not_regressed_coordinates(self):
        # vertices[i] is the junction between edge i-1 and edge i.
        roles = [4, 1, 2, 3, 6, 7]
        vertices = np.arange(12, dtype=np.float32).reshape(6, 2)
        decoded = decode_landmarks(roles, vertices, 0)
        np.testing.assert_array_equal(decoded["FNP"], vertices[1])
        np.testing.assert_array_equal(decoded["SNP"], vertices[2])
        np.testing.assert_array_equal(decoded["SP"], vertices[3])


if __name__ == "__main__":
    unittest.main()
