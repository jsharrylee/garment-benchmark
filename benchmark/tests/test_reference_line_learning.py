import unittest

import torch

from benchmark.drafting_semantics.reference_line_learning import build_reference_line_model


class ReferenceLineModelTest(unittest.TestCase):
    def test_shape(self):
        model = build_reference_line_model(width=32, heads=4, layers=1)
        output = model(torch.randn(3, 12, 17), torch.ones(3, 12, dtype=torch.bool), torch.zeros(3, dtype=torch.long))
        self.assertEqual(output.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
