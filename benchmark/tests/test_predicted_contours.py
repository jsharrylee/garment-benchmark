import unittest

import numpy as np

from benchmark.gcdv2_exact.predicted_contours import resample_closed_contour, symmetric_chamfer


class PredictedContourTest(unittest.TestCase):
    def test_resampling_is_closed_without_duplicate_endpoint(self):
        square = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        result = resample_closed_contour(square, 16)
        self.assertEqual(result.shape, (16, 2))
        self.assertFalse(np.allclose(result[0], result[-1]))
        self.assertAlmostEqual(symmetric_chamfer(result, np.roll(result[::-1], 5, axis=0)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
