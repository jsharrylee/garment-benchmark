from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from benchmark.preprocessing.masking import mask_statistics
from benchmark.preprocessing.normalization import contain_square


class PreprocessingTests(unittest.TestCase):
    def test_mask_statistics(self):
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:8, 3:9] = 255
        stats = mask_statistics(mask)
        self.assertEqual(stats["bbox_xyxy"], [3, 2, 9, 8])
        self.assertEqual(stats["foreground_pixels"], 36)

    def test_contain_square_produces_expected_resolution(self):
        image = Image.new("RGB", (100, 80), "red")
        mask = Image.new("L", (100, 80), 0)
        mask.paste(255, (30, 10, 70, 70))
        result, transform = contain_square(image, mask, size=518)
        self.assertEqual(result.size, (518, 518))
        self.assertEqual(transform["mask_bbox_xyxy"], [30, 10, 70, 70])


if __name__ == "__main__":
    unittest.main()
