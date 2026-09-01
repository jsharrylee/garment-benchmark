import tempfile
import unittest
from pathlib import Path

from PIL import Image

from benchmark.adapters.synbody import CAMERA_SUFFIXES, discover_bundles, validate_bundle


class SynBodyAdapterTests(unittest.TestCase):
    def test_discovers_only_four_camera_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, suffix in enumerate(CAMERA_SUFFIXES):
                rgb = root / "scene" / f"seq{suffix}" / "rgb"
                rgb.mkdir(parents=True)
                Image.new("RGB", (8, 8), (index * 50, 0, 0)).save(rgb / "0001.jpeg")
                if index != 3:
                    Image.new("RGB", (8, 8), (index, 1, 0)).save(rgb / "0002.jpeg")
            bundles = discover_bundles(root)
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0].frame, "0001.jpeg")
            self.assertTrue(validate_bundle(bundles[0])["valid"])


if __name__ == "__main__":
    unittest.main()
