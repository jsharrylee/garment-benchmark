from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.adapters.reweaver import summarize_output, validate_input_directory
from benchmark.evaluation.binding import compare_reweaver_outputs
from benchmark.scripts.run_reweaver import resolve_input_files


class ReWeaverAdapterTests(unittest.TestCase):
    def test_resolves_official_gcd_ts_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = root / "sample" / "render_output" / "rgb"
            rgb.mkdir(parents=True)
            for index in range(1, 5):
                Image.new("RGB", (518, 518), (index * 40, 0, 0)).save(rgb / f"view_{index:03}.png")
            files = resolve_input_files(root, "sample", "gcd-ts-tileable")
            self.assertEqual([path.name for path in files], [f"view_{index:03}.png" for index in range(1, 5)])

            Image.new("RGB", (256, 256)).save(rgb / "view_004.png")
            with self.assertRaisesRegex(ValueError, "expected 518x518"):
                resolve_input_files(root, "sample", "gcd-ts-tileable")

    def test_requires_four_distinct_518_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, camera in enumerate(("CAM000", "CAM001", "CAM002", "CAM003")):
                Image.new("RGB", (518, 518), (index * 50, 0, 0)).save(root / f"{camera}.png")
            result = validate_input_directory(root)
            self.assertTrue(result["valid"])
            (root / "CAM003.png").unlink()
            self.assertEqual(validate_input_directory(root)["failure"], "INPUT_ADAPTER")

    def test_validates_official_object_array_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "prediction.npz"
            np.savez_compressed(
                output,
                flatten_pred=np.array(
                    {
                        0: {"edge_points": np.ones((2, 2, 2))},
                        1: {"edge_points": np.ones((2, 2, 2))},
                    },
                    dtype=object,
                ),
                patch_curve_connectivity=np.ones((2, 3), dtype=bool),
                curve_points=np.ones((3, 4, 3)),
                curve_valid_prob=np.ones(3),
                patch_points=np.ones((2, 4, 3)),
                patch_valid_prob=np.ones(2),
                patch_points_scaled=np.array([[np.ones((5, 3)), np.ones((6, 3))]], dtype=object),
            )
            summary = summarize_output(output)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["panel_count"], 2)

    def test_binding_comparison_detects_geometry_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = {
                "flatten_pred": np.array({0: {"edge_points": np.ones((1, 2, 2))}}, dtype=object),
                "patch_curve_connectivity": np.ones((1, 1), dtype=bool),
                "curve_valid_prob": np.ones(1),
                "patch_valid_prob": np.ones(1),
                "patch_points_scaled": np.array([np.ones((2, 3))], dtype=object),
            }
            first = root / "first.npz"
            second = root / "second.npz"
            np.savez_compressed(first, curve_points=np.zeros((1, 2, 3)), patch_points=np.zeros((1, 2, 3)), **common)
            np.savez_compressed(second, curve_points=np.ones((1, 2, 3)), patch_points=np.ones((1, 2, 3)), **common)
            self.assertTrue(compare_reweaver_outputs(first, second)["valid"])


if __name__ == "__main__":
    unittest.main()
