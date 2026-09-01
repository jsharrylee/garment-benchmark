from __future__ import annotations

import math
import unittest

from benchmark.gcdv2_exact.geometry import _curve_payload, sample_curve


class GCDv2ExactGeometryTests(unittest.TestCase):
    def test_cubic_controls_are_decoded_from_source_relative_frame(self) -> None:
        vertices = ((0.0, 0.0), (10.0, 0.0))
        curve = _curve_payload(
            vertices,
            {"endpoints": [0, 1], "curvature": {"type": "cubic", "params": [[0.25, 0.2], [0.75, 0.2]]}},
        )
        self.assertEqual(curve["type"], "cubic_bezier")
        self.assertEqual(curve["controls_cm"], [[2.5, 2.0], [7.5, 2.0]])
        points = sample_curve(vertices[0], vertices[1], curve, samples=7)
        self.assertEqual(points[0], vertices[0])
        self.assertEqual(points[-1], vertices[1])

    def test_circle_source_is_rendered_as_arc_not_chord(self) -> None:
        vertices = ((-1.0, 0.0), (1.0, 0.0))
        curve = _curve_payload(
            vertices,
            {"endpoints": [0, 1], "curvature": {"type": "circle", "params": [1.0, 0, 1]}},
        )
        points = sample_curve(vertices[0], vertices[1], curve, samples=9)
        self.assertEqual(curve["type"], "circular_arc")
        self.assertGreater(max(abs(point[1]) for point in points), 0.9)
        self.assertLess(math.dist(points[0], vertices[0]), 1e-8)
        self.assertLess(math.dist(points[-1], vertices[1]), 1e-8)


if __name__ == "__main__":
    unittest.main()
