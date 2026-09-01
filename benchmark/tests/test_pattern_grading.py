from __future__ import annotations

import unittest

from benchmark.pattern_pipeline.grading import grade_pattern, grading_scales
from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument, Placement, Stitch, StitchSide


class PatternGradingTests(unittest.TestCase):
    def test_pants_use_hip_waist_and_leg_ratios(self):
        width, length, depth = grading_scales(
            "pants",
            {"hips": 100.0, "waist": 80.0, "leg_length": 80.0},
            {"hips": 110.0, "waist": 88.0, "leg_length": 88.0},
        )
        self.assertAlmostEqual(width, 1.1)
        self.assertAlmostEqual(length, 1.1)
        self.assertGreater(depth, 1.0)

    def test_grading_preserves_topology_and_stitches(self):
        edge = Edge("front.edge_0", ((0.0, 0.0), (10.0, 20.0)))
        panel = Panel("front", (edge,), Placement((2.0, 50.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        stitch = Stitch("s0", StitchSide("front", "front.edge_0"), StitchSide("front", "front.edge_0"))
        source = PatternDocument("sample", "test", (panel,), (stitch,))
        graded = grade_pattern(
            source,
            category="top",
            source_measurements={"bust": 100.0, "height": 170.0},
            target_measurements={"bust": 120.0, "height": 187.0},
        )
        self.assertEqual(len(graded.panels), 1)
        self.assertEqual(graded.stitches, source.stitches)
        self.assertEqual(graded.panels[0].edges[0].points[-1], (12.0, 22.0))
        self.assertEqual(graded.annotations["refinement_status"], "body_measurement_graded_anchor")


if __name__ == "__main__":
    unittest.main()
