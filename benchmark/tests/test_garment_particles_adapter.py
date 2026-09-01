from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.evaluation.binding import compare_garment_particles_outputs

from benchmark.adapters.garment_particles import (
    N_CURVES,
    N_EDGE_PARAMS,
    N_PANELS,
    N_POINTS,
    parse_predictions,
    stitch_pairs,
    summarize_output,
)


class GarmentParticlesAdapterTests(unittest.TestCase):
    def test_parse_and_validate_structured_output(self):
        particles = np.zeros((N_POINTS, 6), dtype=np.float32)
        edges = np.zeros((N_PANELS, N_CURVES, N_EDGE_PARAMS), dtype=np.float32)
        edges[0, 1:5, 7] = 1
        edges[0, 1:3, 11] = 1
        parsed = parse_predictions(particles, edges)
        pairs = stitch_pairs(parsed["stitch_flags"], parsed["stitch_tags"], parsed["edge_valid_mask"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.npz"
            np.savez_compressed(output, **parsed, stitch_pairs=pairs)
            summary = summarize_output(output)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["panel_count"], 1)
            self.assertEqual(summary["edge_count"], 4)

    def test_rejects_nonfinite_output(self):
        particles = np.zeros((N_POINTS, 6), dtype=np.float32)
        edges = np.zeros((N_PANELS, N_CURVES, N_EDGE_PARAMS), dtype=np.float32)
        edges[0, 1, 7] = 1
        parsed = parse_predictions(particles, edges)
        parsed["panel_rotations"][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.npz"
            np.savez_compressed(output, **parsed, stitch_pairs=np.empty((0, 4), dtype=np.int16))
            self.assertEqual(summarize_output(output)["failure"], "NONFINITE_OUTPUT")

    def test_rejects_stitch_reference_to_invalid_edge(self):
        particles = np.zeros((N_POINTS, 6), dtype=np.float32)
        edges = np.zeros((N_PANELS, N_CURVES, N_EDGE_PARAMS), dtype=np.float32)
        edges[0, 1:5, 7] = 1
        parsed = parse_predictions(particles, edges)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.npz"
            invalid_pair = np.array([[0, 1, 0, 12]], dtype=np.int16)
            np.savez_compressed(output, **parsed, stitch_pairs=invalid_pair)
            self.assertEqual(summarize_output(output)["failure"], "INVALID_STITCH_REFERENCE")

    def test_binding_comparison_detects_change(self):
        particles = np.zeros((N_POINTS, 6), dtype=np.float32)
        edges = np.zeros((N_PANELS, N_CURVES, N_EDGE_PARAMS), dtype=np.float32)
        edges[0, 1:5, 7] = 1
        parsed = parse_predictions(particles, edges)
        pairs = stitch_pairs(parsed["stitch_flags"], parsed["stitch_tags"], parsed["edge_valid_mask"])
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "baseline.npz"
            candidate = Path(temp) / "candidate.npz"
            np.savez_compressed(baseline, **parsed, stitch_pairs=pairs)
            changed = dict(parsed)
            changed["particles"] = parsed["particles"].copy()
            changed["particles"][0, 2] += 1.0
            np.savez_compressed(candidate, **changed, stitch_pairs=pairs)
            result = compare_garment_particles_outputs(baseline, candidate)
            self.assertTrue(result["valid"])
            self.assertGreater(result["particle_rms"], 0)

    def test_binding_comparison_rejects_identical_output(self):
        particles = np.zeros((N_POINTS, 6), dtype=np.float32)
        edges = np.zeros((N_PANELS, N_CURVES, N_EDGE_PARAMS), dtype=np.float32)
        edges[0, 1:5, 7] = 1
        parsed = parse_predictions(particles, edges)
        pairs = stitch_pairs(parsed["stitch_flags"], parsed["stitch_tags"], parsed["edge_valid_mask"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "same.npz"
            np.savez_compressed(output, **parsed, stitch_pairs=pairs)
            result = compare_garment_particles_outputs(output, output)
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure"], "OUTPUT_NOT_BOUND_TO_INPUT")


if __name__ == "__main__":
    unittest.main()
