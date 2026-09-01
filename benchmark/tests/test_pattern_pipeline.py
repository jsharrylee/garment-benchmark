import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.pattern_pipeline.evaluation import compare_orthogonal_masks, silhouette_iou
from benchmark.pattern_pipeline.garment_particles_convert import convert_garment_particles_npz, sample_generated_edge
from benchmark.pattern_pipeline.export import export_bundle
from benchmark.pattern_pipeline.repair import snap_boundary_junctions
from benchmark.pattern_pipeline.refinement import candidate_rank, select_generated_candidate
from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument, Placement, Stitch, StitchSide
from benchmark.pattern_pipeline.sewing_mesh import build_sewing_mesh_plan
from benchmark.pattern_pipeline.validation import validate_pattern


def rectangle(panel_id: str, *, gap: float = 0.0) -> Panel:
    return Panel(
        panel_id,
        (
            Edge(f"{panel_id}.e0", ((0.0, 0.0), (2.0, 0.0))),
            Edge(f"{panel_id}.e1", ((2.0 + gap, 0.0), (2.0, 1.0))),
            Edge(f"{panel_id}.e2", ((2.0, 1.0), (0.0, 1.0))),
            Edge(f"{panel_id}.e3", ((0.0, 1.0), (0.0, 0.0))),
        ),
    )


class PatternPipelineTests(unittest.TestCase):
    def test_variable_topology_round_trip_has_no_template_fields(self):
        document = PatternDocument("generated", "test_generator", (rectangle("a"), rectangle("b")), ())
        restored = PatternDocument.from_dict(document.to_dict())
        self.assertEqual([len(panel.edges) for panel in restored.panels], [4, 4])
        encoded = json.dumps(restored.to_dict()).lower()
        self.assertNotIn("template_id", encoded)
        self.assertNotIn("nearest_pattern", encoded)

    def test_validation_and_single_hypothesis_repair(self):
        document = PatternDocument("gapped", "generator", (rectangle("a", gap=0.05),), ())
        self.assertFalse(validate_pattern(document, closure_ratio_limit=0.01).accepted)
        repaired, receipt = snap_boundary_junctions(document)
        self.assertTrue(receipt.accepted)
        self.assertTrue(validate_pattern(repaired).accepted)

        too_large, rejected = snap_boundary_junctions(PatternDocument("large", "generator", (rectangle("a", gap=0.5),), ()))
        self.assertFalse(rejected.accepted)
        self.assertFalse(validate_pattern(too_large).accepted)

    def test_invalid_stitch_reference_is_rejected(self):
        stitch = Stitch("s", StitchSide("a", "missing"), StitchSide("a", "a.e0"))
        report = validate_pattern(PatternDocument("bad", "generator", (rectangle("a"),), (stitch,)))
        self.assertFalse(report.accepted)
        self.assertIn("INVALID_STITCH_REFERENCE", {issue.code for issue in report.issues})

    def test_exports_are_real_structured_files(self):
        document = PatternDocument("ok", "generator", (rectangle("a"),), ())
        with tempfile.TemporaryDirectory() as directory:
            paths = export_bundle(document, Path(directory))
            self.assertTrue(all(path.stat().st_size > 20 for path in paths.values()))
            self.assertIn("POLYLINE", paths["dxf_outline"].read_text())
            self.assertEqual(PatternDocument.read_json(paths["canonical_json"]).pattern_id, "ok")

    def test_four_view_iou_requires_all_views(self):
        mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        self.assertEqual(silhouette_iou(mask, mask), 1.0)
        result = compare_orthogonal_masks({view: mask for view in ("front", "back", "left", "right")}, {view: mask for view in ("front", "back", "left", "right")})
        self.assertEqual(result["mean_iou"], 1.0)
        self.assertFalse(compare_orthogonal_masks({"front": mask}, {"front": mask})["accepted"])

    def test_garment_particles_edge_sampling_preserves_endpoints(self):
        vector = np.array([2.0, 1.0, 0.5, 0.2, 0.0, 0.0, 1.0])
        points = sample_generated_edge(np.array([3.0, 4.0]), vector, samples=8)
        np.testing.assert_allclose(points[0], [3.0, 4.0])
        np.testing.assert_allclose(points[-1], [5.0, 5.0])

    def test_garment_particles_conversion_is_variable_topology(self):
        edges = np.zeros((1, 3, 7), dtype=float)
        edges[0, :, :2] = [[2.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prediction.npz"
            np.savez_compressed(
                source,
                edges=edges,
                edge_valid_mask=np.ones((1, 3), dtype=bool),
                panel_translations=np.zeros((1, 3)),
                panel_rotations=np.zeros((1, 3)),
                stitch_pairs=np.empty((0, 4), dtype=np.int16),
            )
            document = convert_garment_particles_npz(source)
        self.assertEqual(document.generator, "Garment Particles")
        self.assertEqual([len(panel.edges) for panel in document.panels], [3])
        self.assertTrue(validate_pattern(document).accepted)

    def test_refiner_prefers_structurally_valid_generation_without_templates(self):
        invalid = {
            "candidate_id": "seed_a",
            "validation": {"accepted": False, "metrics": {"error_count": 1, "max_closure_gap_cm": 0.1, "mean_seam_length_mismatch": 0.0}},
            "particle_silhouette_proxy": {"mean_iou": 0.9},
        }
        valid = {
            "candidate_id": "seed_b",
            "validation": {"accepted": True, "metrics": {"error_count": 0, "max_closure_gap_cm": 0.0, "mean_seam_length_mismatch": 0.1}},
            "particle_silhouette_proxy": {"mean_iou": 0.2},
        }
        self.assertLess(candidate_rank(valid), candidate_rank(invalid))
        selected = select_generated_candidate([invalid, valid])
        self.assertEqual(selected["selected_candidate_id"], "seed_b")
        self.assertFalse(selected["template_retrieval"])
        self.assertFalse(selected["nearest_pattern_selection"])

    def test_sewing_mesh_plan_has_panel_faces_and_explicit_springs(self):
        front, back = rectangle("front"), rectangle("back")
        stitch = Stitch("side", StitchSide("front", "front.e1"), StitchSide("back", "back.e3", reversed=True))
        plan = build_sewing_mesh_plan(PatternDocument("shirt", "generator", (front, back), (stitch,)))
        self.assertEqual(len(plan.panel_loops), 2)
        self.assertGreaterEqual(len(plan.vertices), 8)
        self.assertGreater(len(plan.sewing_edges), 1)
        self.assertTrue(all(a != b for a, b in plan.sewing_edges))

    def test_dense_panel_meshing_adds_interior_vertices_without_changing_boundary(self):
        panel = rectangle("front")
        document = PatternDocument("dense", "retrieved", (panel,), (), annotations={"panel_mesh_spacing_cm": 0.25})
        plan = build_sewing_mesh_plan(document)
        self.assertGreater(len(plan.panel_faces), 2)
        self.assertGreater(len(plan.vertices), len(plan.panel_loops[0]))
        self.assertEqual(plan.edge_vertices[("front", "front.e0")][0], 0)

    def test_sewing_mesh_plan_preserves_semantic_attachment_pins(self):
        front = rectangle("front")
        document = PatternDocument(
            "shirt",
            "retrieved",
            (front,),
            (),
            annotations={"edge_labels": {"front/front.e1": "lower_interface"}, "pin_semantics": ["lower_interface"]},
        )
        plan = build_sewing_mesh_plan(document)
        self.assertGreater(len(plan.pinned_vertices), 1)
        midpoint = PatternDocument(
            "shirt_midpoint",
            "retrieved",
            (front,),
            (),
            annotations={**document.annotations, "pin_strategy": "edge_midpoints"},
        )
        self.assertEqual(len(build_sewing_mesh_plan(midpoint).pinned_vertices), 1)

    def test_sewing_mesh_plan_preserves_retrieved_anchor_placement_when_requested(self):
        panel = rectangle("front")
        panel = Panel(
            panel.id,
            panel.edges,
            placement=Placement((100.0, 200.0, 300.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "test"),
        )
        document = PatternDocument(
            "anchor",
            "retrieved",
            (panel,),
            (),
            annotations={"preserve_absolute_placement": True},
        )
        plan = build_sewing_mesh_plan(document)
        # Schema XYZ maps to Blender XZY and centimetres map to metres.
        np.testing.assert_allclose(plan.vertices[0], (1.0, 3.0, 2.0))
        self.assertGreater(float(np.linalg.norm(np.mean(np.asarray(plan.vertices), axis=0))), 1.0)


if __name__ == "__main__":
    unittest.main()
