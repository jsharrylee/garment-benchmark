from __future__ import annotations

import unittest

import numpy as np

from benchmark.gcdv2_exact.pattern_dsl_learning import (
    CURVE_COMMANDS,
    EDGE_FEATURE_DIMENSION,
    EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2,
    EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
    MASK_COMMAND,
    build_pattern_dsl_model,
    invariant_edge_features,
    tensorize_pattern_program,
    validate_edge_feature_schema,
)
from benchmark.gcdv2_exact.pattern_dsl import CurveCommand, compile_formal_graph
from benchmark.tests.test_pattern_dsl import _graph


class PatternDSLLearningTests(unittest.TestCase):
    def test_feature_schema_guard_rejects_v1_checkpoint_on_v2_corpus(self):
        arrays = {"edge_feature_schema": np.asarray(EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2)}
        checkpoint = {"edge_feature_schema": EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1}
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_edge_feature_schema(arrays, checkpoint)
        checkpoint["edge_feature_schema"] = EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2
        self.assertEqual(
            validate_edge_feature_schema(arrays, checkpoint),
            EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2,
        )

    def test_canonical_tensorizer_reads_chord_turn_from_dsl_without_ids(self):
        graph = _graph()
        # Make the old tangent-discontinuity feature observably different from
        # the DSL's chord-turn fact. This models the curved junctions for which
        # the two v1/v2 contracts diverge in the real corpus.
        graph["curves"][1]["start_tangent_deg_y_up"] += 23.0
        program = compile_formal_graph(graph)
        tensors = tensorize_pattern_program(program)
        self.assertEqual(
            tuple(tensors),
            ("edge_features", "edge_commands", "edge_valid", "panel_valid"),
        )
        self.assertEqual(tensors["edge_features"].shape[-1], EDGE_FEATURE_DIMENSION)
        commands = [value for value in program.commands if isinstance(value, CurveCommand)]
        for index, command in enumerate(commands):
            feature = tensors["edge_features"][0, index]
            self.assertAlmostEqual(float(feature[0]), command.length_ratio, places=6)
            self.assertAlmostEqual(float(feature[1]), command.chord_ratio, places=6)
            self.assertAlmostEqual(float(feature[14]), command.turn_cos, places=6)
            self.assertAlmostEqual(float(feature[15]), command.turn_sin, places=6)
        legacy = invariant_edge_features(graph, 1)
        canonical = tensors["edge_features"][0, 1]
        self.assertGreater(float(np.linalg.norm(legacy[14:16] - canonical[14:16])), 0.1)
        self.assertNotEqual(
            EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
            EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2,
        )

    def test_dsl_v2_corpus_equivalence_except_explicit_turn_migration(self):
        """The DSL compiler must not silently change the other 16 features."""

        graphs = [_graph(), _graph(length_scale=2.5)]
        graphs[0]["curves"][1]["start_tangent_deg_y_up"] += 23.0
        graphs[1]["curves"][3]["end_tangent_deg_y_up"] -= 17.0
        retained_slots = np.asarray(
            [index for index in range(EDGE_FEATURE_DIMENSION) if index not in (14, 15)]
        )
        for graph in graphs:
            canonical = tensorize_pattern_program(compile_formal_graph(graph))
            count = len(graph["curves"])
            self.assertEqual(int(canonical["edge_valid"][0].sum()), count)
            expected_commands = [
                {"line": 0, "quadratic_bezier": 1, "cubic_bezier": 2, "circular_arc": 3}[
                    curve["primitive"]
                ]
                for curve in graph["curves"]
            ]
            np.testing.assert_array_equal(
                canonical["edge_commands"][0, :count], expected_commands
            )
            for edge_index in range(count):
                legacy = invariant_edge_features(graph, edge_index)
                migrated = canonical["edge_features"][0, edge_index]
                np.testing.assert_allclose(
                    migrated[retained_slots], legacy[retained_slots], atol=1e-6
                )

    def test_edge_features_are_translation_free_and_finite(self):
        graph = {
            "points": [
                {"xy_cm": [10.0, 20.0]},
                {"xy_cm": [20.0, 20.0]},
                {"xy_cm": [20.0, 30.0]},
            ],
            "curves": [
                {"primitive": "line", "length_cm": 10.0, "chord_direction_deg_y_up": 0.0, "start_tangent_deg_y_up": 0.0, "end_tangent_deg_y_up": 0.0, "parameters": {}},
                {"primitive": "line", "length_cm": 10.0, "chord_direction_deg_y_up": 90.0, "start_tangent_deg_y_up": 90.0, "end_tangent_deg_y_up": 90.0, "parameters": {}},
                {"primitive": "line", "length_cm": 14.142, "chord_direction_deg_y_up": -135.0, "start_tangent_deg_y_up": -135.0, "end_tangent_deg_y_up": -135.0, "parameters": {}},
            ],
        }
        first = invariant_edge_features(graph, 0)
        for point in graph["points"]:
            point["xy_cm"][0] += 500.0; point["xy_cm"][1] -= 300.0
        second = invariant_edge_features(graph, 0)
        self.assertEqual(first.shape, (EDGE_FEATURE_DIMENSION,))
        np.testing.assert_allclose(first, second, atol=1e-6)
        self.assertTrue(np.isfinite(first).all())

    def test_unified_model_emits_all_fact_heads(self):
        import torch

        model = build_pattern_dsl_model(width=32, heads=4, edge_layers=1, garment_layers=1)
        features = torch.zeros((2, 3, 5, EDGE_FEATURE_DIMENSION))
        commands = torch.full((2, 3, 5), MASK_COMMAND, dtype=torch.long)
        edge_valid = torch.zeros((2, 3, 5), dtype=torch.bool); edge_valid[:, :, :3] = True
        panel_valid = torch.ones((2, 3), dtype=torch.bool)
        output = model(features, commands, edge_valid, panel_valid)
        self.assertEqual(output["command_logits"].shape, (2, 3, 5, len(CURVE_COMMANDS)))
        self.assertEqual(output["seam_logits"].shape, (2, 15, 15))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_panel_permutation_equivariance_has_no_panel_position_leakage(self):
        import torch

        torch.manual_seed(17)
        model = build_pattern_dsl_model(width=32, heads=4, edge_layers=1, garment_layers=1).eval()
        batch, panels, edges = 1, 3, 5
        features = torch.randn((batch, panels, edges, EDGE_FEATURE_DIMENSION))
        commands = torch.randint(0, len(CURVE_COMMANDS), (batch, panels, edges))
        panel_valid = torch.tensor([[True, True, False]])
        edge_valid = torch.zeros((batch, panels, edges), dtype=torch.bool)
        edge_valid[0, 0, :4] = True
        edge_valid[0, 1, :3] = True
        commands[~edge_valid] = MASK_COMMAND
        features[~edge_valid] = 0.0

        permutation = torch.tensor([2, 0, 1])
        flat_permutation = torch.cat(
            (permutation[:, None] * edges + torch.arange(edges)[None]).unbind(0)
        )
        with torch.no_grad():
            original = model(features, commands, edge_valid, panel_valid)
            permuted = model(
                features[:, permutation],
                commands[:, permutation],
                edge_valid[:, permutation],
                panel_valid[:, permutation],
            )

        torch.testing.assert_close(permuted["category_logits"], original["category_logits"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            permuted["panel_role_logits"], original["panel_role_logits"][:, permutation], atol=1e-5, rtol=1e-5
        )
        for name in ("edge_role_logits", "command_logits", "edge_hidden"):
            torch.testing.assert_close(permuted[name], original[name][:, permutation], atol=1e-5, rtol=1e-5)
        expected_seams = original["seam_logits"][:, flat_permutation][:, :, flat_permutation]
        torch.testing.assert_close(permuted["seam_logits"], expected_seams, atol=1e-5, rtol=1e-5)

    def test_cyclic_edge_rotation_equivariance_has_no_boundary_start_leakage(self):
        import torch

        torch.manual_seed(23)
        model = build_pattern_dsl_model(width=32, heads=4, edge_layers=1, garment_layers=1).eval()
        batch, panels, edges, valid_count = 1, 1, 7, 5
        features = torch.randn((batch, panels, edges, EDGE_FEATURE_DIMENSION))
        commands = torch.randint(0, len(CURVE_COMMANDS), (batch, panels, edges))
        panel_valid = torch.ones((batch, panels), dtype=torch.bool)
        edge_valid = torch.zeros((batch, panels, edges), dtype=torch.bool)
        edge_valid[..., :valid_count] = True
        commands[~edge_valid] = MASK_COMMAND
        features[~edge_valid] = 0.0

        shift = 2
        rotated_features = features.clone()
        rotated_commands = commands.clone()
        rotated_features[..., :valid_count, :] = torch.roll(
            features[..., :valid_count, :], shifts=shift, dims=2
        )
        rotated_commands[..., :valid_count] = torch.roll(
            commands[..., :valid_count], shifts=shift, dims=2
        )
        with torch.no_grad():
            original = model(features, commands, edge_valid, panel_valid)
            rotated = model(rotated_features, rotated_commands, edge_valid, panel_valid)

        torch.testing.assert_close(rotated["category_logits"], original["category_logits"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rotated["panel_role_logits"], original["panel_role_logits"], atol=1e-5, rtol=1e-5)
        for name in ("edge_role_logits", "command_logits", "edge_hidden"):
            expected = torch.roll(original[name][..., :valid_count, :], shifts=shift, dims=2)
            torch.testing.assert_close(rotated[name][..., :valid_count, :], expected, atol=1e-5, rtol=1e-5)
        valid_rotation = torch.roll(torch.arange(valid_count), shifts=shift)
        expected_seams = original["seam_logits"][:, valid_rotation][:, :, valid_rotation]
        torch.testing.assert_close(
            rotated["seam_logits"][:, :valid_count, :valid_count], expected_seams, atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
