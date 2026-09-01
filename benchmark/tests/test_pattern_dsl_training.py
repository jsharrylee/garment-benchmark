from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES, PANEL_ROLES
from benchmark.gcdv2_exact.pattern_dsl_learning import (
    CATEGORIES,
    CURVE_COMMANDS,
    EDGE_FEATURE_DIMENSION,
    MASK_COMMAND,
    PatternDSLArrayDataset,
    build_pattern_dsl_model,
)
from benchmark.scripts.train_gcdv2_pattern_dsl_transformer import (
    compute_loss,
    evaluate,
    training_class_weights,
)


class PatternDSLTrainingTests(unittest.TestCase):
    def test_padded_command_slots_are_not_training_targets(self):
        arrays = {
            "categories": np.asarray([0], np.int8),
            "panel_valid": np.asarray([[True, False]], bool),
            "panel_roles": np.full((1, 2), -100, np.int8),
            "edge_valid": np.asarray([[[True, True, False], [False, False, False]]], bool),
            "edge_commands": np.asarray([[[0, 2, MASK_COMMAND], [MASK_COMMAND] * 3]], np.int8),
            "edge_features": np.zeros((1, 2, 3, EDGE_FEATURE_DIMENSION), np.float16),
            "edge_roles": np.full((1, 2, 3), -100, np.int8),
            "landmarks": np.full((1, 2, 3), -1, np.int8),
            "stitch_pairs": np.full((1, 1, 4), -1, np.int16),
            "stitch_valid": np.zeros((1, 1), bool),
        }
        item = PatternDSLArrayDataset(arrays, [0], mask_commands=False)[0]
        self.assertEqual(item["command_targets"].tolist(), [[0, 2, -100], [-100, -100, -100]])

    def test_all_missing_semantics_produce_finite_zero_semantic_losses(self):
        import torch

        batch_size, panels, edges = 1, 2, 3
        output = {
            "category_logits": torch.zeros((batch_size, len(CATEGORIES)), requires_grad=True),
            "panel_role_logits": torch.zeros((batch_size, panels, len(PANEL_ROLES)), requires_grad=True),
            "edge_role_logits": torch.zeros((batch_size, panels, edges, len(EDGE_ROLES)), requires_grad=True),
            "command_logits": torch.zeros((batch_size, panels, edges, len(CURVE_COMMANDS)), requires_grad=True),
            "seam_logits": torch.zeros((batch_size, panels * edges, panels * edges), requires_grad=True),
        }
        batch = {
            "category": torch.tensor([0]),
            "panel_roles": torch.full((batch_size, panels), -100),
            "edge_roles": torch.full((batch_size, panels, edges), -100),
            "command_targets": torch.full((batch_size, panels, edges), -100),
            "command_mask": torch.zeros((batch_size, panels, edges), dtype=torch.bool),
            "edge_valid": torch.tensor([[[True, True, False], [False, False, False]]]),
            "landmarks": torch.full((batch_size, panels, edges), -1),
            "stitch_pairs": torch.full((batch_size, 1, 4), -1),
            "stitch_valid": torch.zeros((batch_size, 1), dtype=torch.bool),
        }
        weights = {
            "panel": torch.ones(len(PANEL_ROLES)),
            "edge": torch.ones(len(EDGE_ROLES)),
        }
        allowed = torch.ones((len(EDGE_ROLES), len(EDGE_ROLES)), dtype=torch.bool)
        loss, parts = compute_loss(output, batch, torch.device("cpu"), weights, allowed)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(parts["panel"]), 0.0)
        self.assertEqual(float(parts["edge"]), 0.0)
        self.assertEqual(float(parts["command"]), 0.0)
        loss.backward()

    def test_class_weights_use_only_train_split(self):
        labels = np.asarray([[0, 0], [1, -100], [2, 2], [2, 2]], np.int16)
        splits = np.asarray([0, 0, 1, 2], np.int8)
        first = training_class_weights(labels, splits, 3)
        changed_holdout = labels.copy()
        changed_holdout[2:] = 1
        second = training_class_weights(changed_holdout, splits, 3)
        np.testing.assert_allclose(first, second)
        self.assertEqual(float(first[2]), 0.0)

    def test_model_is_finite_with_padded_panels(self):
        import torch

        model = build_pattern_dsl_model(width=16, heads=4, edge_layers=1, garment_layers=1)
        features = torch.zeros((2, 3, 4, EDGE_FEATURE_DIMENSION))
        commands = torch.full((2, 3, 4), MASK_COMMAND, dtype=torch.long)
        edge_valid = torch.zeros((2, 3, 4), dtype=torch.bool)
        panel_valid = torch.tensor([[True, False, False], [True, True, False]])
        edge_valid[0, 0, :2] = True
        edge_valid[1, 0, :3] = True
        edge_valid[1, 1, :2] = True
        output = model(features, commands, edge_valid, panel_valid)
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

        # A malformed/empty record is not a training sample, but attention
        # masking should still fail closed with finite outputs rather than NaN.
        empty = model(
            features[:1], commands[:1], torch.zeros_like(edge_valid[:1]), torch.zeros_like(panel_valid[:1])
        )
        self.assertTrue(all(torch.isfinite(value).all() for value in empty.values()))

    def test_evaluation_excludes_padded_commands(self):
        import torch

        class ConstantModel(torch.nn.Module):
            def forward(self, features, commands, edge_valid, panel_valid):
                batch, panels, edges = commands.shape
                command_logits = torch.full(
                    (batch, panels, edges, len(CURVE_COMMANDS)), -10.0, device=commands.device
                )
                command_logits[..., 0] = 10.0
                return {
                    "category_logits": torch.zeros((batch, len(CATEGORIES)), device=commands.device),
                    "panel_role_logits": torch.zeros(
                        (batch, panels, len(PANEL_ROLES)), device=commands.device
                    ),
                    "edge_role_logits": torch.zeros(
                        (batch, panels, edges, len(EDGE_ROLES)), device=commands.device
                    ),
                    "command_logits": command_logits,
                    "seam_logits": torch.full(
                        (batch, panels * edges, panels * edges), -20.0, device=commands.device
                    ),
                }

        batch = {
            "category": torch.tensor([0]),
            "features": torch.zeros((1, 2, 3, EDGE_FEATURE_DIMENSION)),
            "commands": torch.tensor([[[0, 0, MASK_COMMAND], [MASK_COMMAND] * 3]]),
            # Deliberately wrong, non-ignore targets in padding: edge_valid must
            # still be the authority used by evaluation.
            "command_targets": torch.tensor([[[0, 0, 1], [1, 1, 1]]]),
            "panel_valid": torch.tensor([[True, False]]),
            "panel_roles": torch.full((1, 2), -100),
            "edge_valid": torch.tensor([[[True, True, False], [False, False, False]]]),
            "edge_roles": torch.full((1, 2, 3), -100),
            "landmarks": torch.full((1, 2, 3), -1),
            "stitch_pairs": torch.full((1, 1, 4), -1),
            "stitch_valid": torch.zeros((1, 1), dtype=torch.bool),
        }
        allowed = np.ones((len(EDGE_ROLES), len(EDGE_ROLES)), bool)
        result, _ = evaluate(ConstantModel(), [batch], torch.device("cpu"), allowed)
        self.assertEqual(result["masked_command_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
