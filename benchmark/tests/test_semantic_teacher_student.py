from __future__ import annotations

import unittest

from benchmark.drafting_semantics.semantic_teacher_student import (
    CATEGORY_NAMES,
    MAX_COORDINATE_DIM,
    REFERENCE_LINE_COORDINATES,
    SEMANTIC_QUERY_INVENTORY,
    SEMANTIC_QUERY_KEYS,
    SEMANTIC_QUERY_SCHEMA_VERSION,
    ModalityContractError,
    build_four_view_semantic_student,
    build_vector_graph_teacher,
    category_query_mask,
    detached_teacher_forward,
    freeze_semantic_teacher,
    infer_four_view_semantics,
    query_coordinate_mask,
    semantic_distillation_loss,
)


def _config() -> dict[str, object]:
    return {
        "edge_feature_dim": 6,
        "spatial_feature_dim": 8,
        "global_feature_dim": 10,
        "width": 16,
        "token_dim": 8,
        "heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "feedforward_multiplier": 2,
        "dropout": 0.0,
        "max_views": 4,
        "panel_role_count": 7,
        "edge_role_count": 11,
    }


class SemanticTeacherStudentTests(unittest.TestCase):
    def test_inventory_is_unique_disjoint_and_coordinate_typed(self) -> None:
        self.assertEqual(
            SEMANTIC_QUERY_SCHEMA_VERSION,
            "basic-semantic-query/v3-reference-construction-lines",
        )
        self.assertEqual(len(SEMANTIC_QUERY_INVENTORY), 128)
        self.assertEqual(len(SEMANTIC_QUERY_KEYS), len(set(SEMANTIC_QUERY_KEYS)))
        masks = [category_query_mask(category) for category in CATEGORY_NAMES]
        self.assertTrue(all(any(mask) for mask in masks))
        for query_index in range(len(SEMANTIC_QUERY_INVENTORY)):
            self.assertEqual(sum(mask[query_index] for mask in masks), 1)
        coordinates = query_coordinate_mask()
        self.assertEqual(len(coordinates), len(SEMANTIC_QUERY_INVENTORY))
        for query, mask in zip(SEMANTIC_QUERY_INVENTORY, coordinates):
            self.assertEqual(len(mask), MAX_COORDINATE_DIM)
            self.assertEqual(sum(mask), len(query.coordinate_names))
        skirt_panels = {
            query.name
            for query in SEMANTIC_QUERY_INVENTORY
            if query.category == "skirt" and query.kind == "panel"
        }
        self.assertIn("skirt_panel", skirt_panels)
        self.assertNotIn("front_skirt", skirt_panels)
        self.assertNotIn("back_skirt", skirt_panels)
        references = [
            query for query in SEMANTIC_QUERY_INVENTORY if query.kind == "reference_line"
        ]
        self.assertEqual(len(references), 22)
        self.assertTrue(
            all(query.coordinate_names == REFERENCE_LINE_COORDINATES for query in references)
        )

    def test_vector_graph_teacher_shapes_and_explicit_freeze(self) -> None:
        import torch

        torch.manual_seed(5)
        teacher = build_vector_graph_teacher(_config())
        edge_features = torch.randn(3, 3, 5, 6)
        edge_valid = torch.tensor(
            [
                [[True] * 5, [True, True, False, False, False], [False] * 5],
                [[True] * 5, [True] * 5, [True, False, False, False, False]],
                [[True] * 5, [False] * 5, [False] * 5],
            ]
        )
        category_ids = torch.tensor([0, 1, 2])
        output = teacher(
            edge_features,
            edge_valid=edge_valid,
            category_ids=category_ids,
        )
        expected_queries = len(SEMANTIC_QUERY_INVENTORY)
        self.assertEqual(tuple(output["element_tokens"].shape), (3, expected_queries, 8))
        self.assertEqual(tuple(output["presence_logits"].shape), (3, expected_queries))
        self.assertEqual(
            tuple(output["coordinates"].shape),
            (3, expected_queries, MAX_COORDINATE_DIM),
        )
        self.assertEqual(tuple(output["panel_role_logits"].shape), (3, 3, 7))
        self.assertEqual(tuple(output["edge_role_logits"].shape), (3, 3, 5, 11))
        self.assertEqual(
            output["query_mask"].sum(dim=1).tolist(),
            [sum(category_query_mask(category)) for category in CATEGORY_NAMES],
        )
        with self.assertRaisesRegex(RuntimeError, "freeze_semantic_teacher"):
            detached_teacher_forward(
                teacher,
                edge_features,
                edge_valid=edge_valid,
                category_ids=category_ids,
            )
        freeze_semantic_teacher(teacher)
        detached = detached_teacher_forward(
            teacher,
            edge_features,
            edge_valid=edge_valid,
            category_ids=category_ids,
        )
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        self.assertFalse(detached["element_tokens"].requires_grad)
        self.assertFalse(detached["coordinates"].requires_grad)

    def test_student_accepts_spatial_and_global_but_rejects_pattern_input(self) -> None:
        import torch

        torch.manual_seed(7)
        student = build_four_view_semantic_student(_config())
        spatial = torch.randn(2, 4, 6, 8)
        global_features = torch.randn(2, 4, 10)
        view_valid = torch.tensor(
            [[True, True, True, True], [True, False, True, True]]
        )
        category_ids = torch.tensor([0, 2])
        output = student(
            spatial_features=spatial,
            global_features=global_features,
            view_valid=view_valid,
            category_ids=category_ids,
        )
        expected_queries = len(SEMANTIC_QUERY_INVENTORY)
        self.assertEqual(tuple(output["element_tokens"].shape), (2, expected_queries, 8))
        self.assertEqual(
            tuple(output["coordinates"].shape),
            (2, expected_queries, MAX_COORDINATE_DIM),
        )
        with self.assertRaises(ModalityContractError):
            student(
                spatial_features=spatial,
                category_ids=category_ids,
                edge_features=torch.randn(2, 1, 1, 6),
            )
        with self.assertRaises(ModalityContractError):
            infer_four_view_semantics(
                student,
                spatial_features=spatial,
                global_features=global_features,
                category_ids=category_ids,
                pattern_graph={"panels": []},
            )
        inferred = infer_four_view_semantics(
            student,
            spatial_features=spatial,
            global_features=global_features,
            category_ids=category_ids,
            view_valid=view_valid,
        )
        self.assertFalse(inferred["element_tokens"].requires_grad)

    def test_distillation_masks_absent_elements_and_detaches_teacher(self) -> None:
        import torch

        batch = 1
        queries = len(SEMANTIC_QUERY_INVENTORY)
        token_dim = 3
        query_mask = torch.zeros((batch, queries), dtype=torch.bool)
        query_mask[:, :2] = True
        coordinate_applicability = torch.ones(
            (batch, queries, MAX_COORDINATE_DIM), dtype=torch.bool
        )
        student_tokens = torch.zeros(
            (batch, queries, token_dim), requires_grad=True
        )
        with torch.no_grad():
            student_tokens[:, 0] = 1.0
            student_tokens[:, 1] = 100.0
        student_presence = torch.zeros((batch, queries), requires_grad=True)
        student_coordinates = torch.zeros(
            (batch, queries, MAX_COORDINATE_DIM), requires_grad=True
        )
        with torch.no_grad():
            student_coordinates[:, 1] = 100.0
        teacher_tokens = torch.zeros(
            (batch, queries, token_dim), requires_grad=True
        )
        teacher_presence = torch.zeros((batch, queries), requires_grad=True)
        teacher_coordinates = torch.zeros(
            (batch, queries, MAX_COORDINATE_DIM), requires_grad=True
        )
        student_output = {
            "element_tokens": student_tokens,
            "presence_logits": student_presence,
            "coordinates": student_coordinates,
            "query_mask": query_mask,
            "coordinate_mask": coordinate_applicability,
        }
        teacher_output = {
            "element_tokens": teacher_tokens,
            "presence_logits": teacher_presence,
            "coordinates": teacher_coordinates,
            "query_mask": query_mask,
            "coordinate_mask": coordinate_applicability,
        }
        presence_targets = torch.zeros((batch, queries))
        presence_targets[:, 0] = 1.0
        loss = semantic_distillation_loss(
            student_output,
            teacher_output,
            presence_targets=presence_targets,
        )
        self.assertAlmostEqual(float(loss["distillation_loss"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss["coordinate_loss"]), 0.0, places=6)
        self.assertEqual(int(loss["active_element_count"]), 1)
        loss["loss"].backward()
        self.assertGreater(float(student_tokens.grad[:, 0].abs().sum()), 0.0)
        self.assertEqual(float(student_tokens.grad[:, 1].abs().sum()), 0.0)
        self.assertEqual(float(student_coordinates.grad[:, 1].abs().sum()), 0.0)
        self.assertIsNone(teacher_tokens.grad)
        self.assertIsNone(teacher_presence.grad)
        self.assertIsNone(teacher_coordinates.grad)

    def test_shape_contracts_fail_fast(self) -> None:
        import torch

        teacher = build_vector_graph_teacher(_config())
        with self.assertRaisesRegex(ValueError, "edge_features"):
            teacher(torch.randn(2, 3, 6), category_ids=torch.tensor([0, 1]))
        student = build_four_view_semantic_student(_config())
        with self.assertRaisesRegex(ValueError, "at least one visual"):
            student(category_ids=torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "category_ids"):
            student(
                spatial_features=torch.randn(2, 4, 3, 8),
                category_ids=torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
