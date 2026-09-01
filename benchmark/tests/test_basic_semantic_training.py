from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_basic_block,
    stack_semantic_targets,
)
from benchmark.drafting_semantics.dataset import edge_features as gcd_edge_features
from benchmark.drafting_semantics.multigarment_learning import (
    EDGE_FEATURE_DIM,
    MULTIGARMENT_EDGE_ROLES,
    MULTIGARMENT_PANEL_ROLES,
    padded_garment_batch,
)
from benchmark.drafting_semantics.multiview_curve_parameters import (
    CURVE_QUERY_NAMES,
    CurveFormulaTargets,
)
from benchmark.drafting_semantics.schema import EdgeAnnotation, PanelAnnotation
from benchmark.drafting_semantics.semantic_teacher_student import (
    SEMANTIC_QUERY_INDEX,
    build_four_view_semantic_student,
    build_vector_graph_teacher,
    freeze_semantic_teacher,
)
from benchmark.scripts.train_basic_semantic_teacher_student import (
    CALIBRATION_STATUS,
    CONSTRUCTION_LINE_FEATURE_INDEX,
    CategoryMeanBaseline,
    SemanticTrainingExample,
    TrainOnlyCoordinateCalibrator,
    _apply_dense_curve_formula_overlay,
    _prediction_rows,
    basic_block_to_graph,
    coordinate_confidence_from_validation,
    deterministic_category_oversample,
    deterministic_category_split,
    graph_padding_audit,
    semantic_edit_calibration_from_validation,
    semantic_metrics,
    student_training_step,
    supervised_teacher_loss,
)


def _model_config() -> dict[str, object]:
    return {
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "global_feature_dim": 10,
        "spatial_feature_dim": 8,
        "width": 16,
        "token_dim": 8,
        "heads": 4,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "feedforward_multiplier": 2,
        "dropout": 0.0,
        "max_views": 4,
        "panel_role_count": len(MULTIGARMENT_PANEL_ROLES),
        "edge_role_count": len(MULTIGARMENT_EDGE_ROLES),
    }


def _example(category: str, index: int) -> SemanticTrainingExample:
    block = build_basic_block(category, sample_id=f"{category}_{index}")
    return SemanticTrainingExample(
        sample_id=block.sample_id,
        category=category,
        graph=basic_block_to_graph(block),
        target=semantic_target_from_basic_block(block),
        global_features=np.random.default_rng(index).normal(size=(4, 10)).astype(np.float32),
    )


class BasicSemanticTrainingTests(unittest.TestCase):
    def test_stratified_split_is_repeatable_disjoint_and_keeps_each_category(self) -> None:
        ids = [f"{category}_{index}" for category in ("tshirt", "pants", "skirt") for index in range(10)]
        categories = [value.split("_", 1)[0] for value in ids]
        fractions = {"train": 0.6, "validation": 0.2, "test": 0.2}
        first = deterministic_category_split(ids, categories, seed=17, fractions=fractions)
        second = deterministic_category_split(ids, categories, seed=17, fractions=fractions)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(ids))
        for category in ("tshirt", "pants", "skirt"):
            category_splits = {first[value] for value in ids if value.startswith(category)}
            self.assertEqual(category_splits, {"train", "validation", "test"})

    def test_training_only_oversampling_balances_categories(self) -> None:
        examples = [
            _example("tshirt", 1),
            *(_example("pants", index) for index in range(2, 5)),
            *(_example("skirt", index) for index in range(5, 10)),
        ]
        balanced = deterministic_category_oversample(examples, seed=3)
        self.assertEqual(
            {category: sum(item.category == category for item in balanced) for category in ("tshirt", "pants", "skirt")},
            {"tshirt": 5, "pants": 5, "skirt": 5},
        )

    def test_train_only_calibration_maps_source_median_to_provisional_median(self) -> None:
        reference = semantic_target_from_basic_block(build_basic_block("tshirt", sample_id="reference"))
        query = SEMANTIC_QUERY_INDEX["tshirt:landmark:FNP"]
        channel = 0
        sources, references = [], []
        for index, offset in enumerate((-0.1, 0.0, 0.1)):
            source_coordinates = reference.coordinates.copy()
            source_coordinates[query, channel] = 0.75 + offset
            sources.append(
                replace(reference, sample_id=f"source_{index}", coordinates=source_coordinates)
            )
            reference_coordinates = reference.coordinates.copy()
            reference_coordinates[query, channel] = 0.25 + offset * 0.5
            references.append(
                replace(reference, sample_id=f"reference_{index}", coordinates=reference_coordinates)
            )
        calibrator = TrainOnlyCoordinateCalibrator.fit(
            sources,
            references,
            minimum_support=3,
            minimum_scale=1e-4,
            clamp_standard_deviations=4.0,
        )
        transformed = calibrator.transform(sources[1])
        self.assertAlmostEqual(float(transformed.coordinates[query, channel]), 0.25, places=5)
        self.assertAlmostEqual(float(sources[1].coordinates[query, channel]), 0.75, places=5)
        self.assertIn(CALIBRATION_STATUS, transformed.provenance_status)
        self.assertEqual(calibrator.to_dict()["fit_partition"], "train_only")

        path = SEMANTIC_QUERY_INDEX["tshirt:path:front_neckline"]
        for endpoint_channel in range(4):
            self.assertFalse(calibrator.fitted_mask[0, path, endpoint_channel])
            self.assertEqual(
                float(calibrator.transform(sources[1]).coordinates[path, endpoint_channel]),
                float(sources[1].coordinates[path, endpoint_channel]),
            )
        construction = SEMANTIC_QUERY_INDEX["tshirt:reference_line:front_BL"]
        self.assertFalse(calibrator.fitted_mask[0, construction].any())

    def test_basic_blocks_use_existing_vector_graph_contract(self) -> None:
        for category in ("tshirt", "pants", "skirt"):
            graph = basic_block_to_graph(build_basic_block(category))
            self.assertTrue(graph.panels)
            for panel in graph.panels:
                self.assertEqual(panel.features.shape[1], EDGE_FEATURE_DIM)
                self.assertEqual(len(panel.features), len(panel.edge_targets))
                self.assertTrue(np.isfinite(panel.features).all())

    def test_basic_graph_local_features_match_gcd_vertex_convention(self) -> None:
        block = build_basic_block("tshirt")
        source_panel = block.panels[0]
        graph_panel = basic_block_to_graph(block).panels[0]
        paths = {path.name: path for path in source_panel.paths}
        names = []
        for path_name in source_panel.boundary_order:
            for name in paths[path_name].landmark_sequence:
                if name not in names:
                    names.append(name)
        vertices = tuple(source_panel.landmark(name).xy_cm for name in names)
        indices = {name: index for index, name in enumerate(names)}
        primitives = []
        for path_name in source_panel.boundary_order:
            path = paths[path_name]
            if path.geometry_kind == "cubic_bezier":
                primitives.append(
                    (path, path.landmark_sequence[0], path.landmark_sequence[-1], path_name, "cubic")
                )
            else:
                primitives.extend(
                    (path, start_name, end_name, f"{path_name}#{segment}", "line")
                    for segment, (start_name, end_name) in enumerate(
                        zip(path.landmark_sequence, path.landmark_sequence[1:])
                    )
                )
        edges = []
        for index, (path, start_name, end_name, edge_id, curvature) in enumerate(primitives):
            edges.append(
                EdgeAnnotation(
                    id=edge_id,
                    index=index,
                    endpoints=(indices[start_name], indices[end_name]),
                    start_cm=source_panel.landmark(start_name).xy_cm,
                    end_cm=source_panel.landmark(end_name).xy_cm,
                    curvature_type=curvature,
                    role=path.role,
                    stitched=False,
                    self_stitched=False,
                    length_cm=float(graph_panel.edge_lengths_cm[index]),
                    evidence="recipe_reconstruction",
                    confidence=1.0,
                )
            )
        panel = PanelAnnotation(
            id=source_panel.id,
            role=source_panel.role,
            vertices_cm=vertices,
            edges=tuple(edges),
        )
        local, _ = gcd_edge_features(panel, include_stitch_features=False)
        # Reference/construction lines are appended after the exact boundary
        # primitives.  Their presence must not alter the GCD-compatible
        # boundary feature convention.
        np.testing.assert_allclose(
            graph_panel.features[: len(local), :17], local, atol=1e-6
        )

    def test_basic_graph_contains_non_boundary_construction_tokens(self) -> None:
        for category, expected in (
            ("tshirt", {"BL", "WL", "HL"}),
            ("pants", {"WL", "HL", "CL", "KL", "GRAIN"}),
            ("skirt", {"WL", "HL", "GRAIN"}),
        ):
            graph = basic_block_to_graph(build_basic_block(category))
            for panel in graph.panels:
                observed = {
                    edge_id.removeprefix("@reference:")
                    for edge_id in panel.edge_ids
                    if edge_id.startswith("@reference:")
                }
                if category == "tshirt" and panel.panel_id == "sleeve":
                    self.assertEqual(observed, set())
                else:
                    self.assertEqual(observed, expected)
                for index, edge_id in enumerate(panel.edge_ids):
                    marker = float(
                        panel.features[index, CONSTRUCTION_LINE_FEATURE_INDEX]
                    )
                    if edge_id.startswith("@reference:"):
                        self.assertEqual(marker, 1.0)
                        self.assertEqual(int(panel.edge_targets[index]), -100)
                    else:
                        self.assertEqual(marker, 0.0)

    def test_graph_padding_audit_fails_before_any_token_can_be_truncated(self) -> None:
        examples = [_example("tshirt", 1), _example("pants", 2), _example("skirt", 3)]
        result = graph_padding_audit(
            examples, maximum_panels=10, maximum_edges=32
        )
        self.assertEqual(result["status"], "PASS_NO_TRUNCATION")
        self.assertLessEqual(result["observed_maximum_edges"], 32)

        panel = examples[0].graph.panels[0]
        repeats = 33
        oversized_panel = replace(
            panel,
            features=np.repeat(panel.features[:1], repeats, axis=0),
            edge_targets=np.repeat(panel.edge_targets[:1], repeats, axis=0),
            edge_lengths_cm=np.repeat(panel.edge_lengths_cm[:1], repeats, axis=0),
            edge_ids=tuple(f"oversized_{index}" for index in range(repeats)),
        )
        oversized = replace(
            examples[0],
            graph=replace(
                examples[0].graph,
                panels=(oversized_panel, *examples[0].graph.panels[1:]),
            ),
        )
        with self.assertRaisesRegex(ValueError, "would truncate"):
            graph_padding_audit(
                [oversized], maximum_panels=10, maximum_edges=32
            )

    def test_basic_graph_exposes_hip_and_knee_polyline_breakpoints(self) -> None:
        base = basic_block_to_graph(build_basic_block("pants")).panels[0]
        wider_knee = basic_block_to_graph(
            build_basic_block("pants", design={"knee_circumference_cm": 52.0})
        ).panels[0]
        deeper_hip = basic_block_to_graph(
            build_basic_block("pants", design={"hip_depth_cm": 24.0})
        ).panels[0]
        self.assertIn("outseam#0", base.edge_ids)
        self.assertIn("outseam#1", base.edge_ids)
        self.assertIn("outseam#2", base.edge_ids)
        self.assertIn("inseam#0", base.edge_ids)
        self.assertIn("inseam#1", base.edge_ids)

        def row(panel, edge_id):
            return panel.features[panel.edge_ids.index(edge_id), :6]

        # outseam#1 terminates at SIDE_KNEE; changing knee width must alter its
        # explicit endpoint token rather than only a collapsed-path length.
        self.assertFalse(np.allclose(row(base, "outseam#1"), row(wider_knee, "outseam#1")))
        # outseam#0 terminates at SIDE_HIP; hip-depth variation is likewise
        # visible as an authored primitive endpoint.
        self.assertFalse(np.allclose(row(base, "outseam#0"), row(deeper_hip, "outseam#0")))

    def test_dense_curve_overlay_never_replaces_garment_frame_endpoints(self) -> None:
        target = semantic_target_from_basic_block(build_basic_block("tshirt"))
        endpoints_before = target.coordinates[:, :4].copy()
        values = np.zeros((len(CURVE_QUERY_NAMES), 16), dtype=np.float32)
        values[:, 0:4] = 99.0  # Deliberately incompatible panel-frame UV.
        values[:, 4] = 0.4
        values[:, 5] = 0.5
        # Valid, shallow two-cubic controls in the chord-local frame.
        values[:, 6:] = np.asarray(
            [0.5, 0.1, 0.15, 0.03, 0.35, 0.08, 0.65, 0.08, 0.85, 0.03],
            dtype=np.float32,
        )
        formula = CurveFormulaTargets(
            values=values,
            role_mask=np.ones(len(CURVE_QUERY_NAMES), dtype=np.bool_),
            fit_rmse_over_chord=np.zeros(len(CURVE_QUERY_NAMES), dtype=np.float32),
            observation_count=np.ones(len(CURVE_QUERY_NAMES), dtype=np.int64),
            provenance=tuple("DENSE_CURVE_TWO_CUBIC_APPROXIMATION" for _ in CURVE_QUERY_NAMES),
        )
        overlaid = _apply_dense_curve_formula_overlay(target, formula)
        np.testing.assert_array_equal(overlaid.coordinates[:, :4], endpoints_before)

    def test_tiny_teacher_and_student_steps_have_finite_gradients(self) -> None:
        import torch

        torch.manual_seed(9)
        examples = [_example("tshirt", 1), _example("pants", 2), _example("skirt", 3)]
        graph = padded_garment_batch(
            [item.graph for item in examples], maximum_panels=6, maximum_edges=32
        )
        targets = stack_semantic_targets([item.target for item in examples])
        teacher = build_vector_graph_teacher(_model_config())
        optimizer = torch.optim.SGD(teacher.parameters(), lr=0.05)
        token_weight_before = teacher.element_token_head[1].weight.detach().clone()
        output = teacher(
            torch.from_numpy(graph["features"]),
            edge_valid=torch.from_numpy(graph["edge_valid"]),
            panel_valid=torch.from_numpy(graph["panel_valid"]),
            category_ids=torch.from_numpy(targets["category_ids"]),
        )
        loss = supervised_teacher_loss(
            output,
            targets,
            graph,
            weights={"presence": 1.0, "coordinate": 1.0, "panel_role": 1.0, "edge_role": 1.0},
        )
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in teacher.parameters()))
        token_grad = teacher.element_token_head[1].weight.grad
        self.assertIsNotNone(token_grad)
        self.assertGreater(float(token_grad.abs().sum()), 0.0)
        optimizer.step()
        self.assertFalse(
            torch.equal(token_weight_before, teacher.element_token_head[1].weight.detach())
        )

        teacher.zero_grad(set_to_none=True)
        freeze_semantic_teacher(teacher)
        student = build_four_view_semantic_student(_model_config())
        student_loss = student_training_step(
            student,
            teacher,
            examples,
            mode="global",
            maximum_panels=6,
            maximum_edges=32,
            device=torch.device("cpu"),
            weights={"distillation": 1.0, "presence": 1.0, "coordinate": 1.0},
        )
        self.assertTrue(torch.isfinite(student_loss["loss"]))
        student_loss["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_category_mean_baseline_and_metrics_are_numeric_only(self) -> None:
        targets = [_example("tshirt", 4).target, _example("pants", 5).target, _example("skirt", 6).target]
        baseline = CategoryMeanBaseline.fit(targets)
        presence, coordinates = baseline.predict(targets)
        result = semantic_metrics(presence, coordinates, targets)
        self.assertEqual(result["sample_count"], 3)
        self.assertGreaterEqual(result["presence_macro_f1"], 0.0)
        self.assertAlmostEqual(result["coordinate_normalized_mae"], 0.0, places=7)
        tshirt = result["per_category"]["tshirt"]
        self.assertIn("reference_line", tshirt["coordinate_normalized_mae"])
        bl = tshirt["queries"]["tshirt:reference_line:front_BL"]
        self.assertEqual(bl["presence_metric_status"], "POSITIVE_ONLY_NO_ABSENCE_EVIDENCE")
        self.assertEqual(bl["negative_support"], 0)

    def test_prediction_export_uses_static_schema_not_ground_truth_mask(self) -> None:
        example = _example("pants", 12)
        count = len(example.target.presence)
        probabilities = np.full((1, count), 0.75, dtype=np.float32)
        coordinates = np.arange(count * 8, dtype=np.float32).reshape(1, count, 8) / 100.0
        metrics = semantic_metrics(probabilities, coordinates, [example.target])
        calibration = coordinate_confidence_from_validation(metrics)
        rows = _prediction_rows([example], probabilities, coordinates, calibration)
        legacy = next(
            query for query in rows[0]["queries"] if query["query"] == "pants:path:dart_leg"
        )
        self.assertFalse(legacy["query_supervised_in_ground_truth"])
        self.assertEqual(len(legacy["predicted_coordinates"]), 8)
        self.assertTrue(all(value is not None for value in legacy["predicted_coordinates"].values()))
        self.assertEqual(calibration["per_query"]["pants:path:dart_leg"], 0.0)

    def test_semantic_edit_calibration_selects_student_or_anchor_from_validation(self) -> None:
        base = _example("tshirt", 31)
        target_coordinates = base.target.coordinates.copy()
        fnp = SEMANTIC_QUERY_INDEX["tshirt:landmark:FNP"]
        bnp = SEMANTIC_QUERY_INDEX["tshirt:landmark:BNP"]
        target_coordinates[fnp, 0] += 0.10
        shifted = replace(
            base,
            target=replace(base.target, coordinates=target_coordinates),
        )
        examples = [replace(shifted, sample_id=f"validation_{index}") for index in range(4)]
        probabilities = np.ones((4, len(base.target.presence)), dtype=np.float32)
        coordinates = np.repeat(base.target.coordinates[None], 4, axis=0)
        coordinates[:, fnp] = target_coordinates[fnp]
        coordinates[:, bnp, 0] += 0.08
        metrics = semantic_metrics(
            probabilities,
            coordinates,
            [example.target for example in examples],
        )
        reliability = coordinate_confidence_from_validation(metrics)
        calibration = semantic_edit_calibration_from_validation(
            metrics,
            examples,
            reliability,
        )
        fnp_row = calibration["per_query"]["tshirt:landmark:FNP"]
        bnp_row = calibration["per_query"]["tshirt:landmark:BNP"]
        self.assertTrue(fnp_row["allow_student_edit"])
        self.assertEqual(fnp_row["anchor_retention_weight"], 0.0)
        self.assertFalse(bnp_row["allow_student_edit"])
        self.assertGreater(bnp_row["anchor_retention_weight"], 0.0)
        self.assertFalse(calibration["test_ground_truth_used"])


if __name__ == "__main__":
    unittest.main()
