from __future__ import annotations

import unittest

from benchmark.drafting_semantics.semantic_paths import merge_predicted_semantic_paths


class SemanticPathTests(unittest.TestCase):
    def test_two_armhole_primitives_become_one_predicted_path(self) -> None:
        paths = merge_predicted_semantic_paths(
            ("hemline", "side_seam", "armhole", "armhole", "shoulder", "neckline", "center_front"),
            edge_ids=tuple(f"e{index}" for index in range(7)),
            edge_lengths_cm=(20, 30, 8, 9, 12, 11, 40),
        )
        armhole = [path for path in paths if path.role == "armhole"]
        self.assertEqual(len(armhole), 1)
        self.assertEqual(armhole[0].edge_ids, ("e2", "e3"))
        self.assertEqual(armhole[0].primitive_count, 2)
        self.assertEqual(armhole[0].length_cm, 17)

    def test_five_sleeve_head_primitives_become_one_path(self) -> None:
        paths = merge_predicted_semantic_paths(
            ("sleeve_hem", "sleeve_underarm", *("sleeve_head",) * 5, "sleeve_underarm")
        )
        sleeve_head = [path for path in paths if path.role == "sleeve_head"]
        self.assertEqual(len(sleeve_head), 1)
        self.assertEqual(sleeve_head[0].primitive_count, 5)

    def test_predicted_link_can_keep_adjacent_equal_roles_separate(self) -> None:
        paths = merge_predicted_semantic_paths(
            ("armhole", "armhole", "shoulder"),
            same_path_links=(False, False, False),
            closed_boundary=False,
        )
        self.assertEqual([path.role for path in paths], ["armhole", "armhole", "shoulder"])

    def test_predicted_link_joins_armhole_without_geometry_loss(self) -> None:
        paths = merge_predicted_semantic_paths(
            ("armhole", "armhole", "shoulder"),
            same_path_links=(True, False, False),
            edge_ids=("curve_a", "curve_b", "shoulder"),
            closed_boundary=False,
        )
        self.assertEqual(paths[0].edge_ids, ("curve_a", "curve_b"))

    def test_closed_boundary_joins_wraparound_run(self) -> None:
        paths = merge_predicted_semantic_paths(("side_seam", "hemline", "side_seam"))
        self.assertEqual(paths[0].role, "side_seam")
        self.assertEqual(paths[0].edge_indices, (2, 0))

    def test_open_boundary_preserves_wraparound_split(self) -> None:
        paths = merge_predicted_semantic_paths(
            ("side_seam", "hemline", "side_seam"), closed_boundary=False
        )
        self.assertEqual([path.role for path in paths], ["side_seam", "hemline", "side_seam"])


if __name__ == "__main__":
    unittest.main()
