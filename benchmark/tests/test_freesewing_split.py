from __future__ import annotations

import unittest

import numpy as np

from benchmark.drafting_semantics.freesewing_split import (
    EXPECTED_SPLIT_COUNTS,
    TEST_BODY_MODELS,
    TEST_DESIGNS,
    TRAIN_BODY_MODELS,
    TRAIN_DESIGNS,
    VALIDATION_BODY_MODELS,
    VALIDATION_DESIGNS,
    teagan_training_split,
)
from benchmark.drafting_semantics.tshirt_learning import (
    EDGE_FEATURE_DIM,
    EDGE_ROLES,
    FEATURE_SLICES,
    LANDMARK_NAMES,
    PANEL_ROLES,
    decode_structural_semantics,
)


class FreeSewingTeaganSplitTests(unittest.TestCase):
    def test_body_and_design_groups_are_disjoint(self):
        self.assertFalse(set(TRAIN_BODY_MODELS) & set(VALIDATION_BODY_MODELS))
        self.assertFalse(set(TRAIN_BODY_MODELS) & set(TEST_BODY_MODELS))
        self.assertFalse(set(VALIDATION_BODY_MODELS) & set(TEST_BODY_MODELS))
        self.assertFalse(set(TRAIN_DESIGNS) & set(VALIDATION_DESIGNS))
        self.assertFalse(set(TRAIN_DESIGNS) & set(TEST_DESIGNS))
        self.assertFalse(set(VALIDATION_DESIGNS) & set(TEST_DESIGNS))

    def test_complete_80_record_matrix_has_expected_splits(self):
        observed = {name: 0 for name in EXPECTED_SPLIT_COUNTS}
        bodies = (*TRAIN_BODY_MODELS, *VALIDATION_BODY_MODELS, *TEST_BODY_MODELS)
        designs = (*TRAIN_DESIGNS, *VALIDATION_DESIGNS, *TEST_DESIGNS)
        for body in bodies:
            for design in designs:
                try:
                    split = teagan_training_split(f"freesewing_teagan__{body}__{design}")
                except ValueError:
                    continue
                observed[split] += 1
        self.assertEqual(observed, EXPECTED_SPLIT_COUNTS)

    def test_rejects_unknown_or_unsupported_crosses(self):
        with self.assertRaises(ValueError):
            teagan_training_split("freesewing_teagan__unknown__default")
        with self.assertRaises(ValueError):
            teagan_training_split("freesewing_teagan__cisFemaleAdult40__unknown")

    def test_structural_decoder_accepts_two_segment_armhole(self):
        segments = (
            ("hemline", (0, 10), (8, 10)),
            ("side_seam", (8, 10), (8, 5)),
            ("armhole", (8, 5), (7, 3)),
            ("armhole", (7, 3), (7, 1)),
            ("shoulder", (7, 1), (4, 0)),
            ("neckline", (4, 0), (0, 2)),
            ("center_front", (0, 2), (0, 10)),
        )
        features = np.zeros((len(segments), EDGE_FEATURE_DIM), dtype=np.float32)
        roles = []
        for index, (role, start, end) in enumerate(segments):
            features[index, FEATURE_SLICES["start"]] = start
            features[index, FEATURE_SLICES["end"]] = end
            roles.append(EDGE_ROLES.index(role))
        panel, exists, coordinates = decode_structural_semantics(features, roles)
        self.assertEqual(PANEL_ROLES[panel], "front")
        decoded = {
            name: tuple(coordinates[index])
            for index, name in enumerate(LANDMARK_NAMES)
            if exists[index]
        }
        self.assertEqual(decoded, {"FNP": (0.0, 2.0), "SNP": (4.0, 0.0), "SP": (7.0, 1.0)})


if __name__ == "__main__":
    unittest.main()
