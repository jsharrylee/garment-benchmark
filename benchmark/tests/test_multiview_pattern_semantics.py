from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multiview_pattern_semantics import (
    PATTERN_TARGET_NAMES,
    VIEW_FEATURE_DIM,
    MultiviewPatternExample,
    TargetStandardizer,
    _split_lookup,
    build_multiview_pattern_model,
    multiview_batch,
)


class MultiviewPatternSemanticsTests(unittest.TestCase):
    def test_split_lookup_scopes_reused_basename_to_the_selected_archive(self) -> None:
        payload = {
            "training": ["garments_5000_0/default_body/reused"],
            "test": ["garments_5000_1/default_body/reused"],
            "validation": ["garments_5000_0/default_body/unique"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            split = _split_lookup(path, split_prefix="garments_5000_0/default_body")
        self.assertEqual(split, {"reused": "train", "unique": "validation"})

    def test_standardizer_round_trip(self) -> None:
        values = []
        for index in range(3):
            target = np.arange(len(PATTERN_TARGET_NAMES), dtype=np.float32) + index
            values.append(MultiviewPatternExample(str(index), "train", 0, np.zeros((4, VIEW_FEATURE_DIM), np.float32), target, ("a",) * 4, "p"))
        standardizer = TargetStandardizer.fit(values)
        batch = multiview_batch(values, standardizer)
        np.testing.assert_allclose(standardizer.decode(batch["pattern_targets"]), batch["raw_pattern_targets"], atol=1e-5)

    def test_model_captures_one_matrix_per_layer_and_head(self) -> None:
        import torch

        config = {"width": 40, "heads": 5, "layers": 2, "feedforward_multiplier": 2, "dropout": 0.0, "contrastive_dimension": 12}
        model = build_multiview_pattern_model(config)
        output = model(torch.zeros((3, 4, VIEW_FEATURE_DIM)), pattern_targets=torch.zeros((3, len(PATTERN_TARGET_NAMES))), capture_attention=True)
        self.assertEqual(output["pattern_prediction"].shape, (3, len(PATTERN_TARGET_NAMES)))
        self.assertEqual(output["attention"][0].shape, (3, 5, 5, 5))


if __name__ == "__main__":
    unittest.main()
