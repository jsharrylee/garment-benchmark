from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multiview_curve_parameters import (
    CURVE_PARAMETER_NAMES,
    CURVE_QUERY_NAMES,
    build_spatial_curve_model,
)
from benchmark.scripts.convert_resnet50_global_curve_features import (
    convert_global_feature_archive,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GlobalCurveFeatureAblationTests(unittest.TestCase):
    def test_conversion_only_inserts_singleton_token_axis(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "source.npz"
            source_manifest = directory / "source_manifest.json"
            output = directory / "global_tokens.npz"
            manifest = directory / "global_tokens_manifest.json"
            sample_ids = np.asarray(("a", "b", "c"))
            features = np.arange(3 * 4 * 8, dtype=np.float16).reshape(3, 4, 8)
            np.savez_compressed(source, sample_ids=sample_ids, features=features)
            source_manifest.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")

            payload = convert_global_feature_archive(
                source, output, manifest, source_manifest=source_manifest
            )

            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(tuple(archive["features"].shape), (3, 4, 1, 8))
                np.testing.assert_array_equal(archive["features"][:, :, 0, :], features)
                np.testing.assert_array_equal(archive["sample_ids"], sample_ids)
            self.assertFalse(payload["numerical_values_changed"])
            self.assertFalse(payload["images_recomputed"])
            self.assertEqual(payload["source_archive_sha256"], _sha256(source))
            self.assertEqual(payload["output_sha256"], _sha256(output))
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), payload)

    def test_global_config_uses_same_curve_model_with_one_token_per_view(self) -> None:
        import torch

        config_directory = Path(__file__).parents[1] / "configs"
        config_path = config_directory / "multiview_curve_parameters_resnet50_global.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        spatial_config = json.loads(
            (config_directory / "multiview_curve_parameters_fpn.json").read_text(
                encoding="utf-8"
            )
        )
        identical_keys = (
            "target",
            "image_size",
            "width",
            "heads",
            "memory_layers",
            "decoder_layers",
            "feedforward_multiplier",
            "dropout",
            "precomputed_feature_batch_size",
            "end_to_end_image_batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "view_dropout_probability",
            "seed",
            "loss_weights",
            "network_download",
        )
        for key in identical_keys:
            with self.subTest(key=key):
                self.assertEqual(config[key], spatial_config[key])
        model = build_spatial_curve_model(config)
        self.assertEqual(model.tokens_per_view, 1)
        features = torch.randn(2, 4, 1, 2048)
        output = model(spatial_features=features, capture_attention=True)
        self.assertEqual(
            tuple(output["curve_prediction"].shape),
            (2, len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES)),
        )
        self.assertEqual(
            tuple(output["spatial_attention"][-1].shape),
            (2, config["heads"], len(CURVE_QUERY_NAMES), 4, 1),
        )

    def test_conversion_rejects_already_spatial_or_wrong_view_archives(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = directory / "bad.npz"
            np.savez(
                source,
                sample_ids=np.asarray(("a",)),
                features=np.zeros((1, 4, 1, 8), dtype=np.float16),
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                convert_global_feature_archive(
                    source, directory / "out.npz", directory / "manifest.json"
                )


if __name__ == "__main__":
    unittest.main()
