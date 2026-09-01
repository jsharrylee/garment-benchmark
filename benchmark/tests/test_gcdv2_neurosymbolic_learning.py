import unittest

import torch

from benchmark.gcdv2_exact.neurosymbolic_learning import build_visual_model, visual_loss, visual_metrics


class VisualGeometryLearningTest(unittest.TestCase):
    def test_model_shapes_and_finite_loss(self):
        model = build_visual_model(base_width=8)
        image = torch.rand(2, 1, 128, 128)
        batch = {
            "mask": (image > 0.5).float(),
            "sdf": image * 2 - 1,
            "junction": torch.zeros_like(image),
        }
        output = model(image)
        self.assertEqual(output["mask_logits"].shape, image.shape)
        self.assertEqual(output["sdf"].shape, image.shape)
        self.assertEqual(output["junction_logits"].shape, image.shape)
        self.assertTrue(torch.isfinite(visual_loss(output, batch)["loss"]))
        metrics = visual_metrics(output, batch)
        self.assertIn("silhouette_iou", metrics)


if __name__ == "__main__":
    unittest.main()
