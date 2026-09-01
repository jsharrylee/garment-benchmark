from __future__ import annotations

import numpy as np


class TorchvisionPersonSegmenter:
    """COCO person-instance masker selected by overlap with a tracked box."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()

    @staticmethod
    def _iou(box: np.ndarray, target: tuple[int, int, int, int]) -> float:
        tx0, ty0, tx1, ty1 = target
        x0, y0 = max(float(box[0]), tx0), max(float(box[1]), ty0)
        x1, y1 = min(float(box[2]), tx1), min(float(box[3]), ty1)
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])) + (tx1 - tx0) * (ty1 - ty0) - intersection)
        return intersection / union

    def __call__(self, rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.inference_mode():
            prediction = self.model([tensor])[0]
        boxes = prediction["boxes"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        candidates = [
            (self._iou(box, bbox_xyxy), float(score), index)
            for index, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=True))
            if label == 1 and score >= 0.35
        ]
        if not candidates:
            raise ValueError("MASK_FAILURE: Mask R-CNN found no person instance")
        overlap, score, index = max(candidates)
        if overlap < 0.05:
            raise ValueError(f"MASK_FAILURE: best person detection misses tracked box (IoU={overlap:.4f}, score={score:.4f})")
        probability = prediction["masks"][index, 0].detach().cpu().numpy()
        return np.where(probability >= 0.5, 255, 0).astype(np.uint8)


def grabcut_mask(rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int], *, iterations: int = 8) -> np.ndarray:
    """Return a deterministic uint8 foreground mask from a tracked person box."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("opencv-python-headless is required for GrabCut masking") from exc

    x0, y0, x1, y1 = bbox_xyxy
    height, width = rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox after clipping: {(x0, y0, x1, y1)}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    labels = np.full((height, width), cv2.GC_BGD, np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    labels[y0:y1, x0:x1] = cv2.GC_PR_FGD
    box_width, box_height = x1 - x0, y1 - y0
    seed_x0 = x0 + max(1, round(box_width * 0.35))
    seed_x1 = x1 - max(1, round(box_width * 0.35))
    seed_y0 = y0 + max(1, round(box_height * 0.12))
    seed_y1 = y1 - max(1, round(box_height * 0.30))
    labels[seed_y0:seed_y1, seed_x0:seed_x1] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(bgr, labels, None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
    foreground = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.dilate(foreground, kernel, iterations=1)
    return foreground


def mask_statistics(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"foreground_pixels": 0, "foreground_fraction": 0.0, "bbox_xyxy": None}
    return {
        "foreground_pixels": int(len(xs)),
        "foreground_fraction": float(len(xs) / mask.size),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
    }
