from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


IMAGE_SIZE = 128


def read_panel_rows(index_path: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(index_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows if split is None else [row for row in rows if row["split"] == split]


class VisualGeometryDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["input_panel_image"]) as image:
            image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, np.float32)[None] / 255.0
        with np.load(row["visual_truth_path"]) as truth:
            mask = truth["mask_u8"].astype(np.float32)[None]
            if mask.max() > 1.0:
                mask /= 255.0
            sdf_cm = truth["sdf_cm_f16"].astype(np.float32)[None]
            junction = truth["visible_junction_heatmap_f16"].astype(np.float32)[None]
        # A bounded physical-distance target prevents distant background pixels
        # from dominating the boundary-reconstruction objective.
        sdf_scale_cm = 5.0
        sdf = np.clip(sdf_cm / sdf_scale_cm, -1.0, 1.0)
        return {
            "image": pixels,
            "mask": mask,
            "sdf": sdf,
            "junction": junction,
            "cm_per_pixel": np.float32(row["input_scale_cm_per_pixel"]),
            "panel_uid": row["panel_uid"],
        }


def build_visual_model(base_width: int = 24):
    import torch
    from torch import nn
    import torch.nn.functional as F

    def block(input_channels: int, output_channels: int) -> nn.Sequential:
        groups = max(1, min(8, output_channels // 4))
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    class VisualGeometryUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            w = base_width
            self.enc1 = block(1, w)
            self.enc2 = block(w, w * 2)
            self.enc3 = block(w * 2, w * 4)
            self.bottleneck = block(w * 4, w * 8)
            self.dec3 = block(w * 8 + w * 4, w * 4)
            self.dec2 = block(w * 4 + w * 2, w * 2)
            self.dec1 = block(w * 2 + w, w)
            self.head = nn.Conv2d(w, 3, 1)

        def forward(self, image):
            e1 = self.enc1(image)
            e2 = self.enc2(F.avg_pool2d(e1, 2))
            e3 = self.enc3(F.avg_pool2d(e2, 2))
            latent = self.bottleneck(F.avg_pool2d(e3, 2))
            d3 = self.dec3(torch.cat((F.interpolate(latent, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3), 1))
            d2 = self.dec2(torch.cat((F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), 1))
            d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), 1))
            raw = self.head(d1)
            return {"mask_logits": raw[:, 0:1], "sdf": raw[:, 1:2].tanh(), "junction_logits": raw[:, 2:3]}

    return VisualGeometryUNet()


def visual_loss(output: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    mask = batch["mask"]
    mask_probability = output["mask_logits"].sigmoid()
    bce = F.binary_cross_entropy_with_logits(output["mask_logits"], mask)
    intersection = (mask_probability * mask).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (mask_probability.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1.0)).mean()
    # Boundary-adjacent SDF values carry more exact geometric information.
    sdf_weight = 1.0 + 3.0 * torch.exp(-batch["sdf"].abs() * 8.0)
    sdf = (F.smooth_l1_loss(output["sdf"], batch["sdf"], reduction="none") * sdf_weight).mean()
    junction_target = batch["junction"]
    junction_probability = output["junction_logits"].sigmoid()
    focal_weight = torch.where(junction_target > 0.05, 8.0, 1.0)
    junction = (F.binary_cross_entropy_with_logits(output["junction_logits"], junction_target, reduction="none") * focal_weight).mean()
    total = bce + dice + 2.0 * sdf + junction
    return {"loss": total, "mask_bce": bce, "mask_dice": dice, "sdf": sdf, "junction": junction}


def visual_metrics(output: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    predicted = output["mask_logits"].sigmoid() >= 0.5
    target = batch["mask"] >= 0.5
    intersection = (predicted & target).sum(dim=(1, 2, 3)).float()
    union = (predicted | target).sum(dim=(1, 2, 3)).float().clamp_min(1)
    # Morphological boundaries make the metric independent of filled area.
    pred_eroded = -F.max_pool2d(-predicted.float(), 3, 1, 1) >= 0.5
    true_eroded = -F.max_pool2d(-target.float(), 3, 1, 1) >= 0.5
    pred_boundary = predicted & ~pred_eroded
    true_boundary = target & ~true_eroded
    boundary_f1 = []
    for radius in (1, 2):
        kernel = radius * 2 + 1
        pred_near = F.max_pool2d(pred_boundary.float(), kernel, 1, radius) > 0
        true_near = F.max_pool2d(true_boundary.float(), kernel, 1, radius) > 0
        precision = (pred_boundary & true_near).sum().float() / pred_boundary.sum().clamp_min(1)
        recall = (true_boundary & pred_near).sum().float() / true_boundary.sum().clamp_min(1)
        boundary_f1.append(2 * precision * recall / (precision + recall).clamp_min(1e-8))
    sdf_mae_cm = (output["sdf"] - batch["sdf"]).abs().mean() * 5.0
    return {
        "silhouette_iou": float((intersection / union).mean()),
        "boundary_f1_1px": float(boundary_f1[0]),
        "boundary_f1_2px": float(boundary_f1[1]),
        "sdf_mae_cm_clipped": float(sdf_mae_cm),
    }


__all__ = [
    "IMAGE_SIZE",
    "VisualGeometryDataset",
    "build_visual_model",
    "read_panel_rows",
    "visual_loss",
    "visual_metrics",
]
