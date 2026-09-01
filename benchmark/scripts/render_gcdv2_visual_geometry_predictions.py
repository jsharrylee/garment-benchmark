from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from benchmark.gcdv2_exact.neurosymbolic_learning import VisualGeometryDataset, build_visual_model, read_panel_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render frozen-test visual geometry predictions beside truth.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_neurosymbolic/visual_geometry.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_training/visual_geometry_test_10.png"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    rows = read_panel_rows(args.index, "test")
    chosen = random.Random(args.seed).sample(rows, min(args.count, len(rows)))
    dataset = VisualGeometryDataset(chosen)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_visual_model(int(checkpoint["base_width"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    figure, axes = plt.subplots(len(chosen), 5, figsize=(15, 3 * len(chosen)), constrained_layout=True)
    for row_index, sample in enumerate(dataset):
        image = torch.from_numpy(sample["image"])[None]
        with torch.no_grad():
            output = model(image)
        predicted_mask = output["mask_logits"][0, 0].sigmoid().numpy()
        predicted_sdf = output["sdf"][0, 0].numpy() * 5.0
        predicted_junction = output["junction_logits"][0, 0].sigmoid().numpy()
        values = (
            (sample["image"][0], "input panel"),
            (sample["mask"][0], "true mask"),
            (predicted_mask, "predicted mask"),
            (predicted_sdf, "predicted SDF cm"),
            (predicted_junction, "predicted visible corners"),
        )
        for column, (value, title) in enumerate(values):
            axis = axes[row_index, column]
            axis.imshow(value, cmap="coolwarm" if column == 3 else "gray", vmin=-5 if column == 3 else None, vmax=5 if column == 3 else None)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title)
        axes[row_index, 0].text(0, 140, sample["panel_uid"], fontsize=8)
    figure.suptitle("GCDv2 garment-disjoint frozen test — learned raster-observable panel geometry", fontsize=15, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140, facecolor="white")
    plt.close(figure)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
