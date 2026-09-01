from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from benchmark.drafting_semantics.merged_visible_learning import LANDMARK_NAMES, MERGED_EDGE_ROLES, MergedVisibleDataset, build_merged_semantic_model, decode_landmarks
from benchmark.gcdv2_exact.intrinsic_graph_learning import nearest_contour_indices


COLORS = {"other": "#b0bec5", "neckline": "#d81b60", "shoulder": "#00897b", "armhole": "#7b1fa2", "center_front": "#fb8c00", "center_back": "#f4511e", "side_seam": "#1e88e5", "waistline": "#6d4c41", "dart_leg": "#7cb342"}


def draw_graph(axis, contour, vertices, roles, landmarks, title):
    indices = nearest_contour_indices(contour, vertices)
    for local, start in enumerate(indices):
        end = indices[(local + 1) % len(indices)]
        points = np.vstack((contour[start:], contour[: end + 1])) if end <= start else contour[start : end + 1]
        axis.plot(points[:, 0], points[:, 1], color=COLORS[MERGED_EDGE_ROLES[int(roles[local])]], linewidth=2.7)
    axis.scatter(vertices[:, 0], vertices[:, 1], color="#111", edgecolor="white", s=22, linewidth=0.5, zorder=4)
    for name, point in landmarks.items():
        axis.scatter(*point, color="white", edgecolor="#111", s=38, zorder=5)
        axis.annotate(name, point, xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    axis.set_aspect("equal"); axis.invert_yaxis(); axis.axis("off"); axis.set_title(title, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render predicted-contour merged semantic results.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/merged_semantics.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/metadata.jsonl"))
    parser.add_argument("--panel-index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--predicted-contours", type=Path, default=Path("artifacts/gcdv2_predicted_contours_v1/predicted_contours.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_end_to_end/merged_visible_semantics.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_end_to_end/frozen_test_semantics_10.png"))
    parser.add_argument("--count", type=int, default=10); parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    raw = np.load(args.dataset); arrays = {key: raw[key] for key in raw.files}
    metadata = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    panel_rows = [json.loads(line) for line in args.panel_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    contours = np.load(args.predicted_contours)["contours"]
    test_indices = np.flatnonzero(arrays["splits"] == 2).tolist(); chosen = random.Random(args.seed).sample(test_indices, min(args.count, len(test_indices)))
    dataset = MergedVisibleDataset(arrays, chosen, augment=False)
    model = build_merged_semantic_model(); model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state"]); model.eval()
    figure, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.2 * len(chosen)), constrained_layout=True)
    for display_row, (dataset_row, source) in enumerate(zip(dataset, chosen, strict=True)):
        count = int(dataset_row["valid"].sum()); vertices = dataset_row["vertices_uv"][:count]; contour = contours[metadata[source]["source_panel_index"]]
        with Image.open(panel_rows[metadata[source]["source_panel_index"]]["input_panel_image"]) as image:
            axes[display_row, 0].imshow(image.convert("L"), cmap="gray")
        axes[display_row, 0].axis("off"); axes[display_row, 0].set_title(f"panel.png input\n{metadata[source]['panel_uid']}", fontsize=9)
        target_roles = dataset_row["roles"][:count]
        target_landmarks = {name: dataset_row["landmark_uv"][index] for index, name in enumerate(LANDMARK_NAMES) if dataset_row["landmark_mask"][index]}
        with torch.no_grad():
            logits = model(torch.from_numpy(dataset_row["features"])[None], torch.from_numpy(dataset_row["valid"])[None], torch.tensor([dataset_row["panel_role"]]))
        predicted_roles = logits[0, :count].argmax(-1).numpy(); predicted_landmarks = decode_landmarks(predicted_roles, vertices, dataset_row["panel_role"])
        draw_graph(axes[display_row, 1], contour, vertices, target_roles, target_landmarks, "merged semantic target")
        draw_graph(axes[display_row, 2], contour, vertices, predicted_roles, predicted_landmarks, "Transformer prediction")
    figure.suptitle("Learned mask contour → intrinsic merged graph → semantic edges → named landmarks", fontsize=14, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(args.output, dpi=150, facecolor="white"); plt.close(figure)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
