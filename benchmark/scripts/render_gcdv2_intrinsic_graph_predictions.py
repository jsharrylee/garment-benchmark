from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from benchmark.gcdv2_exact.intrinsic_graph_learning import (
    PRIMITIVES,
    build_corner_model,
    build_segment_model,
    intrinsic_contour_features,
    intrinsic_segment_features,
    segment_between,
    select_cyclic_peaks,
)


COLORS = {"line": "#00a6a6", "quadratic_bezier": "#f5a623", "cubic_bezier": "#ef4c9a", "circular_arc": "#65b95b"}


def _draw_segments(axis, contour, indices, kinds=None):
    for local, start in enumerate(indices):
        end = indices[(local + 1) % len(indices)]
        if end <= start:
            points = np.vstack((contour[start:], contour[: end + 1]))
        else:
            points = contour[start : end + 1]
        kind = kinds[local] if kinds else "line"
        axis.plot(points[:, 0], points[:, 1], color=COLORS.get(kind, "#455a64") if kinds else "#90a4ae", linewidth=2.2)
    axis.scatter(contour[indices, 0], contour[indices, 1], s=28, color="#111111", edgecolor="white", linewidth=0.5, zorder=3)
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render intrinsic visible graph predictions without absolute-coordinate model inputs.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_v1/intrinsic_graph.npz"))
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gcdv2_intrinsic_graph"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_training/frozen_test_10.png"))
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    data = np.load(args.dataset)
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_indices = np.flatnonzero(data["panel_splits"] == 2).tolist()
    chosen = random.Random(args.seed).sample(test_indices, min(args.count, len(test_indices)))
    corner_checkpoint = torch.load(args.checkpoint_dir / "visible_corners.pt", map_location="cpu", weights_only=True)
    segment_checkpoint = torch.load(args.checkpoint_dir / "segment_geometry.pt", map_location="cpu", weights_only=True)
    corner_model, segment_model = build_corner_model(), build_segment_model()
    corner_model.load_state_dict(corner_checkpoint["model_state"])
    segment_model.load_state_dict(segment_checkpoint["model_state"])
    corner_model.eval(); segment_model.eval()

    figure, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.2 * len(chosen)), constrained_layout=True)
    for row_number, panel_index in enumerate(chosen):
        contour = data["contours"][panel_index].astype(np.float32)
        target_indices = np.flatnonzero(data["corner_targets"][panel_index] >= 0.999).tolist()
        feature = torch.from_numpy(intrinsic_contour_features(contour))[None]
        with torch.no_grad():
            corner_output = corner_model(feature)
        predicted_count = int(np.clip(corner_output["count_logits"].argmax(-1).item(), 3, 36))
        predicted_indices = select_cyclic_peaks(corner_output["corner_logits"][0].sigmoid().numpy(), predicted_count)
        segment_inputs = []
        for local, start in enumerate(predicted_indices):
            segment = segment_between(contour, start, predicted_indices[(local + 1) % len(predicted_indices)])
            current, _ = intrinsic_segment_features(segment)
            segment_inputs.append(current)
        with torch.no_grad():
            segment_output = segment_model(torch.from_numpy(np.asarray(segment_inputs, np.float32)))
        predicted_kinds = [PRIMITIVES[int(value)] for value in segment_output["primitive_logits"].argmax(-1).numpy()]

        axes[row_number, 0].plot(contour[:, 0], contour[:, 1], color="#263238", linewidth=2)
        axes[row_number, 0].set_aspect("equal"); axes[row_number, 0].invert_yaxis(); axes[row_number, 0].axis("off")
        axes[row_number, 0].set_title(f"intrinsic contour input\n{rows[panel_index]['panel_uid']}", fontsize=9)
        _draw_segments(axes[row_number, 1], contour, target_indices)
        axes[row_number, 1].set_title(f"visible-graph target · {len(target_indices)} vertices", fontsize=9)
        _draw_segments(axes[row_number, 2], contour, predicted_indices, predicted_kinds)
        axes[row_number, 2].set_title(f"prediction · {len(predicted_indices)} vertices · closed cycle", fontsize=9)
    figure.suptitle("GCDv2 frozen test — intrinsic contour graph (no absolute x/y model input)", fontsize=15, weight="bold")
    figure.text(0.5, 0.003, "Predicted edge colors: cyan=line · orange=quadratic · pink=cubic · green=arc", ha="center", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor="white")
    plt.close(figure)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
