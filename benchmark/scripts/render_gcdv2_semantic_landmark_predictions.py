from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from benchmark.drafting_semantics.dataset import padded_batch, panel_examples, read_records
from benchmark.drafting_semantics.decoding import decode_named_landmarks
from benchmark.drafting_semantics.model import build_model
from benchmark.drafting_semantics.schema import EDGE_ROLES


PALETTE = {
    "neckline": "#d81b60",
    "shoulder": "#00897b",
    "armhole": "#7b1fa2",
    "center_front": "#fb8c00",
    "center_back": "#f4511e",
    "side_seam": "#1e88e5",
    "waistline": "#6d4c41",
    "dart_leg": "#7cb342",
    "other": "#b0bec5",
}


def _draw(axis, example, roles, landmarks, title):
    panel = example.panel
    for edge, role in zip(panel.edges, roles, strict=True):
        axis.plot(
            [edge.start_cm[0], edge.end_cm[0]],
            [edge.start_cm[1], edge.end_cm[1]],
            color=PALETTE.get(role, "#455a64"),
            linewidth=3,
        )
    for name, point in landmarks.items():
        axis.scatter(*point, color="#111111", edgecolor="white", linewidth=0.7, s=38, zorder=4)
        axis.annotate(name, point, xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render held-out GCDv2 semantic edge and named-landmark predictions.")
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/records.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_neurosymbolic/edge_semantics.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_training/semantic_landmark_test_10.png"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    records = read_records(args.records)
    examples = list(panel_examples(records, splits={"test"}, include_stitch_features=False))
    chosen = random.Random(args.seed).sample(examples, min(args.count, len(examples)))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    figure, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.2 * len(chosen)), constrained_layout=True)
    for row, example in enumerate(chosen):
        features, _, valid, panel_roles = padded_batch((example,), int(checkpoint["config"]["maximum_edges"]))
        with torch.no_grad():
            logits = model(torch.from_numpy(features), torch.from_numpy(valid), torch.from_numpy(panel_roles))[0, : len(example.targets)]
        predicted = logits.argmax(-1).numpy()
        target_roles = [EDGE_ROLES[int(value)] for value in example.targets]
        predicted_roles = [EDGE_ROLES[int(value)] for value in predicted]
        target_landmarks = {item.name: item.xy_cm for item in example.panel.landmarks if item.name in {"FNP", "BNP", "SNP", "SP"}}
        predicted_landmarks = decode_named_landmarks(example.panel, predicted_roles)
        raw_roles = ["other"] * len(example.targets)
        _draw(axes[row, 0], example, raw_roles, {}, f"INPUT vector panel\n{example.sample_id}:{example.panel.role}")
        _draw(axes[row, 1], example, target_roles, target_landmarks, "rule-derived target")
        _draw(axes[row, 2], example, predicted_roles, predicted_landmarks, "Transformer prediction")
    figure.suptitle("GCDv2 official frozen test — semantic edges → FNP/BNP/SNP/SP shared-point decoding", fontsize=14, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor="white")
    plt.close(figure)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
