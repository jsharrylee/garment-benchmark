from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import GARMENT_ROLES
from benchmark.drafting_semantics.multiview_pattern_semantics import (
    PATTERN_TARGET_NAMES,
    TargetStandardizer,
    VIEW_NAMES,
    build_multiview_pattern_model,
    multiview_batch,
    read_multiview_pattern_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render actual 4-view inputs, paired 2D patterns, and structural predictions.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--semantic-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multiview_pattern_semantics_resnet50.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/prediction_board.png"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/prediction_board.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    standardizer = TargetStandardizer(
        tuple(checkpoint["target_standardizer"]["means"]),
        tuple(checkpoint["target_standardizer"]["standard_deviations"]),
    )
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = build_multiview_pattern_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    examples = tuple(
        item
        for item in read_multiview_pattern_examples(args.index, args.split, args.semantic_records, args.features)
        if item.split == "test"
    )
    predictions = []
    with torch.no_grad():
        for example in examples:
            batch = multiview_batch((example,), standardizer)
            output = model(torch.from_numpy(batch["view_features"]).to(device))
            category = int(output["category_logits"].argmax(dim=-1).cpu().item())
            pattern = standardizer.decode(output["pattern_prediction"].cpu().float().numpy())[0]
            normalized_error = np.mean(
                np.abs(pattern - example.pattern_target)
                / np.asarray(standardizer.standard_deviations, dtype=np.float32)
            )
            predictions.append((example, category, pattern, float(normalized_error)))

    selected = []
    for category_name in GARMENT_ROLES:
        category_id = GARMENT_ROLES.index(category_name)
        candidates = [value for value in predictions if value[0].category_target == category_id and value[1] == category_id]
        if not candidates:
            candidates = [value for value in predictions if value[0].category_target == category_id]
        candidates.sort(key=lambda value: (value[3], value[0].sample_id))
        selected.append(candidates[len(candidates) // 2])

    fig = plt.figure(figsize=(18, 3.5 * len(selected)), constrained_layout=True)
    grid = fig.add_gridspec(len(selected), 6, width_ratios=(1, 1, 1, 1, 1.5, 1.55))
    manifest_rows = []
    for row, (example, predicted_category, predicted_pattern, normalized_error) in enumerate(selected):
        for column, (view_name, path) in enumerate(zip(VIEW_NAMES, example.view_paths)):
            ax = fig.add_subplot(grid[row, column])
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(view_name, fontsize=9)
            ax.axis("off")
        pattern_ax = fig.add_subplot(grid[row, 4])
        pattern_png = Path("data/processed/garmentcode_v2/batch_0_full") / example.sample_id / f"{example.sample_id}_pattern.png"
        if pattern_png.is_file():
            with Image.open(pattern_png) as image:
                pattern_ax.imshow(image.convert("RGB"))
        else:
            pattern_ax.text(0.5, 0.5, "pattern PNG unavailable", ha="center", va="center")
        pattern_ax.set_title("GROUND-TRUTH 2D target (not generated)", fontsize=9)
        pattern_ax.axis("off")

        text_ax = fig.add_subplot(grid[row, 5])
        text_ax.axis("off")
        truth = example.pattern_target
        present = [
            index
            for index, name in enumerate(PATTERN_TARGET_NAMES)
            if index >= 2 and truth[index] > 0
        ]
        present.sort(key=lambda index: (-truth[index], PATTERN_TARGET_NAMES[index]))
        shown = [0, 1, *present[:6]]
        lines = [
            f"{example.sample_id}",
            f"garment: {GARMENT_ROLES[example.category_target]} → {GARMENT_ROLES[predicted_category]}",
            f"mean normalized error: {normalized_error:.3f}",
            "",
            "2D inventory    target → prediction",
        ]
        for index in shown:
            label = PATTERN_TARGET_NAMES[index].replace("panel:", "P:").replace("path:", "L:")
            lines.append(f"{label:<24} {truth[index]:5.1f} → {predicted_pattern[index]:5.1f}")
        text_ax.text(0.0, 0.98, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=8.2)
        manifest_rows.append(
            {
                "sample_id": example.sample_id,
                "target_category": GARMENT_ROLES[example.category_target],
                "predicted_category": GARMENT_ROLES[predicted_category],
                "mean_normalized_pattern_error": normalized_error,
                "target": {PATTERN_TARGET_NAMES[index]: float(truth[index]) for index in shown},
                "prediction": {PATTERN_TARGET_NAMES[index]: float(predicted_pattern[index]) for index in shown},
                "view_paths": list(example.view_paths),
                "pattern_png": str(pattern_png),
            }
        )
    fig.suptitle(
        "Actual four-view GCDv2 renders → frozen ResNet-50 + view Transformer → 2D semantic inventory\n"
        "Representative correctly-classified frozen-test sample nearest each category median error",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)
    args.manifest.write_text(json.dumps({"records": manifest_rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": manifest_rows}, indent=2))


if __name__ == "__main__":
    main()
