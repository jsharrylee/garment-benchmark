from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import GARMENT_ROLES
from benchmark.drafting_semantics.multiview_element_geometry import (
    GEOMETRY_TARGET_NAMES,
    PRESENCE_TARGET_NAMES,
    MaskedTargetStandardizer,
    build_multiview_geometry_model,
    multiview_geometry_batch,
    read_multiview_geometry_examples,
)
from benchmark.drafting_semantics.multiview_pattern_semantics import VIEW_NAMES


# Keep the board compact while preferring measurements that explain a garment's
# main silhouette.  Missing elements (for example a sleeve on a sleeveless top)
# are skipped rather than displayed as zero-valued geometry.
DISPLAY_TARGETS_BY_CATEGORY = {
    "top": (
        "panel:front_bodice:mean_major_extent",
        "panel:front_bodice:mean_minor_extent",
        "panel:front_bodice:mean_polygon_area",
        "path:neckline:mean_length",
        "path:armhole:mean_length",
        "path:armhole:mean_primitive_curvedness",
        "path:sleeve_head:mean_length",
        "seam:sleeve_head_to_armhole_ratio",
    ),
    "pants": (
        "panel:front_pants:mean_major_extent",
        "panel:front_pants:mean_minor_extent",
        "panel:front_pants:mean_polygon_area",
        "path:waistline:mean_length",
        "path:inseam:mean_length",
        "path:outseam:mean_length",
        "path:crotch_curve:mean_length",
        "path:crotch_curve:mean_primitive_curvedness",
    ),
    "skirt": (
        "panel:front_skirt:mean_major_extent",
        "panel:front_skirt:mean_minor_extent",
        "panel:front_skirt:mean_polygon_area",
        "path:waistline:mean_length",
        "path:side_seam:mean_length",
        "path:hemline:mean_length",
    ),
    "dress": (
        "panel:front_bodice:mean_major_extent",
        "panel:front_bodice:mean_minor_extent",
        "panel:front_skirt:mean_major_extent",
        "panel:front_skirt:mean_minor_extent",
        "path:neckline:mean_length",
        "path:armhole:mean_length",
        "path:side_seam:mean_length",
        "path:hemline:mean_length",
    ),
    "jumpsuit": (
        "panel:front_bodice:mean_major_extent",
        "panel:front_bodice:mean_minor_extent",
        "panel:front_pants:mean_major_extent",
        "panel:front_pants:mean_minor_extent",
        "path:armhole:mean_length",
        "path:inseam:mean_length",
        "path:crotch_curve:mean_length",
        "seam:sleeve_head_to_armhole_ratio",
    ),
}


def _presence_name_for_geometry(target_name: str) -> str:
    fields = target_name.split(":")
    if fields[0] in {"panel", "path"}:
        return ":".join(fields[:2])
    return "seam:sleeve_head_to_armhole"


def _short_target_name(target_name: str) -> str:
    fields = target_name.split(":")
    if fields[0] == "panel":
        component = {
            "mean_major_extent": "major extent",
            "mean_minor_extent": "minor extent",
            "mean_polygon_area": "polygon area",
        }[fields[2]]
        return f"P {fields[1]} {component}"
    if fields[0] == "path":
        component = {
            "mean_length": "length",
            "mean_chord": "chord",
            "mean_primitive_curvedness": "curvedness",
        }[fields[2]]
        return f"L {fields[1]} {component}"
    return "S sleeve-head / armhole"


def _short_presence_name(presence_name: str) -> str:
    prefix, role = presence_name.split(":", 1)
    marker = {"panel": "P", "path": "L", "seam": "S"}[prefix]
    if prefix == "seam":
        role = "sleeve/armhole"
    return f"{marker} {role}"


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def _display_indices(category_name: str, mask: np.ndarray, limit: int = 8) -> tuple[int, ...]:
    preferred = [
        GEOMETRY_TARGET_NAMES.index(name)
        for name in DISPLAY_TARGETS_BY_CATEGORY[category_name]
        if name in GEOMETRY_TARGET_NAMES
    ]
    selected = [index for index in preferred if bool(mask[index])]
    # Some randomized recipes omit a preferred element.  Fill any remaining
    # rows with other observed measurements, keeping the board truthful.
    if len(selected) < min(6, limit):
        for index, present in enumerate(mask):
            if present and index not in selected:
                selected.append(index)
                if len(selected) >= limit:
                    break
    return tuple(selected[:limit])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render four-view inputs, paired ground-truth patterns, continuous predictions, and role attention."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"),
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json")
    )
    parser.add_argument(
        "--semantic-records",
        type=Path,
        default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drafting_semantics/multiview_element_geometry_resnet50.pt"),
    )
    parser.add_argument(
        "--pattern-images-root",
        type=Path,
        default=Path("data/processed/garmentcode_v2/batch_0_full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_element_geometry/prediction_board.png"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_element_geometry/prediction_board.json"),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    standardizer = MaskedTargetStandardizer(
        tuple(checkpoint["target_standardizer"]["means"]),
        tuple(checkpoint["target_standardizer"]["standard_deviations"]),
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_multiview_geometry_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)

    examples = tuple(
        item
        for item in read_multiview_geometry_examples(
            args.index, args.split, args.semantic_records, args.features
        )
        if item.split == "test"
    )
    deviations = np.asarray(standardizer.standard_deviations, dtype=np.float32)
    predictions: list[dict] = []
    batch_size = int(config.get("batch_size", 96))
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            current = examples[start : start + batch_size]
            batch = multiview_geometry_batch(current, standardizer)
            output = model(
                torch.from_numpy(batch["view_features"]).to(device), capture_attention=True
            )
            categories = output["category_logits"].argmax(dim=-1).cpu().numpy()
            geometry = standardizer.decode(
                output["geometry_prediction"].cpu().float().numpy()
            )
            presence = output["presence_logits"].sigmoid().cpu().float().numpy()
            # Use the final decoder layer.  Its per-head cross-attention is
            # [sample, head, semantic-role query, front/back/left/right].
            role_attention = output["role_attention"][-1].cpu().float().numpy().mean(axis=1)
            for offset, example in enumerate(current):
                normalized = np.abs(geometry[offset] - example.geometry_target) / deviations
                valid = example.geometry_mask
                error = float(normalized[valid].mean())
                predictions.append(
                    {
                        "example": example,
                        "category": int(categories[offset]),
                        "geometry": geometry[offset],
                        "presence": presence[offset],
                        "role_attention": role_attention[offset],
                        "normalized_error": error,
                    }
                )

    selected: list[dict] = []
    for category_name in GARMENT_ROLES:
        category_id = GARMENT_ROLES.index(category_name)
        candidates = [
            value
            for value in predictions
            if value["example"].category_target == category_id and value["category"] == category_id
        ]
        if not candidates:
            candidates = [
                value for value in predictions if value["example"].category_target == category_id
            ]
        if not candidates:
            raise RuntimeError(f"test split has no {category_name} sample")
        candidates.sort(key=lambda value: (value["normalized_error"], value["example"].sample_id))
        selected.append(candidates[len(candidates) // 2])

    fig = plt.figure(figsize=(24, 4.25 * len(selected)), constrained_layout=True)
    grid = fig.add_gridspec(
        len(selected), 6, width_ratios=(1, 1, 1, 1, 1.55, 3.45)
    )
    manifest_rows = []
    for row, result in enumerate(selected):
        example = result["example"]
        truth = example.geometry_target
        predicted = result["geometry"]
        category_name = GARMENT_ROLES[example.category_target]
        predicted_category_name = GARMENT_ROLES[result["category"]]
        for column, (view_name, path) in enumerate(zip(VIEW_NAMES, example.view_paths)):
            ax = fig.add_subplot(grid[row, column])
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"{view_name} input", fontsize=9)
            ax.axis("off")

        pattern_ax = fig.add_subplot(grid[row, 4])
        pattern_png = (
            args.pattern_images_root
            / example.sample_id
            / f"{example.sample_id}_pattern.png"
        )
        if pattern_png.is_file():
            with Image.open(pattern_png) as image:
                pattern_ax.imshow(image.convert("RGB"))
        else:
            pattern_ax.text(
                0.5, 0.5, "pattern PNG unavailable", ha="center", va="center", fontsize=9
            )
        pattern_ax.set_title("GROUND-TRUTH 2D target (not generated)", fontsize=9)
        pattern_ax.axis("off")

        display_indices = _display_indices(category_name, example.geometry_mask)
        descriptor_lines = [
            example.sample_id,
            f"garment: {category_name} -> {predicted_category_name}",
            f"masked normalized MAE: {result['normalized_error']:.3f}",
            "",
            "role-aggregated normalized descriptor",
            "target -> prediction",
        ]
        descriptor_payload = {}
        for index in display_indices:
            name = GEOMETRY_TARGET_NAMES[index]
            descriptor_lines.append(
                f"{_short_target_name(name):<33} {truth[index]:6.3f} -> {predicted[index]:6.3f}"
            )
            descriptor_payload[name] = {
                "target": float(truth[index]),
                "prediction": float(predicted[index]),
            }

        presence_names = _ordered_unique(
            _presence_name_for_geometry(GEOMETRY_TARGET_NAMES[index])
            for index in display_indices
        )
        attention_lines = [
            "final role-query attention, mean of 8 heads",
            "semantic role             front  back  left right",
        ]
        attention_payload = {}
        for presence_name in presence_names:
            query_index = PRESENCE_TARGET_NAMES.index(presence_name)
            weights = result["role_attention"][query_index]
            attention_lines.append(
                f"{_short_presence_name(presence_name):<24} "
                + " ".join(f"{100.0 * value:5.1f}" for value in weights)
            )
            attention_payload[presence_name] = {
                view_name: float(weights[index]) for index, view_name in enumerate(VIEW_NAMES)
            }
        attention_lines.extend(
            (
                "",
                "Values are scale-normalized aggregates.",
                "Attention is global-view weight, not image-region evidence.",
                "No panel vertices, splines, stitches, or CAD are generated.",
            )
        )

        text_ax = fig.add_subplot(grid[row, 5])
        text_ax.axis("off")
        text_ax.text(
            0.00,
            0.98,
            "\n".join(descriptor_lines),
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.2,
        )
        text_ax.text(
            0.55,
            0.98,
            "\n".join(attention_lines),
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.2,
        )
        manifest_rows.append(
            {
                "sample_id": example.sample_id,
                "target_category": category_name,
                "predicted_category": predicted_category_name,
                "masked_normalized_mae": result["normalized_error"],
                "descriptors": descriptor_payload,
                "role_query_attention": attention_payload,
                "view_paths": list(example.view_paths),
                "ground_truth_pattern_png": str(pattern_png),
            }
        )

    fig.suptitle(
        "Actual four-view GCDv2 renders -> role-query Transformer -> continuous semantic 2D descriptors\n"
        "Correctly classified frozen-test example nearest each category's median error; central 2D pattern is paired ground truth",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)
    args.manifest.write_text(
        json.dumps(
            {
                "status": "VISUAL_AUDIT_ONLY_NOT_PATTERN_GENERATION",
                "selection": "correctly categorized frozen-test sample nearest category median masked normalized MAE",
                "records": manifest_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(args.manifest),
                "records": [
                    {
                        "sample_id": row["sample_id"],
                        "category": row["target_category"],
                        "masked_normalized_mae": row["masked_normalized_mae"],
                    }
                    for row in manifest_rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
