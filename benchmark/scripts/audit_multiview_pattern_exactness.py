from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multiview_pattern_semantics import (
    PANEL_COUNT_NAMES,
    PATTERN_TARGET_NAMES,
    SEMANTIC_COUNT_NAMES,
    TargetStandardizer,
    build_multiview_pattern_model,
    multiview_batch,
    read_multiview_pattern_examples,
)


def _subset_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    predicted_category: np.ndarray,
    target_category: np.ndarray,
    standard_deviations: np.ndarray,
) -> dict[str, float | int]:
    projected = np.rint(np.clip(predicted, 0.0, None)).astype(np.int64)
    expected = target.astype(np.int64)
    normalized = np.abs(predicted - target) / standard_deviations[None, :]
    panel_slice = slice(2, 2 + len(PANEL_COUNT_NAMES))
    path_slice = slice(2 + len(PANEL_COUNT_NAMES), len(PATTERN_TARGET_NAMES))
    return {
        "sample_count": int(len(target)),
        "category_accuracy": float(np.mean(predicted_category == target_category)),
        "mean_normalized_pattern_mae": float(normalized.mean()),
        "panel_count_mae": float(np.abs(predicted[:, 0] - target[:, 0]).mean()),
        "edge_count_mae": float(np.abs(predicted[:, 1] - target[:, 1]).mean()),
        "rounded_panel_count_exact_rate": float(np.mean(projected[:, 0] == expected[:, 0])),
        "rounded_edge_count_exact_rate": float(np.mean(projected[:, 1] == expected[:, 1])),
        "all_panel_role_counts_exact_rate": float(
            np.mean(np.all(projected[:, panel_slice] == expected[:, panel_slice], axis=1))
        ),
        "all_semantic_path_counts_exact_rate": float(
            np.mean(np.all(projected[:, path_slice] == expected[:, path_slice], axis=1))
        ),
        "all_29_counts_exact_rate": float(np.mean(np.all(projected == expected, axis=1))),
    }


def _relation(targets: np.ndarray, left: str, right: str, factor: float = 1.0) -> dict[str, object]:
    left_index = PATTERN_TARGET_NAMES.index(left)
    right_index = PATTERN_TARGET_NAMES.index(right)
    return {
        "relation": f"{left} == {factor:g} * {right}",
        "holds_for_all_records": bool(np.allclose(targets[:, left_index], factor * targets[:, right_index])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact and unseen-inventory behavior of the four-view baseline.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--semantic-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multiview_pattern_semantics_resnet50.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/exactness_audit.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    standardizer = TargetStandardizer(
        tuple(checkpoint["target_standardizer"]["means"]),
        tuple(checkpoint["target_standardizer"]["standard_deviations"]),
    )
    model = build_multiview_pattern_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    examples = read_multiview_pattern_examples(
        args.index, args.split, args.semantic_records, args.features
    )
    train = tuple(item for item in examples if item.split == "train")
    test = tuple(item for item in examples if item.split == "test")

    predictions: list[np.ndarray] = []
    category_predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(test), int(config["batch_size"])):
            batch = multiview_batch(test[start : start + int(config["batch_size"])], standardizer)
            output = model(torch.from_numpy(batch["view_features"]).to(device))
            predictions.append(
                standardizer.decode(output["pattern_prediction"].cpu().float().numpy())
            )
            category_predictions.append(output["category_logits"].argmax(dim=-1).cpu().numpy())
    predicted = np.concatenate(predictions)
    target = np.stack([item.pattern_target for item in test])
    category_prediction = np.concatenate(category_predictions)
    category_target = np.asarray([item.category_target for item in test], dtype=np.int64)
    train_targets = np.stack([item.pattern_target for item in train])
    train_signatures = {tuple(value.astype(np.int64).tolist()) for value in train_targets}
    seen = np.asarray(
        [tuple(value.astype(np.int64).tolist()) in train_signatures for value in target],
        dtype=bool,
    )
    deviations = np.asarray(standardizer.standard_deviations, dtype=np.float32)
    negative_values = predicted[predicted < 0]
    all_targets = np.stack([item.pattern_target for item in examples])
    centered = all_targets - all_targets.mean(axis=0, keepdims=True)

    payload = {
        "schema_version": "multiview-pattern-exactness-audit-1.0",
        "interpretation": (
            "All outputs are aggregate counts. Exactness after nonnegative rounding is a stricter deployment audit; "
            "it is not panel geometry, spline, landmark, or stitch-graph accuracy."
        ),
        "test": _subset_metrics(predicted, target, category_prediction, category_target, deviations),
        "seen_train_inventory": _subset_metrics(
            predicted[seen], target[seen], category_prediction[seen], category_target[seen], deviations
        ),
        "unseen_train_inventory": _subset_metrics(
            predicted[~seen], target[~seen], category_prediction[~seen], category_target[~seen], deviations
        ),
        "inventory_overlap": {
            "test_with_exact_train_signature": int(seen.sum()),
            "test_without_exact_train_signature": int((~seen).sum()),
            "exact_train_signature_rate": float(seen.mean()),
        },
        "raw_output_validity": {
            "samples_with_any_negative_count": int(np.any(predicted < 0, axis=1).sum()),
            "samples_with_any_negative_count_rate": float(np.any(predicted < 0, axis=1).mean()),
            "negative_value_count": int(len(negative_values)),
            "negative_value_median": float(np.median(negative_values)) if len(negative_values) else None,
            "minimum_raw_count": float(predicted.min()),
            "values_below_minus_point_five": int((predicted < -0.5).sum()),
        },
        "target_dependency_audit": {
            "target_dimension": len(PATTERN_TARGET_NAMES),
            "centered_matrix_rank": int(np.linalg.matrix_rank(centered)),
            "known_exact_relations": [
                _relation(all_targets, "panel:front_bodice", "panel:back_bodice"),
                _relation(all_targets, "path:center_front", "panel:front_bodice"),
                _relation(all_targets, "path:center_back", "panel:back_bodice"),
                _relation(all_targets, "panel:front_pants", "panel:back_pants"),
                _relation(all_targets, "path:outseam", "panel:front_pants", 2.0),
                _relation(all_targets, "path:crotch_curve", "panel:front_pants", 2.0),
                _relation(all_targets, "path:inseam", "panel:front_pants", 4.0),
                _relation(all_targets, "path:sleeve_head", "panel:sleeve"),
                _relation(all_targets, "path:sleeve_underarm", "panel:sleeve", 2.0),
                _relation(all_targets, "path:collar_attachment", "panel:collar"),
                _relation(all_targets, "path:cuff_attachment", "path:sleeve_hem"),
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

