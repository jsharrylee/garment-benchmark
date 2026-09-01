from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    semantic_target_from_basic_block,
    semantic_target_from_pattern_document,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INDEX,
    SEMANTIC_QUERY_INVENTORY,
)
from benchmark.pattern_pipeline.four_view_semantic_inference import (
    CANONICAL_VIEW_ORDER,
    FourViewFeatureBundle,
    infer_provisional_basic_pattern,
    load_four_view_student_checkpoint,
)


def _target_arrays(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM)
    target = np.full(shape, np.nan, dtype=np.float64)
    predicted = np.full(shape, np.nan, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.bool_)
    for item in row["queries"]:
        index = SEMANTIC_QUERY_INDEX[item["query"]]
        query = SEMANTIC_QUERY_INVENTORY[index]
        for channel, name in enumerate(query.coordinate_names):
            predicted[index, channel] = float(item["predicted_coordinates"][name])
            value = item["target_coordinates"][name]
            active = bool(item["coordinate_supervision_mask"][name]) and value is not None
            if active:
                target[index, channel] = float(value)
                mask[index, channel] = True
    return target, predicted, mask


def _mean_error(values: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(values - target)[mask].mean()) if bool(mask.any()) else float("nan")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate student prediction -> bounded BasicBlock edit on all unseen test IDs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    prediction_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = prediction_payload["rows"]
    with np.load(args.features, allow_pickle=False) as archive:
        sample_ids = tuple(str(value) for value in archive["sample_ids"].tolist())
        features = archive["features"]
        if features.ndim != 4 or features.shape[1] != 4:
            raise ValueError("evaluation requires spatial features [N,4,P,C]")
        feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        if len(feature_index) != len(sample_ids):
            raise ValueError("feature sample IDs are duplicated")
        missing = [row["sample_id"] for row in rows if row["sample_id"] not in feature_index]
        if missing:
            raise ValueError(f"test predictions are missing feature rows: {len(missing)}")

        loaded = load_four_view_student_checkpoint(args.checkpoint, device=args.device)
        per_category: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        per_query: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        statuses: Counter[str] = Counter()
        projection_scales: Counter[str] = Counter()
        gated: Counter[str] = Counter()
        landmark_edits: list[int] = []
        path_edits: list[int] = []
        requested_landmark_edits: list[int] = []
        requested_path_edits: list[int] = []
        started = time.perf_counter()
        for row in rows:
            category = str(row["category"])
            bundle = FourViewFeatureBundle(
                view_order=CANONICAL_VIEW_ORDER,
                spatial_features=np.asarray(features[feature_index[row["sample_id"]]]),
            )
            result = infer_provisional_basic_pattern(
                loaded,
                bundle,
                category=category,
            )
            statuses[result.receipt["status"]] += 1
            projection_scale = float(
                result.receipt.get("semantic_projection", {}).get("selected_scale", 0.0)
            )
            projection_scales[f"{projection_scale:.2f}"] += 1
            landmark_edits.append(len(result.plan.landmark_residuals))
            path_edits.append(len(result.plan.path_residuals))
            requested = result.receipt.get("requested_residual_plan", {})
            requested_landmark_edits.append(
                len(requested.get("landmark_residuals", {}))
            )
            requested_path_edits.append(len(requested.get("path_residuals", {})))
            gated.update(result.plan.gated_queries.values())

            anchor = build_basic_block(category)
            anchor_target = semantic_target_from_basic_block(anchor)
            edited_target = semantic_target_from_pattern_document(
                result.document,
                category=category,
                source="four_view_semantic_edited_provisional_block",
                provenance_status="PROVISIONAL_EXPERT_REVIEW",
                source_y_axis_down=True,
            )
            target, direct_prediction, target_mask = _target_arrays(row)
            comparable = (
                target_mask
                & anchor_target.coordinate_mask
                & edited_target.coordinate_mask
            )
            anchor_error = _mean_error(anchor_target.coordinates, target, comparable)
            edited_error = _mean_error(edited_target.coordinates, target, comparable)
            direct_error = _mean_error(direct_prediction, target, target_mask)
            per_category[category]["anchor_error"].append(anchor_error)
            per_category[category]["edited_error"].append(edited_error)
            per_category[category]["direct_student_error"].append(direct_error)
            per_category[category]["improved"].append(float(edited_error < anchor_error))

            for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
                active = comparable[index]
                if not bool(active.any()):
                    continue
                query_anchor = float(
                    np.abs(anchor_target.coordinates[index] - target[index])[active].mean()
                )
                query_edited = float(
                    np.abs(edited_target.coordinates[index] - target[index])[active].mean()
                )
                per_query[query.key]["anchor"].append(query_anchor)
                per_query[query.key]["edited"].append(query_edited)

    category_summary = {}
    for category, values in sorted(per_category.items()):
        anchor = np.asarray(values["anchor_error"])
        edited = np.asarray(values["edited_error"])
        category_summary[category] = {
            "sample_count": len(anchor),
            "anchor_coordinate_mae": _summary(values["anchor_error"]),
            "edited_coordinate_mae": _summary(values["edited_error"]),
            "direct_student_coordinate_mae": _summary(values["direct_student_error"]),
            "sample_improvement_rate": float(np.mean(values["improved"])),
            "aggregate_relative_change_percent": float(
                100.0 * (edited.mean() - anchor.mean()) / max(anchor.mean(), 1e-12)
            ),
        }
    query_summary = []
    for query_key, values in per_query.items():
        before = float(np.mean(values["anchor"]))
        after = float(np.mean(values["edited"]))
        query_summary.append(
            {
                "query": query_key,
                "sample_count": len(values["anchor"]),
                "anchor_mae": before,
                "edited_mae": after,
                "absolute_improvement": before - after,
                "relative_improvement_percent": 100.0 * (before - after) / max(before, 1e-12),
            }
        )
    query_summary.sort(key=lambda item: item["absolute_improvement"], reverse=True)
    all_anchor = [value for group in per_category.values() for value in group["anchor_error"]]
    all_edited = [value for group in per_category.values() for value in group["edited_error"]]
    payload = {
        "schema_version": "four-view-semantic-edit-evaluation/v1",
        "status": "COMPLETE_SAME_GENERATOR_SAMPLE_ID_UNSEEN",
        "sample_count": len(rows),
        "view_order": list(CANONICAL_VIEW_ORDER),
        "contains_source_images": False,
        "contains_source_paths": False,
        "overall": {
            "anchor_coordinate_mae": _summary(all_anchor),
            "edited_coordinate_mae": _summary(all_edited),
            "sample_improvement_rate": float(
                np.mean(
                    [after < before for before, after in zip(all_anchor, all_edited)]
                )
            ),
            "aggregate_relative_change_percent": float(
                100.0
                * (np.mean(all_edited) - np.mean(all_anchor))
                / max(np.mean(all_anchor), 1e-12)
            ),
        },
        "per_category": category_summary,
        "semantic_edit_selector": dict(
            loaded.confidence_receipt.get("semantic_edit_selector", {})
        ),
        "edit_status_counts": dict(statuses),
        "semantic_projection_scale_counts": dict(projection_scales),
        "mean_requested_landmark_edits": float(np.mean(requested_landmark_edits)),
        "mean_requested_path_edits": float(np.mean(requested_path_edits)),
        "mean_landmark_edits": float(np.mean(landmark_edits)),
        "mean_path_edits": float(np.mean(path_edits)),
        "planner_gated_reason_counts": dict(gated),
        "best_query_improvements": query_summary[:15],
        "worst_query_changes": list(reversed(query_summary[-15:])),
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Targets and test images are same-generator GarmentCodeData v2 sample-ID unseen. "
            "The editable anchor is PROVISIONAL_EXPERT_REVIEW; this is neither family-disjoint "
            "nor cross-domain evidence and does not establish industrial pattern accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
