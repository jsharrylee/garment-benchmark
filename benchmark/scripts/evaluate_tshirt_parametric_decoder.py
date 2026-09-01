from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.basic_semantic_targets import (
    BasicSemanticTarget,
    semantic_target_from_basic_block,
)
from benchmark.drafting_semantics.semantic_teacher_student import (
    MAX_COORDINATE_DIM,
    SEMANTIC_QUERY_INDEX,
    SEMANTIC_QUERY_INVENTORY,
)
from benchmark.drafting_semantics.tshirt_parametric_projection import (
    TSHIRT_DRAFT_PARAMETER_NAMES,
    TShirtProjectionConfig,
    fit_tshirt_drafting_parameters,
    materialize_tshirt_projection_graph,
)
from benchmark.drafting_semantics.tshirt_parametric_decoder import (
    PARAMETER_NAMES,
    CanonicalPatternGraph,
    audit_sampled_tshirt_document_sleeve_constraint,
)
from benchmark.pattern_pipeline.validation import validate_pattern


FROZEN_TSHIRT_TEST_IDS = (
    "rand_1JGSONFALQ",
    "rand_7EGKGR5KZ0",
    "rand_C1WYZBK90A",
    "rand_DGG86GBUL7",
    "rand_K4YVUO220R",
    "rand_KLX8NWAG66",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_arrays(
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inference inputs without consulting target fields or masks."""

    shape = (len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM)
    coordinates = np.full(shape, np.nan, dtype=np.float64)
    presence = np.zeros(len(SEMANTIC_QUERY_INVENTORY), dtype=np.float64)
    confidence = np.zeros(len(SEMANTIC_QUERY_INVENTORY), dtype=np.float64)
    query_mask = np.zeros(len(SEMANTIC_QUERY_INVENTORY), dtype=np.bool_)
    for item in row["queries"]:
        key = str(item["query"])
        index = SEMANTIC_QUERY_INDEX[key]
        query = SEMANTIC_QUERY_INVENTORY[index]
        query_mask[index] = True
        presence[index] = float(item["presence_probability"])
        confidence[index] = float(item["coordinate_confidence"])
        predicted = item["predicted_coordinates"]
        for channel, name in enumerate(query.coordinate_names):
            coordinates[index, channel] = float(predicted[name])
    return coordinates, presence, confidence, query_mask


def _target_arrays(
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM)
    coordinates = np.full(shape, np.nan, dtype=np.float64)
    presence = np.zeros(len(SEMANTIC_QUERY_INVENTORY), dtype=np.float64)
    mask = np.zeros(shape, dtype=np.bool_)
    query_mask = np.zeros(len(SEMANTIC_QUERY_INVENTORY), dtype=np.bool_)
    for item in row["queries"]:
        key = str(item["query"])
        index = SEMANTIC_QUERY_INDEX[key]
        query = SEMANTIC_QUERY_INVENTORY[index]
        supervised = bool(item["query_supervised_in_ground_truth"])
        query_mask[index] = supervised
        target_present = item["target_present"]
        presence[index] = 0.0 if target_present is None else float(bool(target_present))
        target = item["target_coordinates"]
        supervision = item["coordinate_supervision_mask"]
        for channel, name in enumerate(query.coordinate_names):
            value = target[name]
            active = bool(supervision[name]) and value is not None
            if active:
                coordinates[index, channel] = float(value)
                mask[index, channel] = True
    return coordinates, presence, mask, query_mask


def _semantic_mae(
    candidate: BasicSemanticTarget | np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    *,
    additional_mask: np.ndarray | None = None,
) -> float:
    if isinstance(candidate, BasicSemanticTarget):
        values = candidate.coordinates
        mask = target_mask & candidate.coordinate_mask
    else:
        values = np.asarray(candidate, dtype=np.float64)
        mask = target_mask & np.isfinite(values)
    if additional_mask is not None:
        mask &= additional_mask
    return float(np.abs(values - target)[mask].mean()) if bool(mask.any()) else float("nan")


def _kind_mask(kind: str) -> np.ndarray:
    values = np.zeros(
        (len(SEMANTIC_QUERY_INVENTORY), MAX_COORDINATE_DIM), dtype=np.bool_
    )
    for index, query in enumerate(SEMANTIC_QUERY_INVENTORY):
        if query.category == "tshirt" and query.kind == kind:
            values[index, : len(query.coordinate_names)] = True
    return values


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([item for item in values if np.isfinite(item)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
    }


def _aggregate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    anchor = np.asarray([float(row["anchor_coordinate_mae"]) for row in rows])
    candidate = np.asarray(values)
    return {
        f"{field}": _summary(values),
        "sample_improvement_count": int(np.sum(candidate < anchor)),
        "sample_improvement_rate": float(np.mean(candidate < anchor)),
        "aggregate_relative_change_vs_anchor_percent": float(
            100.0 * (candidate.mean() - anchor.mean()) / max(anchor.mean(), 1e-12)
        ),
    }


def _optimizer_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    receipts = [row[key] for row in rows]
    return {
        "success_count": int(sum(bool(item["optimizer"]["success"]) for item in receipts)),
        "mean_function_evaluations": float(
            np.mean([int(item["optimizer"]["function_evaluations"]) for item in receipts])
        ),
        "mean_selected_coordinate_count": float(
            np.mean([int(item["selected_coordinate_count"]) for item in receipts])
        ),
        "constraint_pass_count": int(
            sum(bool(item["constraint_audit"]["passed"]) for item in receipts)
        ),
        "maximum_total_sleeve_ease_absolute_error": float(
            max(
                float(item["constraint_audit"]["total_ease_absolute_error"])
                for item in receipts
            )
        ),
    }


def _canonical_graph_audit(graph: CanonicalPatternGraph) -> dict[str, Any]:
    """Return compact structural evidence without serializing source paths."""

    graph.validate()
    constraint = graph.sleeve_head_constraint
    return {
        "schema_version": graph.schema_version,
        "parameter_count": len(graph.parameters.to_vector()),
        "panel_count": len(graph.panels),
        "path_count": len(graph.paths),
        "stitch_count": len(graph.stitches),
        "symmetry_relation_count": len(graph.symmetry_relations),
        "instance_aware_path_count": sum("#" in path.id for path in graph.paths),
        "all_paths_instance_aware": all("#" in path.id for path in graph.paths),
        "exact_reflection_validated": True,
        "shared_landmark_references_validated": True,
        "sleeve_head_constraint": {
            "equation": constraint.equation,
            "front_armhole_length_cm": constraint.front_armhole_length_cm,
            "back_armhole_length_cm": constraint.back_armhole_length_cm,
            "sleeve_ease_cm": constraint.sleeve_ease_cm,
            "target_length_cm": constraint.target_length_cm,
            "actual_length_cm": constraint.actual_length_cm,
            "residual_cm": constraint.residual_cm,
            "tolerance_cm": constraint.tolerance_cm,
            "converged": constraint.converged,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen four-view semantics -> bounded T-shirt drafting "
            "parameters -> shared-point/seam-constrained BasicBlock."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern-output", type=Path)
    args = parser.parse_args()

    config = TShirtProjectionConfig()
    config.validate()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["category"] == "tshirt"]
    identifiers = tuple(sorted(str(row["sample_id"]) for row in rows))
    if identifiers != tuple(sorted(FROZEN_TSHIRT_TEST_IDS)):
        raise ValueError(
            "evaluation input is not the frozen six-sample T-shirt test split: "
            f"{identifiers}"
        )
    baseline_payload = None
    if args.baseline_evaluation is not None:
        baseline_payload = json.loads(
            args.baseline_evaluation.read_text(encoding="utf-8")
        )
        if baseline_payload.get("schema_version") != "four-view-semantic-edit-evaluation/v1":
            raise ValueError("baseline evaluation schema is not the guarded v1 contract")
        if baseline_payload.get("status") != "COMPLETE_SAME_GENERATOR_SAMPLE_ID_UNSEEN":
            raise ValueError("baseline evaluation status is not the frozen unseen contract")
        if tuple(baseline_payload.get("view_order", ())) != (
            "front",
            "back",
            "left",
            "right",
        ):
            raise ValueError("baseline evaluation view order is not canonical")
        baseline_tshirt = baseline_payload["per_category"]["tshirt"]
        if int(baseline_tshirt["sample_count"]) != len(rows):
            raise ValueError("baseline evaluation does not contain the frozen T-shirt split")
        if not np.isclose(float(baseline_tshirt["sample_improvement_rate"]), 0.0):
            raise ValueError("baseline T-shirt edit was not the declared zero-improvement gate")
        if not np.isclose(float(baseline_tshirt["aggregate_relative_change_percent"]), 0.0):
            raise ValueError("baseline T-shirt aggregate edit was not identity")

    pattern_output = args.pattern_output or args.output.parent / "patterns"
    pattern_output.mkdir(parents=True, exist_ok=True)
    anchor = semantic_target_from_basic_block(build_basic_block("tshirt"))
    per_sample: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        sample_id = str(row["sample_id"])
        predicted, predicted_presence, confidence, query_mask = _prediction_arrays(row)
        target, target_presence, target_mask, target_query_mask = _target_arrays(row)

        # Primary image-only inference.  This call cannot receive target
        # coordinates or target supervision masks by construction.
        student_projection = fit_tshirt_drafting_parameters(
            predicted,
            predicted_presence,
            confidence,
            query_mask=query_mask,
            config=config,
            sample_id=f"tshirt_parametric_student_{sample_id}",
        )
        student_semantic = semantic_target_from_basic_block(student_projection.block)
        student_graph = materialize_tshirt_projection_graph(
            student_projection,
            pattern_id=f"tshirt_parametric_student_{sample_id}",
            samples_per_cubic=49,
        )
        student_document = student_graph.to_pattern_document()
        student_validation = validate_pattern(student_document)
        student_sampled_constraint = audit_sampled_tshirt_document_sleeve_constraint(
            student_document,
            sleeve_ease_cm=student_graph.sleeve_head_constraint.sleeve_ease_cm,
            samples_per_cubic=student_graph.samples_per_cubic,
        )

        # Explicitly separate oracle representational ceiling.  It is fitted
        # only after the primary candidate is frozen and is never an inference
        # result.
        oracle_confidence = target_mask.any(axis=1).astype(np.float64)
        oracle_projection = fit_tshirt_drafting_parameters(
            target,
            target_presence,
            oracle_confidence,
            query_mask=target_query_mask,
            config=config,
            sample_id=f"tshirt_parametric_oracle_{sample_id}",
        )
        oracle_semantic = semantic_target_from_basic_block(oracle_projection.block)
        oracle_graph = materialize_tshirt_projection_graph(
            oracle_projection,
            pattern_id=f"tshirt_parametric_oracle_{sample_id}",
            samples_per_cubic=49,
        )
        oracle_document = oracle_graph.to_pattern_document()
        oracle_validation = validate_pattern(oracle_document)
        oracle_sampled_constraint = audit_sampled_tshirt_document_sleeve_constraint(
            oracle_document,
            sleeve_ease_cm=oracle_graph.sleeve_head_constraint.sleeve_ease_cm,
            samples_per_cubic=oracle_graph.samples_per_cubic,
        )

        comparable = target_mask & anchor.coordinate_mask
        sample = {
            "sample_id": sample_id,
            "anchor_coordinate_mae": _semantic_mae(anchor, target, comparable),
            # The guarded baseline selected scale zero for all six T-shirts;
            # keep the per-sample identity explicit instead of fabricating an
            # unavailable edited value from an aggregate JSON.
            "guarded_residual_coordinate_mae": _semantic_mae(
                anchor, target, comparable
            ),
            "direct_independent_semantic_coordinate_mae": _semantic_mae(
                predicted, target, target_mask
            ),
            "student_parametric_coordinate_mae": _semantic_mae(
                student_semantic, target, comparable
            ),
            "oracle_parametric_coordinate_mae": _semantic_mae(
                oracle_semantic, target, comparable
            ),
            "per_kind_mae": {},
            "student_projection": student_projection.to_receipt(),
            "oracle_projection_representational_ceiling": oracle_projection.to_receipt(),
            "student_canonical_graph": _canonical_graph_audit(student_graph),
            "oracle_canonical_graph": _canonical_graph_audit(oracle_graph),
            "student_sampled_document_constraint": student_sampled_constraint.to_dict(),
            "oracle_sampled_document_constraint": oracle_sampled_constraint.to_dict(),
            "student_pattern_validation": student_validation.to_dict(),
            "oracle_pattern_validation": oracle_validation.to_dict(),
            "pattern_artifacts": {
                "student": f"{sample_id}_student_parametric.json",
                "oracle": f"{sample_id}_oracle_parametric.json",
            },
        }
        for kind in ("landmark", "path", "panel", "reference_line"):
            mask = comparable & _kind_mask(kind)
            sample["per_kind_mae"][kind] = {
                "anchor": _semantic_mae(anchor, target, mask),
                "student_parametric": _semantic_mae(student_semantic, target, mask),
                "oracle_parametric": _semantic_mae(oracle_semantic, target, mask),
                "direct_independent_semantic": _semantic_mae(
                    predicted, target, target_mask & _kind_mask(kind)
                ),
            }
        student_document.write_json(pattern_output / sample["pattern_artifacts"]["student"])
        oracle_document.write_json(pattern_output / sample["pattern_artifacts"]["oracle"])
        per_sample.append(sample)

    if baseline_payload is not None:
        baseline_tshirt = baseline_payload["per_category"]["tshirt"]
        recomputed = {
            "anchor_coordinate_mae": float(
                np.mean([row["anchor_coordinate_mae"] for row in per_sample])
            ),
            "edited_coordinate_mae": float(
                np.mean([row["guarded_residual_coordinate_mae"] for row in per_sample])
            ),
            "direct_student_coordinate_mae": float(
                np.mean(
                    [row["direct_independent_semantic_coordinate_mae"] for row in per_sample]
                )
            ),
        }
        for key, value in recomputed.items():
            recorded = float(baseline_tshirt[key]["mean"])
            if not np.isclose(recorded, value, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"baseline T-shirt {key} does not match the frozen prediction rows"
                )

    output: dict[str, Any] = {
        "schema_version": "tshirt-four-view-parametric-decoder-evaluation/v2",
        "status": "COMPLETE_FROZEN_SAME_GENERATOR_SAMPLE_ID_UNSEEN",
        "frozen_split": {
            "seed": 20260902,
            "train_count": 25,
            "validation_count": 6,
            "test_count": 6,
            "test_ids": list(FROZEN_TSHIRT_TEST_IDS),
            "family_disjoint": False,
        },
        "input": {
            "prediction_artifact_sha256": _sha256_file(args.predictions),
            "source": "frozen four-view spatial Transformer test predictions",
            "target_fields_passed_to_primary_fit": False,
            "guarded_baseline_identity_validated": baseline_payload is not None,
            "source_images_embedded": False,
            "source_paths_embedded": False,
        },
        "decoder": {
            "parameter_order": list(TSHIRT_DRAFT_PARAMETER_NAMES),
            "optimized_parameter_count": len(TSHIRT_DRAFT_PARAMETER_NAMES),
            "final_physical_graph_parameter_count": len(PARAMETER_NAMES),
            "body_prior": "fixed default neutral T-shirt measurements",
            "configuration": {
                "confidence_threshold": config.confidence_threshold,
                "presence_threshold": config.presence_threshold,
                "prior_strength": config.prior_strength,
                "maximum_function_evaluations": config.max_function_evaluations,
                "robust_loss": config.robust_loss,
                "robust_scale": config.robust_scale,
            },
            "joint_construction": (
                "FNP/BNP/SNP/SP/underarm and cubic paths regenerated from one "
                "parameter vector; no post-decoder affine residual warp"
            ),
            "physical_materialization": (
                "front/back plus two exact sleeve instances; all physical path ids "
                "carry #left/#right or #center identity and share landmark ids"
            ),
            "sleeve_constraint": (
                "L(sleeve head) = L(front armhole) + L(back armhole) + ease_cm; "
                "ease_cm is the archetype's 1% allowance converted to centimetres "
                "and cap height is solved by bounded bisection"
            ),
        },
        "evaluation_metric": {
            "name": "normalized semantic coordinate MAE",
            "geometry": (
                "positive-x BasicBlock archetype used by the frozen semantic schema"
            ),
            "final_physical_graph_scored_in_this_mae": False,
            "reason": (
                "the GCD source and canonical export use different mirrored/split "
                "panel layout conventions; graph validity is audited separately"
            ),
        },
        "baseline": {
            "anchor_coordinate_mae": _summary(
                [float(row["anchor_coordinate_mae"]) for row in per_sample]
            ),
            "guarded_residual_coordinate_mae": _summary(
                [float(row["guarded_residual_coordinate_mae"]) for row in per_sample]
            ),
            "guarded_residual_improvement_count": 0,
            "direct_independent_semantic_coordinate_mae": _summary(
                [
                    float(row["direct_independent_semantic_coordinate_mae"])
                    for row in per_sample
                ]
            ),
            "loaded_baseline_evaluation_sha256": (
                _sha256_file(args.baseline_evaluation)
                if args.baseline_evaluation is not None
                else None
            ),
        },
        "student_parametric": {
            **_aggregate(per_sample, "student_parametric_coordinate_mae"),
            **_optimizer_summary(per_sample, "student_projection"),
            "valid_pattern_count": int(
                sum(bool(row["student_pattern_validation"]["accepted"]) for row in per_sample)
            ),
            "canonical_graph_valid_count": int(
                sum(
                    bool(row["student_canonical_graph"]["exact_reflection_validated"])
                    and bool(
                        row["student_canonical_graph"]["sleeve_head_constraint"][
                            "converged"
                        ]
                    )
                    for row in per_sample
                )
            ),
            "all_paths_instance_aware_count": int(
                sum(
                    bool(row["student_canonical_graph"]["all_paths_instance_aware"])
                    for row in per_sample
                )
            ),
            "maximum_exact_sleeve_constraint_abs_residual_cm": float(
                max(
                    abs(
                        float(
                            row["student_canonical_graph"]["sleeve_head_constraint"][
                                "residual_cm"
                            ]
                        )
                    )
                    for row in per_sample
                )
            ),
            "sampled_document_constraint_pass_count": int(
                sum(
                    bool(row["student_sampled_document_constraint"]["passed"])
                    for row in per_sample
                )
            ),
            "maximum_sampled_document_constraint_abs_residual_cm": float(
                max(
                    abs(
                        float(row["student_sampled_document_constraint"]["residual_cm"])
                    )
                    for row in per_sample
                )
            ),
            "maximum_individual_armhole_relative_mismatch": float(
                max(
                    float(
                        row["student_sampled_document_constraint"][
                            "maximum_individual_armhole_relative_mismatch"
                        ]
                    )
                    for row in per_sample
                )
            ),
        },
        "oracle_parametric_representational_ceiling": {
            **_aggregate(per_sample, "oracle_parametric_coordinate_mae"),
            **_optimizer_summary(
                per_sample, "oracle_projection_representational_ceiling"
            ),
            "uses_test_target_for_fit": True,
            "inference_result": False,
        },
        "per_kind": {},
        "samples": per_sample,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": [
            "The primary candidate uses frozen four-view student predictions only; test targets are read after fitting for evaluation.",
            "The oracle fit is a decoder-family capacity ceiling and is not an image-only benchmark result.",
            "The six samples are same-generator, same-render-style, same-neutral-body sample-ID unseen, not family-disjoint or cross-domain.",
            "The decoder is a provisional basic T-shirt construction, not expert-approved industrial pattern truth.",
            "Normalized semantic MAE includes the current shared schema's source-layout convention; structural seam checks are reported independently.",
            "The 22-value final graph contains fixed body-prior values, 12 fitted design values, and derived sleeve ease; only the 12 design values are optimized from visual semantics.",
            "Analytic cubic and sampled PatternDocument sleeve constraints have separate tolerances and receipts.",
        ],
    }
    for kind in ("landmark", "path", "panel", "reference_line"):
        output["per_kind"][kind] = {
            name: _summary(
                [float(row["per_kind_mae"][kind][name]) for row in per_sample]
            )
            for name in (
                "anchor",
                "student_parametric",
                "oracle_parametric",
                "direct_independent_semantic",
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
