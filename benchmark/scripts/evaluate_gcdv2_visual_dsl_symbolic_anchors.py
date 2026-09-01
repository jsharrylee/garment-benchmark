from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.pattern_dsl_learning import (
    LANDMARK_NAMES,
    MAXIMUM_EDGES,
    build_pattern_dsl_model,
    validate_edge_feature_schema,
)
from benchmark.gcdv2_exact.pattern_dsl_solver import symbolic_project_and_verify
from benchmark.gcdv2_exact.visual_dsl_retrieval import read_metadata


def _safe_ratio(value: int, total: int) -> float:
    return value / max(total, 1)


def _f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _target_seams(arrays: dict[str, np.ndarray], source: int) -> set[tuple[int, int]]:
    output: set[tuple[int, int]] = set()
    for pair, valid in zip(arrays["stitch_pairs"][source], arrays["stitch_valid"][source], strict=True):
        if valid:
            first = int(pair[0]) * MAXIMUM_EDGES + int(pair[1])
            second = int(pair[2]) * MAXIMUM_EDGES + int(pair[3])
            output.add(tuple(sorted((first, second))))
    return output


def _target_landmarks(arrays: dict[str, np.ndarray], source: int) -> set[tuple[int, int, str]]:
    output = set()
    for panel, vertex in np.argwhere(arrays["landmarks"][source] >= 0):
        value = int(arrays["landmarks"][source, panel, vertex])
        output.add((int(panel), int(vertex), LANDMARK_NAMES[value]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify train-bank anchors selected by four-view-to-DSL retrieval."
    )
    parser.add_argument("--programs", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/metadata.jsonl"))
    parser.add_argument(
        "--retrieval-predictions",
        type=Path,
        default=Path("artifacts/gcdv2_visual_pattern_dsl_retrieval/test_train_bank_predictions.jsonl"),
    )
    parser.add_argument(
        "--dsl-checkpoint", type=Path, default=Path("checkpoints/gcdv2_pattern_dsl/unified_transformer.pt")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_visual_pattern_dsl_retrieval/symbolic_anchor_verification.json"),
    )
    parser.add_argument("--proof-samples", type=int, default=5)
    parser.add_argument("--seam-top-k", type=int, default=16)
    args = parser.parse_args()

    import torch

    archive = np.load(args.programs, allow_pickle=False)
    arrays = {key: archive[key] for key in archive.files}
    metadata = read_metadata(args.metadata)
    source_lookup = {str(row["sample_id"]): index for index, row in enumerate(metadata)}
    retrieval_rows = [
        json.loads(line)
        for line in args.retrieval_predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint = torch.load(args.dsl_checkpoint, map_location="cpu", weights_only=False)
    validate_edge_feature_schema(arrays, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pattern_dsl_model(width=int(checkpoint["width"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    allowed = np.asarray(checkpoint["allowed_transitions"], dtype=bool)
    seam_threshold = float(checkpoint["seam_threshold"])

    totals = {
        "queries": 0,
        "valid": 0,
        "raw_grammar": 0,
        "projected_grammar": 0,
        "changed_roles": 0,
        "role_correct": 0,
        "role_supervised": 0,
        "seam_tp": 0,
        "seam_fp": 0,
        "seam_fn": 0,
        "landmark_tp": 0,
        "landmark_fp": 0,
        "landmark_fn": 0,
        "accepted_seam_facts": 0,
        "derived_landmark_facts": 0,
    }
    proof_samples: list[dict] = []
    anchors: list[str] = []
    with torch.inference_mode():
        for retrieval in retrieval_rows:
            query_id = str(retrieval["sample_id"])
            anchor_id = str(retrieval["retrieved_sample_id"])
            anchors.append(anchor_id)
            source = source_lookup[anchor_id]
            features = torch.from_numpy(arrays["edge_features"][source].astype(np.float32))[None].to(device)
            commands = torch.from_numpy(arrays["edge_commands"][source].astype(np.int64))[None].to(device)
            edge_valid = torch.from_numpy(arrays["edge_valid"][source])[None].to(device)
            panel_valid = torch.from_numpy(arrays["panel_valid"][source])[None].to(device)
            prediction = model(features, commands, edge_valid, panel_valid)
            report = symbolic_project_and_verify(
                prediction["edge_role_logits"][0].float().cpu().numpy(),
                prediction["seam_logits"][0].sigmoid().float().cpu().numpy(),
                arrays["edge_valid"][source],
                allowed,
                seam_threshold=seam_threshold,
                seam_top_k_per_edge=args.seam_top_k,
            )
            totals["queries"] += 1
            totals["valid"] += int(report.valid)
            totals["raw_grammar"] += len(report.roles.raw_violations)
            totals["projected_grammar"] += len(report.roles.projected_violations)
            totals["changed_roles"] += report.roles.changed_edges
            totals["accepted_seam_facts"] += len(report.seams.pairs)
            totals["derived_landmark_facts"] += len(report.landmarks)

            target_roles = arrays["edge_roles"][source]
            supervised = arrays["edge_valid"][source] & (target_roles >= 0)
            totals["role_supervised"] += int(supervised.sum())
            totals["role_correct"] += int(
                (report.roles.projected_roles[supervised] == target_roles[supervised]).sum()
            )
            target_seams = _target_seams(arrays, source)
            predicted_seams = {
                (pair.first.flat_index, pair.second.flat_index) for pair in report.seams.pairs
            }
            totals["seam_tp"] += len(target_seams & predicted_seams)
            totals["seam_fp"] += len(predicted_seams - target_seams)
            totals["seam_fn"] += len(target_seams - predicted_seams)
            target_landmarks = _target_landmarks(arrays, source)
            predicted_landmarks = {
                (value.panel_index, value.vertex_index, value.base_name) for value in report.landmarks
            }
            totals["landmark_tp"] += len(target_landmarks & predicted_landmarks)
            totals["landmark_fp"] += len(predicted_landmarks - target_landmarks)
            totals["landmark_fn"] += len(target_landmarks - predicted_landmarks)
            if len(proof_samples) < args.proof_samples:
                proof_samples.append(
                    {
                        "query_4view_sample_id": query_id,
                        "retrieved_train_anchor_id": anchor_id,
                        "category_match": bool(retrieval["category_match"]),
                        "target_anchor_topology_match": bool(
                            retrieval["exact_closed_cycle_primitive_topology_match"]
                        ),
                        "anchor_symbolically_valid": report.valid,
                        "accepted_fact_count": len(report.facts()),
                        "accepted_facts": list(report.facts()),
                    }
                )

    result = {
        "schema_version": "gcdv2-visual-dsl-symbolic-anchor-verification-1.0",
        "status": "PASS_RETRIEVED_ANCHORS_SYMBOLICALLY_VERIFIED",
        "device": str(device),
        "test_query_count": totals["queries"],
        "unique_retrieved_train_anchors": len(set(anchors)),
        "retrieved_anchor_symbolic_valid_rate": _safe_ratio(totals["valid"], totals["queries"]),
        "grammar": {
            "raw_violations": totals["raw_grammar"],
            "projected_violations": totals["projected_grammar"],
            "changed_role_edges": totals["changed_roles"],
        },
        "retrieved_anchor_internal_label_agreement": {
            "projected_edge_role_accuracy": _safe_ratio(totals["role_correct"], totals["role_supervised"]),
            "supervised_edges": totals["role_supervised"],
            "seams": _f1(totals["seam_tp"], totals["seam_fp"], totals["seam_fn"]),
            "landmarks": _f1(
                totals["landmark_tp"], totals["landmark_fp"], totals["landmark_fn"]
            ),
        },
        "accepted_symbolic_facts": {
            "sewn_to": totals["accepted_seam_facts"],
            "derived_landmarks": totals["derived_landmark_facts"],
        },
        "proof_samples": proof_samples,
        "claim_boundary": [
            "The four-view model selects an existing train-bank anchor; it does not generate the held-out target DSL.",
            "The Pattern DSL proposer and symbolic solver run on the retrieved anchor's exact observed geometry.",
            "Validity and label agreement describe the anchor program, not geometric equality to the held-out target.",
            "All semantics are derived GCDv2-internal labels, not expert cross-source production-pattern approval.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "proof_samples"}, indent=2))


if __name__ == "__main__":
    main()
