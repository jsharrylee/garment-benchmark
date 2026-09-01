from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES
from benchmark.gcdv2_exact.pattern_dsl_learning import (
    LANDMARK_NAMES,
    MAXIMUM_EDGES,
    build_pattern_dsl_model,
    validate_edge_feature_schema,
)
from benchmark.gcdv2_exact.pattern_dsl_solver import symbolic_project_and_verify


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _target_seams(arrays: dict[str, np.ndarray], source: int) -> set[tuple[int, int]]:
    output: set[tuple[int, int]] = set()
    for pair, valid in zip(
        arrays["stitch_pairs"][source], arrays["stitch_valid"][source], strict=True
    ):
        if not bool(valid):
            continue
        first = int(pair[0]) * MAXIMUM_EDGES + int(pair[1])
        second = int(pair[2]) * MAXIMUM_EDGES + int(pair[3])
        output.add(tuple(sorted((first, second))))
    return output


def _target_landmarks(arrays: dict[str, np.ndarray], source: int) -> set[tuple[int, int, str]]:
    output: set[tuple[int, int, str]] = set()
    values = arrays["landmarks"][source]
    for panel, vertex in np.argwhere(values >= 0):
        output.add((int(panel), int(vertex), LANDMARK_NAMES[int(values[panel, vertex])]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run symbolic proof/projection over trained Pattern-DSL propositions."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gcdv2_pattern_dsl/unified_transformer.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_pattern_dsl_training/symbolic_test.json"),
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--seam-top-k", type=int, default=16)
    parser.add_argument("--proof-samples", type=int, default=5)
    args = parser.parse_args()

    import torch

    split_code = {"train": 0, "validation": 1, "test": 2}[args.split]
    archive = np.load(args.dataset)
    arrays = {key: archive[key] for key in archive.files}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_edge_feature_schema(arrays, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pattern_dsl_model(width=int(checkpoint["width"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    allowed = np.asarray(checkpoint["allowed_transitions"], dtype=bool)
    seam_threshold = float(checkpoint["seam_threshold"])

    totals = {
        "samples": 0,
        "symbolically_valid": 0,
        "raw_grammar_violations": 0,
        "projected_grammar_violations": 0,
        "changed_role_edges": 0,
        "role_correct_raw": 0,
        "role_correct_projected": 0,
        "role_supervised": 0,
        "seam_tp": 0,
        "seam_fp": 0,
        "seam_fn": 0,
        "landmark_tp": 0,
        "landmark_fp": 0,
        "landmark_fn": 0,
        "rejected_seam_candidates": 0,
    }
    proof_samples: list[dict] = []
    sources = np.flatnonzero(arrays["splits"] == split_code)
    with torch.no_grad():
        for source in sources:
            source = int(source)
            features = torch.from_numpy(arrays["edge_features"][source].astype(np.float32))[None].to(device)
            commands = torch.from_numpy(arrays["edge_commands"][source].astype(np.int64))[None].to(device)
            edge_valid = torch.from_numpy(arrays["edge_valid"][source])[None].to(device)
            panel_valid = torch.from_numpy(arrays["panel_valid"][source])[None].to(device)
            prediction = model(features, commands, edge_valid, panel_valid)
            logits = prediction["edge_role_logits"][0].float().cpu().numpy()
            seam_scores = prediction["seam_logits"][0].sigmoid().float().cpu().numpy()
            report = symbolic_project_and_verify(
                logits,
                seam_scores,
                arrays["edge_valid"][source],
                allowed,
                seam_threshold=seam_threshold,
                seam_top_k_per_edge=args.seam_top_k,
            )
            totals["samples"] += 1
            totals["symbolically_valid"] += int(report.valid)
            totals["raw_grammar_violations"] += len(report.roles.raw_violations)
            totals["projected_grammar_violations"] += len(report.roles.projected_violations)
            totals["changed_role_edges"] += report.roles.changed_edges
            totals["rejected_seam_candidates"] += report.seams.rejected_candidate_count

            target_roles = arrays["edge_roles"][source]
            supervised = arrays["edge_valid"][source] & (target_roles >= 0)
            totals["role_supervised"] += int(supervised.sum())
            totals["role_correct_raw"] += int((report.roles.raw_roles[supervised] == target_roles[supervised]).sum())
            totals["role_correct_projected"] += int(
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
                (value.panel_index, value.vertex_index, value.base_name)
                for value in report.landmarks
            }
            totals["landmark_tp"] += len(target_landmarks & predicted_landmarks)
            totals["landmark_fp"] += len(predicted_landmarks - target_landmarks)
            totals["landmark_fn"] += len(target_landmarks - predicted_landmarks)
            if len(proof_samples) < args.proof_samples:
                proof_samples.append({"source_index": source, **report.to_dict()})

    result = {
        "status": "PASS_SYMBOLIC_EVALUATION",
        "split": args.split,
        "device": str(device),
        "sample_count": totals["samples"],
        "symbolic_valid_rate": _ratio(totals["symbolically_valid"], totals["samples"]),
        "grammar": {
            "raw_violations": totals["raw_grammar_violations"],
            "projected_violations": totals["projected_grammar_violations"],
            "changed_role_edges": totals["changed_role_edges"],
        },
        "edge_role_accuracy": {
            "raw": _ratio(totals["role_correct_raw"], totals["role_supervised"]),
            "projected": _ratio(totals["role_correct_projected"], totals["role_supervised"]),
            "supervised_edges": totals["role_supervised"],
        },
        "seams": {
            **_f1(totals["seam_tp"], totals["seam_fp"], totals["seam_fn"]),
            "tp": totals["seam_tp"],
            "fp": totals["seam_fp"],
            "fn": totals["seam_fn"],
            "rejected_candidates": totals["rejected_seam_candidates"],
            "threshold": seam_threshold,
            "top_k_per_edge": args.seam_top_k,
        },
        "landmarks": {
            **_f1(totals["landmark_tp"], totals["landmark_fp"], totals["landmark_fn"]),
            "tp": totals["landmark_tp"],
            "fp": totals["landmark_fp"],
            "fn": totals["landmark_fn"],
        },
        "proof_samples": proof_samples,
        "claim_boundary": (
            "Exact GCDv2 vector geometry to derived semantic/seam propositions. "
            "This evaluates symbolic consistency and internal labels, not expert cross-source validity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "proof_samples"}, indent=2))


if __name__ == "__main__":
    main()
