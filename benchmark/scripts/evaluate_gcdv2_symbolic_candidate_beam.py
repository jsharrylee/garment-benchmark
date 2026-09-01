"""Evaluate a four-view -> train DSL -> exact symbolic candidate beam.

This is the strongest honest AlphaGeometry-style bridge supported by the
current checkpoints.  It proposes *whole retrieved train programs* as
discrete topology hypotheses, projects learned role/seam propositions through
the exact verifier, and reranks only with inference-time learned evidence.
It does not synthesize or splice new ``L/Q/C/A`` geometry.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES
from benchmark.gcdv2_exact.pattern_dsl_learning import (
    CURVE_COMMANDS,
    build_pattern_dsl_model,
    validate_edge_feature_schema,
)
from benchmark.gcdv2_exact.pattern_dsl_solver import symbolic_project_and_verify
from benchmark.gcdv2_exact.symbolic_candidate_beam import (
    BeamWeights,
    SymbolicCandidate,
    candidate_beam_metrics,
    rank_symbolic_beam,
    select_validation_weights,
)
from benchmark.gcdv2_exact.visual_dsl_retrieval import (
    build_visual_dsl_corpus,
    build_visual_dsl_retrieval_model,
    make_visual_dsl_batch,
)


def _device(raw: str):
    import torch

    return torch.device(
        "cuda" if raw == "auto" and torch.cuda.is_available() else ("cpu" if raw == "auto" else raw)
    )


def _embed(model, corpus, indices: np.ndarray, batch_size: int, device):
    import torch

    visual_rows: list[np.ndarray] = []
    pattern_rows: list[np.ndarray] = []
    category_rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start : start + batch_size]
            batch = make_visual_dsl_batch(corpus, current)
            with torch.amp.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(
                    torch.from_numpy(batch["views"]).to(device),
                    torch.from_numpy(batch["panel_tokens"]).to(device),
                    torch.from_numpy(batch["panel_valid"]).to(device),
                )
            visual_rows.append(output["visual_embedding"].float().cpu().numpy())
            pattern_rows.append(output["pattern_embedding"].float().cpu().numpy())
            category_rows.append(
                output["category_logits"].float().softmax(-1).cpu().numpy()
            )
    return (
        np.concatenate(visual_rows),
        np.concatenate(pattern_rows),
        np.concatenate(category_rows),
    )


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - np.max(values, axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / np.maximum(result.sum(axis=axis, keepdims=True), 1e-12)


def _panel_command_cycles(
    commands: np.ndarray, edge_valid: np.ndarray, panel_valid: np.ndarray
) -> tuple[tuple[str, ...], ...]:
    output = []
    for panel in np.flatnonzero(panel_valid):
        output.append(
            tuple(CURVE_COMMANDS[int(value)] for value in commands[panel][edge_valid[panel]])
        )
    return tuple(output)


def _anchor_evidence(
    *,
    model,
    arrays: Mapping[str, np.ndarray],
    corpus,
    corpus_indices: Sequence[int],
    allowed: np.ndarray,
    seam_threshold: float,
    seam_top_k: int,
    batch_size: int,
    device,
) -> dict[int, dict[str, Any]]:
    """Cache neural/symbolic evidence once for every anchor used by a beam."""

    import torch

    output: dict[int, dict[str, Any]] = {}
    unique = np.asarray(sorted({int(value) for value in corpus_indices}), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            local_indices = unique[start : start + batch_size]
            source_indices = corpus.dsl_indices[local_indices]
            features = torch.from_numpy(
                arrays["edge_features"][source_indices].astype(np.float32)
            ).to(device)
            commands = torch.from_numpy(
                arrays["edge_commands"][source_indices].astype(np.int64)
            ).to(device)
            edge_valid = torch.from_numpy(arrays["edge_valid"][source_indices]).to(device)
            panel_valid = torch.from_numpy(arrays["panel_valid"][source_indices]).to(device)
            predicted = model(features, commands, edge_valid, panel_valid)
            category_probabilities = predicted["category_logits"].float().softmax(-1).cpu().numpy()
            role_logits = predicted["edge_role_logits"].float().cpu().numpy()
            seam_probabilities = predicted["seam_logits"].float().sigmoid().cpu().numpy()
            for offset, (local, source) in enumerate(
                zip(local_indices, source_indices, strict=True)
            ):
                report = symbolic_project_and_verify(
                    role_logits[offset],
                    seam_probabilities[offset],
                    arrays["edge_valid"][source],
                    allowed,
                    seam_threshold=seam_threshold,
                    seam_top_k_per_edge=seam_top_k,
                )
                valid = arrays["edge_valid"][source].astype(bool)
                probabilities = _softmax(role_logits[offset], axis=-1)
                projected = report.roles.projected_roles
                selected_probability = probabilities[valid, projected[valid]]
                role_cycles = tuple(
                    tuple(EDGE_ROLES[int(value)] for value in panel.projected_roles)
                    for panel in report.roles.panels
                )
                output[int(local)] = {
                    "dsl_category_probabilities": tuple(
                        float(value) for value in category_probabilities[offset]
                    ),
                    "projected_role_confidence": float(selected_probability.mean()),
                    "repair_fraction": report.roles.changed_edges / max(int(valid.sum()), 1),
                    "symbolic_valid": bool(report.valid),
                    "panel_command_cycles": _panel_command_cycles(
                        arrays["edge_commands"][source], valid, arrays["panel_valid"][source]
                    ),
                    "projected_role_cycles": role_cycles,
                    "seam_pair_count": len(report.seams.pairs),
                    "landmark_count": len(report.landmarks),
                    "raw_grammar_violations": len(report.roles.raw_violations),
                    "projected_grammar_violations": len(report.roles.projected_violations),
                }
    return output


def _make_beams(
    *,
    query_indices: np.ndarray,
    visual_embeddings: np.ndarray,
    visual_category_probabilities: np.ndarray,
    train_indices: np.ndarray,
    train_pattern_embeddings: np.ndarray,
    corpus,
    evidence: Mapping[int, Mapping[str, Any]],
    beam_size: int,
) -> tuple[list[list[SymbolicCandidate]], np.ndarray]:
    similarities = np.asarray(visual_embeddings, np.float32) @ np.asarray(
        train_pattern_embeddings, np.float32
    ).T
    order = np.argsort(-similarities, axis=1, kind="stable")[:, :beam_size]
    beams: list[list[SymbolicCandidate]] = []
    for query_offset, candidates in enumerate(order):
        beam: list[SymbolicCandidate] = []
        for rank, train_offset in enumerate(candidates, start=1):
            local = int(train_indices[int(train_offset)])
            anchor = evidence[local]
            beam.append(
                SymbolicCandidate(
                    sample_id=str(corpus.sample_ids[local]),
                    retrieval_rank=rank,
                    similarity=float(similarities[query_offset, int(train_offset)]),
                    visual_category_probabilities=tuple(
                        float(value) for value in visual_category_probabilities[query_offset]
                    ),
                    dsl_category_probabilities=tuple(anchor["dsl_category_probabilities"]),
                    projected_role_confidence=float(anchor["projected_role_confidence"]),
                    repair_fraction=float(anchor["repair_fraction"]),
                    symbolic_valid=bool(anchor["symbolic_valid"]),
                    topology_signature=str(corpus.topology_signatures[local]),
                    # Evaluation metadata. rank_symbolic_beam never reads it.
                    evaluation_category=int(corpus.categories[local]),
                    panel_command_cycles=tuple(anchor["panel_command_cycles"]),
                    projected_role_cycles=tuple(anchor["projected_role_cycles"]),
                    seam_pair_count=int(anchor["seam_pair_count"]),
                    landmark_count=int(anchor["landmark_count"]),
                    raw_grammar_violations=int(anchor["raw_grammar_violations"]),
                    projected_grammar_violations=int(anchor["projected_grammar_violations"]),
                )
            )
        beams.append(beam)
    return beams, order


def _grid() -> list[BeamWeights]:
    return [
        BeamWeights(1.0, category, role, repair)
        for category, role, repair in itertools.product(
            (0.0, 0.025, 0.05, 0.1, 0.2, 0.4),
            (0.0, 0.025, 0.05, 0.1),
            (0.0, 0.025, 0.05, 0.1),
        )
    ]


def _proof_rows(
    beams,
    query_indices,
    corpus,
    target_categories,
    target_topologies,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    output = []
    for index, beam in enumerate(beams[:limit]):
        candidates = []
        for ranked in beam[:3]:
            value = ranked.candidate
            candidates.append(
                {
                    "anchor_sample_id": value.sample_id,
                    "inference_score": ranked.score,
                    "score_components": dict(ranked.component_scores),
                    "symbolic_valid": value.symbolic_valid,
                    "panel_LQCA_cycles": [list(cycle) for cycle in value.panel_command_cycles],
                    "projected_role_cycles": [list(cycle) for cycle in value.projected_role_cycles],
                    "SEWN_TO_count": value.seam_pair_count,
                    "derived_landmark_count": value.landmark_count,
                    # The next two fields are post-selection evaluation only.
                    "evaluation_category_match": value.evaluation_category
                    == int(target_categories[index]),
                    "evaluation_topology_match": value.topology_signature
                    == str(target_topologies[index]),
                }
            )
        output.append(
            {
                "query_4view_sample_id": str(corpus.sample_ids[int(query_indices[index])]),
                "selected_anchor": beam[0].candidate.sample_id,
                "candidate_hypotheses": candidates,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a leakage-safe AlphaGeometry-style train-program candidate beam."
    )
    parser.add_argument("--programs", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/metadata.jsonl"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"))
    parser.add_argument("--dsl-checkpoint", type=Path, default=Path("checkpoints/gcdv2_pattern_dsl/unified_transformer.pt"))
    parser.add_argument("--visual-checkpoint", type=Path, default=Path("checkpoints/gcdv2_pattern_dsl/visual_retrieval.pt"))
    parser.add_argument("--cached-panel-tokens", type=Path, default=Path("artifacts/gcdv2_visual_pattern_dsl_retrieval/frozen_dsl_panel_tokens.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_visual_pattern_dsl_retrieval/symbolic_candidate_beam.json"))
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seam-top-k", type=int, default=16)
    parser.add_argument("--proof-samples", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.beam_size <= 0:
        raise ValueError("beam size must be positive")

    import torch

    device = _device(args.device)
    visual_checkpoint = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    visual_model = build_visual_dsl_retrieval_model(visual_checkpoint["config"])
    visual_model.load_state_dict(visual_checkpoint["model_state"], strict=True)
    visual_model.to(device).eval()
    corpus = build_visual_dsl_corpus(
        args.programs,
        args.metadata,
        args.features,
        args.dsl_checkpoint,
        device=device,
        extraction_batch_size=max(4, args.batch_size // 2),
        cached_panel_tokens_path=args.cached_panel_tokens,
    )
    split = {name: corpus.indices(name) for name in ("train", "validation", "test")}
    _, train_pattern, _ = _embed(visual_model, corpus, split["train"], args.batch_size, device)
    validation_visual, _, validation_category = _embed(
        visual_model, corpus, split["validation"], args.batch_size, device
    )
    test_visual, _, test_category = _embed(
        visual_model, corpus, split["test"], args.batch_size, device
    )

    validation_similarity = validation_visual @ train_pattern.T
    test_similarity = test_visual @ train_pattern.T
    validation_order = np.argsort(-validation_similarity, axis=1, kind="stable")[:, : args.beam_size]
    test_order = np.argsort(-test_similarity, axis=1, kind="stable")[:, : args.beam_size]
    needed_local = {
        int(split["train"][offset]) for offset in np.concatenate((validation_order, test_order)).reshape(-1)
    }

    programs = np.load(args.programs, allow_pickle=False)
    dsl_checkpoint = torch.load(args.dsl_checkpoint, map_location="cpu", weights_only=False)
    validate_edge_feature_schema(programs, dsl_checkpoint)
    dsl_model = build_pattern_dsl_model(width=int(dsl_checkpoint["width"]))
    dsl_model.load_state_dict(dsl_checkpoint["model_state"], strict=True)
    dsl_model.to(device).eval()
    evidence = _anchor_evidence(
        model=dsl_model,
        arrays=programs,
        corpus=corpus,
        corpus_indices=needed_local,
        allowed=np.asarray(dsl_checkpoint["allowed_transitions"], dtype=bool),
        seam_threshold=float(dsl_checkpoint["seam_threshold"]),
        seam_top_k=args.seam_top_k,
        batch_size=max(4, args.batch_size // 2),
        device=device,
    )
    validation_beams, _ = _make_beams(
        query_indices=split["validation"],
        visual_embeddings=validation_visual,
        visual_category_probabilities=validation_category,
        train_indices=split["train"],
        train_pattern_embeddings=train_pattern,
        corpus=corpus,
        evidence=evidence,
        beam_size=args.beam_size,
    )
    test_beams, _ = _make_beams(
        query_indices=split["test"],
        visual_embeddings=test_visual,
        visual_category_probabilities=test_category,
        train_indices=split["train"],
        train_pattern_embeddings=train_pattern,
        corpus=corpus,
        evidence=evidence,
        beam_size=args.beam_size,
    )

    validation_targets_category = corpus.categories[split["validation"]]
    validation_targets_topology = corpus.topology_signatures[split["validation"]]
    weights, validation_metrics, search = select_validation_weights(
        validation_beams,
        validation_targets_category,
        validation_targets_topology,
        _grid(),
    )
    baseline_weights = BeamWeights()
    baseline_ranked = [rank_symbolic_beam(beam, baseline_weights) for beam in test_beams]
    ranked = [rank_symbolic_beam(beam, weights) for beam in test_beams]
    test_targets_category = corpus.categories[split["test"]]
    test_targets_topology = corpus.topology_signatures[split["test"]]
    baseline_metrics = candidate_beam_metrics(
        baseline_ranked, test_targets_category, test_targets_topology
    )
    test_metrics = candidate_beam_metrics(ranked, test_targets_category, test_targets_topology)
    raw_violations = sum(
        value.raw_grammar_violations for beam in test_beams for value in beam
    )
    projected_violations = sum(
        value.projected_grammar_violations for beam in test_beams for value in beam
    )
    changed_role_edges = sum(
        round(value.repair_fraction * sum(len(cycle) for cycle in value.projected_role_cycles))
        for beam in test_beams
        for value in beam
    )
    result = {
        "schema_version": "gcdv2-symbolic-candidate-beam-1.0",
        "status": "PASS_HONEST_RETRIEVED_PROGRAM_SYMBOLIC_BEAM",
        "device": str(device),
        "beam_size": args.beam_size,
        "train_bank_size": int(len(split["train"])),
        "candidate_programs_symbolically_evaluated": int(len(test_beams) * args.beam_size),
        "validation_weight_selection": {
            "query_count": int(len(validation_beams)),
            "objective": "exact canonical primitive topology, then category; labels used only on validation",
            "selected_weights": weights.as_dict(),
            "selected_metrics": validation_metrics,
            "grid_candidate_count": len(search),
            "grid_results": search,
        },
        "frozen_test": {
            "raw_similarity_top1": baseline_metrics,
            "symbolic_reranked_top1": test_metrics,
            "raw_grammar_violations_across_beam": raw_violations,
            "projected_grammar_violations_across_beam": projected_violations,
            "changed_role_edges_across_beam": changed_role_edges,
        },
        "inference_contract": {
            "topology_proposal": "top-k complete train-bank PANEL/M/L/Q/C/A/Z programs",
            "role_proposal": "Pattern DSL Transformer edge-role logits",
            "proof": "exact cyclic role grammar plus symmetric degree<=1 Blossom seam matching and landmark deductions",
            "reranker_inputs": [
                "four-view/DSL cosine similarity",
                "visual-vs-DSL learned category posterior agreement",
                "projected role posterior confidence",
                "fraction of roles not changed by exact projection",
            ],
            "held_out_target_dsl_used_at_inference": False,
            "candidate_source_split": "train only",
        },
        "proof_samples": _proof_rows(
            ranked,
            split["test"],
            corpus,
            test_targets_category,
            test_targets_topology,
            limit=args.proof_samples,
        ),
        "claim_boundary": [
            "This is a beam over retrieved complete train programs, not generation of new target L/Q/C/A commands.",
            "The symbolic verifier proves internal grammar/seam/landmark consistency; it cannot prove visual identity or expert production-pattern correctness.",
            "Beam topology coverage is an evaluation-only oracle diagnostic and is never used to select the test winner.",
            "Validation target topology/category may select four scalar weights; held-out test targets are read only after rankings are frozen.",
            "All results remain inside the GarmentCodeData v2 generator and Blender render domain.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_weights": weights.as_dict(),
                "validation": validation_metrics,
                "test_baseline": baseline_metrics,
                "test_symbolic_reranked": test_metrics,
                "grammar": [raw_violations, projected_violations],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
