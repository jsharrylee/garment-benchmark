from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from benchmark.pattern_pipeline.grading import grade_pattern
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.retrieval.features import multiview_descriptor
from benchmark.retrieval.index import PatternIndex, QueryEvidence
from benchmark.retrieval.anchor_bank import load_procedural_anchors, rank_dataset_anchors


CANONICAL_PANEL_CAPS = {"top": 10, "pants": 8, "shorts": 8, "skirt": 8, "dress": 14, "jumpsuit": 16}


def _views(directory: Path) -> list[Path]:
    files = sorted([*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")])
    if len(files) != 4:
        raise ValueError(f"expected four view images in {directory}, found {len(files)}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve structurally valid starting-pattern candidates.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--category", choices=["top", "pants", "shorts", "skirt", "dress", "outerwear", "jumpsuit"])
    parser.add_argument("--reweaver-summary", type=Path)
    parser.add_argument("--garment-particles-summary", type=Path)
    parser.add_argument("--anchor-bank", type=Path, default=Path("data/manifests/garmentcode_anchor_bank.json"))
    parser.add_argument("--dataset-catalog", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_catalog.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-body-measurements", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--canonical-root", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_full_canonical"))
    parser.add_argument("--graded-output", type=Path)
    parser.add_argument("--panel-mesh-spacing-cm", type=float, default=2.5)
    parser.add_argument("--max-panels", type=int, help="Override the canonical category panel cap.")
    args = parser.parse_args()

    query = QueryEvidence.from_files(
        multiview_descriptor(_views(args.views)),
        category=args.category,
        reweaver_summary=args.reweaver_summary,
        garment_particles_summary=args.garment_particles_summary,
    )
    full_index = PatternIndex.read_json(args.index)
    panel_cap = args.max_panels if args.max_panels is not None else CANONICAL_PANEL_CAPS.get(args.category or "", 16)
    capped_records = [
        record for record in full_index.records if record.panel_count <= panel_cap or (args.category is not None and record.category != args.category)
    ]
    result = PatternIndex(capped_records).search(query, top_k=args.top_k)
    procedural = load_procedural_anchors(args.anchor_bank, category=args.category) if args.anchor_bank.is_file() else ()
    dataset_candidates = (
        rank_dataset_anchors(
            args.dataset_catalog,
            category=args.category,
            reweaver_panel_count=query.reweaver_panel_count,
            reweaver_edge_count=query.reweaver_edge_count,
            reweaver_reliability=query.reweaver_reliability,
            garment_particles_panel_count=query.garment_particles_panel_count,
            garment_particles_edge_count=query.garment_particles_edge_count,
            garment_particles_reliability=query.garment_particles_reliability,
            top_k=args.top_k,
        )
        if args.dataset_catalog.is_file()
        else ()
    )
    decision = result.decision
    final_acceptance = result.final_acceptance
    if result.decision == "NO_SUITABLE_ANCHOR" and dataset_candidates:
        decision = "DATASET_ANCHOR_CANDIDATES_AVAILABLE"
        final_acceptance = "REQUIRES_SIMULATION_RERANK"
    elif result.decision == "NO_SUITABLE_ANCHOR" and procedural:
        decision = "PROCEDURAL_ANCHOR_CANDIDATES_AVAILABLE"
        final_acceptance = "REQUIRES_SIMULATION_RERANK"
    payload = {
        "mode": "retrieval_anchored_v2",
        "query": {
            "category": query.category,
            "reweaver_panel_count": query.reweaver_panel_count,
            "reweaver_edge_count": query.reweaver_edge_count,
            "reweaver_reliability": query.reweaver_reliability,
            "garment_particles_panel_count": query.garment_particles_panel_count,
            "garment_particles_edge_count": query.garment_particles_edge_count,
            "garment_particles_stitch_count": query.garment_particles_stitch_count,
            "garment_particles_reliability": query.garment_particles_reliability,
            "canonical_panel_cap": panel_cap,
        },
        "uncapped_corpus_size": len(full_index.records),
        **result.to_dict(),
        "decision": decision,
        "final_acceptance": final_acceptance,
        "procedural_anchor_candidates": [
            {
                **anchor.__dict__,
                "selection_scope": "category_fallback_without_visual_descriptor",
            }
            for anchor in procedural
        ],
        "procedural_eligible_size": len(procedural),
        "dataset_anchor_candidates": list(dataset_candidates),
        "dataset_eligible_size": len(dataset_candidates),
        "scope": "retrieval_only_not_simulation_accepted",
    }
    if args.target_body_measurements:
        selected_id = result.candidates[0].sample_id if result.candidates else (dataset_candidates[0]["sample_id"] if dataset_candidates else None)
        if selected_id is None or not args.category:
            payload["body_grading"] = {"status": "BLOCKED_NO_SELECTED_DATASET_ANCHOR"}
        else:
            canonical = args.canonical_root / args.category / f"{selected_id}.json"
            source_measurements_path = args.dataset_root / selected_id / f"{selected_id}_body_measurements.yaml"
            source_raw = yaml.safe_load(source_measurements_path.read_text(encoding="utf-8"))
            target_raw = yaml.safe_load(args.target_body_measurements.read_text(encoding="utf-8"))
            graded = grade_pattern(
                PatternDocument.read_json(canonical),
                category=args.category,
                source_measurements=source_raw.get("body", source_raw),
                target_measurements=target_raw.get("body", target_raw),
                panel_mesh_spacing_cm=args.panel_mesh_spacing_cm,
            )
            graded_output = args.graded_output or args.output.with_name(args.output.stem + "_graded_pattern.json")
            graded.write_json(graded_output)
            payload["body_grading"] = {
                "status": "PASS" if validate_pattern(graded).accepted else "FAILED_STRUCTURAL_VALIDATION",
                "selected_sample_id": selected_id,
                "graded_pattern": str(graded_output),
                **graded.annotations["body_grading"],
            }
            payload["scope"] = "retrieval_and_body_grading_complete_simulation_rerank_required"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
