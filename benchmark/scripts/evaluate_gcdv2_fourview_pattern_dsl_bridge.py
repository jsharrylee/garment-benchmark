"""Evaluate 4-view retrieval -> train anchor -> Pattern DSL symbolic facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.fourview_dsl_bridge import (
    adapt_aligned_dsl_prediction,
    aligned_dsl_retrieval_catalog,
    load_retrieval_catalog,
    make_bridge_record,
    metadata_lookup,
    neural_geometry_input,
    read_jsonl,
    select_train_bank_anchor,
    summarize_bridge_records,
)
from benchmark.gcdv2_exact.pattern_dsl_learning import build_pattern_dsl_model
from benchmark.gcdv2_exact.pattern_dsl_solver import symbolic_project_and_verify


def _device(raw: str):
    import torch

    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge held-out four-view retrieval predictions to train-bank anchor "
            "Pattern-DSL propositions and symbolic proof facts."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/test_predictions.jsonl"),
    )
    parser.add_argument(
        "--prediction-schema",
        choices=("auto", "legacy", "aligned-dsl"),
        default="auto",
        help="Auto-detect the legacy crossmodal or DSL-split-aligned prediction schema.",
    )
    parser.add_argument(
        "--retrieval-embeddings",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/embeddings.npz"),
    )
    parser.add_argument(
        "--dsl-dataset",
        type=Path,
        default=Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz"),
    )
    parser.add_argument(
        "--dsl-metadata",
        type=Path,
        default=Path("artifacts/gcdv2_pattern_dsl_v1/metadata.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gcdv2_pattern_dsl/unified_transformer.pt"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("artifacts/gcdv2_fourview_pattern_dsl_bridge/summary.json"),
    )
    parser.add_argument(
        "--output-records",
        type=Path,
        default=Path("artifacts/gcdv2_fourview_pattern_dsl_bridge/records.jsonl"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seam-top-k", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch

    raw_predictions = read_jsonl(args.predictions)
    schema = args.prediction_schema
    if schema == "auto":
        schema = (
            "aligned-dsl"
            if raw_predictions and "top_train_bank_sample_ids" in raw_predictions[0]
            else "legacy"
        )
    predictions = (
        [adapt_aligned_dsl_prediction(row) for row in raw_predictions]
        if schema == "aligned-dsl"
        else raw_predictions
    )
    if args.limit is not None:
        predictions = predictions[: args.limit]
    metadata_rows = read_jsonl(args.dsl_metadata)
    dsl_lookup = metadata_lookup(metadata_rows)
    archive = np.load(args.dsl_dataset, allow_pickle=False)
    # Deliberately omit stitch_pairs, stitch_valid, edge_roles, panel_roles and
    # landmarks.  This bridge reviews predictions without source-label access.
    arrays = {
        key: archive[key]
        for key in ("edge_features", "edge_commands", "edge_valid", "panel_valid")
    }
    catalog = (
        aligned_dsl_retrieval_catalog(metadata_rows, archive)
        if schema == "aligned-dsl"
        else load_retrieval_catalog(args.retrieval_embeddings)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = _device(args.device)
    model = build_pattern_dsl_model(width=int(checkpoint["width"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    allowed = np.asarray(checkpoint["allowed_transitions"], dtype=bool)
    seam_threshold = float(checkpoint["seam_threshold"])

    records: list[dict] = []
    with torch.inference_mode():
        for prediction in predictions:
            query_id = str(prediction["sample_id"])
            if query_id not in catalog:
                raise ValueError(f"query is missing from retrieval catalogue: {query_id}")
            if catalog[query_id].split != "test":
                raise ValueError(
                    f"bridge expects held-out test queries, got {query_id}:{catalog[query_id].split}"
                )
            selection = select_train_bank_anchor(
                prediction,
                retrieval_catalog=catalog,
                dsl_lookup=dsl_lookup,
            )
            source_index, dsl_metadata = dsl_lookup[selection.anchor_sample_id]
            inputs = neural_geometry_input(arrays, source_index)
            output = model(
                torch.from_numpy(inputs["edge_features"].astype(np.float32))[None].to(device),
                torch.from_numpy(inputs["edge_commands"].astype(np.int64))[None].to(device),
                torch.from_numpy(inputs["edge_valid"])[None].to(device),
                torch.from_numpy(inputs["panel_valid"])[None].to(device),
            )
            role_logits = output["edge_role_logits"][0].float().cpu().numpy()
            seam_scores = output["seam_logits"][0].sigmoid().float().cpu().numpy()
            report = symbolic_project_and_verify(
                role_logits,
                seam_scores,
                inputs["edge_valid"],
                allowed,
                seam_threshold=seam_threshold,
                seam_top_k_per_edge=args.seam_top_k,
            )
            records.append(
                make_bridge_record(
                    prediction=prediction,
                    selection=selection,
                    retrieval_catalog=catalog,
                    dsl_metadata=dsl_metadata,
                    arrays=arrays,
                    source_index=source_index,
                    predicted_category_index=int(output["category_logits"][0].argmax().item()),
                    report=report,
                )
            )

    summary = summarize_bridge_records(records)
    summary.update(
        {
            "device": str(device),
            "prediction_schema": schema,
            "seam_threshold": seam_threshold,
            "seam_top_k_per_edge": args.seam_top_k,
            "artifacts": {
                "records_ignored": str(args.output_records),
                "summary_ignored": str(args.output_summary),
            },
        }
    )
    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
