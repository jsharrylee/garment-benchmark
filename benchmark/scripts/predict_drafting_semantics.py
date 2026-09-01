from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.dataset import PanelExample, edge_features, padded_batch
from benchmark.drafting_semantics.decoding import decode_darts, decode_named_landmarks, decode_path_measurements
from benchmark.drafting_semantics.garmentcode import annotate_garmentcode_sample
from benchmark.drafting_semantics.model import build_model
from benchmark.drafting_semantics.schema import EDGE_ROLES, PANEL_ROLES


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict drafting semantics on one GarmentCode vector pattern.")
    parser.add_argument("specification", type=Path)
    parser.add_argument("body_measurements", type=Path)
    parser.add_argument("design_params", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/gcdv2_edge_semantics.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/prediction.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-source-comparison", action="store_true")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    record = annotate_garmentcode_sample(
        args.specification,
        args.body_measurements,
        args.design_params,
        split="inference",
    )

    panels = []
    for panel in record.panels:
        if panel.role not in {"front_bodice", "back_bodice"}:
            continue
        features, targets = edge_features(panel, include_stitch_features=bool(config.get("include_stitch_features", False)))
        example = PanelExample(
            sample_id=record.sample_id,
            split="inference",
            panel_id=panel.id,
            panel_role_id=PANEL_ROLES.index(panel.role),
            panel=panel,
            features=features,
            targets=targets,
            edge_indices=np.arange(len(targets), dtype=np.int64),
        )
        padded_features, _, valid, panel_roles = padded_batch((example,), int(config["maximum_edges"]))
        with torch.no_grad():
            logits = model(
                torch.from_numpy(padded_features).to(device),
                torch.from_numpy(valid).to(device),
                torch.from_numpy(panel_roles).to(device),
            )[0, : len(targets)]
            probabilities = logits.softmax(dim=-1)
            predicted_ids = probabilities.argmax(dim=-1).cpu().numpy()
            confidence = probabilities.max(dim=-1).values.cpu().numpy()
        predicted_roles = [EDGE_ROLES[int(value)] for value in predicted_ids]
        panel_result = {
            "panel_id": panel.id,
            "panel_role": panel.role,
            "edges": [
                {
                    "edge_index": index,
                    "predicted_role": role,
                    "confidence": float(confidence[index]),
                    "start_cm": list(panel.edges[index].start_cm),
                    "end_cm": list(panel.edges[index].end_cm),
                }
                for index, role in enumerate(predicted_roles)
            ],
            "decoded_landmarks_cm": {name: list(point) for name, point in decode_named_landmarks(panel, predicted_roles).items()},
            "decoded_darts": decode_darts(panel, predicted_roles),
            "decoded_measurements": decode_path_measurements(panel, predicted_roles),
            "conditioned_reference_lines": [
                {
                    "name": line.name,
                    "points_cm": [list(point) for point in line.points_cm],
                    "evidence": line.evidence,
                    "training_eligible": line.training_eligible,
                }
                for line in panel.reference_lines
            ],
        }
        if args.include_source_comparison:
            panel_result["source_annotation_roles"] = [edge.role for edge in panel.edges]
        panels.append(panel_result)

    result = {
        "schema_version": "drafting-semantic-prediction-1.0",
        "sample_id": record.sample_id,
        "device": str(device),
        "model_scope": "GarmentCodeData_v2_symmetric_FittedShirt_and_Shirt_vector_panels",
        "panels": panels,
        "warnings": [
            "Reference lines are conditioned GarmentCode formulas, not predictions from the edge model.",
            "This checkpoint has not been validated on external CAD, textbook, raster, RGB, raglan, or princess-seam patterns.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sample_id": record.sample_id, "panel_count": len(panels), "output": args.output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
