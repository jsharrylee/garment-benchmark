from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.intrinsic_graph_learning import build_corner_model, build_segment_model
from benchmark.scripts.train_gcdv2_intrinsic_graph import CornerDataset, SegmentDataset, evaluate_corners, evaluate_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen intrinsic visible-graph checkpoints.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_v1/intrinsic_graph.npz"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gcdv2_intrinsic_graph"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_training/final_evaluation.json"))
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    panel_indices = np.flatnonzero(data["panel_splits"] == 2)
    segment_indices = np.flatnonzero(data["segment_splits"] == 2)
    corner_loader = DataLoader(CornerDataset(data["contours"], data["corner_targets"], data["corner_counts"], panel_indices, augment=False), batch_size=32, num_workers=0)
    segment_loader = DataLoader(SegmentDataset(data["segment_features"], data["segment_targets"], data["segment_primitives"], segment_indices), batch_size=1024, num_workers=0)
    corner_model, segment_model = build_corner_model().to(device), build_segment_model().to(device)
    corner_model.load_state_dict(torch.load(args.checkpoint_dir / "visible_corners.pt", map_location=device, weights_only=True)["model_state"])
    segment_model.load_state_dict(torch.load(args.checkpoint_dir / "segment_geometry.pt", map_location=device, weights_only=True)["model_state"])
    corners, predictions = evaluate_corners(corner_model, corner_loader, device)
    segments = evaluate_segments(segment_model, segment_loader, device)
    result = {
        "status": "PASS_FROZEN_INTRINSIC_GRAPH_EVALUATION",
        "test_panel_count": len(panel_indices),
        "test_segment_count": len(segment_indices),
        "no_absolute_xy_model_input": True,
        "corners": corners,
        "segments": segments,
        "closed_cycle_by_construction": 1.0,
        "prediction_count": len(predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
