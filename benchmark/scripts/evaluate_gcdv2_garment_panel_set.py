from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.gcdv2_exact.garment_panel_set_learning import (
    GarmentPanelDataset,
    build_model,
    collate_garments,
    read_garments,
)
from benchmark.scripts.train_gcdv2_garment_panel_set import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate the selected garment-panel-set checkpoint.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_exact/garment_panel_set.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/test_metrics.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/test_predictions.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    garments, source_ids = read_garments(args.index)
    if tuple(source_ids) != tuple(checkpoint["source_ids"]):
        raise ValueError("source panel vocabulary drift")
    assignments = checkpoint["split_assignments"]
    test = [garment for garment in garments if assignments[garment.sample_id] == "test"]
    dataset = GarmentPanelDataset(test, source_ids, shuffle_panels=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=device.type == "cuda", collate_fn=collate_garments)
    model = build_model(len(source_ids), checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions = evaluate(model, loader, device, source_ids, export_predictions=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps({"status": "PASS", "checkpoint_epoch": checkpoint["epoch"], "test": metrics}, indent=2) + "\n", encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    print(json.dumps({"test_garments": len(test), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
