from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from benchmark.gcdv2_exact.neurosymbolic_learning import (
    VisualGeometryDataset,
    build_visual_model,
    read_panel_rows,
    visual_loss,
    visual_metrics,
)


def _move(batch, device):
    import torch

    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _evaluate(model, loader, device):
    import torch

    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["image"])
                losses.append(float(visual_loss(output, batch)["loss"]))
            metrics.append(visual_metrics(output, batch))
    return {
        "loss": float(np.mean(losses)),
        **{key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train raster-observable GCDv2 panel geometry before formal graph decoding.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_neurosymbolic/visual_geometry.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_training/visual_geometry_metrics.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rows = read_panel_rows(args.index)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    loaders = {
        split: DataLoader(
            VisualGeometryDataset(values),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=device.type == "cuda",
        )
        for split, values in split_rows.items()
    }
    model = build_visual_model(args.base_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for raw in loaders["train"]:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["image"])
                losses = visual_loss(output, batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(losses["loss"].detach()))
        validation = _evaluate(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch, stale = validation["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "base_width": args.base_width, "epoch": epoch}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = _evaluate(model, loaders["test"], device)
    result = {
        "status": "PASS_TRAINED_VISUAL_GEOMETRY",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "training_seconds": time.perf_counter() - started,
        "split_panel_counts": {key: len(value) for key, value in split_rows.items()},
        "best_epoch": best_epoch,
        "history": history,
        "test": test,
        "claim_boundary": "Raster-observable geometry only; exact source subdivisions and semantic drafting names are excluded.",
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "best_epoch", "training_seconds", "test")}, indent=2))


if __name__ == "__main__":
    main()
