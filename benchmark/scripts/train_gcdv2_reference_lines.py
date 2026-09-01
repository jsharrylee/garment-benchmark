from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from benchmark.drafting_semantics.dataset import read_records
from benchmark.drafting_semantics.reference_line_learning import (
    REFERENCE_LINES,
    build_reference_line_model,
    collate_reference_lines,
    reference_line_examples,
)


def evaluate(model, examples, device, batch_size):
    import torch

    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate_reference_lines(examples[start : start + batch_size])
            predicted = model(batch["features"].to(device), batch["valid"].to(device), batch["panel_roles"].to(device)).cpu()
            errors.append((predicted - batch["targets"]).abs() * batch["scales_cm"][:, None])
    values = torch.cat(errors).numpy()
    return {"mean_absolute_error_cm": {name: float(values[:, index].mean()) for index, name in enumerate(REFERENCE_LINES)}, "overall_mae_cm": float(values.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BL/WL drafting reference-line positions from vector panel geometry.")
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/records.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_neurosymbolic/reference_lines.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_training/reference_line_metrics.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_records(args.records)
    splits = {name: list(reference_line_examples(records, {source})) for name, source in (("train", "training"), ("validation", "validation"), ("test", "test"))}
    model = build_reference_line_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(splits["train"])
        model.train()
        losses = []
        for start in range(0, len(splits["train"]), args.batch_size):
            batch = collate_reference_lines(splits["train"][start : start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["features"].to(device), batch["valid"].to(device), batch["panel_roles"].to(device))
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch["targets"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, splits["validation"], device, args.batch_size)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation})
        if validation["overall_mae_cm"] < best:
            best = validation["overall_mae_cm"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    test = evaluate(model, splits["test"], device, args.batch_size)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "reference_lines": REFERENCE_LINES, "width": 96, "heads": 4, "layers": 2}, args.checkpoint)
    result = {
        "status": "PASS_TRAINED_REFERENCE_LINES",
        "device": str(device),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "training_seconds": time.perf_counter() - started,
        "split_panel_counts": {key: len(value) for key, value in splits.items()},
        "history": history,
        "test": test,
        "HL_status": "NOT_LEARNED_CONDITIONED_BODY_PRIOR",
        "HL_reason": "GCDv2 source marks HL as outside these bodice panels and the extracted batch contains one unique body payload.",
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "training_seconds", "test", "HL_status")}, indent=2))


if __name__ == "__main__":
    main()
