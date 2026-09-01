from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from benchmark.drafting_semantics.merged_visible_learning import (
    LANDMARK_NAMES,
    MERGED_EDGE_ROLES,
    MergedVisibleDataset,
    build_merged_semantic_model,
    decode_landmarks,
)


def role_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    per_role, f1s = {}, []
    for index, role in enumerate(MERGED_EDGE_ROLES):
        tp = int(((predicted == index) & (target == index)).sum())
        fp = int(((predicted == index) & (target != index)).sum())
        fn = int(((predicted != index) & (target == index)).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = int((target == index).sum())
        per_role[role] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {"accuracy": float((predicted == target).mean()), "macro_f1": float(np.mean(f1s)), "per_role": per_role}


def evaluate(model, loader, device):
    import torch

    model.eval()
    predictions, targets = [], []
    landmark_target = landmark_decoded = landmark_pck = 0
    normalized_errors = []
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"].to(device), batch["valid"].to(device), batch["panel_role"].to(device))
            predicted = logits.argmax(-1).cpu().numpy()
            valid = batch["valid"].numpy()
            predictions.append(predicted[valid])
            targets.append(batch["roles"].numpy()[valid])
            for row in range(len(predicted)):
                count = int(valid[row].sum())
                roles = predicted[row, :count]
                vertices = batch["vertices_uv"][row, :count].numpy()
                decoded = decode_landmarks(roles, vertices, int(batch["panel_role"][row]))
                span = max(float(np.ptp(vertices, axis=0).max()), 1e-8)
                current = {"source": int(batch["source"][row]), "predicted_roles": roles.tolist(), "decoded_landmarks": {key: value.tolist() for key, value in decoded.items()}}
                for landmark_index, name in enumerate(LANDMARK_NAMES):
                    if not bool(batch["landmark_mask"][row, landmark_index]):
                        continue
                    landmark_target += 1
                    if name not in decoded:
                        continue
                    landmark_decoded += 1
                    error = float(np.linalg.norm(decoded[name] - batch["landmark_uv"][row, landmark_index].numpy()) / span)
                    normalized_errors.append(error)
                    landmark_pck += int(error <= 0.02)
                prediction_rows.append(current)
    predicted, target = np.concatenate(predictions), np.concatenate(targets)
    return {
        "edge_roles": role_metrics(predicted, target),
        "landmarks": {
            "target_count": landmark_target,
            "decoded_count": landmark_decoded,
            "decode_coverage": landmark_decoded / max(landmark_target, 1),
            "pck_at_2pct_panel_span": landmark_pck / max(landmark_target, 1),
            "mean_normalized_error_when_decoded": float(np.mean(normalized_errors)) if normalized_errors else None,
        },
    }, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train merged visible-edge semantics on learned-mask predicted contours.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/merged_semantics.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_end_to_end/merged_visible_semantics.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/gcdv2_end_to_end/merged_visible_semantics_metrics.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_end_to_end/merged_visible_semantics_test_predictions.jsonl"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.dataset)
    arrays = {key: raw[key] for key in raw.files}
    split = {"train": 0, "validation": 1, "test": 2}
    datasets = {name: MergedVisibleDataset(arrays, np.flatnonzero(arrays["splits"] == code), augment=name == "train") for name, code in split.items()}
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train", num_workers=0, pin_memory=True) for name, dataset in datasets.items()}
    model = build_merged_semantic_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    valid_train_roles = arrays["edge_roles"][(arrays["splits"] == 0)[:, None] & arrays["valid_edges"]]
    counts = np.bincount(valid_train_roles, minlength=len(MERGED_EDGE_ROLES)).astype(np.float32)
    weights = torch.from_numpy(np.sqrt(counts.sum() / np.maximum(counts, 1))).to(device); weights /= weights.mean()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_state, history = (-1.0, -1.0), None, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch["features"].to(device), batch["valid"].to(device), batch["panel_role"].to(device))
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), batch["roles"].to(device), weight=weights, ignore_index=-100)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        validation, _ = evaluate(model, loaders["validation"], device)
        score = (validation["landmarks"]["pck_at_2pct_panel_span"], validation["edge_roles"]["macro_f1"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        print(json.dumps({"epoch": epoch, "loss": row["train_loss"], "edge_macro_f1": score[1], "landmark_pck": score[0]}), flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test, predictions = evaluate(model, loaders["test"], device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "edge_roles": MERGED_EDGE_ROLES, "panel_roles": ("front_bodice", "back_bodice"), "input_contract": "predicted-contour unit-chord segments"}, args.checkpoint)
    result = {
        "status": "PASS_TRAINED_PREDICTED_CONTOUR_SEMANTICS", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(value.numel() for value in model.parameters()), "training_seconds": time.perf_counter() - started,
        "split_panel_counts": {key: len(value) for key, value in datasets.items()}, "history": history, "test": test,
        "pipeline_input": "panel.png -> learned mask -> predicted contour -> source-visible segmentation targets during training -> intrinsic segment graph",
        "remaining_oracle": "front/back bodice panel role is supplied; visible-vertex test below must use predicted corner model for fully predicted topology",
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True); args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.predictions.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    print(json.dumps({"status": result["status"], "test": test}, indent=2))


if __name__ == "__main__":
    main()
