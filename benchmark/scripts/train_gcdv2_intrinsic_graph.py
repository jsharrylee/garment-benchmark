from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from benchmark.gcdv2_exact.intrinsic_graph_learning import (
    PRIMITIVES,
    build_corner_model,
    build_segment_model,
    intrinsic_contour_features,
    select_cyclic_peaks,
)


class CornerDataset:
    def __init__(self, contours, targets, counts, indices, *, augment: bool) -> None:
        self.contours, self.targets, self.counts = contours, targets, counts
        self.indices = np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = int(self.indices[index])
        contour = self.contours[source].astype(np.float32)
        target = self.targets[source].astype(np.float32)
        if self.augment and random.random() < 0.5:
            contour = contour[::-1].copy()
            target = target[::-1].copy()
        if self.augment:
            shift = random.randrange(len(contour))
            contour = np.roll(contour, shift, axis=0)
            target = np.roll(target, shift)
        return intrinsic_contour_features(contour), target, int(self.counts[source]), source


class SegmentDataset:
    def __init__(self, features, targets, primitives, indices) -> None:
        self.features, self.targets, self.primitives = features, targets, primitives
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = int(self.indices[index])
        return self.features[source].astype(np.float32), self.targets[source].astype(np.float32), int(self.primitives[source])


def corner_loss(output, target, count):
    import torch
    import torch.nn.functional as F

    probability = output["corner_logits"].sigmoid()
    bce = F.binary_cross_entropy_with_logits(output["corner_logits"], target, reduction="none")
    pt = target * probability + (1 - target) * (1 - probability)
    weight = 1.0 + 7.0 * target
    focal = (bce * (1 - pt).square() * weight).mean()
    count_loss = F.cross_entropy(output["count_logits"], count)
    # Total probability should remain compatible with the visible vertex count.
    mass = F.smooth_l1_loss(probability.sum(1), count.float())
    return focal + count_loss + 0.05 * mass


def _circular_distance(a: int, b: int, size: int = 256) -> int:
    return min((a - b) % size, (b - a) % size)


def _match_peaks(predicted, target, tolerance=2):
    remaining = set(int(value) for value in target)
    matched, distances = 0, []
    for value in predicted:
        if not remaining:
            break
        candidate = min(remaining, key=lambda item: _circular_distance(value, item))
        distance = _circular_distance(value, candidate)
        if distance <= tolerance:
            matched += 1
            distances.append(distance)
            remaining.remove(candidate)
    return matched, distances


def evaluate_corners(model, loader, device):
    import torch

    model.eval()
    tp = predicted_total = target_total = count_exact = 0
    distances = []
    rows = []
    with torch.no_grad():
        for features, target, count, source in loader:
            output = model(features.to(device))
            probability = output["corner_logits"].sigmoid().cpu().numpy()
            predicted_counts = output["count_logits"].argmax(-1).cpu().numpy()
            for row in range(len(probability)):
                expected = np.flatnonzero(target[row].numpy() >= 0.999)
                predicted_count = int(np.clip(predicted_counts[row], 3, 36))
                predicted = select_cyclic_peaks(probability[row], predicted_count)
                matched, current_distances = _match_peaks(predicted, expected)
                tp += matched
                predicted_total += len(predicted)
                target_total += len(expected)
                count_exact += int(predicted_count == len(expected))
                distances.extend(current_distances)
                rows.append({"panel_index": int(source[row]), "target_indices": expected.tolist(), "predicted_indices": predicted, "predicted_count": predicted_count})
    precision = tp / max(predicted_total, 1)
    recall = tp / max(target_total, 1)
    return {
        "visible_vertex_precision_at_2": precision,
        "visible_vertex_recall_at_2": recall,
        "visible_vertex_f1_at_2": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_matched_index_error": float(np.mean(distances)) if distances else None,
        "count_exact_accuracy": count_exact / max(len(rows), 1),
    }, rows


def classification_metrics(predicted, target):
    per_class, f1s = {}, []
    for index, name in enumerate(PRIMITIVES):
        tp = int(((predicted == index) & (target == index)).sum())
        fp = int(((predicted == index) & (target != index)).sum())
        fn = int(((predicted != index) & (target == index)).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = int((target == index).sum())
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {"accuracy": float((predicted == target).mean()), "macro_f1": float(np.mean(f1s)), "per_class": per_class}


def evaluate_segments(model, loader, device):
    import torch

    model.eval()
    predictions, targets, geometry_errors, tangent_angles = [], [], [], []
    curve_rmse, curve_hausdorff = [], []
    with torch.no_grad():
        for features, geometry, primitive in loader:
            output = model(features.to(device))
            predictions.append(output["primitive_logits"].argmax(-1).cpu().numpy())
            targets.append(primitive.numpy())
            predicted_geometry = output["geometry"].cpu()
            geometry_errors.append((predicted_geometry[:, :5] - geometry[:, :5]).abs().numpy())
            true_curve = features[:, :, :2]
            t = torch.linspace(0.0, 1.0, true_curve.shape[1])[None, :, None]
            p0 = torch.zeros((len(features), 1, 2))
            p3 = torch.tensor([1.0, 0.0])[None, None].expand(len(features), -1, -1)
            p1 = predicted_geometry[:, None, 0:2]
            p2 = predicted_geometry[:, None, 2:4]
            predicted_curve = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            curve_rmse.extend(torch.sqrt(torch.square(predicted_curve - true_curve).sum(-1).mean(-1)).numpy().tolist())
            distances = torch.cdist(predicted_curve, true_curve)
            hausdorff = torch.maximum(distances.min(-1).values.max(-1).values, distances.min(-2).values.max(-1).values)
            curve_hausdorff.extend(hausdorff.numpy().tolist())
            for offset in (5, 7):
                cosine = (predicted_geometry[:, offset : offset + 2] * geometry[:, offset : offset + 2]).sum(1).clamp(-1, 1)
                tangent_angles.extend(torch.rad2deg(torch.acos(cosine)).numpy().tolist())
    predicted, target = np.concatenate(predictions), np.concatenate(targets)
    errors = np.concatenate(geometry_errors)
    return {
        "primitive": classification_metrics(predicted, target),
        "relative_control_component_mae": float(errors[:, :4].mean()),
        "arc_over_chord_mae": float(errors[:, 4].mean()),
        "endpoint_tangent_mae_deg": float(np.mean(tangent_angles)),
        "cubic_reconstruction_rmse_over_chord": float(np.mean(curve_rmse)),
        "cubic_reconstruction_hausdorff_over_chord": float(np.mean(curve_hausdorff)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intrinsic visible-vertex and segment-geometry models without absolute x/y.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_v1/intrinsic_graph.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_intrinsic_graph_training"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gcdv2_intrinsic_graph"))
    parser.add_argument("--corner-epochs", type=int, default=8)
    parser.add_argument("--segment-epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    split = {name: code for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    corner_datasets = {
        name: CornerDataset(data["contours"], data["corner_targets"], data["corner_counts"], np.flatnonzero(data["panel_splits"] == code), augment=name == "train")
        for name, code in split.items()
    }
    corner_loaders = {
        name: DataLoader(dataset, batch_size=32, shuffle=name == "train", num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
        for name, dataset in corner_datasets.items()
    }
    corner_model = build_corner_model().to(device)
    optimizer = torch.optim.AdamW(corner_model.parameters(), lr=3e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1, best_corner_state, corner_history = -1.0, None, []
    started = time.perf_counter()
    for epoch in range(1, args.corner_epochs + 1):
        corner_model.train()
        losses = []
        for features, target, count, _ in corner_loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = corner_model(features.to(device))
                loss = corner_loss(output, target.to(device), count.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(corner_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation, _ = evaluate_corners(corner_model, corner_loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        corner_history.append(row)
        print(json.dumps({"corner": row}), flush=True)
        if validation["visible_vertex_f1_at_2"] > best_f1:
            best_f1 = validation["visible_vertex_f1_at_2"]
            best_corner_state = {key: value.detach().cpu().clone() for key, value in corner_model.state_dict().items()}
    corner_seconds = time.perf_counter() - started
    corner_model.load_state_dict(best_corner_state)
    corner_test, corner_predictions = evaluate_corners(corner_model, corner_loaders["test"], device)

    segment_datasets = {
        name: SegmentDataset(data["segment_features"], data["segment_targets"], data["segment_primitives"], np.flatnonzero(data["segment_splits"] == code))
        for name, code in split.items()
    }
    # On Windows these in-memory arrays must not be copied into several worker
    # processes.  One large pinned GPU batch is substantially faster and uses
    # less host memory than worker-side copies of the 93k segment corpus.
    segment_loaders = {
        name: DataLoader(dataset, batch_size=1024, shuffle=name == "train", num_workers=0, pin_memory=True)
        for name, dataset in segment_datasets.items()
    }
    segment_model = build_segment_model().to(device)
    optimizer = torch.optim.AdamW(segment_model.parameters(), lr=3e-4, weight_decay=1e-3)
    class_counts = np.bincount(data["segment_primitives"][data["segment_splits"] == 0], minlength=len(PRIMITIVES)).astype(np.float32)
    class_weights = torch.from_numpy(np.sqrt(class_counts.sum() / np.maximum(class_counts, 1))).to(device)
    class_weights /= class_weights.mean()
    best_macro, best_segment_state, segment_history = -1.0, None, []
    segment_started = time.perf_counter()
    for epoch in range(1, args.segment_epochs + 1):
        segment_model.train()
        losses = []
        for features, geometry, primitive in segment_loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = segment_model(features.to(device))
                classification = torch.nn.functional.cross_entropy(output["primitive_logits"], primitive.to(device), weight=class_weights)
                regression = torch.nn.functional.smooth_l1_loss(output["geometry"], geometry.to(device))
                loss = classification + 2.0 * regression
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(segment_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation = evaluate_segments(segment_model, segment_loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        segment_history.append(row)
        print(json.dumps({"segment": {"epoch": epoch, "train_loss": row["train_loss"], "macro_f1": validation["primitive"]["macro_f1"]}}), flush=True)
        if validation["primitive"]["macro_f1"] > best_macro:
            best_macro = validation["primitive"]["macro_f1"]
            best_segment_state = {key: value.detach().cpu().clone() for key, value in segment_model.state_dict().items()}
    segment_seconds = time.perf_counter() - segment_started
    segment_model.load_state_dict(best_segment_state)
    segment_test = evaluate_segments(segment_model, segment_loaders["test"], device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_corner_state, "feature_contract": "intrinsic_contour_25", "no_absolute_xy": True}, args.checkpoint_dir / "visible_corners.pt")
    torch.save({"model_state": best_segment_state, "feature_contract": "unit_chord_segment_8", "no_absolute_xy": True}, args.checkpoint_dir / "segment_geometry.pt")
    result = {
        "status": "PASS_TRAINED_INTRINSIC_VISIBLE_GRAPH",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "input_contract": "no absolute x/y; cyclic length ratios, turning invariants, unit-chord local geometry",
        "corner_parameter_count": sum(value.numel() for value in corner_model.parameters()),
        "segment_parameter_count": sum(value.numel() for value in segment_model.parameters()),
        "corner_training_seconds": corner_seconds,
        "segment_training_seconds": segment_seconds,
        "corner_history": corner_history,
        "segment_history": segment_history,
        "test": {"corners": corner_test, "segments": segment_test, "closed_cycle_by_construction": 1.0},
        "test_corner_predictions": corner_predictions,
        "claim_boundary": (
            "Learned raster mask/SDF contours feed this phase; the downstream full-pipeline evaluator must also use predicted vertices."
            if "predicted" in args.dataset.as_posix().lower()
            else "Ground-truth dense contours feed this phase; connection to the learned raster mask/SDF contour remains a later end-to-end step."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
