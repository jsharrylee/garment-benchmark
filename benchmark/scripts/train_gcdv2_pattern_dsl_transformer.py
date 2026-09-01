from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES, PANEL_ROLES
from benchmark.gcdv2_exact.pattern_dsl_learning import (
    CATEGORIES,
    CURVE_COMMANDS,
    EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
    LANDMARK_NAMES,
    MASK_COMMAND,
    PatternDSLArrayDataset,
    build_pattern_dsl_model,
    grammar_transition_matrix,
)


LANDMARK_ROLE_PAIRS = (
    (EDGE_ROLES.index("center_front"), EDGE_ROLES.index("neckline")),
    (EDGE_ROLES.index("center_back"), EDGE_ROLES.index("neckline")),
    (EDGE_ROLES.index("neckline"), EDGE_ROLES.index("shoulder")),
    (EDGE_ROLES.index("shoulder"), EDGE_ROLES.index("armhole")),
)


def classification_metrics(predicted: np.ndarray, target: np.ndarray, names) -> dict:
    valid = target >= 0
    predicted, target = predicted[valid], target[valid]
    per_class, f1s = {}, []
    for index, name in enumerate(names):
        tp = int(((predicted == index) & (target == index)).sum())
        fp = int(((predicted == index) & (target != index)).sum())
        fn = int(((predicted != index) & (target == index)).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = int((target == index).sum())
        per_class[str(name)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {
        "accuracy": float((predicted == target).mean()) if len(target) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
    }


def seam_targets(batch, device):
    import torch

    batch_size = len(batch["category"])
    edges_per_panel = int(batch["edge_valid"].shape[2])
    size = batch["edge_valid"].shape[1] * edges_per_panel
    target = torch.zeros((batch_size, size, size), dtype=torch.bool, device=device)
    for row in range(batch_size):
        for pair, valid in zip(batch["stitch_pairs"][row], batch["stitch_valid"][row], strict=True):
            if not bool(valid):
                continue
            first = int(pair[0]) * edges_per_panel + int(pair[1])
            second = int(pair[2]) * edges_per_panel + int(pair[3])
            target[row, first, second] = True; target[row, second, first] = True
    return target


def masked_cross_entropy(logits, target, *, weight=None):
    """Cross entropy that is exactly zero when a batch has no supervision.

    ``torch.nn.functional.cross_entropy(..., ignore_index=-100)`` returns NaN
    when every target is ignored because its mean has a zero denominator.  The
    semantic corpus intentionally omits some garments, so all-unlabelled
    batches are expected rather than exceptional.
    """
    import torch.nn.functional as F

    valid = target >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid], weight=weight)


def training_class_weights(labels: np.ndarray, splits: np.ndarray, class_count: int):
    """Build balanced weights from the training partition only.

    Validation and test label frequencies are deliberately excluded so model
    selection does not consume holdout distribution information.
    """
    train_values = np.asarray(labels)[np.asarray(splits) == 0].reshape(-1)
    train_values = train_values[train_values >= 0]
    counts = np.bincount(train_values, minlength=class_count).astype(np.float32)
    weights = np.zeros(class_count, np.float32)
    present = counts > 0
    if present.any():
        weights[present] = np.sqrt(counts[present].sum() / counts[present])
        weights[present] /= weights[present].mean()
    return weights


def sampled_seam_loss(logits, target, edge_valid):
    import torch
    import torch.nn.functional as F

    flat_valid = edge_valid.flatten(1)
    pair_valid = flat_valid[:, :, None] & flat_valid[:, None, :]
    upper = torch.triu(torch.ones_like(pair_valid), diagonal=1)
    eligible = pair_valid & upper
    positive = target & eligible
    negative = ~target & eligible
    selected = positive.clone()
    for row in range(len(logits)):
        negative_indices = torch.nonzero(negative[row].flatten(), as_tuple=False).flatten()
        amount = min(len(negative_indices), max(int(positive[row].sum()) * 5, 32))
        if amount:
            choice = negative_indices[torch.randperm(len(negative_indices), device=logits.device)[:amount]]
            selected[row].view(-1)[choice] = True
    values, labels = logits[selected], target[selected].to(logits.dtype)
    if not len(values):
        return logits.sum() * 0.0
    positive_count = labels.sum().clamp_min(1)
    negative_count = (1 - labels).sum().clamp_min(1)
    positive_weight = (negative_count / positive_count).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(values, labels, pos_weight=positive_weight)


def junction_loss(edge_logits, landmarks, edge_valid):
    import torch

    log_probability = edge_logits.log_softmax(-1)
    losses = []
    for batch in range(edge_logits.shape[0]):
        for panel in range(edge_logits.shape[1]):
            count = int(edge_valid[batch, panel].sum())
            for vertex in range(count):
                landmark = int(landmarks[batch, panel, vertex])
                if landmark < 0:
                    continue
                first, second = LANDMARK_ROLE_PAIRS[landmark]
                previous = (vertex - 1) % count
                score = torch.logsumexp(
                    torch.stack((
                        log_probability[batch, panel, previous, first] + log_probability[batch, panel, vertex, second],
                        log_probability[batch, panel, previous, second] + log_probability[batch, panel, vertex, first],
                    )), dim=0,
                )
                losses.append(-score)
    return torch.stack(losses).mean() if losses else edge_logits.sum() * 0.0


def grammar_loss(edge_logits, edge_roles, edge_valid, allowed):
    import torch

    probability = edge_logits.softmax(-1)
    illegal = (~allowed).to(probability.dtype)
    values = []
    for batch in range(edge_logits.shape[0]):
        for panel in range(edge_logits.shape[1]):
            count = int(edge_valid[batch, panel].sum())
            if count < 2:
                continue
            targets = edge_roles[batch, panel, :count]
            supervised = (targets >= 0) & (torch.roll(targets, -1) >= 0)
            current, following = probability[batch, panel, :count], torch.roll(probability[batch, panel, :count], -1, 0)
            penalty = torch.einsum("ir,rs,is->i", current, illegal, following)
            if supervised.any():
                values.append(penalty[supervised].mean())
    return torch.stack(values).mean() if values else edge_logits.sum() * 0.0


def compute_loss(output, batch, device, weights, allowed):
    import torch.nn.functional as F

    category = F.cross_entropy(output["category_logits"], batch["category"].to(device))
    panel_target = batch["panel_roles"].to(device)
    panel = masked_cross_entropy(output["panel_role_logits"], panel_target, weight=weights["panel"])
    edge_target = batch["edge_roles"].to(device)
    edge = masked_cross_entropy(output["edge_role_logits"], edge_target, weight=weights["edge"])
    command_mask = batch["command_mask"].to(device)
    command = masked_cross_entropy(
        output["command_logits"],
        batch["command_targets"].to(device).masked_fill(~command_mask, -100),
    )
    seam_target = seam_targets(batch, device)
    seam = sampled_seam_loss(output["seam_logits"], seam_target, batch["edge_valid"].to(device))
    junction = junction_loss(output["edge_role_logits"], batch["landmarks"].to(device), batch["edge_valid"].to(device))
    grammar = grammar_loss(output["edge_role_logits"], edge_target, batch["edge_valid"].to(device), allowed)
    total = category * 0.3 + panel * 0.6 + edge + command * 0.25 + seam * 0.7 + junction * 0.45 + grammar * 0.2
    return total, {"category": category, "panel": panel, "edge": edge, "command": command, "seam": seam, "junction": junction, "grammar": grammar}


def _best_threshold(scores: np.ndarray, target: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 37):
        predicted = scores >= threshold
        tp = int((predicted & target).sum()); fp = int((predicted & ~target).sum()); fn = int((~predicted & target).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[0]: best = (f1, float(threshold))
    return best[1]


def evaluate(model, loader, device, allowed, *, seam_threshold: float | None = None):
    import torch

    model.eval()
    category_pred, category_target = [], []
    panel_pred, panel_target = [], []
    edge_pred, edge_target = [], []
    command_pred, command_target = [], []
    command_valid = []
    seam_scores, seam_truth = [], []
    seam_top1_correct = seam_top1_total = grammar_bad = grammar_total = 0
    landmark_target = landmark_covered = landmark_exact = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device); edge_valid = batch["edge_valid"].to(device); panel_valid = batch["panel_valid"].to(device)
            output = model(features, batch["commands"].to(device), edge_valid, panel_valid)
            category_pred.append(output["category_logits"].argmax(-1).cpu().numpy()); category_target.append(batch["category"].numpy())
            panel_values = output["panel_role_logits"].argmax(-1).cpu().numpy(); edge_values = output["edge_role_logits"].argmax(-1).cpu().numpy()
            panel_pred.append(panel_values); panel_target.append(batch["panel_roles"].numpy())
            edge_pred.append(edge_values); edge_target.append(batch["edge_roles"].numpy())
            masked_commands = torch.full_like(batch["commands"], MASK_COMMAND).to(device)
            primitive = model(features, masked_commands, edge_valid, panel_valid)["command_logits"].argmax(-1).cpu().numpy()
            command_pred.append(primitive); command_target.append(batch["command_targets"].numpy())
            command_valid.append(batch["edge_valid"].numpy())
            seam_target_batch = seam_targets(batch, device)
            flat_valid = edge_valid.flatten(1); pair_valid = flat_valid[:, :, None] & flat_valid[:, None, :]
            upper = torch.triu(torch.ones_like(pair_valid), 1); eligible = pair_valid & upper
            probability = output["seam_logits"].sigmoid()
            seam_scores.append(probability[eligible].cpu().numpy()); seam_truth.append(seam_target_batch[eligible].cpu().numpy())
            for row in range(len(probability)):
                truth = seam_target_batch[row]
                valid_indices = torch.nonzero(flat_valid[row], as_tuple=False).flatten()
                for first in valid_indices:
                    mates = torch.nonzero(truth[first], as_tuple=False).flatten()
                    if not len(mates): continue
                    candidate = probability[row, first].clone(); candidate[~flat_valid[row]] = -1; candidate[first] = -1
                    seam_top1_correct += int(int(candidate.argmax()) in mates.tolist()); seam_top1_total += 1
            for row in range(len(edge_values)):
                for panel in range(edge_values.shape[1]):
                    count = int(batch["edge_valid"][row, panel].sum())
                    if not count: continue
                    roles = edge_values[row, panel, :count]
                    grammar_bad += sum(not bool(allowed[int(a), int(b)]) for a, b in zip(roles, np.roll(roles, -1)))
                    grammar_total += count
                    candidates = {name: [] for name in LANDMARK_NAMES}
                    for vertex in range(count):
                        adjacent = {int(roles[(vertex - 1) % count]), int(roles[vertex])}
                        for landmark_index, pair in enumerate(LANDMARK_ROLE_PAIRS):
                            if adjacent == set(pair): candidates[LANDMARK_NAMES[landmark_index]].append(vertex)
                    for vertex in range(count):
                        target_landmark = int(batch["landmarks"][row, panel, vertex])
                        if target_landmark < 0: continue
                        landmark_target += 1; values = candidates[LANDMARK_NAMES[target_landmark]]
                        landmark_covered += int(bool(values)); landmark_exact += int(vertex in values)
    category_pred, category_target = np.concatenate(category_pred), np.concatenate(category_target)
    panel_pred, panel_target = np.concatenate(panel_pred), np.concatenate(panel_target)
    edge_pred, edge_target = np.concatenate(edge_pred), np.concatenate(edge_target)
    command_pred, command_target = np.concatenate(command_pred), np.concatenate(command_target)
    command_valid = np.concatenate(command_valid)
    scores, truth = np.concatenate(seam_scores), np.concatenate(seam_truth).astype(bool)
    threshold = _best_threshold(scores, truth) if seam_threshold is None else seam_threshold
    seam_prediction = scores >= threshold
    tp = int((seam_prediction & truth).sum()); fp = int((seam_prediction & ~truth).sum()); fn = int((~seam_prediction & truth).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result = {
        "category_accuracy": float((category_pred == category_target).mean()),
        "panel_roles": classification_metrics(panel_pred, panel_target, PANEL_ROLES),
        "edge_roles": classification_metrics(edge_pred, edge_target, EDGE_ROLES),
        "masked_command_accuracy": float(
            (command_pred[command_valid] == command_target[command_valid]).mean()
        ) if command_valid.any() else 0.0,
        "seams": {"threshold": threshold, "pair_precision": precision, "pair_recall": recall, "pair_f1": 2 * precision * recall / max(precision + recall, 1e-12), "mate_recall_at_1": seam_top1_correct / max(seam_top1_total, 1), "positive_pairs": int(truth.sum())},
        "grammar_violation_rate": grammar_bad / max(grammar_total, 1),
        "landmarks": {"target_count": landmark_target, "decode_coverage": landmark_covered / max(landmark_target, 1), "exact_vertex_accuracy": landmark_exact / max(landmark_target, 1)},
    }
    return result, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaGeometry-style Pattern DSL proposition model.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gcdv2_pattern_dsl/unified_transformer.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_training/metrics.json"))
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.dataset); arrays = {key: raw[key] for key in raw.files}
    feature_schema = str(
        arrays.get(
            "edge_feature_schema",
            np.asarray(EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1),
        ).item()
    )
    split = {"train": 0, "validation": 1, "test": 2}
    datasets = {name: PatternDSLArrayDataset(arrays, np.flatnonzero(arrays["splits"] == code), mask_commands=name == "train") for name, code in split.items()}
    loaders = {name: DataLoader(value, batch_size=args.batch_size, shuffle=name == "train", num_workers=args.workers, pin_memory=True) for name, value in datasets.items()}
    model = build_pattern_dsl_model(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = {
        "panel": torch.from_numpy(training_class_weights(arrays["panel_roles"], arrays["splits"], len(PANEL_ROLES))).to(device),
        "edge": torch.from_numpy(training_class_weights(arrays["edge_roles"], arrays["splits"], len(EDGE_ROLES))).to(device),
    }
    allowed_np = grammar_transition_matrix(arrays["edge_roles"], arrays["edge_valid"], arrays["splits"])
    allowed = torch.from_numpy(allowed_np).to(device)
    best_score, best_state, best_threshold, history = -1.0, None, 0.5, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["features"].to(device), batch["commands"].to(device), batch["edge_valid"].to(device), batch["panel_valid"].to(device))
                loss, parts = compute_loss(output, batch, device, weights, allowed)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        validation, threshold = evaluate(model, loaders["validation"], device, allowed_np)
        score = validation["edge_roles"]["macro_f1"] + 0.5 * validation["panel_roles"]["macro_f1"] + 0.5 * validation["seams"]["mate_recall_at_1"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(row); print(json.dumps({"epoch": epoch, "loss": row["train_loss"], "edge_f1": validation["edge_roles"]["macro_f1"], "panel_f1": validation["panel_roles"]["macro_f1"], "seam_r1": validation["seams"]["mate_recall_at_1"], "landmark": validation["landmarks"]["exact_vertex_accuracy"]}), flush=True)
        if score > best_score:
            best_score = score; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; best_threshold = threshold
    model.load_state_dict(best_state)
    test, _ = evaluate(model, loaders["test"], device, allowed_np, seam_threshold=best_threshold)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "width": args.width, "edge_roles": EDGE_ROLES, "panel_roles": PANEL_ROLES, "curve_commands": CURVE_COMMANDS, "allowed_transitions": allowed_np, "seam_threshold": best_threshold, "edge_feature_schema": feature_schema, "representation": "Pattern DSL exact geometry facts; no raster or absolute xy"}, args.checkpoint)
    result = {"status": "PASS_TRAINED_PATTERN_DSL_PROPOSER", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "parameter_count": sum(value.numel() for value in model.parameters()), "training_seconds": time.perf_counter() - started, "edge_feature_schema": feature_schema, "split_counts": {key: len(value) for key, value in datasets.items()}, "history": history, "test": test, "claim_boundary": "Exact GCDv2 geometry/SVG facts -> derived semantic and seam propositions. This is not image-to-DSL or expert cross-source validation."}
    args.metrics.parent.mkdir(parents=True, exist_ok=True); args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "test": test}, indent=2))


if __name__ == "__main__":
    main()
