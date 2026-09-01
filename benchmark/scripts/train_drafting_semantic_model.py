from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.dataset import (
    augment_boundary_serialization,
    balanced_class_weights,
    padded_batch,
    panel_examples,
    read_records,
    reindex_panel_example,
)
from benchmark.drafting_semantics.decoding import landmark_error_summary
from benchmark.drafting_semantics.model import DEFAULT_MODEL_CONFIG, build_model
from benchmark.drafting_semantics.schema import EDGE_ROLES


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    valid = targets >= 0
    predictions, targets = predictions[valid], targets[valid]
    per_role = {}
    f1_values = []
    for role_id, role in enumerate(EDGE_ROLES):
        tp = int(np.sum((predictions == role_id) & (targets == role_id)))
        fp = int(np.sum((predictions == role_id) & (targets != role_id)))
        fn = int(np.sum((predictions != role_id) & (targets == role_id)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        support = int(np.sum(targets == role_id))
        per_role[role] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if role != "other" and support:
            f1_values.append(f1)
    return {
        "accuracy": float(np.mean(predictions == targets)) if len(targets) else 0.0,
        "semantic_macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_role": per_role,
        "edge_count": int(len(targets)),
    }


def _evaluate(model, examples, config, device):
    import torch

    model.eval()
    all_predictions, all_targets = [], []
    landmark_totals = {"target_count": 0, "decoded_count": 0, "exact_count": 0, "normalized_distance_sum": 0.0}
    batch_size = int(config["batch_size"])
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            features, targets, valid, panel_roles = padded_batch(examples[start : start + batch_size], int(config["maximum_edges"]))
            logits = model(
                torch.from_numpy(features).to(device),
                torch.from_numpy(valid).to(device),
                torch.from_numpy(panel_roles).to(device),
            )
            prediction = logits.argmax(dim=-1).cpu().numpy()
            all_predictions.append(prediction[valid])
            all_targets.append(targets[valid])
            for row, example in enumerate(examples[start : start + batch_size]):
                count = min(len(example.targets), int(config["maximum_edges"]))
                canonical_prediction = np.full(len(example.panel.edges), EDGE_ROLES.index("other"), dtype=np.int64)
                canonical_prediction[example.edge_indices[:count]] = prediction[row, :count]
                summary = landmark_error_summary(example.panel, canonical_prediction)
                for key in landmark_totals:
                    landmark_totals[key] += summary[key]
    if not all_predictions:
        return {"accuracy": 0.0, "semantic_macro_f1": 0.0, "per_role": {}, "edge_count": 0}
    result = _metrics(np.concatenate(all_predictions), np.concatenate(all_targets))
    target_count = int(landmark_totals["target_count"])
    decoded_count = int(landmark_totals["decoded_count"])
    result["landmarks"] = {
        "target_count": target_count,
        "decoded_count": decoded_count,
        "decode_coverage": decoded_count / max(target_count, 1),
        "exact_vertex_accuracy": int(landmark_totals["exact_count"]) / max(target_count, 1),
        "mean_normalized_distance_when_decoded": float(landmark_totals["normalized_distance_sum"]) / max(decoded_count, 1),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small vector-pattern semantic edge model.")
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/records.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/drafting_semantics_gcdv2.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/gcdv2_edge_semantics.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_batch0/training_metrics.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    config = dict(DEFAULT_MODEL_CONFIG)
    config.update({"batch_size": 64, "epochs": 12, "learning_rate": 3e-4, "weight_decay": 1e-3, "seed": 2026, "include_stitch_features": False, "boundary_reindex_augmentations": 2})
    if args.config.is_file():
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    records = read_records(args.records)
    include_stitches = bool(config.get("include_stitch_features", False))
    train_base = panel_examples(records, splits={"training"}, include_stitch_features=include_stitches)
    train = list(
        augment_boundary_serialization(
            train_base,
            variants=int(config.get("boundary_reindex_augmentations", 0)),
            seed=seed,
        )
    )
    validation = panel_examples(records, splits={"validation"}, include_stitch_features=include_stitches)
    test = panel_examples(records, splits={"test"}, include_stitch_features=include_stitches)
    test_reindexed = tuple(
        reindex_panel_example(example, shift=max(len(example.targets) // 3, 1), reverse=True)
        for example in test
    )
    if not train:
        raise SystemExit("no official training examples in semantic records")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    weights = torch.from_numpy(balanced_class_weights(train)).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(seed)

    history = []
    training_started = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        generator.shuffle(train)
        model.train()
        losses = []
        for start in range(0, len(train), int(config["batch_size"])):
            features, targets, valid, panel_roles = padded_batch(train[start : start + int(config["batch_size"])], int(config["maximum_edges"]))
            features_t = torch.from_numpy(features).to(device)
            targets_t = torch.from_numpy(targets).to(device)
            valid_t = torch.from_numpy(valid).to(device)
            panel_roles_t = torch.from_numpy(panel_roles).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(features_t, valid_t, panel_roles_t)
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets_t, weight=weights, ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_metrics = _evaluate(model, validation, config, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation_metrics}
        history.append(row)
        print(json.dumps({"epoch": epoch + 1, "train_loss": row["train_loss"], "validation_macro_f1": validation_metrics["semantic_macro_f1"]}), flush=True)
    training_seconds = time.perf_counter() - training_started

    test_metrics = _evaluate(model, test, config, device)
    test_reindexed_metrics = _evaluate(model, test_reindexed, config, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "edge_roles": EDGE_ROLES,
            "records_sha256": __import__("hashlib").sha256(args.records.read_bytes()).hexdigest(),
        },
        args.checkpoint,
    )
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    result = {
        "status": "PASS_TRAINED_VECTOR_SEMANTIC_BASELINE",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": peak_memory,
        "training_seconds": training_seconds,
        "training_panel_presentations_per_second": len(train) * int(config["epochs"]) / max(training_seconds, 1e-9),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_base_panel_count": len(train_base),
        "train_panel_count": len(train),
        "validation_panel_count": len(validation),
        "test_panel_count": len(test),
        "include_stitch_features": include_stitches,
        "history": history,
        "test": test_metrics,
        "test_reindexed_boundary": test_reindexed_metrics,
        "limitations": [
            "predicts semantic roles on 2D vector pattern edges, not directly from a clothed RGB image",
            "body-measurement regression is disabled because this extracted batch has one body",
            "construction sequence is generator provenance, not a learned target in this baseline",
            "test labels are generated by the same annotation policy as training labels, not by independent pattern experts",
            "official sample split is an in-generator test, not an unseen-recipe or cross-CAD-source split",
            "no real commercial or textbook pattern generalization claim is supported by this run",
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "gpu",
                    "peak_cuda_memory_bytes",
                    "training_seconds",
                    "parameter_count",
                    "test",
                    "test_reindexed_boundary",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
