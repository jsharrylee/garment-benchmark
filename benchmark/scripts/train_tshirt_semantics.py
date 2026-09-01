"""Train/evaluate the construction-traced basic T-shirt semantic baseline.

Checkpoints and detailed metrics default to ignored ``checkpoints/`` and
``artifacts/`` directories.  The metrics file also contains a compact
``manifest_safe`` block with hashes, counts, and configuration only; it can be
copied into a tracked report without exposing local paths or source payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from benchmark.drafting_semantics.tshirt_learning import (
    DEFAULT_TSHIRT_MODEL_CONFIG,
    EDGE_ROLES,
    BodyFeatureSpec,
    balanced_edge_weights,
    build_tshirt_model,
    dataset_audit,
    deterministic_split,
    evaluate_by_split_and_source,
    evaluate_model,
    padded_batch,
    panel_examples,
    random_augmentation,
    read_tshirt_records,
    source_value,
)


DEFAULT_RECORDS_PATH = Path("artifacts/drafting_semantics/tshirt_traces.jsonl.gz")
DEFAULT_CONFIG_PATH = Path("benchmark/configs/tshirt_creation_trace_training.json")
DEFAULT_TRAIN_SPLITS = "train"
DEFAULT_VALIDATION_SPLITS = "iid_validation,validation_body,validation_design,validation_double"


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _split_values(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _device(raw: str):
    import torch

    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _seed_everything(seed: int) -> None:
    # CUDA requires this before the first cuBLAS workspace is initialized for
    # reproducible matrix products.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These settings make repeated corpus/split comparisons reproducible.  A
    # Transformer operation that has no deterministic CUDA kernel will fail
    # loudly rather than silently changing a reported score.
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _guard_hash_split_reallocation(records, train_splits: set[str]) -> None:
    """Keep explicit cross-source holdouts frozen under optional hash splitting.

    Hash repartitioning is useful for an otherwise unsplit single-source smoke
    corpus.  It is scientifically invalid once records already identify an
    unseen source, or once a source absent from the original training split is
    present, because it would leak that external source into fitting.
    """

    explicit_unseen = [record.sample_id for record in records if record.split == "unseen_source"]
    all_sources = {source_value(record) for record in records}
    training_sources = {source_value(record) for record in records if record.split in train_splits}
    external_sources = (
        all_sources - training_sources if training_sources else (all_sources if len(all_sources) > 1 else set())
    )
    if explicit_unseen or external_sources:
        details = []
        if explicit_unseen:
            details.append(f"{len(explicit_unseen)} record(s) labelled unseen_source")
        if external_sources:
            details.append("external source(s): " + ", ".join(sorted(external_sources)))
        raise ValueError(
            "--split-mode hash is forbidden because it would reallocate frozen cross-source evaluation data ("
            + "; ".join(details)
            + ")"
        )


def _validation_selection_key(metrics: Mapping[str, Any], objective: str = "edge-primary") -> tuple[float, ...]:
    """Stable task-specific checkpoint order using validation-only metrics."""

    edge = metrics.get("edge_semantics", {})
    landmarks = metrics.get("landmarks", {})
    if objective == "landmark-primary":
        return (
            float(landmarks.get("detection_aware_success_pck_panel_span_2pct", 0.0) or 0.0),
            float(landmarks.get("existence_micro_f1", 0.0) or 0.0),
            -float(landmarks.get("gt_positive_conditional_median_euclidean_error_cm", float("inf"))),
            float(edge.get("length_weighted_macro_f1_supported_semantics", 0.0) or 0.0),
        )
    if objective != "edge-primary":
        raise ValueError(f"unsupported checkpoint selection objective: {objective}")
    return (
        float(edge.get("length_weighted_macro_f1_supported_semantics", 0.0) or 0.0),
        float(landmarks.get("existence_micro_f1", 0.0) or 0.0),
        float(landmarks.get("detection_aware_success_pck_panel_span_2pct", 0.0) or 0.0),
    )


def _compact_split_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Headline-sized frozen result; never mixes training and test panels."""

    if metrics.get("status") == "NO_EXAMPLES":
        return {"status": "NO_EXAMPLES", "panel_count": 0}
    edge = metrics["edge_semantics"]
    landmarks = metrics["landmarks"]
    structural = metrics.get("structural_decoding", {})
    structural_landmarks = structural.get("landmarks", {})
    return {
        "panel_count": metrics["panel_count"],
        "garment_count": metrics["garment_count"],
        "edge_macro_f1": edge["macro_f1_supported_semantics"],
        "edge_length_weighted_macro_f1": edge["length_weighted_macro_f1_supported_semantics"],
        "panel_accuracy": metrics["panel_role"]["accuracy"],
        "landmark_existence_f1": landmarks["existence_micro_f1"],
        "landmark_gt_positive_conditional_mae_cm": landmarks[
            "gt_positive_conditional_mean_euclidean_error_cm"
        ],
        "landmark_detection_aware_pck_2pct": landmarks[
            "detection_aware_success_pck_panel_span_2pct"
        ],
        "structural_panel_accuracy": structural.get("panel_role_accuracy"),
        "structural_landmark_existence_f1": structural_landmarks.get("existence_micro_f1"),
        "structural_landmark_mae_cm": structural_landmarks.get(
            "gt_positive_conditional_mean_euclidean_error_cm"
        ),
        "structural_landmark_pck_2pct": structural_landmarks.get(
            "detection_aware_success_pck_panel_span_2pct"
        ),
        "dart_false_positive_garment_rate": metrics["dart_false_positive"][
            "false_positive_garment_rate"
        ],
    }


def _tensor_batch(batch: Mapping[str, Any], device):
    import torch

    return {
        key: torch.from_numpy(batch[key]).to(device)
        for key in (
            "features",
            "edge_targets",
            "valid_mask",
            "panel_targets",
            "landmark_exists",
            "landmark_xy_normalized",
            "landmark_coordinate_mask",
            "body_features",
        )
    }


def _loss(outputs, tensors, edge_weights, config: Mapping[str, Any]):
    import torch

    weights = config["loss_weights"]
    # The mathematically equivalent N-D cross-entropy path dispatches to
    # nll_loss2d on CUDA, which has no deterministic implementation in the
    # pinned torch build.  Flatten tokens explicitly so strict deterministic
    # training remains enabled instead of silently accepting nondeterminism.
    edge_logits = outputs["edge_logits"].reshape(-1, outputs["edge_logits"].shape[-1])
    edge_targets = tensors["edge_targets"].reshape(-1)
    edge = torch.nn.functional.cross_entropy(
        edge_logits,
        edge_targets,
        weight=edge_weights,
        ignore_index=-100,
    )
    panel = torch.nn.functional.cross_entropy(outputs["panel_logits"], tensors["panel_targets"])
    existence = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["landmark_existence_logits"], tensors["landmark_exists"]
    )
    coordinate_mask = tensors["landmark_coordinate_mask"]
    if bool(coordinate_mask.any()):
        expanded = coordinate_mask.unsqueeze(-1).expand_as(outputs["landmark_xy_normalized"])
        coordinate = torch.nn.functional.smooth_l1_loss(
            outputs["landmark_xy_normalized"][expanded],
            tensors["landmark_xy_normalized"][expanded],
            beta=0.02,
        )
    else:
        coordinate = outputs["landmark_xy_normalized"].sum() * 0.0
    total = (
        float(weights["edge"]) * edge
        + float(weights["panel"]) * panel
        + float(weights["landmark_existence"]) * existence
        + float(weights["landmark_coordinate"]) * coordinate
    )
    return total, {
        "edge": float(edge.detach().cpu()),
        "panel": float(panel.detach().cpu()),
        "landmark_existence": float(existence.detach().cpu()),
        "landmark_coordinate": float(coordinate.detach().cpu()),
    }


def train(
    *,
    records_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    config: dict[str, Any],
    train_splits: set[str],
    validation_splits: set[str],
    split_mode: str,
    device_name: str,
) -> dict[str, Any]:
    import torch

    seed = int(config["seed"])
    _seed_everything(seed)
    device = _device(device_name)
    raw_bytes = records_path.read_bytes()
    records = read_tshirt_records(records_path)
    if split_mode == "hash":
        _guard_hash_split_reallocation(records, train_splits)
        records = tuple(replace(record, split=deterministic_split(record.sample_id, seed=seed)) for record in records)
    available_splits = sorted({record.split for record in records})
    training_records = tuple(record for record in records if record.split in train_splits)
    if not training_records:
        raise ValueError(
            f"no training records for {sorted(train_splits)}; available record split labels are {available_splits}"
        )
    mode = str(config["mode"])
    body_spec = BodyFeatureSpec.fit(training_records) if mode == "pattern+body" else BodyFeatureSpec()
    all_examples = panel_examples(records, body_spec=body_spec)
    training_examples = tuple(example for example in all_examples if example.split in train_splits)
    validation_examples = tuple(example for example in all_examples if example.split in validation_splits)
    if not training_examples:
        raise ValueError("training records contain no non-empty panels")
    if not validation_examples:
        raise ValueError(
            f"no validation panels for {sorted(validation_splits)}; available record split labels are {available_splits}"
        )
    audit = dataset_audit(records, all_examples)
    if audit["sample_ids_present_in_multiple_splits"]:
        raise ValueError("split leakage: one or more sample ids occur in multiple record-provided splits")
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be at least one so a validation-selected checkpoint can be created")

    model = build_tshirt_model(config, body_feature_dim=body_spec.feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    edge_weights = torch.from_numpy(balanced_edge_weights(training_examples)).to(device)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    maximum_edges = int(config["maximum_edges"])
    batch_size = int(config["batch_size"])
    best_selection_key: tuple[float, float, float] | None = None
    best_model_state: dict[str, Any] | None = None
    selected_epoch: int | None = None
    selection_objective = str(config.get("selection_objective", "edge-primary"))

    for epoch in range(int(config["epochs"])):
        order = generator.permutation(len(training_examples))
        augmented = [
            random_augmentation(training_examples[int(index)], generator, config["augmentation"]) for index in order
        ]
        model.train()
        losses: list[float] = []
        component_rows: list[dict[str, float]] = []
        for start in range(0, len(augmented), batch_size):
            batch = padded_batch(augmented[start : start + batch_size], maximum_edges)
            tensors = _tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(tensors["features"], tensors["valid_mask"], tensors["body_features"])
                loss, components = _loss(outputs, tensors, edge_weights, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            component_rows.append(components)
        validation = evaluate_model(model, validation_examples, config, device)
        selection_key = _validation_selection_key(validation, selection_objective)
        is_best = best_selection_key is None or selection_key > best_selection_key
        if is_best:
            best_selection_key = selection_key
            selected_epoch = epoch + 1
            best_model_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch + 1,
            "training_loss": float(np.mean(losses)),
            "training_loss_components": {
                key: float(np.mean([item[key] for item in component_rows])) for key in component_rows[0]
            },
            "validation": validation,
            "selected_as_best_validation_checkpoint": is_best,
            "validation_selection_key": list(selection_key),
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "training_loss": row["training_loss"],
                    "validation_edge_macro_f1": validation.get("edge_semantics", {}).get(
                        "length_weighted_macro_f1_supported_semantics"
                    ),
                    "validation_landmark_gt_positive_conditional_mae_cm": validation.get("landmarks", {}).get(
                        "gt_positive_conditional_mean_euclidean_error_cm"
                    ),
                    "selected_as_best_validation_checkpoint": is_best,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    training_seconds = time.perf_counter() - started
    if best_model_state is None or selected_epoch is None or best_selection_key is None:
        raise RuntimeError("training completed without a validation-selected model state")
    model.load_state_dict(best_model_state)
    evaluation = evaluate_by_split_and_source(model, all_examples, config, device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selection_metric = (
        "lexicographic(validation detection-aware landmark PCK@2% panel span, landmark existence micro F1, "
        "negative GT-positive conditional landmark median error, length-weighted macro edge F1)"
        if selection_objective == "landmark-primary"
        else (
            "lexicographic(validation length-weighted macro edge F1, landmark existence micro F1, "
            "detection-aware landmark PCK@2% panel span)"
        )
    )
    checkpoint_payload = {
        "model_state": model.state_dict(),
        "config": config,
        "body_feature_spec": body_spec.to_dict(),
        "edge_roles": EDGE_ROLES,
        "dataset_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "schema_version": "tshirt-construction-trace-1.0",
        "selected_epoch": selected_epoch,
        "selection_objective": selection_objective,
        "selection_metric": selection_metric,
        "selection_key": list(best_selection_key),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    training_sources = sorted({example.source for example in training_examples})
    source_scope = {
        source: ("SEEN_TRAINING_SOURCE" if source in training_sources else "ZERO_SHOT_UNSEEN_SOURCE")
        for source in sorted({example.source for example in all_examples})
    }
    manifest_safe = {
        "status": "TRAINED_BASIC_TSHIRT_SEMANTIC_BASELINE",
        "dataset_sha256": checkpoint_payload["dataset_sha256"],
        "record_count": len(records),
        "panel_count": len(all_examples),
        "training_panel_count": len(training_examples),
        "validation_panel_count": len(validation_examples),
        "split_mode": split_mode,
        "train_split_labels": sorted(train_splits),
        "validation_split_labels": sorted(validation_splits),
        "mode": mode,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": selected_epoch,
        "validation_selection_key": list(best_selection_key),
        "selection_objective": selection_objective,
        "config": config,
        "feature_contract": audit["feature_contract"],
        "construction_dag_metric": "NOT_GENERALIZABLE_RECIPE_MEMORIZATION",
        "unseen_source_group_count": sum(scope == "ZERO_SHOT_UNSEEN_SOURCE" for scope in source_scope.values()),
    }
    result = {
        "status": "TRAINED_BASIC_TSHIRT_SEMANTIC_BASELINE",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": peak_memory,
        "training_seconds": training_seconds,
        "dataset_file_name": records_path.name,
        "dataset_sha256": checkpoint_payload["dataset_sha256"],
        "checkpoint_file_name": checkpoint_path.name,
        "selected_epoch": selected_epoch,
        "validation_selection_key": list(best_selection_key),
        "checkpoint_policy": f"BEST_VALIDATION_{selection_objective.upper().replace('-', '_')}_LEXICOGRAPHIC",
        "selection_objective": selection_objective,
        "config": config,
        "body_feature_spec": body_spec.to_dict(),
        "audit": audit,
        "source_evaluation_scope": source_scope,
        "history": history,
        "evaluation": evaluation,
        "manifest_safe": manifest_safe,
        "interpretation_contract": {
            "generated_trace_truth": "PROVISIONAL_GENERATOR_GROUND_TRUTH_PENDING_EXPERT_VALIDATION",
            "test_scope": "same trace ontology; split/source groups are reported separately",
            "unseen_source": "valid zero-shot evidence only when source is absent from every training record",
            "bp": "UNAVAILABLE_NOT_A_TRAIN_TARGET",
            "darts": "NOT_APPLICABLE_FOR_BASIC_TSHIRT; only false-positive dart_leg predictions are counted",
            "construction_dag": "not scored because a fixed recipe DAG is memorization, not drafting generalization",
        },
        "limitations": [
            "Generator-created labels are provisional truth until the nearby pattern expert reviews them.",
            "A held-out body or design from the same recipe is not evidence of cross-recipe drafting knowledge.",
            "A FreeSewing or other unseen-source score must be frozen before any source-specific adaptation.",
            "The model starts from vector pattern geometry; it does not infer a pattern directly from a clothed RGB image.",
            "BP, darts, and notches cannot be learned from this no-dart basic T-shirt recipe alone.",
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train construction-time T-shirt panel/edge/landmark semantics without trace-label leakage."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drafting_semantics/tshirt_semantics.pt"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/drafting_semantics/tshirt_traces/training_metrics.json"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("pattern-only", "pattern+body"))
    parser.add_argument("--train-splits", default=DEFAULT_TRAIN_SPLITS)
    parser.add_argument("--validation-splits", default=DEFAULT_VALIDATION_SPLITS)
    parser.add_argument("--split-mode", choices=("record", "hash"), default="record")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--selection-objective", choices=("edge-primary", "landmark-primary"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(json.dumps(DEFAULT_TSHIRT_MODEL_CONFIG))
    if args.config is not None:
        config = _deep_update(config, json.loads(args.config.read_text(encoding="utf-8")))
    if args.mode is not None:
        config["mode"] = args.mode
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.selection_objective is not None:
        config["selection_objective"] = args.selection_objective
    result = train(
        records_path=args.records,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        config=config,
        train_splits=_split_values(args.train_splits),
        validation_splits=_split_values(args.validation_splits),
        split_mode=args.split_mode,
        device_name=args.device,
    )
    split_contract = config.get("split_contract", {})
    validation_headline_splits = tuple(split_contract.get("validation_only", sorted(_split_values(args.validation_splits))))
    frozen_headline_splits = tuple(
        split_contract.get(
            "frozen_test",
            sorted(set(result["evaluation"]["by_split"]) - _split_values(args.train_splits) - set(validation_headline_splits)),
        )
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gpu": result["gpu"],
                "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
                "training_seconds": result["training_seconds"],
                "selected_epoch": result["selected_epoch"],
                "validation_split_summaries": {
                    split: _compact_split_metrics(result["evaluation"]["by_split"][split])
                    for split in validation_headline_splits
                    if split in result["evaluation"]["by_split"]
                },
                "frozen_test_split_summaries": {
                    split: _compact_split_metrics(result["evaluation"]["by_split"][split])
                    for split in frozen_headline_splits
                    if split in result["evaluation"]["by_split"]
                },
                "headline_scope": "validation and frozen test splits only; no mixed train/test aggregate",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
