from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.retrieval_learning import (
    CATEGORY_NAMES,
    EDGE_FEATURE_NAMES,
    FPN_CACHE_TO_SEMANTIC_VIEW_ORDER,
    SEMANTIC_VIEW_NAMES,
    SCHEMA_VERSION,
    bidirectional_infonce,
    build_crossmodal_retrieval_model,
    load_exact_retrieval_corpus,
    make_retrieval_batch,
    paired_retrieval_metrics,
    train_bank_retrieval,
)


def _json_safe_paired(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key not in {"ordering", "similarity", "ranks"}}


def _device(raw: str):
    import torch

    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _embed(model, corpus, indices, batch_size, device):
    import torch

    images, patterns = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start : start + batch_size]
            batch = make_retrieval_batch(corpus, current)
            views = torch.from_numpy(batch["view_features"]).to(device)
            tokens = torch.from_numpy(batch["pattern_tokens"]).to(device)
            mask = torch.from_numpy(batch["pattern_mask"]).to(device)
            with torch.amp.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(views, tokens, mask)
            images.append(output["image_embedding"].float().cpu().numpy())
            patterns.append(output["pattern_embedding"].float().cpu().numpy())
    return np.concatenate(images), np.concatenate(patterns)


def _paired_bidirectional(model, corpus, indices, batch_size, device, *, rankings=False):
    image, pattern = _embed(model, corpus, indices, batch_size, device)
    ids = [corpus.examples[index].sample_id for index in indices]
    image_to_pattern = paired_retrieval_metrics(
        image, pattern, ids, ids, return_rankings=rankings
    )
    pattern_to_image = paired_retrieval_metrics(
        pattern, image, ids, ids, return_rankings=rankings
    )
    summary = {
        "image_to_pattern": _json_safe_paired(image_to_pattern),
        "pattern_to_image": _json_safe_paired(pattern_to_image),
        "mean_mrr": 0.5 * (image_to_pattern["mrr"] + pattern_to_image["mrr"]),
        "mean_recall_at_1": 0.5
        * (image_to_pattern["recall_at_1"] + pattern_to_image["recall_at_1"]),
    }
    return summary, image, pattern, image_to_pattern, pattern_to_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train four-view FPN to exact GCDv2 geometry retrieval with bidirectional InfoNCE."
    )
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl")
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gcdv2_exact_crossmodal_retrieval")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gcdv2_exact/crossmodal_retrieval.pt"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--pattern-layers", type=int, default=3)
    parser.add_argument("--view-layers", type=int, default=2)
    parser.add_argument("--pool-queries-per-view", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, help="Deterministic total sample limit for smoke tests only."
    )
    args = parser.parse_args()

    import torch

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(args.device)
    corpus = load_exact_retrieval_corpus(
        args.index,
        args.features,
        seed=seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        limit=args.limit,
    )
    split_indices = {
        split: [index for index, example in enumerate(corpus.examples) if example.split == split]
        for split in ("train", "validation", "test")
    }
    if any(not split_indices[split] for split in split_indices):
        raise SystemExit(f"every split must be nonempty: { {key: len(value) for key, value in split_indices.items()} }")
    config = {
        "schema_version": SCHEMA_VERSION,
        "spatial_feature_dim": int(corpus.view_features.shape[-1]),
        "max_spatial_tokens": int(corpus.view_features.shape[-2]),
        "max_edges": int(corpus.max_edges),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "num_heads": int(args.num_heads),
        "pattern_layers": int(args.pattern_layers),
        "view_layers": int(args.view_layers),
        "pool_queries_per_view": int(args.pool_queries_per_view),
        "dropout": float(args.dropout),
        "temperature": float(args.temperature),
        "seed": seed,
        "semantic_view_names": SEMANTIC_VIEW_NAMES,
        "fpn_cache_to_semantic_view_order": FPN_CACHE_TO_SEMANTIC_VIEW_ORDER,
    }
    model = build_crossmodal_retrieval_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(seed)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_order = np.asarray(split_indices["train"], dtype=np.int64)
        generator.shuffle(train_order)
        model.train()
        losses = []
        for start in range(0, len(train_order), args.batch_size):
            current = train_order[start : start + args.batch_size]
            if len(current) < 2:
                continue
            batch = make_retrieval_batch(corpus, current)
            views = torch.from_numpy(batch["view_features"]).to(device)
            tokens = torch.from_numpy(batch["pattern_tokens"]).to(device)
            mask = torch.from_numpy(batch["pattern_mask"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(views, tokens, mask)
                loss = bidirectional_infonce(
                    model, output["image_embedding"], output["pattern_embedding"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation, _, _, _, _ = _paired_bidirectional(
            model,
            corpus,
            split_indices["validation"],
            args.batch_size,
            device,
        )
        score = float(validation["mean_mrr"])
        row = {
            "epoch": epoch,
            "train_bidirectional_infonce": float(np.mean(losses)),
            "validation": validation,
            "logit_scale": float(model.logit_scale.exp().detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)

    all_indices = list(range(len(corpus.examples)))
    all_image, all_pattern = _embed(model, corpus, all_indices, args.batch_size, device)
    offset_lookup = {example_index: offset for offset, example_index in enumerate(all_indices)}
    validation_summary, _, _, _, _ = _paired_bidirectional(
        model, corpus, split_indices["validation"], args.batch_size, device
    )
    test_summary, test_image, test_pattern, test_i2p, test_p2i = _paired_bidirectional(
        model, corpus, split_indices["test"], args.batch_size, device, rankings=True
    )
    train_pattern = np.stack([all_pattern[offset_lookup[index]] for index in split_indices["train"]])
    train_examples = [corpus.examples[index] for index in split_indices["train"]]
    test_examples = [corpus.examples[index] for index in split_indices["test"]]
    train_bank_metrics, predictions = train_bank_retrieval(
        test_image, test_examples, train_pattern, train_examples
    )
    test_ids = [item.sample_id for item in test_examples]
    paired_order = test_i2p["ordering"]
    for index, row in enumerate(predictions):
        row["paired_gallery_target_present"] = True
        row["paired_gallery_target_rank"] = int(test_i2p["ranks"][index])
        row["paired_gallery_top"] = [
            {
                "sample_id": test_ids[int(value)],
                "similarity": float(test_i2p["similarity"][index, int(value)]),
            }
            for value in paired_order[index, : min(10, len(test_ids))]
        ]

    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    split_assignments = {item.sample_id: item.split for item in corpus.examples}
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_state": best_state,
            "config": config,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "split_assignments": split_assignments,
            "best_epoch": best_epoch,
            "index_path": str(args.index),
            "feature_cache_path": str(args.features),
        },
        args.checkpoint,
    )
    np.savez(
        args.output / "embeddings.npz",
        sample_ids=np.asarray([item.sample_id for item in corpus.examples]),
        categories=np.asarray([item.category for item in corpus.examples]),
        splits=np.asarray([item.split for item in corpus.examples]),
        topology_signatures=np.asarray(
            [item.topology_signature for item in corpus.examples]
        ),
        image_embeddings=all_image.astype(np.float32),
        pattern_embeddings=all_pattern.astype(np.float32),
    )
    with (args.output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "split_assignments.json").write_text(
        json.dumps(split_assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_TECHNICAL_EXECUTION_EXACT_GEOMETRY_CROSSMODAL_RETRIEVAL",
        "benchmark_quality_gate": "NOT_PREDECLARED_REPORT_METRICS_WITHOUT_PASS_CLAIM",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "split_counts": {key: len(value) for key, value in split_indices.items()},
        "excluded_missing_fpn_feature_count": len(corpus.missing_feature_sample_ids),
        "excluded_missing_fpn_feature_sample_ids": corpus.missing_feature_sample_ids,
        "category_counts": {
            split: {
                category: sum(
                    corpus.examples[index].category == category for index in indices
                )
                for category in CATEGORY_NAMES
            }
            for split, indices in split_indices.items()
        },
        "input_contract": {
            "image": "four frozen ResNet-50-FPN tensors, [4,85,256]",
            "semantic_view_names": SEMANTIC_VIEW_NAMES,
            "fpn_cache_to_semantic_view_order": FPN_CACHE_TO_SEMANTIC_VIEW_ORDER,
            "pattern": (
                "variable-length exact edge tokens: endpoints, native line/Bezier/arc kind, "
                "controls, lengths, directions/tangents and panel dimensions; no 3D placement"
            ),
            "objective": "paired bidirectional InfoNCE",
            "target_leakage": False,
        },
        "validation_paired_gallery": validation_summary,
        "test_paired_gallery": test_summary,
        "test_train_bank_target_absent": train_bank_metrics,
        "history": history,
        "artifacts": {
            "checkpoint": str(args.checkpoint),
            "embeddings": str(args.output / "embeddings.npz"),
            "predictions": str(args.output / "test_predictions.jsonl"),
            "split_assignments": str(args.output / "split_assignments.json"),
        },
        "claim_boundary": [
            "paired-gallery Recall/MRR includes the held-out target pattern in the held-out gallery",
            "train-bank evaluation removes every held-out target ID before nearest-pattern search",
            "the split is sample-disjoint but remains within one GarmentCode generator and render domain",
            "normalized geometry distance is exact-token RMSE for compatible topology and symmetric token Chamfer plus edge-count penalty otherwise",
            "retrieval selects an anchor only; it does not itself reconstruct or edit the target pattern",
        ],
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "best_epoch": best_epoch,
                "split_counts": metrics["split_counts"],
                "test_paired_gallery": test_summary,
                "test_train_bank_target_absent": train_bank_metrics,
                "artifacts": metrics["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
