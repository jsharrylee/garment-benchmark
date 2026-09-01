from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from benchmark.gcdv2_exact.visual_dsl_retrieval import (
    FPN_CACHE_TO_SEMANTIC_VIEW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_VIEW_NAMES,
    bidirectional_infonce,
    build_visual_dsl_corpus,
    build_visual_dsl_retrieval_model,
    make_visual_dsl_batch,
    paired_retrieval_metrics,
    train_bank_retrieval_metrics,
)


def _device(name: str):
    import torch

    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _embed(model, corpus, indices, batch_size: int, device):
    import torch

    visual_rows, pattern_rows = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start : start + batch_size]
            batch = make_visual_dsl_batch(corpus, current)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    torch.from_numpy(batch["views"]).to(device),
                    torch.from_numpy(batch["panel_tokens"]).to(device),
                    torch.from_numpy(batch["panel_valid"]).to(device),
                )
            visual_rows.append(output["visual_embedding"].float().cpu().numpy())
            pattern_rows.append(output["pattern_embedding"].float().cpu().numpy())
    return np.concatenate(visual_rows), np.concatenate(pattern_rows)


def _paired_summary(model, corpus, indices, batch_size: int, device):
    visual, pattern = _embed(model, corpus, indices, batch_size, device)
    image_to_dsl = paired_retrieval_metrics(visual, pattern)
    dsl_to_image = paired_retrieval_metrics(pattern, visual)
    return (
        {
            "image_to_dsl": image_to_dsl,
            "dsl_to_image": dsl_to_image,
            "mean_mrr": 0.5 * (image_to_dsl["mrr"] + dsl_to_image["mrr"]),
            "mean_recall_at_1": 0.5 * (image_to_dsl["recall_at_1"] + dsl_to_image["recall_at_1"]),
        },
        visual,
        pattern,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align four-view frozen FPN tokens to the frozen coordinate-free Pattern DSL encoder."
    )
    parser.add_argument("--programs", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/gcdv2_pattern_dsl_v1/metadata.jsonl"))
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz")
    )
    parser.add_argument(
        "--dsl-checkpoint", type=Path, default=Path("checkpoints/gcdv2_pattern_dsl/unified_transformer.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gcdv2_visual_pattern_dsl_retrieval")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/gcdv2_pattern_dsl/visual_retrieval.pt")
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--visual-layers", type=int, default=2)
    parser.add_argument("--pattern-layers", type=int, default=2)
    parser.add_argument("--pool-queries-per-view", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--category-loss-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    from torch.nn import functional as F

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = build_visual_dsl_corpus(
        args.programs,
        args.metadata,
        args.features,
        args.dsl_checkpoint,
        device=device,
        extraction_batch_size=max(4, args.batch_size // 2),
        cached_panel_tokens_path=args.output / "frozen_dsl_panel_tokens.npz",
    )
    # Token extraction constructs and loads the frozen teacher on a cold run,
    # whereas a warm run reads its cache.  Reset RNGs here so bridge
    # initialization is identical in both cases.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    split_indices = {name: corpus.indices(name) for name in ("train", "validation", "test")}
    if any(not len(value) for value in split_indices.values()):
        raise RuntimeError(f"empty authoritative DSL split after FPN intersection: { {k: len(v) for k,v in split_indices.items()} }")

    config = {
        "spatial_dim": int(corpus.view_features.shape[-1]),
        "dsl_dim": int(corpus.dsl_panel_tokens.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "heads": int(args.heads),
        "visual_layers": int(args.visual_layers),
        "pattern_layers": int(args.pattern_layers),
        "max_spatial_tokens": int(corpus.view_features.shape[-2]),
        "max_panels": int(corpus.panel_valid.shape[-1]),
        "pool_queries_per_view": int(args.pool_queries_per_view),
        "dropout": float(args.dropout),
        "temperature": float(args.temperature),
    }
    model = build_visual_dsl_retrieval_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(args.seed)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = split_indices["train"].copy()
        generator.shuffle(order)
        model.train()
        losses, contrastive_losses, category_losses = [], [], []
        for start in range(0, len(order), args.batch_size):
            current = order[start : start + args.batch_size]
            if len(current) < 2:
                continue
            batch = make_visual_dsl_batch(corpus, current)
            views = torch.from_numpy(batch["views"]).to(device)
            panels = torch.from_numpy(batch["panel_tokens"]).to(device)
            panel_valid = torch.from_numpy(batch["panel_valid"]).to(device)
            categories = torch.from_numpy(batch["categories"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(views, panels, panel_valid)
                contrastive = bidirectional_infonce(
                    model, output["visual_embedding"], output["pattern_embedding"]
                )
                category = F.cross_entropy(output["category_logits"], categories)
                loss = contrastive + float(args.category_loss_weight) * category
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            contrastive_losses.append(float(contrastive.detach().cpu()))
            category_losses.append(float(category.detach().cpu()))

        validation, _, _ = _paired_summary(
            model, corpus, split_indices["validation"], args.batch_size, device
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_bidirectional_infonce": float(np.mean(contrastive_losses)),
            "train_category_cross_entropy": float(np.mean(category_losses)),
            "validation": validation,
            "logit_scale": float(model.logit_scale.exp().detach().cpu()),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        score = float(validation["mean_mrr"])
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)

    validation_summary, _, _ = _paired_summary(
        model, corpus, split_indices["validation"], args.batch_size, device
    )
    test_summary, test_visual, test_pattern = _paired_summary(
        model, corpus, split_indices["test"], args.batch_size, device
    )
    train_visual, train_pattern = _embed(
        model, corpus, split_indices["train"], args.batch_size, device
    )
    del train_visual
    train_bank_metrics, order = train_bank_retrieval_metrics(
        test_visual,
        train_pattern,
        corpus.categories[split_indices["test"]],
        corpus.categories[split_indices["train"]],
        corpus.topology_signatures[split_indices["test"]],
        corpus.topology_signatures[split_indices["train"]],
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_state": best_state,
            "config": config,
            "best_epoch": best_epoch,
            "dsl_checkpoint": str(args.dsl_checkpoint),
            "dsl_encoder_frozen": True,
            "split_authority": str(args.programs),
        },
        args.checkpoint,
    )
    np.savez_compressed(
        args.output / "test_embeddings.npz",
        sample_ids=corpus.sample_ids[split_indices["test"]],
        visual_embeddings=test_visual.astype(np.float32),
        pattern_embeddings=test_pattern.astype(np.float32),
    )
    train_ids = corpus.sample_ids[split_indices["train"]]
    test_ids = corpus.sample_ids[split_indices["test"]]
    train_categories = corpus.categories[split_indices["train"]]
    train_topologies = corpus.topology_signatures[split_indices["train"]]
    with (args.output / "test_train_bank_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for query_index, candidates in enumerate(order):
            winner = int(candidates[0])
            row = {
                "sample_id": str(test_ids[query_index]),
                "target_category": int(corpus.categories[split_indices["test"]][query_index]),
                "retrieved_sample_id": str(train_ids[winner]),
                "retrieved_category": int(train_categories[winner]),
                "category_match": bool(
                    corpus.categories[split_indices["test"]][query_index] == train_categories[winner]
                ),
                "exact_closed_cycle_primitive_topology_match": bool(
                    corpus.topology_signatures[split_indices["test"]][query_index] == train_topologies[winner]
                ),
                "top_train_bank_sample_ids": [str(train_ids[int(value)]) for value in candidates[:10]],
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    intersection_counts = {name: int(len(values)) for name, values in split_indices.items()}
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_TECHNICAL_VISUAL_TO_DSL_RETRIEVAL",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count_trainable_bridge": int(sum(value.numel() for value in model.parameters())),
        "dsl_teacher_parameter_count_frozen": 0,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "split_authority": "artifacts/gcdv2_pattern_dsl_v1/programs.npz; metadata split checked row-for-row",
        "intersection_split_counts": intersection_counts,
        "excluded_dsl_without_fpn": int(len(np.load(args.programs)["splits"]) - len(corpus.sample_ids)),
        "input_contract": {
            "visual": "four frozen ResNet50-FPN feature maps [4,85,256], reordered to front/back/left/right",
            "pattern": "frozen pretrained Pattern DSL encoder panel tokens from L/Q/C/A plus 18 intrinsic geometric features",
            "dsl_teacher_frozen": True,
            "excluded_from_neural_input": [
                "absolute x/y",
                "sample/source/panel/edge IDs",
                "semantic role labels",
                "stitch targets",
                "split labels",
                "topology signature",
            ],
            "objective": "bidirectional paired InfoNCE plus 0.1 visual category auxiliary loss",
            "semantic_view_order": SEMANTIC_VIEW_NAMES,
            "fpn_reorder": FPN_CACHE_TO_SEMANTIC_VIEW_ORDER,
        },
        "validation_paired_gallery": validation_summary,
        "test_paired_gallery_target_present": test_summary,
        "test_train_bank_target_absent": train_bank_metrics,
        "random_paired_gallery_recall_at_1": 1.0 / len(split_indices["test"]),
        "history": history,
        "artifacts": {
            "checkpoint_ignored": str(args.checkpoint),
            "frozen_dsl_tokens_ignored": str(args.output / "frozen_dsl_panel_tokens.npz"),
            "test_embeddings_ignored": str(args.output / "test_embeddings.npz"),
            "predictions_ignored": str(args.output / "test_train_bank_predictions.jsonl"),
        },
        "claim_boundary": [
            "The DSL split is garment-ID-disjoint, but all samples remain inside the same GarmentCode generator/render domain.",
            "Paired-gallery retrieval includes the exact held-out target DSL program; train-bank retrieval does not.",
            "Topology compatibility is an evaluation-only equality of category plus canonical closed-panel L/Q/C/A command cycles; it is not a neural input.",
            "This is image-to-program retrieval/alignment, not autoregressive SVG command generation and not expert cross-source validation.",
            "The pretrained DSL teacher uses derived GCDv2 semantic supervision, not expert-verified production pattern terminology.",
        ],
    }
    # Replace the deliberately zero placeholder with the actual frozen teacher
    # count without retaining a second model in GPU memory.
    frozen_checkpoint = torch.load(args.dsl_checkpoint, map_location="cpu", weights_only=False)
    metrics["dsl_teacher_parameter_count_frozen"] = int(
        sum(value.numel() for value in frozen_checkpoint["model_state"].values())
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "best_epoch": best_epoch,
                "training_seconds": training_seconds,
                "split_counts": intersection_counts,
                "test_paired_gallery": test_summary,
                "test_train_bank_target_absent": train_bank_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
