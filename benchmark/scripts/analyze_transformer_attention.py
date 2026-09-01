from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import (
    MULTIGARMENT_EDGE_ROLES,
    build_multigarment_model,
    padded_garment_batch,
    read_gcd_multigarment_examples,
    read_teagan_multigarment_examples,
)
from benchmark.drafting_semantics.multiview_pattern_semantics import (
    TargetStandardizer,
    VIEW_NAMES,
    build_multiview_pattern_model,
    multiview_batch,
    read_multiview_pattern_examples,
)


def _multigarment_attention(model, examples, config, device):
    import torch

    heads = int(config["heads"])
    roles = len(MULTIGARMENT_EDGE_ROLES)
    sums = np.zeros((heads, roles, roles), dtype=np.float64)
    counts = np.zeros((roles, roles), dtype=np.int64)
    occurrences: dict[tuple[int, int], list[tuple[object, int, int, int]]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), 4):
            current = examples[start : start + 4]
            batch = padded_garment_batch(current, maximum_panels=int(config["maximum_panels"]), maximum_edges=int(config["maximum_edges"]))
            output = model(
                torch.from_numpy(batch["features"]).to(device),
                torch.from_numpy(batch["edge_valid"]).to(device),
                torch.from_numpy(batch["panel_valid"]).to(device),
                capture_attention=True,
            )
            predictions = output["edge_logits"].argmax(dim=-1).cpu().numpy()
            attention = output["local_attention"][-1].cpu().float().numpy()
            for row, example in enumerate(current):
                for panel_index, panel in enumerate(example.panels):
                    flat = row * int(config["maximum_panels"]) + panel_index
                    targets = panel.edge_targets
                    for query, query_role in enumerate(targets):
                        if query_role < 0 or predictions[row, panel_index, query] != query_role:
                            continue
                        for key, key_role in enumerate(targets):
                            if key_role < 0:
                                continue
                            sums[:, query_role, key_role] += attention[flat, :, query + 1, key + 1]
                            counts[query_role, key_role] += 1
                            pair = (int(query_role), int(key_role))
                            if len(occurrences[pair]) < 16:
                                occurrences[pair].append((example, panel_index, query, key))
    averages = sums / np.maximum(counts[None, :, :], 1)
    top_pairs = []
    for head in range(heads):
        candidates = []
        for query in range(roles):
            for key in range(roles):
                if query == key or query == 0 or key == 0 or counts[query, key] < 20:
                    continue
                candidates.append((float(averages[head, query, key]), query, key, int(counts[query, key])))
        value, query, key, support = max(candidates)
        top_pairs.append({
            "head": head,
            "query_role": MULTIGARMENT_EDGE_ROLES[query],
            "key_role": MULTIGARMENT_EDGE_ROLES[key],
            "mean_attention": value,
            "support_pairs": support,
            "role_ids": (query, key),
        })
    return averages, top_pairs, occurrences


def _causal_edge_deletion(model, top_pairs, occurrences, config, device):
    import torch

    results = []
    model.eval()
    with torch.no_grad():
        for item in top_pairs:
            query_role, key_role = item["role_ids"]
            generator = np.random.default_rng(20260828 + int(item["head"]))
            top_drops, random_drops = [], []
            for example, panel_index, query, key in occurrences[(query_role, key_role)][:16]:
                batch = padded_garment_batch((example,), maximum_panels=int(config["maximum_panels"]), maximum_edges=int(config["maximum_edges"]))
                features = torch.from_numpy(batch["features"]).to(device)
                edge_valid = torch.from_numpy(batch["edge_valid"]).to(device)
                panel_valid = torch.from_numpy(batch["panel_valid"]).to(device)
                baseline = model(features, edge_valid, panel_valid)
                baseline_probability = torch.softmax(baseline["edge_logits"][0, panel_index, query], dim=-1)[query_role]

                modified_features = features.clone()
                modified_valid = edge_valid.clone()
                modified_features[0, panel_index, key] = 0
                modified_valid[0, panel_index, key] = False
                changed = model(modified_features, modified_valid, panel_valid)
                changed_probability = torch.softmax(changed["edge_logits"][0, panel_index, query], dim=-1)[query_role]
                top_drops.append(float((baseline_probability - changed_probability).cpu()))

                panel = example.panels[panel_index]
                random_candidates = [
                    index
                    for index, role in enumerate(panel.edge_targets)
                    if index not in {query, key} and role >= 0
                ]
                if random_candidates:
                    random_key = int(generator.choice(random_candidates))
                    random_features = features.clone()
                    random_valid = edge_valid.clone()
                    random_features[0, panel_index, random_key] = 0
                    random_valid[0, panel_index, random_key] = False
                    random_output = model(random_features, random_valid, panel_valid)
                    random_probability = torch.softmax(random_output["edge_logits"][0, panel_index, query], dim=-1)[query_role]
                    random_drops.append(float((baseline_probability - random_probability).cpu()))
            results.append({
                **{key: value for key, value in item.items() if key != "role_ids"},
                "top_key_deletion_probability_drop": float(np.mean(top_drops)) if top_drops else None,
                "matched_random_key_deletion_probability_drop": float(np.mean(random_drops)) if random_drops else None,
                "causal_samples": len(top_drops),
            })
    return results


def _multiview_attention(model, examples, standardizer, config, device):
    import torch

    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), int(config["batch_size"])):
            batch = multiview_batch(examples[start : start + int(config["batch_size"])], standardizer)
            output = model(torch.from_numpy(batch["view_features"]).to(device), capture_attention=True)
            attention = output["attention"][-1].cpu().float().numpy()[:, :, 0, 1:]
            attention /= np.maximum(attention.sum(axis=-1, keepdims=True), 1e-12)
            values.append(attention)
    return np.concatenate(values, axis=0).mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit headwise attention with edge and view ablations.")
    parser.add_argument("--multigarment-checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multigarment_graph_transformer.pt"))
    parser.add_argument("--gcd-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--teagan-records", type=Path, default=Path("artifacts/drafting_semantics/teagan_diverse.jsonl.gz"))
    parser.add_argument("--multiview-checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multiview_pattern_semantics_resnet50.pt"))
    parser.add_argument("--multiview-features", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_features.npz"))
    parser.add_argument("--multiview-index", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"))
    parser.add_argument("--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--multiview-metrics", type=Path, default=Path("artifacts/drafting_semantics/multiview_pattern_semantics/resnet50_training_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/attention_audit/attention_audit.json"))
    parser.add_argument("--image", type=Path, default=Path("artifacts/drafting_semantics/attention_audit/attention_audit.png"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    graph_checkpoint = torch.load(args.multigarment_checkpoint, map_location="cpu", weights_only=False)
    graph_config = graph_checkpoint["config"]
    graph_model = build_multigarment_model(graph_config)
    graph_model.load_state_dict(graph_checkpoint["model_state"])
    graph_model.to(device)
    graph_examples = tuple(
        item
        for item in (*read_gcd_multigarment_examples(args.gcd_records), *read_teagan_multigarment_examples(args.teagan_records))
        if item.split == "test"
    )
    _, top_pairs, occurrences = _multigarment_attention(graph_model, graph_examples, graph_config, device)
    causal = _causal_edge_deletion(graph_model, top_pairs, occurrences, graph_config, device)

    view_checkpoint = torch.load(args.multiview_checkpoint, map_location="cpu", weights_only=False)
    view_config = view_checkpoint["config"]
    standardizer = TargetStandardizer(
        tuple(view_checkpoint["target_standardizer"]["means"]),
        tuple(view_checkpoint["target_standardizer"]["standard_deviations"]),
    )
    view_model = build_multiview_pattern_model(view_config)
    view_model.load_state_dict(view_checkpoint["model_state"])
    view_model.to(device)
    view_examples = tuple(
        item
        for item in read_multiview_pattern_examples(args.multiview_index, args.split, args.gcd_records, args.multiview_features)
        if item.split == "test"
    )
    view_attention = _multiview_attention(view_model, view_examples, standardizer, view_config, device)
    metrics = json.loads(args.multiview_metrics.read_text(encoding="utf-8"))
    baseline = metrics["test"]
    view_ablation = {}
    for view in VIEW_NAMES:
        value = metrics["test_leave_one_view_out"][view]
        view_ablation[view] = {
            "category_macro_f1_drop": baseline["category"]["macro_f1"] - value["category"]["macro_f1"],
            "normalized_pattern_mae_increase": value["mean_normalized_pattern_mae"] - baseline["mean_normalized_pattern_mae"],
        }

    payload = {
        "schema_version": "transformer-attention-audit-1.1",
        "multigarment_edge_heads": causal,
        "multiview_head_attention": [
            {"head": head, "view_distribution": {view: float(view_attention[head, index]) for index, view in enumerate(VIEW_NAMES)}}
            for head in range(view_attention.shape[0])
        ],
        "multiview_leave_one_out": view_ablation,
        "interpretation_contract": (
            "exploratory attention is paired with matched random edge deletion or leave-one-view-out ablation; "
            "the multiview model has global view tokens, no spatial or target-specific attention"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes[0, 0].axis("off")
    lines = ["2D edge heads: strongest cross-role connection on correct predictions", ""]
    for item in causal:
        lines.append(
            f"H{item['head']}: {item['query_role']} → {item['key_role']}  "
            f"attn={item['mean_attention']:.3f}  Δp(top/matched-random)="
            f"{item['top_key_deletion_probability_drop']:+.3f}/{item['matched_random_key_deletion_probability_drop']:+.3f}"
        )
    axes[0, 0].text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)

    image = axes[0, 1].imshow(view_attention, vmin=0, vmax=max(0.4, float(view_attention.max())), cmap="viridis", aspect="auto")
    axes[0, 1].set_title("4-view Transformer: final-layer CLS attention per head")
    axes[0, 1].set_xticks(range(len(VIEW_NAMES)), VIEW_NAMES)
    axes[0, 1].set_yticks(range(view_attention.shape[0]), [f"H{index}" for index in range(view_attention.shape[0])])
    for head in range(view_attention.shape[0]):
        for view in range(len(VIEW_NAMES)):
            axes[0, 1].text(view, head, f"{view_attention[head, view]:.2f}", ha="center", va="center", color="white" if view_attention[head, view] > 0.25 else "black", fontsize=8)
    fig.colorbar(image, ax=axes[0, 1], fraction=0.04, label="attention share among four views")

    heads = [item["head"] for item in causal]
    top_drop = [item["top_key_deletion_probability_drop"] for item in causal]
    random_drop = [item["matched_random_key_deletion_probability_drop"] for item in causal]
    x = np.arange(len(heads))
    axes[1, 0].bar(x - 0.18, top_drop, width=0.36, label="top-attended key deletion")
    axes[1, 0].bar(x + 0.18, random_drop, width=0.36, label="matched random key deletion")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(x, [f"H{head}" for head in heads])
    axes[1, 0].set_ylabel("correct-role probability drop")
    axes[1, 0].set_title("Edge deletion check: attention is useful only when prediction changes")
    axes[1, 0].legend(fontsize=8)

    f1_drop = [view_ablation[view]["category_macro_f1_drop"] for view in VIEW_NAMES]
    mae_increase = [view_ablation[view]["normalized_pattern_mae_increase"] for view in VIEW_NAMES]
    axes[1, 1].bar(x[:4] - 0.18, f1_drop, width=0.36, label="category macro-F1 drop")
    axes[1, 1].bar(x[:4] + 0.18, mae_increase, width=0.36, label="pattern normalized-MAE increase")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x[:4], VIEW_NAMES)
    axes[1, 1].set_title("Leave-one-view-out ablation (trained with view dropout)")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Transformer attention audit · frozen test only", fontsize=15)
    fig.savefig(args.image, dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "image": str(args.image), "edge_heads": causal, "view_ablation": view_ablation}, indent=2))


if __name__ == "__main__":
    main()
