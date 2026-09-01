from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.multigarment_learning import (
    GARMENT_ROLES,
    MULTIGARMENT_EDGE_ROLES,
    MULTIGARMENT_PANEL_ROLES,
    build_multigarment_model,
    padded_garment_batch,
    read_gcd_multigarment_examples,
    read_teagan_multigarment_examples,
)
from benchmark.drafting_semantics.semantic_paths import merge_predicted_semantic_paths


def _predict(model, examples, config, device):
    import torch

    model.eval()
    output = []
    with torch.no_grad():
        for example in examples:
            batch = padded_garment_batch((example,), maximum_panels=int(config["maximum_panels"]), maximum_edges=int(config["maximum_edges"]))
            result = model(
                torch.from_numpy(batch["features"]).to(device),
                torch.from_numpy(batch["edge_valid"]).to(device),
                torch.from_numpy(batch["panel_valid"]).to(device),
            )
            edge = result["edge_logits"].argmax(dim=-1).cpu().numpy()[0]
            same_path = (result["same_path_logits"].sigmoid() >= 0.5).cpu().numpy()[0]
            panel = result["panel_logits"].argmax(dim=-1).cpu().numpy()[0]
            garment = int(result["garment_logits"].argmax(dim=-1).cpu().item())
            correct, total = 0, 0
            for panel_index, value in enumerate(example.panels):
                mask = value.edge_targets >= 0
                correct += int(np.sum(edge[panel_index, : len(value.features)][mask] == value.edge_targets[mask]))
                total += int(mask.sum())
            panel_accuracy = float(
                np.mean(panel[: len(example.panels)] == np.asarray([value.panel_target for value in example.panels]))
            )
            output.append((example, edge, panel, garment, correct / max(total, 1), panel_accuracy, same_path))
    return output


def _choose(predictions):
    selected = []
    gcd = [value for value in predictions if value[0].source == "garmentcode_v2"]
    for category in GARMENT_ROLES:
        candidates = [value for value in gcd if GARMENT_ROLES[value[0].garment_target] == category and len(value[0].panels) <= 16]
        if not candidates:
            candidates = [value for value in gcd if GARMENT_ROLES[value[0].garment_target] == category]
        candidates.sort(key=lambda value: (0.7 * value[4] + 0.3 * value[5], value[0].sample_id))
        selected.append(candidates[len(candidates) // 2])
    teagan = [value for value in predictions if value[0].source.startswith("freesewing")]
    teagan.sort(key=lambda value: (0.7 * value[4] + 0.3 * value[5], value[0].sample_id))
    selected.append(teagan[len(teagan) // 2])
    return selected


def _draw(ax, example, edge_ids, panel_ids, *, truth: bool, colors) -> None:
    panel_count = len(example.panels)
    columns = max(1, int(np.ceil(np.sqrt(panel_count))))
    rows = int(np.ceil(panel_count / columns))
    for panel_index, panel in enumerate(example.panels):
        col, row = panel_index % columns, panel_index // columns
        offset = np.asarray((col * 1.35, -row * 1.35), dtype=np.float32)
        for edge_index, feature in enumerate(panel.features):
            if truth:
                role_id = int(panel.edge_targets[edge_index])
                if role_id < 0:
                    color = "#b8b8b8"
                else:
                    color = colors[role_id]
            else:
                role_id = int(edge_ids[panel_index, edge_index])
                color = colors[role_id]
            start = feature[0:2] + offset
            end = feature[2:4] + offset
            ax.plot((start[0], end[0]), (start[1], end[1]), color=color, linewidth=1.7, solid_capstyle="round")
        role = MULTIGARMENT_PANEL_ROLES[panel.panel_target if truth else int(panel_ids[panel_index])]
        ax.text(offset[0], offset[1] + 0.66, role.replace("_", " "), ha="center", va="bottom", fontsize=5.8)
    ax.set_xlim(-0.75, columns * 1.35 - 0.60)
    ax.set_ylim(-rows * 1.35 + 0.55, 0.82)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render representative unified 2D semantic predictions.")
    parser.add_argument("--gcd-records", type=Path, default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"))
    parser.add_argument("--teagan-records", type=Path, default=Path("artifacts/drafting_semantics/teagan_diverse.jsonl.gz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drafting_semantics/multigarment_graph_transformer.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/multigarment/prediction_board.png"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/drafting_semantics/multigarment/prediction_board.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = build_multigarment_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    examples = tuple(
        item
        for item in (*read_gcd_multigarment_examples(args.gcd_records), *read_teagan_multigarment_examples(args.teagan_records))
        if item.split == "test"
    )
    selected = _choose(_predict(model, examples, config, device))
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(MULTIGARMENT_EDGE_ROLES)))
    colors = {index: palette[index] for index in range(len(MULTIGARMENT_EDGE_ROLES))}
    fig, axes = plt.subplots(len(selected), 2, figsize=(15, 4.1 * len(selected)), constrained_layout=True)
    rows = []
    for row, (example, edges, panels, garment, accuracy, panel_accuracy, same_path) in enumerate(selected):
        _draw(axes[row, 0], example, edges, panels, truth=True, colors=colors)
        _draw(axes[row, 1], example, edges, panels, truth=False, colors=colors)
        category = GARMENT_ROLES[example.garment_target]
        predicted_category = GARMENT_ROLES[garment]
        axes[row, 0].set_title(f"TARGET · {category} · {example.sample_id}", fontsize=10)
        axes[row, 1].set_title(
            f"PREDICTED · {predicted_category} · edge {accuracy:.1%} · panel {panel_accuracy:.1%}",
            fontsize=10,
        )
        armhole_groups = 0
        armhole_primitives = 0
        for panel_index, panel in enumerate(example.panels):
            roles = tuple(MULTIGARMENT_EDGE_ROLES[int(value)] for value in edges[panel_index, : len(panel.features)])
            paths = merge_predicted_semantic_paths(
                roles,
                same_path_links=same_path[panel_index, : len(panel.features)],
                edge_ids=panel.edge_ids,
                edge_lengths_cm=panel.edge_lengths_cm,
            )
            armhole_groups += sum(path.role == "armhole" for path in paths)
            armhole_primitives += sum(path.primitive_count for path in paths if path.role == "armhole")
        rows.append(
            {
                "sample_id": example.sample_id,
                "source": example.source,
                "target_garment": category,
                "predicted_garment": predicted_category,
                "labeled_edge_accuracy": accuracy,
                "panel_accuracy": panel_accuracy,
                "predicted_armhole_semantic_paths": armhole_groups,
                "predicted_armhole_primitives": armhole_primitives,
            }
        )
    fig.suptitle("Hierarchical 2D Transformer · representative frozen-test predictions\nGray target edges are unlabeled and excluded from loss/metrics", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    args.manifest.write_text(json.dumps({"records": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": rows}, indent=2))


if __name__ == "__main__":
    main()
