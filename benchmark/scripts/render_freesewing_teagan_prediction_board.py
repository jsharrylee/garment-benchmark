from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmark.drafting_semantics.tshirt_learning import (
    EDGE_ROLES,
    LANDMARK_NAMES,
    build_tshirt_model,
    decode_structural_semantics,
    padded_batch,
    panel_examples,
    read_tshirt_records,
)


DEFAULT_SAMPLES = (
    "freesewing_teagan__cisMaleAdult48__wide_deep_neck",
    "freesewing_teagan__cisFemaleAdult44__wide_deep_neck",
)


def _path(edge):
    from matplotlib.path import Path as MplPath

    geometry = edge.geometry
    vertices = [geometry.start_cm]
    codes = [MplPath.MOVETO]
    if geometry.kind == "cubic_bezier" and len(geometry.control_points_cm) >= 2:
        vertices.extend((*geometry.control_points_cm[:2], geometry.end_cm))
        codes.extend((MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4))
    elif geometry.control_points_cm:
        vertices.extend((geometry.control_points_cm[0], geometry.end_cm))
        codes.extend((MplPath.CURVE3, MplPath.CURVE3))
    else:
        vertices.append(geometry.end_cm)
        codes.append(MplPath.LINETO)
    return MplPath(np.asarray(vertices, dtype=float), codes)


def _draw_panel(axis, panel, roles=None, landmarks=None, *, title: str) -> None:
    from matplotlib.patches import PathPatch

    palette = {
        "hemline": "#16a6a0",
        "side_seam": "#7950f2",
        "armhole": "#228be6",
        "shoulder": "#e64980",
        "neckline": "#2f9e44",
        "center_front": "#f08c00",
        "center_back": "#7048e8",
        "sleeve_head": "#228be6",
        "sleeve_underarm": "#7950f2",
        "sleeve_hem": "#16a6a0",
    }
    for index, edge in enumerate(panel.edges):
        role = roles[index] if roles is not None else None
        color = palette.get(role, "#202124") if role is not None else "#202124"
        axis.add_patch(PathPatch(_path(edge), fill=False, color=color, linewidth=2.8, capstyle="round"))
        if role is not None:
            start = np.asarray(edge.geometry.start_cm)
            end = np.asarray(edge.geometry.end_cm)
            midpoint = (start + end) / 2.0
            axis.text(midpoint[0], midpoint[1], role.replace("center_", "C-").replace("side_seam", "side"), fontsize=8)
    if landmarks:
        for name, point in landmarks.items():
            axis.scatter([point[0]], [point[1]], s=38, color="#d9480f", zorder=3)
            axis.annotate(name, point, xytext=(5, 5), textcoords="offset points", fontsize=9, weight="bold")
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis()
    axis.set_title(title, fontsize=11)
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render actual FreeSewing Teagan frozen-test predictions.")
    parser.add_argument("--records", type=Path, default=Path("artifacts/drafting_semantics/teagan_training.jsonl.gz"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drafting_semantics/freesewing_teagan_pattern_only_edge_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/freesewing_teagan_prediction_board.png"),
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    records = read_tshirt_records(args.records)
    examples = panel_examples(records)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_tshirt_model(checkpoint["config"], body_feature_dim=0)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        "FreeSewing Teagan source-specific model — frozen unseen-body + unseen-design examples",
        fontsize=16,
        weight="bold",
    )
    for row, sample_id in enumerate(DEFAULT_SAMPLES):
        record = next(item for item in records if item.sample_id == sample_id)
        by_panel = {panel.id: panel for panel in record.panels}
        sample_examples = {example.panel_id: example for example in examples if example.sample_id == sample_id}
        predictions = {}
        decoded = {}
        for panel_id, example in sample_examples.items():
            batch = padded_batch([example], int(checkpoint["config"]["maximum_edges"]))
            with torch.no_grad():
                output = model(torch.from_numpy(batch["features"]), torch.from_numpy(batch["valid_mask"]))
            count = len(example.edge_targets)
            role_ids = output["edge_logits"][0, :count].argmax(dim=-1).numpy()
            predictions[panel_id] = [EDGE_ROLES[int(value)] for value in role_ids]
            _, exists, xy = decode_structural_semantics(example.features, role_ids)
            xy_cm = xy * example.normalization_scale_cm + example.normalization_center_cm
            decoded[panel_id] = {
                name: tuple(xy_cm[index])
                for index, name in enumerate(LANDMARK_NAMES)
                if exists[index]
            }

        success = row == 0
        row_label = "PASS example" if success else "Known failure example"
        _draw_panel(
            axes[row, 0],
            by_panel["teagan.front"],
            title=f"{row_label}\nraw front input",
        )
        _draw_panel(
            axes[row, 1],
            by_panel["teagan.front"],
            predictions["teagan.front"],
            decoded["teagan.front"],
            title="front prediction",
        )
        _draw_panel(
            axes[row, 2],
            by_panel["teagan.back"],
            title="raw back input",
        )
        _draw_panel(
            axes[row, 3],
            by_panel["teagan.back"],
            predictions["teagan.back"],
            decoded["teagan.back"],
            title=("back prediction" if success else "back failure: center-back → center-front"),
        )
    fig.text(
        0.5,
        0.005,
        "Top: male size 48, all shown roles correct. Bottom: female size 44; front correct, back center line misclassified.",
        ha="center",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white")
    plt.close(fig)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
