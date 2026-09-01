from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from benchmark.drafting_semantics.merged_visible_learning import (
    LANDMARK_NAMES,
    MAXIMUM_VISIBLE_EDGES,
    MERGED_EDGE_ROLES,
    build_merged_semantic_model,
    decode_landmarks,
)
from benchmark.gcdv2_exact.intrinsic_graph_learning import (
    build_corner_model,
    intrinsic_contour_features,
    intrinsic_segment_features,
    nearest_contour_indices,
    segment_between,
    select_cyclic_peaks,
)
from benchmark.scripts.train_gcdv2_merged_visible_semantics import role_metrics


COLORS = {
    "other": "#b0bec5", "neckline": "#d81b60", "shoulder": "#00897b", "armhole": "#7b1fa2",
    "center_front": "#fb8c00", "center_back": "#f4511e", "side_seam": "#1e88e5",
    "waistline": "#6d4c41", "dart_leg": "#7cb342",
}


def _cyclic_indices(start: int, end: int, size: int) -> np.ndarray:
    if end <= start:
        return np.concatenate((np.arange(start, size), np.arange(end)))
    return np.arange(start, end)


def _sample_role_map(contour: np.ndarray, vertices: np.ndarray, roles: np.ndarray) -> np.ndarray:
    indices = nearest_contour_indices(contour, vertices)
    output = np.full(len(contour), -1, np.int64)
    for local, start in enumerate(indices):
        end = int(indices[(local + 1) % len(indices)])
        output[_cyclic_indices(int(start), end, len(contour))] = int(roles[local])
    return output


def _predicted_graph(contour, corner_model, semantic_model, panel_role, device):
    import torch

    corner_features = torch.from_numpy(intrinsic_contour_features(contour))[None].to(device)
    with torch.no_grad():
        corner_output = corner_model(corner_features)
        count = int(np.clip(corner_output["count_logits"].argmax(-1).item(), 3, MAXIMUM_VISIBLE_EDGES))
        probabilities = corner_output["corner_logits"].sigmoid()[0].cpu().numpy()
    indices = np.asarray(select_cyclic_peaks(probabilities, count), np.int64)
    indices.sort()
    features = np.zeros((1, MAXIMUM_VISIBLE_EDGES, 32, 8), np.float32)
    valid = np.zeros((1, MAXIMUM_VISIBLE_EDGES), bool)
    for local, start in enumerate(indices):
        end = int(indices[(local + 1) % len(indices)])
        features[0, local] = intrinsic_segment_features(segment_between(contour, int(start), end))[0]
        valid[0, local] = True
    with torch.no_grad():
        logits = semantic_model(
            torch.from_numpy(features).to(device),
            torch.from_numpy(valid).to(device),
            torch.tensor([panel_role], device=device),
        )
    roles = logits[0, :count].argmax(-1).cpu().numpy()
    vertices = contour[indices]
    return indices, vertices, roles, decode_landmarks(roles, vertices, panel_role)


def _draw(axis, contour, vertices, roles, landmarks, title):
    indices = nearest_contour_indices(contour, vertices)
    for local, start in enumerate(indices):
        end = int(indices[(local + 1) % len(indices)])
        values = _cyclic_indices(int(start), end, len(contour))
        points = contour[np.append(values, end)]
        axis.plot(points[:, 0], points[:, 1], color=COLORS[MERGED_EDGE_ROLES[int(roles[local])]], linewidth=2.5)
    axis.scatter(vertices[:, 0], vertices[:, 1], color="#111", edgecolor="white", s=20, linewidth=0.5, zorder=4)
    for name, point in landmarks.items():
        axis.scatter(*point, color="white", edgecolor="#111", s=35, zorder=5)
        axis.annotate(name, point, xytext=(3, 3), textcoords="offset points", fontsize=7, weight="bold")
    axis.set_aspect("equal"); axis.invert_yaxis(); axis.axis("off"); axis.set_title(title, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fully predicted raster-to-landmark pipeline.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/merged_semantics.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/gcdv2_merged_visible_semantics_v1/metadata.jsonl"))
    parser.add_argument("--panel-index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--contours", type=Path, default=Path("artifacts/gcdv2_predicted_contours_v1/predicted_contours.npz"))
    parser.add_argument("--corner-checkpoint", type=Path, default=Path("checkpoints/gcdv2_intrinsic_graph_predicted/visible_corners.pt"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path("checkpoints/gcdv2_end_to_end/merged_visible_semantics.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_end_to_end/full_predicted_metrics.json"))
    parser.add_argument("--board", type=Path, default=Path("artifacts/gcdv2_end_to_end/full_predicted_frozen_test_10.png"))
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.dataset); arrays = {key: raw[key] for key in raw.files}
    contours = np.load(args.contours)["contours"]
    metadata = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    panel_rows = [json.loads(line) for line in args.panel_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    corner_model = build_corner_model().to(device)
    corner_model.load_state_dict(torch.load(args.corner_checkpoint, map_location=device, weights_only=True)["model_state"])
    semantic_model = build_merged_semantic_model().to(device)
    semantic_model.load_state_dict(torch.load(args.semantic_checkpoint, map_location=device, weights_only=True)["model_state"])
    corner_model.eval(); semantic_model.eval()

    predicted_roles_all, target_roles_all = [], []
    target_total = decoded_total = correct_total = count_exact = 0
    normalized_errors, rows = [], []
    test_indices = np.flatnonzero(arrays["splits"] == 2)
    for source in test_indices:
        panel_index = int(arrays["source_panel_indices"][source]); contour = contours[panel_index]
        indices, vertices, roles, decoded = _predicted_graph(
            contour, corner_model, semantic_model, int(arrays["panel_roles"][source]), device
        )
        target_count = int(arrays["valid_edges"][source].sum())
        target_vertices = arrays["vertices_uv"][source, :target_count]
        target_roles = arrays["edge_roles"][source, :target_count]
        sample_roles = _sample_role_map(contour, target_vertices, target_roles)
        segment_targets = []
        for local, start in enumerate(indices):
            end = int(indices[(local + 1) % len(indices)])
            values = sample_roles[_cyclic_indices(int(start), end, len(contour))]
            values = values[values >= 0]
            segment_targets.append(int(np.bincount(values, minlength=len(MERGED_EDGE_ROLES)).argmax()) if len(values) else 0)
        predicted_roles_all.extend(roles.tolist()); target_roles_all.extend(segment_targets)
        count_exact += int(len(indices) == target_count)
        span = max(float(np.ptp(target_vertices, axis=0).max()), 1e-8)
        for landmark_index, name in enumerate(LANDMARK_NAMES):
            if not arrays["landmark_mask"][source, landmark_index]:
                continue
            target_total += 1
            if name not in decoded:
                continue
            decoded_total += 1
            error = float(np.linalg.norm(decoded[name] - arrays["landmark_uv"][source, landmark_index]) / span)
            normalized_errors.append(error); correct_total += int(error <= 0.02)
        rows.append({
            "source": int(source), "panel_index": panel_index, "predicted_indices": indices.tolist(),
            "predicted_roles": roles.tolist(), "decoded_landmarks": {key: value.tolist() for key, value in decoded.items()},
        })

    metrics = {
        "status": "PASS_FULLY_PREDICTED_PIPELINE_EVALUATED",
        "test_panel_count": int(len(test_indices)),
        "pipeline": "panel.png -> learned mask/SDF -> predicted contour -> predicted vertices -> predicted semantic edges -> named landmarks",
        "edge_roles_on_predicted_segments": role_metrics(np.asarray(predicted_roles_all), np.asarray(target_roles_all)),
        "vertex_count_exact_accuracy": count_exact / max(len(test_indices), 1),
        "landmarks": {
            "target_count": target_total, "decoded_count": decoded_total,
            "decode_coverage": decoded_total / max(target_total, 1),
            "pck_at_2pct_panel_span": correct_total / max(target_total, 1),
            "mean_normalized_error_when_decoded": float(np.mean(normalized_errors)) if normalized_errors else None,
        },
        "remaining_oracle": "front/back bodice panel role is supplied; it is not inferred from the image in this phase",
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    chosen = random.Random(args.seed).sample(rows, min(10, len(rows)))
    figure, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.0 * len(chosen)), constrained_layout=True)
    for display, row in enumerate(chosen):
        source, panel_index = row["source"], row["panel_index"]
        contour = contours[panel_index]; vertices = contour[np.asarray(row["predicted_indices"])]
        roles = np.asarray(row["predicted_roles"]); decoded = {key: np.asarray(value) for key, value in row["decoded_landmarks"].items()}
        with Image.open(panel_rows[panel_index]["input_panel_image"]) as image:
            axes[display, 0].imshow(image.convert("L"), cmap="gray")
        axes[display, 0].axis("off"); axes[display, 0].set_title(f"actual panel.png input\n{metadata[source]['panel_uid']}", fontsize=8)
        target_count = int(arrays["valid_edges"][source].sum())
        target_landmarks = {name: arrays["landmark_uv"][source, index] for index, name in enumerate(LANDMARK_NAMES) if arrays["landmark_mask"][source, index]}
        _draw(axes[display, 1], contour, arrays["vertices_uv"][source, :target_count], arrays["edge_roles"][source, :target_count], target_landmarks, "target graph / names")
        _draw(axes[display, 2], contour, vertices, roles, decoded, "fully predicted graph / names")
    figure.suptitle("Frozen test: image-only contour + intrinsic graph + semantic landmark decoding", fontsize=13, weight="bold")
    figure.savefig(args.board, dpi=150, facecolor="white"); plt.close(figure)
    print(json.dumps({key: value for key, value in metrics.items() if key != "predictions"}, indent=2))
    print(args.board.as_posix())


if __name__ == "__main__":
    main()
