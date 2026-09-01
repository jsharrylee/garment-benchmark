from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.adapters.reweaver import order_panel_edges


def render_inputs(sample_root: Path, output: Path) -> None:
    files = sorted((sample_root / "render_output" / "rgb").glob("view_*.png"))
    canvas = Image.new("RGB", (1036, 1036), "white")
    for index, path in enumerate(files):
        with Image.open(path) as image:
            tile = image.convert("RGB")
        canvas.paste(tile, ((index % 2) * 518, (index // 2) * 518))
    canvas.save(output, "PNG")


def render_pattern_overlay(metric, output: Path, title: str) -> None:
    count = len(metric.gt_panels)
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
        axis.set_aspect("equal")
    for panel_id in range(count):
        axis = axes.flat[panel_id]
        gt, _ = order_panel_edges(np.asarray(metric.gt_panels[panel_id]["edge_points"], dtype=np.float64))
        pred, _ = order_panel_edges(np.asarray(metric.pred_panels[panel_id]["edge_points"], dtype=np.float64))
        for edge in gt:
            axis.plot(edge[:, 0], edge[:, 1], color="#16a34a", linewidth=4, alpha=0.62)
        for edge in pred:
            axis.plot(edge[:, 0], edge[:, 1], color="#dc2626", linewidth=1.8, linestyle="--")
        points = np.concatenate([gt.reshape(-1, 2), pred.reshape(-1, 2)], axis=0)
        span = np.ptp(points, axis=0)
        margin = max(float(span.max()) * 0.08, 0.02)
        axis.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        axis.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
        axis.set_title(f"panel {panel_id} · GT green / prediction red", fontsize=10)
        axis.set_axis_on()
        axis.grid(alpha=0.12)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=140, facecolor="white")
    plt.close(figure)


def render_gt_geometry(sample_root: Path, output: Path, title: str) -> None:
    archive = np.load(sample_root / f"{sample_root.name}_3d_geo.npz")
    points = archive["pc_sampled"]
    labels = archive["pc_labels"]
    curves = archive["curves_sampled"]
    valid_points = points[labels >= 0]
    minimum = valid_points.min(axis=0)
    maximum = valid_points.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max(float((maximum - minimum).max()) / 2, 0.01)
    figure = plt.figure(figsize=(12, 6))
    for slot, (azimuth, label) in enumerate(((35, "three-quarter"), (180, "opposite")), start=1):
        axis = figure.add_subplot(1, 2, slot, projection="3d")
        for patch_label in sorted(set(labels[labels >= 0])):
            patch = points[labels == patch_label]
            axis.scatter(patch[:, 0], patch[:, 1], patch[:, 2], s=0.8, alpha=0.22)
        for curve in curves:
            axis.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=0.65, alpha=0.75, color="black")
        axis.view_init(elev=16, azim=azimuth)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)


def contain(source: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(source) as image:
        tile = image.convert("RGB")
    tile.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(tile, ((size[0] - tile.width) // 2, (size[1] - tile.height) // 2))
    return canvas


def build_board(sample: str, input_image: Path, overlay: Path, prediction: Path, gt: Path, metrics: dict, output: Path) -> None:
    width = 1080
    canvas = Image.new("RGB", (width, 3180), "white")
    draw = ImageDraw.Draw(canvas)
    regular_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    title_font = ImageFont.truetype(str(bold_path), 38)
    section_font = ImageFont.truetype(str(bold_path), 30)
    text_font = ImageFont.truetype(str(regular_path), 22)
    draw.text((40, 28), f"OFFICIAL REWEAVER TEST · {sample}", fill="black", font=title_font)
    y = 92

    draw.text((40, y), "1. OFFICIAL FOUR-VIEW INPUT", fill="black", font=section_font)
    y += 46
    canvas.paste(contain(input_image, (1000, 1000)), (40, y))
    y += 1020

    draw.text((40, y), "2. PREDICTION OVER GROUND TRUTH", fill="black", font=section_font)
    y += 42
    draw.text((40, y), "GREEN = ground truth   RED DASH = ReWeaver prediction", fill=(70, 70, 70), font=text_font)
    y += 34
    canvas.paste(contain(overlay, (1000, 900)), (40, y))
    y += 920
    draw.text(
        (40, y),
        f"Panel count accuracy {metrics['panel_accuracy']:.0%}   Edge-count accuracy {metrics['edge_count_accuracy']:.0%}   Panel IoU {metrics['panel_iou']:.3f}",
        fill=(20, 20, 20),
        font=text_font,
    )
    y += 54

    draw.text((40, y), "3. 3D SHAPE · PREDICTION THEN GROUND TRUTH", fill="black", font=section_font)
    y += 46
    canvas.paste(contain(prediction, (1000, 480)), (40, y))
    y += 495
    canvas.paste(contain(gt, (1000, 480)), (40, y))
    y += 500
    result = canvas.crop((0, 0, width, y + 24))
    result.save(output, "JPEG", quality=88, optimize=True, progressive=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("external/ReWeaver-Code"))
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str((args.repo / "metric").resolve()))
    from metric import Metric

    metrics_by_sample = {item["sample_id"]: item for item in json.loads(args.metrics.read_text())["samples"]}
    for sample in args.samples:
        output_root = args.prediction_root / sample
        input_image = output_root / "official_inputs.png"
        overlay = output_root / "gt_prediction_overlay.png"
        gt_geometry = output_root / "ground_truth_geometry.png"
        board = output_root / "official_comparison_mobile.jpg"
        metric = Metric(args.ground_truth_root, args.prediction_root, sample)
        render_inputs(args.ground_truth_root / sample, input_image)
        render_pattern_overlay(metric, overlay, f"2D panel comparison · {sample}")
        render_gt_geometry(args.ground_truth_root / sample, gt_geometry, f"Ground-truth 3D patches and curves · {sample}")
        build_board(sample, input_image, overlay, output_root / "geometry.png", gt_geometry, metrics_by_sample[sample], board)
        print(board)


if __name__ == "__main__":
    main()
