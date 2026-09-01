"""Render an evidence-first QA board for T-shirt 2D-to-surface correspondence.

The board deliberately keeps three kinds of evidence visually distinct:

* solid coloured mesh-edge traces and dots are exact GCDv2 source-edge
  correspondences carried through the shared boxmesh/simulated-mesh topology;
* dashed coloured squares are the finest 8x8 top-1 locations predicted by the
  sample's out-of-fold checkpoint;
* the table contains exact 2D measurements and their out-of-fold predictions.

The script reads query/measurement names from the corpus and checkpoints.  It
therefore does not assume the historical merged-element schema and remains
usable when front/back element queries are separated.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.drafting_semantics.gcdv2_surface_correspondence import (
    build_element_vertex_index,
    build_tshirt_visual_correspondence_model,
    load_shared_topology_mesh,
    project_gcdv2_orthographic,
    read_jsonl,
    visible_projected_vertices,
)


VIEW_NAMES = ("front", "back", "left", "right")
VIEW_CAMERAS = ("CAM001", "CAM000", "CAM002", "CAM003")

BACKGROUND = "#07111f"
CARD = "#111c2d"
CARD_INNER = "#0a1422"
TEXT = "#f7f9fc"
MUTED = "#a9b6c9"
SUBTLE = "#637188"
GREEN = "#58d6a9"
AMBER = "#f1c75b"
RED = "#ff7a87"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render exact semantic surface traces and OOF 8x8 predictions."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/tshirt_visual_causality/"
            "surface_correspondence.npz"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/tshirt_visual_causality/"
            "correspondence_model"
        ),
    )
    parser.add_argument(
        "--fpn-cache",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/processed/garmentcode_v2/batch_0_full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/tshirt_visual_causality/"
            "surface_correspondence_qa.png"
        ),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        help="Explicit sample ids. Otherwise representative OOF quantiles are used.",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=300)
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filenames = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for filename in filenames:
        if Path(filename).is_file():
            return ImageFont.truetype(filename, size=size)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), value, font=font)
    return int(box[2] - box[0])


def _query_palette(names: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Produce a stable high-contrast palette without binding colours to names."""

    output: dict[str, tuple[int, int, int]] = {}
    golden = 0.618033988749895
    for index, name in enumerate(names):
        hue = (0.03 + golden * index) % 1.0
        saturation = 0.62 + 0.12 * (index % 2)
        value = 0.96 if index % 3 else 0.88
        output[str(name)] = tuple(
            int(round(channel * 255.0))
            for channel in colorsys.hsv_to_rgb(hue, saturation, value)
        )
    return output


def _mean_location_score(row: Mapping[str, Any]) -> float:
    values = [
        float(view["top1_target_score"])
        for per_element in row["element_location"].values()
        for view in per_element.values()
    ]
    return float(np.mean(values)) if values else math.nan


def _normalized_parameter_error(
    row: Mapping[str, Any], metrics: Mapping[str, Any]
) -> float:
    values = []
    parameter_metrics = metrics.get("parameter_metrics", {})
    for name, item in row["parameters"].items():
        if not str(item.get("status", "")).startswith("EVALUATED"):
            continue
        scale = float(parameter_metrics.get(name, {}).get("truth_standard_deviation", 0.0))
        if scale > 1e-8:
            values.append(abs(float(item["prediction"]) - float(item["truth"])) / scale)
    return float(np.mean(values)) if values else math.nan


def _sample_quality(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> float:
    location = _mean_location_score(row)
    parameter_error = _normalized_parameter_error(row, metrics)
    if not math.isfinite(location):
        location = 0.0
    if not math.isfinite(parameter_error):
        parameter_error = 0.0
    return location - 0.15 * parameter_error


def _select_rows(
    rows: Sequence[dict[str, Any]],
    metrics: Mapping[str, Any],
    sample_ids: Sequence[str] | None,
    count: int,
) -> list[tuple[str, dict[str, Any]]]:
    by_id = {str(row["sample_id"]): row for row in rows}
    if sample_ids:
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise KeyError(f"sample ids absent from OOF predictions: {missing}")
        return [("requested", by_id[sample_id]) for sample_id in sample_ids]

    count = max(1, min(int(count), len(rows)))
    ranked = sorted(rows, key=lambda row: _sample_quality(row, metrics))
    if count == 1:
        positions = [len(ranked) // 2]
    else:
        positions = [round(value) for value in np.linspace(len(ranked) - 1, 0, count)]
    labels = {
        0: "high OOF sample score",
        count - 1: "low OOF sample score",
    }
    if count % 2:
        labels[count // 2] = "median OOF sample score"
    return [
        (labels.get(index, f"OOF quantile {index + 1}/{count}"), ranked[position])
        for index, position in enumerate(positions)
    ]


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before weights_only was introduced
        return torch.load(path, map_location="cpu")


def _infer_oof_cells(
    selected: Sequence[tuple[str, dict[str, Any]]],
    *,
    corpus: Mapping[str, np.ndarray],
    fpn: Mapping[str, np.ndarray],
    model_dir: Path,
    element_names: Sequence[str],
    parameter_names: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Run each selected sample only through its held-out fold checkpoint."""

    import torch

    sample_lookup = {
        str(sample_id): index for index, sample_id in enumerate(corpus["sample_ids"])
    }
    metadata = json.loads(str(fpn["metadata_json"]))
    reorder = np.asarray(metadata.get("semantic_reorder", (1, 0, 2, 3)), dtype=np.int64)
    output: dict[str, dict[str, np.ndarray]] = {}
    models: dict[int, tuple[Any, Mapping[str, Any]]] = {}

    for _, row in selected:
        sample_id = str(row["sample_id"])
        fold = int(row["fold"])
        if fold not in models:
            checkpoint = _load_checkpoint(model_dir / f"fold_{fold:02d}.pt")
            checkpoint_elements = tuple(str(value) for value in checkpoint["element_names"])
            checkpoint_parameters = tuple(str(value) for value in checkpoint["parameter_names"])
            if checkpoint_elements != tuple(element_names):
                raise ValueError(
                    "checkpoint/corpus element schema mismatch: "
                    f"{checkpoint_elements} != {tuple(element_names)}"
                )
            if checkpoint_parameters != tuple(parameter_names):
                raise ValueError("checkpoint/corpus parameter schema mismatch")
            model = build_tshirt_visual_correspondence_model(
                feature_dim=int(fpn["features"].shape[-1]),
                width=int(checkpoint["width"]),
                layers=int(checkpoint["layers"]),
                heads=int(checkpoint["heads"]),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            models[fold] = (model, checkpoint)

        model, checkpoint = models[fold]
        corpus_index = sample_lookup[sample_id]
        feature_index = int(corpus["feature_indices"][corpus_index])
        features = np.asarray(fpn["features"][feature_index], dtype=np.float32)[reorder]
        with torch.inference_mode():
            prediction = model(torch.from_numpy(features[None]))
        location = prediction["element_location_logits"][0, :, :, :64]
        cells = torch.argmax(location, dim=-1).cpu().numpy().astype(np.int16)
        normalized = prediction["parameter_mean"][0].cpu().numpy()
        parameter_prediction = (
            normalized * np.asarray(checkpoint["parameter_std"])
            + np.asarray(checkpoint["parameter_mean"])
        ).astype(np.float32)

        # The JSONL is the evaluation receipt.  Re-running the fold must agree
        # with it before its cell locations are allowed onto the QA board.
        for index, name in enumerate(parameter_names):
            receipt = row["parameters"][name]
            if str(receipt.get("status", "")).startswith("EVALUATED") and not np.isclose(
                parameter_prediction[index], float(receipt["prediction"]), atol=2e-4
            ):
                raise ValueError(f"OOF checkpoint/receipt mismatch for {sample_id}:{name}")
        output[sample_id] = {
            "cells": cells,
            "parameter_prediction": parameter_prediction,
        }
    return output


def _raw_to_original_edges(boxmesh_path: Path) -> np.ndarray:
    """Recover exact original-index mesh adjacency after UV seam expansion."""

    import trimesh

    loaded = trimesh.load_mesh(boxmesh_path, process=False)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    key_to_original: dict[tuple[float, float, float], int] = {}
    raw_to_original = np.empty(len(vertices), dtype=np.int64)
    for raw_index, row in enumerate(np.round(vertices, decimals=6)):
        key = (float(row[0]), float(row[1]), float(row[2]))
        if key not in key_to_original:
            key_to_original[key] = len(key_to_original)
        raw_to_original[raw_index] = key_to_original[key]
    mapped = raw_to_original[faces]
    edges = np.concatenate(
        (mapped[:, (0, 1)], mapped[:, (1, 2)], mapped[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _sample_surface_geometry(
    sample_id: str,
    *,
    raw_root: Path,
    semantic_record: Mapping[str, Any],
    element_names: Sequence[str],
) -> dict[str, Any]:
    sample_root = raw_root / sample_id
    prefix = sample_root / sample_id
    specification_path = prefix.with_name(f"{sample_id}_specification.json")
    boxmesh_path = prefix.with_name(f"{sample_id}_boxmesh.ply")
    simmesh_path = prefix.with_name(f"{sample_id}_sim.ply")
    segmentation_path = prefix.with_name(f"{sample_id}_sim_segmentation.txt")
    labels_path = prefix.with_name(f"{sample_id}_vertex_labels.yaml")
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    mesh = load_shared_topology_mesh(boxmesh_path, simmesh_path, segmentation_path)
    indices, audit = build_element_vertex_index(
        semantic_record, specification, mesh, labels_path
    )
    if set(element_names) - set(indices):
        raise ValueError(
            f"surface mapper did not produce corpus queries: {set(element_names) - set(indices)}"
        )
    xy, depth = project_gcdv2_orthographic(mesh.sim_vertices_cm)
    adjacency = _raw_to_original_edges(boxmesh_path)
    per_element: dict[str, dict[str, Any]] = {}
    for name in element_names:
        candidates = np.asarray(indices[name], dtype=np.int64)
        visible = visible_projected_vertices(xy, depth, candidates)
        per_view = []
        for view_index in range(len(VIEW_NAMES)):
            visible_candidates = candidates[visible[view_index]]
            visible_mask = np.zeros(mesh.original_vertex_count, dtype=bool)
            visible_mask[visible_candidates] = True
            edge_mask = visible_mask[adjacency[:, 0]] & visible_mask[adjacency[:, 1]]
            per_view.append(
                {
                    "points": xy[view_index, visible_candidates],
                    "segments": xy[view_index, adjacency[edge_mask]],
                }
            )
        per_element[name] = {"views": per_view, "vertex_count": int(len(candidates))}
    return {
        "elements": per_element,
        "audit": audit,
        "vertex_count": mesh.original_vertex_count,
    }


def _composite_render(path: Path, size: int) -> Image.Image:
    source = Image.open(path).convert("RGBA")
    black = Image.new("RGBA", source.size, (0, 0, 0, 255))
    black.alpha_composite(source)
    return black.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: tuple[int, int, int, int],
    colour: tuple[int, int, int, int],
    width: int = 2,
    dash: int = 7,
    gap: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=fill)
    for start in np.arange(x0, x1, dash + gap):
        draw.line((start, y0, min(start + dash, x1), y0), fill=colour, width=width)
        draw.line((start, y1, min(start + dash, x1), y1), fill=colour, width=width)
    for start in np.arange(y0, y1, dash + gap):
        draw.line((x0, start, x0, min(start + dash, y1)), fill=colour, width=width)
        draw.line((x1, start, x1, min(start + dash, y1)), fill=colour, width=width)


def _overlay_view(
    image: Image.Image,
    *,
    geometry: Mapping[str, Any],
    prediction_cells: np.ndarray,
    target_heatmaps: np.ndarray,
    element_names: Sequence[str],
    colours: Mapping[str, tuple[int, int, int]],
    view_index: int,
) -> Image.Image:
    width, height = image.size
    base = image.convert("RGBA")
    # Draw into a transparent layer and alpha-composite it afterwards.  Drawing
    # low-alpha fills directly onto an opaque RGBA image and then dropping the
    # alpha channel would turn the cells into misleading solid colour blocks.
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    # Predictions first, so exact source geometry remains the visually dominant
    # evidence if multiple cells and curves overlap.
    for element_index, name in enumerate(element_names):
        if float(np.sum(target_heatmaps[element_index, view_index])) <= 1e-6:
            continue
        cell = int(prediction_cells[element_index, view_index])
        cell_x, cell_y = cell % 8, cell // 8
        colour = colours[name]
        x0, y0 = cell_x * width / 8.0, cell_y * height / 8.0
        x1, y1 = (cell_x + 1) * width / 8.0, (cell_y + 1) * height / 8.0
        _draw_dashed_rectangle(
            draw,
            (x0 + 1, y0 + 1, x1 - 1, y1 - 1),
            fill=(*colour, 15),
            colour=(*colour, 190),
        )
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        draw.line((cx - 3, cy, cx + 3, cy), fill=(*colour, 220), width=1)
        draw.line((cx, cy - 3, cx, cy + 3), fill=(*colour, 220), width=1)

    for name in element_names:
        colour = colours[name]
        data = geometry["elements"][name]["views"][view_index]
        segments = np.asarray(data["segments"], dtype=np.float32)
        points = np.asarray(data["points"], dtype=np.float32)
        for segment in segments:
            values = (
                float(segment[0, 0] * (width - 1)),
                float(segment[0, 1] * (height - 1)),
                float(segment[1, 0] * (width - 1)),
                float(segment[1, 1] * (height - 1)),
            )
            draw.line(values, fill=(0, 0, 0, 230), width=5)
            draw.line(values, fill=(*colour, 245), width=3)
        for point in points:
            x = float(point[0] * (width - 1))
            y = float(point[1] * (height - 1))
            draw.ellipse((x - 2.3, y - 2.3, x + 2.3, y + 2.3), fill=(0, 0, 0, 230))
            draw.ellipse((x - 1.4, y - 1.4, x + 1.4, y + 1.4), fill=(*colour, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def _draw_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    value: str,
    *,
    width: int,
) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 56), radius=12, fill=CARD)
    draw.text((x + 16, y + 9), label, font=_font(14, bold=True), fill=MUTED)
    value_font = _font(21, bold=True)
    draw.text(
        (x + width - 16 - _text_width(draw, value, value_font), y + 25),
        value,
        font=value_font,
        fill=TEXT,
    )


def _short_parameter(name: str) -> str:
    replacements = {
        "neck_width_cm": "neck width",
        "front_neck_depth_cm": "front neck depth",
        "shoulder_slope_deg": "shoulder slope",
        "armhole_depth_cm": "armhole depth",
        "body_length_cm": "body length",
        "sleeve_cap_height_cm": "sleeve-cap height",
        "sleeve_length_cm": "sleeve length",
        "sleeve_width_cm": "sleeve opening width",
    }
    return replacements.get(name, name.replace("_cm", "").replace("_deg", "").replace("_", " "))


def _render_board(
    *,
    selected: Sequence[tuple[str, dict[str, Any]]],
    corpus: Mapping[str, np.ndarray],
    prediction_cells: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[str, Any],
    index_by_id: Mapping[str, Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    raw_root: Path,
    element_names: Sequence[str],
    parameter_names: Sequence[str],
    output: Path,
    image_size: int,
) -> None:
    margin = 42
    view_card_width = image_size + 26
    view_gap = 12
    views_width = 4 * view_card_width + 3 * view_gap
    table_gap = 28
    table_width = 900
    canvas_width = margin * 2 + views_width + table_gap + table_width
    legend_columns = 5
    legend_rows = int(math.ceil(len(element_names) / legend_columns))
    header_height = 215 + legend_rows * 34
    sample_height = image_size + 190
    sample_gap = 18
    footer_height = 72
    canvas_height = (
        header_height
        + len(selected) * sample_height
        + max(0, len(selected) - 1) * sample_gap
        + footer_height
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    colours = _query_palette(element_names)

    draw.text(
        (margin, 30),
        "T-shirt 2D element -> simulated-surface correspondence",
        font=_font(38, bold=True),
        fill=TEXT,
    )
    draw.text(
        (margin, 80),
        (
            f"{metrics['sample_count']} real GCDv2 four-view garments | deterministic OOF evaluation | "
            f"{len(element_names)} semantic queries | {sum(1 for p in metrics['parameter_metrics'].values() if p.get('count', 0))} varying measurements"
        ),
        font=_font(19),
        fill=MUTED,
    )
    location = metrics["location_metrics"]
    prior = metrics["mean_pattern_prior_location_metrics"]
    _draw_chip(
        draw,
        (margin, 118),
        "OOF top-1 target score",
        f"{float(location['top1_target_score']):.3f}",
        width=260,
    )
    _draw_chip(
        draw,
        (margin + 274, 118),
        "mean-pattern prior",
        f"{float(prior['top1_target_score']):.3f}",
        width=240,
    )
    _draw_chip(
        draw,
        (margin + 528, 118),
        "OOF top-3 hit",
        f"{100.0 * float(location['top3_hit']):.1f}%",
        width=220,
    )
    draw.text(
        (margin + 780, 129),
        "Solid trace + dots = exact shared-topology truth\nDashed square + cross = OOF model's finest 8x8 top-1 cell",
        font=_font(17),
        fill=TEXT,
        spacing=6,
    )

    legend_y = 191
    legend_width = (canvas_width - margin * 2) // legend_columns
    for index, name in enumerate(element_names):
        column, row_index = index % legend_columns, index // legend_columns
        x = margin + column * legend_width
        y = legend_y + row_index * 34
        colour = colours[name]
        draw.line((x, y + 10, x + 28, y + 10), fill=colour, width=4)
        draw.ellipse((x + 11, y + 6, x + 19, y + 14), fill=colour)
        draw.text((x + 38, y), name, font=_font(16), fill=TEXT)

    corpus_lookup = {
        str(sample_id): index for index, sample_id in enumerate(corpus["sample_ids"])
    }
    parameter_metrics = metrics["parameter_metrics"]
    current_y = header_height
    for selection_label, prediction_row in selected:
        sample_id = str(prediction_row["sample_id"])
        corpus_index = corpus_lookup[sample_id]
        heatmaps = np.asarray(corpus["element_heatmaps"][corpus_index], dtype=np.float32)
        row_cells = prediction_cells[sample_id]["cells"]
        row_index = index_by_id[sample_id]
        geometry = _sample_surface_geometry(
            sample_id,
            raw_root=raw_root,
            semantic_record=records_by_id[sample_id],
            element_names=element_names,
        )
        card_box = (
            margin,
            current_y,
            canvas_width - margin,
            current_y + sample_height,
        )
        draw.rounded_rectangle(card_box, radius=18, fill=CARD)
        draw.text(
            (margin + 20, current_y + 17),
            sample_id,
            font=_font(23, bold=True),
            fill=TEXT,
        )
        location_score = _mean_location_score(prediction_row)
        normalized_error = _normalized_parameter_error(prediction_row, metrics)
        draw.text(
            (margin + 260, current_y + 22),
            (
                f"{selection_label}  |  sample top-1 target score {location_score:.3f}  |  "
                f"measurement normalized MAE {normalized_error:.2f} sigma"
            ),
            font=_font(16),
            fill=MUTED,
        )

        image_y = current_y + 78
        for view_index, (view_name, camera, raw_path) in enumerate(
            zip(VIEW_NAMES, VIEW_CAMERAS, row_index["view_paths"])
        ):
            card_x = margin + 14 + view_index * (view_card_width + view_gap)
            draw.rounded_rectangle(
                (
                    card_x,
                    image_y - 28,
                    card_x + view_card_width,
                    image_y + image_size + 32,
                ),
                radius=12,
                fill=CARD_INNER,
            )
            title = f"{view_name} ({camera})"
            title_font = _font(16, bold=True)
            draw.text(
                (
                    card_x + (view_card_width - _text_width(draw, title, title_font)) // 2,
                    image_y - 23,
                ),
                title,
                font=title_font,
                fill=TEXT,
            )
            rendered = _composite_render(Path(raw_path), image_size)
            rendered = _overlay_view(
                rendered,
                geometry=geometry,
                prediction_cells=row_cells,
                target_heatmaps=heatmaps,
                element_names=element_names,
                colours=colours,
                view_index=view_index,
            )
            canvas.paste(rendered, (card_x + 13, image_y))
            view_scores = [
                float(per_view[str(view_index)]["top1_target_score"])
                for per_view in prediction_row["element_location"].values()
                if str(view_index) in per_view
            ]
            view_score = float(np.mean(view_scores)) if view_scores else math.nan
            score_text = f"view mean target score {view_score:.3f}"
            score_font = _font(14)
            draw.text(
                (
                    card_x
                    + (view_card_width - _text_width(draw, score_text, score_font)) // 2,
                    image_y + image_size + 8,
                ),
                score_text,
                font=score_font,
                fill=MUTED,
            )

        table_x = margin + views_width + table_gap
        table_y = current_y + 67
        draw.text(
            (table_x, table_y),
            "Seven varying 2D measurements (held-out fold)",
            font=_font(20, bold=True),
            fill=TEXT,
        )
        table_y += 37
        columns = (table_x, table_x + 420, table_x + 565, table_x + 710)
        for x, heading in zip(columns, ("measurement", "truth", "prediction", "absolute error")):
            draw.text((x, table_y), heading, font=_font(14, bold=True), fill=MUTED)
        table_y += 27
        evaluated_names = [
            name
            for name in parameter_names
            if str(prediction_row["parameters"][name].get("status", "")).startswith(
                "EVALUATED"
            )
        ]
        for row_number, name in enumerate(evaluated_names):
            item = prediction_row["parameters"][name]
            truth = float(item["truth"])
            prediction = float(item["prediction"])
            error = abs(prediction - truth)
            scale = float(parameter_metrics.get(name, {}).get("truth_standard_deviation", 0.0))
            normalized = error / scale if scale > 1e-8 else math.nan
            error_colour = GREEN if normalized < 0.5 else AMBER if normalized < 1.0 else RED
            y = table_y + row_number * 39
            if row_number % 2:
                draw.rounded_rectangle(
                    (table_x - 8, y - 5, table_x + table_width - 12, y + 29),
                    radius=7,
                    fill="#142238",
                )
            unit = "deg" if name.endswith("_deg") else "cm"
            draw.text((columns[0], y), _short_parameter(name), font=_font(16), fill=TEXT)
            draw.text((columns[1], y), f"{truth:7.2f} {unit}", font=_font(16), fill=TEXT)
            draw.text((columns[2], y), f"{prediction:7.2f} {unit}", font=_font(16), fill=TEXT)
            draw.text(
                (columns[3], y),
                f"{error:6.2f} ({normalized:.2f} sigma)",
                font=_font(16, bold=True),
                fill=error_colour,
            )
        constant_names = [
            name
            for name in parameter_names
            if name not in evaluated_names
        ]
        note_y = table_y + len(evaluated_names) * 39 + 10
        if constant_names:
            draw.text(
                (table_x, note_y),
                "Not scored because constant in each training fold: "
                + ", ".join(_short_parameter(name) for name in constant_names),
                font=_font(14),
                fill=SUBTLE,
            )
            note_y += 27
        mapped_count = sum(
            int(geometry["elements"][name]["vertex_count"]) for name in element_names
        )
        draw.text(
            (table_x, note_y),
            (
                f"Exact overlay evidence: {mapped_count:,} semantic vertex memberships; "
                f"mesh {geometry['vertex_count']:,} vertices."
            ),
            font=_font(14),
            fill=SUBTLE,
        )
        draw.text(
            (table_x, note_y + 23),
            "Cells are re-inferred from this sample's held-out checkpoint, not copied from truth.",
            font=_font(14),
            fill=SUBTLE,
        )
        current_y += sample_height + sample_gap

    footer_y = canvas_height - footer_height + 12
    draw.rounded_rectangle(
        (margin, footer_y - 5, canvas_width - margin, canvas_height - 18),
        radius=12,
        fill="#291923",
    )
    draw.text(
        (margin + 18, footer_y + 8),
        (
            "Claim boundary: observational same-generator/fixed-body correspondence only. "
            "No edited-pattern counterfactual renders -> no causal influence or inverse-edit claim."
        ),
        font=_font(17, bold=True),
        fill="#ffd4d8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    corpus_archive = np.load(args.corpus, allow_pickle=False)
    corpus = {name: corpus_archive[name] for name in corpus_archive.files}
    element_names = tuple(str(value) for value in corpus["element_names"])
    parameter_names = tuple(str(value) for value in corpus["parameter_names"])
    metrics = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(args.model_dir / "cross_validation_predictions.jsonl")
    selected = _select_rows(predictions, metrics, args.sample_ids, args.sample_count)
    index_by_id = {
        str(row["sample_id"]): row for row in read_jsonl(args.index)
    }
    records_by_id = {
        str(row["sample_id"]): row for row in read_jsonl(args.records)
    }
    selected_ids = [str(row["sample_id"]) for _, row in selected]
    for sample_id in selected_ids:
        if sample_id not in index_by_id or sample_id not in records_by_id:
            raise KeyError(f"missing index or semantic record for {sample_id}")

    fpn_archive = np.load(args.fpn_cache, allow_pickle=False, mmap_mode="r")
    fpn = {name: fpn_archive[name] for name in fpn_archive.files}
    inferred = _infer_oof_cells(
        selected,
        corpus=corpus,
        fpn=fpn,
        model_dir=args.model_dir,
        element_names=element_names,
        parameter_names=parameter_names,
    )
    _render_board(
        selected=selected,
        corpus=corpus,
        prediction_cells=inferred,
        metrics=metrics,
        index_by_id=index_by_id,
        records_by_id=records_by_id,
        raw_root=args.raw_root,
        element_names=element_names,
        parameter_names=parameter_names,
        output=args.output,
        image_size=int(args.image_size),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": args.output.as_posix(),
                "sample_ids": selected_ids,
                "element_query_count": len(element_names),
                "evaluated_measurement_count": sum(
                    1
                    for item in metrics["parameter_metrics"].values()
                    if int(item.get("count", 0)) > 0
                ),
                "claim_boundary": metrics.get("claim_boundary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
