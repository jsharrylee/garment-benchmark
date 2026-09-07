"""Build the small, attributed case-study figures used by the public report.

The script reads local experiment artifacts but writes only compact derived PNGs.
All source roots are CLI arguments so no machine-specific path is embedded in the
public snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIEWS = ("front", "back", "left", "right")
EDGE_COLORS = {
    "L": (38, 94, 158),
    "Q": (213, 94, 0),
    "C": (0, 128, 96),
    "A": (128, 72, 170),
}
INK = (31, 43, 55)
MUTED = (91, 104, 117)
GRID = (218, 224, 230)
LIGHT = (246, 248, 250)
GOOD = (24, 128, 88)
BAD = (185, 53, 48)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, *, bold: bool = False, explicit: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            "arialbd.ttf" if bold else "arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, face, fill=INK) -> None:
    box = draw.textbbox((0, 0), value, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=face, fill=fill)


def paste_contain(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int], *, border: bool = False) -> None:
    x0, y0, x1, y1 = box
    available = (max(1, x1 - x0), max(1, y1 - y0))
    image = source.convert("RGB").copy()
    image.thumbnail(available, Image.Resampling.LANCZOS)
    x = x0 + (available[0] - image.width) // 2
    y = y0 + (available[1] - image.height) // 2
    canvas.paste(image, (x, y))
    if border:
        ImageDraw.Draw(canvas).rectangle(box, outline=GRID, width=2)


def save_public_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    if path.stat().st_size > 1_900_000:
        image.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def build_panel_pretraining_figure(initial_path: Path, best_path: Path, output: Path, font_path: str | None) -> None:
    initial = Image.open(initial_path).convert("RGB")
    best = Image.open(best_path).convert("RGB")
    if initial.size != best.size:
        raise ValueError("initial and best diagnostic figures must have the same size")

    width, height = initial.size
    # The source diagnostic is a stable 3-row contact sheet: RGB / GT / prediction.
    crops = [
        (best.crop((0, int(height * 0.10), width, int(height * 0.40))), "Input RGB"),
        (best.crop((0, int(height * 0.40), width, int(height * 0.70))), "GT visible-panel mask"),
        (initial.crop((0, int(height * 0.70), width, height)), "Before training"),
        (best.crop((0, int(height * 0.70), width, height)), "After epoch 9"),
    ]

    canvas = Image.new("RGB", (1680, 1510), "white")
    draw = ImageDraw.Draw(canvas)
    title = font(38, bold=True, explicit=font_path)
    label = font(27, bold=True, explicit=font_path)
    small = font(18, explicit=font_path)
    text_center(draw, (840, 42), "Panel-mask pretraining: same garment before and after", title)
    text_center(draw, (840, 82), "rand_0BYYF4ETOT | internal diagnostic, not independent test evidence", small, MUTED)

    y = 110
    for crop, name in crops:
        draw.text((42, y + 118), name, font=label, fill=INK)
        paste_contain(canvas, crop, (300, y, 1640, y + 320))
        draw.line((300, y + 322, 1640, y + 322), fill=GRID, width=2)
        y += 335

    draw.text(
        (42, 1442),
        "Gray = GT validity/ignore band around antialiasing, raster disagreement, and panel interfaces.",
        font=small,
        fill=MUTED,
    )
    draw.text(
        (42, 1471),
        "It is neither RGB input nor model output; the same band is overlaid after prediction and excluded from loss/IoU.",
        font=small,
        fill=MUTED,
    )
    save_public_png(canvas, output)


def target_panels(dataset_root: Path, sample_id: str) -> list[dict[str, Any]]:
    sample = dataset_root / "samples" / sample_id
    decoded = read_json(sample / "conversion_retry_03" / "decoded_pattern.json")
    target_info = read_json(sample / "conversion_retry_03" / "panel_targets.json")
    result = []
    type_map = {
        "line": "L",
        "quadratic_bezier": "Q",
        "cubic_bezier": "C",
        "circular_arc": "A",
    }
    for panel in decoded["panels"]:
        short_id = panel["id"].split(":", 1)[-1]
        metadata = target_info.get(short_id, {})
        meta_edges = metadata.get("geometry", {}).get("edges", [])
        edges = []
        for index, edge in enumerate(panel["edges"]):
            op = "L"
            if index < len(meta_edges):
                op = type_map.get(meta_edges[index].get("curve_type"), "L")
            edges.append({"op": op, "points": np.asarray(edge["points"], dtype=np.float64)})
        result.append({"label": short_id, "edges": edges})
    return result


def target_panel(dataset_root: Path, sample_id: str, panel_id: str) -> dict[str, Any]:
    panels = target_panels(dataset_root, sample_id)
    for panel in panels:
        if panel["label"] == panel_id:
            return panel
    raise KeyError(f"panel {panel_id!r} not found for {sample_id}")


def load_prediction(codec_module, record_path: Path, sample_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(record_path)
    record = next(row for row in payload["records"] if row["sample_id"] == sample_id)
    decoded = codec_module.decode_record(record["prediction"])
    panels = []
    for panel_index, panel in enumerate(decoded["panels"]):
        edges = []
        for edge in panel["edges"]:
            points = codec_module.curve_points(edge, np.linspace(0.0, 1.0, 65))
            edges.append(
                {
                    "op": edge["op"],
                    "points": np.column_stack((points.real, points.imag)),
                }
            )
        panels.append({"label": f"P{panel_index + 1}", "edges": edges})
    return panels, record["metrics"]


def panel_bounds(panel: dict[str, Any]) -> tuple[float, float, float, float]:
    points = np.concatenate([edge["points"] for edge in panel["edges"]], axis=0)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def draw_panel_collection(
    canvas: Image.Image,
    panels: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    font_path: str | None,
    *,
    columns: int = 3,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    n = max(1, len(panels))
    columns = min(columns, n)
    rows = math.ceil(n / columns)
    cell_w = (x1 - x0) / columns
    cell_h = (y1 - y0) / rows
    bounds = [panel_bounds(panel) for panel in panels]
    max_w = max(max(b[2] - b[0], 1e-6) for b in bounds)
    max_h = max(max(b[3] - b[1], 1e-6) for b in bounds)
    scale = min((cell_w - 26) / max_w, (cell_h - 42) / max_h)
    label_face = font(16, explicit=font_path)

    for index, (panel, bounds_i) in enumerate(zip(panels, bounds, strict=True)):
        row, col = divmod(index, columns)
        cx0 = x0 + col * cell_w
        cy0 = y0 + row * cell_h
        min_x, min_y, max_x, max_y = bounds_i
        panel_w = (max_x - min_x) * scale
        panel_h = (max_y - min_y) * scale
        ox = cx0 + (cell_w - panel_w) / 2
        oy = cy0 + (cell_h - panel_h) / 2 + 10

        for edge in panel["edges"]:
            pts = []
            for px, py in edge["points"]:
                sx = ox + (float(px) - min_x) * scale
                sy = oy + panel_h - (float(py) - min_y) * scale
                pts.append((int(round(sx)), int(round(sy))))
            if len(pts) >= 2:
                draw.line(pts, fill=EDGE_COLORS.get(edge["op"], INK), width=3, joint="curve")
        text_center(draw, (int(cx0 + cell_w / 2), int(cy0 + cell_h - 12)), panel["label"], label_face, MUTED)


def draw_views(canvas: Image.Image, dataset_root: Path, sample_id: str, box: tuple[int, int, int, int], font_path: str | None) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    gap = 18
    cell_w = (x1 - x0 - gap * 3) // 4
    face = font(18, bold=True, explicit=font_path)
    for index, view in enumerate(VIEWS):
        left = x0 + index * (cell_w + gap)
        image = Image.open(dataset_root / "samples" / sample_id / "rgb" / f"{view}.png")
        paste_contain(canvas, image, (left, y0 + 26, left + cell_w, y1), border=True)
        text_center(draw, (left + cell_w // 2, y0 + 11), view, face, MUTED)


def draw_views_grid(canvas: Image.Image, dataset_root: Path, sample_id: str, box: tuple[int, int, int, int], font_path: str | None) -> None:
    """Draw four views as a compact 2x2 grid for multi-case figures."""
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    gap = 12
    label_h = 22
    cell_w = (x1 - x0 - gap) // 2
    cell_h = (y1 - y0 - gap) // 2
    face = font(15, bold=True, explicit=font_path)
    for index, view in enumerate(VIEWS):
        row, col = divmod(index, 2)
        left = x0 + col * (cell_w + gap)
        top = y0 + row * (cell_h + gap)
        image = Image.open(dataset_root / "samples" / sample_id / "rgb" / f"{view}.png")
        paste_contain(canvas, image, (left, top + label_h, left + cell_w, top + cell_h), border=True)
        text_center(draw, (left + cell_w // 2, top + 9), view, face, MUTED)


def metric_line(metrics: dict[str, Any]) -> str:
    structure = metrics["structure"]
    geometry = metrics["geometry"]
    seams = metrics["seams"]
    return (
        f"panels {structure['predicted_panels']}/{structure['target_panels']} | "
        f"edges {structure['predicted_edges']}/{structure['target_edges']} | "
        f"geom {geometry['global_scale_normalized_error_with_unmatched_penalty']:.3f} | "
        f"seam F1 {seams['selection_f1']:.3f}"
    )


def detailed_metric_line(metrics: dict[str, Any]) -> str:
    structure = metrics["structure"]
    geometry = metrics["geometry"]
    seams = metrics["seams"]
    return (
        f"panels {structure['predicted_panels']}/{structure['target_panels']}  |  "
        f"edges {structure['predicted_edges']}/{structure['target_edges']}  |  "
        f"typed {structure['typed_edge_accuracy_with_unmatched_penalty']:.2f}  |  "
        f"geom {geometry['global_scale_normalized_error_with_unmatched_penalty']:.2f}  |  "
        f"seam F1 {seams['selection_f1']:.2f}"
    )


def build_ordinary_good_cases(
    *,
    dataset_root: Path,
    codec_module,
    panel_records: Path,
    output: Path,
    font_path: str | None,
) -> None:
    """Show familiar silhouettes among the stronger internal-validation cases."""
    cases = [
        ("rand_ZLFD860PWX", "plain trousers"),
        ("rand_2BWWUFZYF0", "shorts"),
        ("rand_NF31FYNRTZ", "long-sleeve top"),
    ]
    canvas = Image.new("RGB", (1900, 1880), "white")
    draw = ImageDraw.Draw(canvas)
    title = font(38, bold=True, explicit=font_path)
    subtitle = font(20, explicit=font_path)
    heading = font(21, bold=True, explicit=font_path)
    metric = font(18, explicit=font_path)
    small = font(16, explicit=font_path)
    text_center(draw, (950, 42), "Familiar silhouettes: selected stronger validation cases", title)
    text_center(
        draw,
        (950, 80),
        "Post-hoc qualitative selection; panel-pretrained set/cycle/graph output, not independent test evidence",
        subtitle,
        MUTED,
    )

    row_height = 565
    for row_index, (sample_id, label) in enumerate(cases):
        y0 = 105 + row_index * row_height
        y1 = y0 + row_height - 18
        target = target_panels(dataset_root, sample_id)
        prediction, metrics = load_prediction(codec_module, panel_records, sample_id)
        draw.rounded_rectangle((28, y0, 1872, y1), radius=14, fill=LIGHT, outline=GRID, width=2)
        draw.text((50, y0 + 18), f"{label}  |  {sample_id}", font=heading, fill=INK)
        draw.text((450, y0 + 20), detailed_metric_line(metrics), font=metric, fill=GOOD)

        columns = 5 if max(len(target), len(prediction)) > 12 else 3
        headings = [
            (50, 570, "Neutral RGB, four views"),
            (590, 1218, "Ground-truth panels"),
            (1238, 1850, "Panel-pretrained prediction"),
        ]
        for left, right, name in headings:
            text_center(draw, ((left + right) // 2, y0 + 67), name, heading, MUTED)
        draw_views_grid(canvas, dataset_root, sample_id, (55, y0 + 88, 565, y1 - 28), font_path)
        draw_panel_collection(canvas, target, (600, y0 + 88, 1208, y1 - 22), font_path, columns=columns)
        draw_panel_collection(canvas, prediction, (1248, y0 + 88, 1840, y1 - 22), font_path, columns=columns)

    draw.text(
        (35, 1810),
        "Lower geometry error is better. Exact counts and high component metrics still do not certify an exact, sewable pattern.",
        font=small,
        fill=MUTED,
    )
    draw.text(
        (35, 1838),
        "L/Q/C/A colors describe boundary primitive type; seam correspondence is reported numerically rather than drawn.",
        font=small,
        fill=MUTED,
    )
    save_public_png(canvas, output)


def build_pattern_comparison(
    *,
    dataset_root: Path,
    codec_module,
    generic_records: Path,
    panel_records: Path,
    sample_id: str,
    verdict: str,
    output: Path,
    font_path: str | None,
) -> None:
    target = target_panels(dataset_root, sample_id)
    generic, generic_metrics = load_prediction(codec_module, generic_records, sample_id)
    panel, panel_metrics = load_prediction(codec_module, panel_records, sample_id)

    canvas = Image.new("RGB", (1800, 1260), "white")
    draw = ImageDraw.Draw(canvas)
    title = font(38, bold=True, explicit=font_path)
    subtitle = font(21, explicit=font_path)
    heading = font(26, bold=True, explicit=font_path)
    metric = font(17, explicit=font_path)
    small = font(17, explicit=font_path)
    verdict_color = GOOD if verdict == "improvement" else BAD
    verdict_text = "Representative relative improvement (still not an exact reconstruction)" if verdict == "improvement" else "Representative regression after panel pretraining"

    text_center(draw, (900, 42), verdict_text, title, verdict_color)
    text_center(draw, (900, 80), f"{sample_id} | internal validation case", subtitle, MUTED)
    text_center(draw, (900, 119), "Input: four neutral RGB views", heading, INK)
    draw_views(canvas, dataset_root, sample_id, (230, 142, 1570, 382), font_path)

    column_width = 552
    lefts = [48, 624, 1200]
    top = 420
    bottom = 1165
    blocks = [
        ("Ground truth", target, f"{len(target)} panels | {sum(len(p['edges']) for p in target)} edges", INK),
        ("Generic ImageNet ViT", generic, metric_line(generic_metrics), INK),
        ("Panel-pretrained ViT", panel, metric_line(panel_metrics), verdict_color),
    ]
    for left, (name, panels, details, accent) in zip(lefts, blocks, strict=True):
        draw.rounded_rectangle((left, top, left + column_width, bottom), radius=12, fill=LIGHT, outline=GRID, width=2)
        text_center(draw, (left + column_width // 2, top + 31), name, heading, accent)
        text_center(draw, (left + column_width // 2, top + 64), details, metric, MUTED)
        draw_panel_collection(canvas, panels, (left + 12, top + 85, left + column_width - 12, bottom - 18), font_path)

    legend_x = 50
    for op in ("L", "Q", "C", "A"):
        draw.line((legend_x, 1215, legend_x + 34, 1215), fill=EDGE_COLORS[op], width=5)
        draw.text((legend_x + 42, 1203), op, font=small, fill=INK)
        legend_x += 105
    draw.text((500, 1203), "Pattern drawings show panel boundaries; seam correspondence is reported numerically.", font=small, fill=MUTED)
    save_public_png(canvas, output)


def draw_single_panel(canvas: Image.Image, panel: dict[str, Any], box: tuple[int, int, int, int], font_path: str | None) -> None:
    draw_panel_collection(canvas, [panel], box, font_path, columns=1)


def draw_signature(draw: ImageDraw.ImageDraw, origin: tuple[int, int], signature: str, face, *, outline: tuple[int, int, int] = GRID) -> None:
    count_text, sequence = signature.split(":", 1)
    x, y = origin
    draw.text((x, y + 4), f"{count_text} edges", font=face, fill=MUTED)
    x += 92
    max_width = 560
    cell = max(18, min(32, max_width // max(1, len(sequence))))
    for token in sequence:
        color = EDGE_COLORS.get(token, INK)
        draw.rounded_rectangle((x, y, x + cell - 3, y + 32), radius=4, fill=(255, 255, 255), outline=color, width=2)
        text_center(draw, (x + (cell - 3) // 2, y + 16), token, face, color)
        x += cell
    draw.line((origin[0], y + 40, origin[0] + 660, y + 40), fill=outline, width=1)


def build_reranker_cases(dataset_root: Path, output: Path, font_path: str | None) -> None:
    cases = [
        {
            "status": "RECOVERED",
            "sample": "rand_166TVX8TI3",
            "panel": "left_sleeve_b",
            "target": "4:LLLC",
            "stage1": "5:LLLCC",
            "reranked": "4:LLLC",
            "note": "Stage-1 rank 2 → reranker selected the target signature",
            "color": GOOD,
        },
        {
            "status": "HARMED",
            "sample": "rand_0V45X4ELM8",
            "panel": "left_sleeve_b",
            "target": "5:LLLCC",
            "stage1": "5:LLLCC",
            "reranked": "4:LLLC",
            "note": "Stage-1 was correct → reranker changed it by one edge",
            "color": BAD,
        },
    ]
    canvas = Image.new("RGB", (1800, 1030), "white")
    draw = ImageDraw.Draw(canvas)
    title = font(38, bold=True, explicit=font_path)
    heading = font(25, bold=True, explicit=font_path)
    body = font(19, explicit=font_path)
    body_bold = font(19, bold=True, explicit=font_path)
    small = font(16, explicit=font_path)
    text_center(draw, (900, 42), "Line-signature reranker: one recovery and one failure", title)
    text_center(draw, (900, 79), "Both are internal-selection examples; signatures are cyclic L/Q/C/A configurations.", body, MUTED)

    for row_index, case in enumerate(cases):
        y0 = 110 + row_index * 430
        y1 = y0 + 400
        draw.rounded_rectangle((35, y0, 1765, y1), radius=14, fill=LIGHT, outline=GRID, width=2)
        draw.text((60, y0 + 22), case["status"], font=heading, fill=case["color"])
        draw.text((205, y0 + 27), f"{case['sample']} | target panel: {case['panel']}", font=body, fill=INK)

        draw_views(canvas, dataset_root, case["sample"], (55, y0 + 72, 585, y0 + 260), font_path)
        panel = target_panel(dataset_root, case["sample"], case["panel"])
        draw_single_panel(canvas, panel, (610, y0 + 72, 875, y0 + 340), font_path)
        text_center(draw, (742, y0 + 355), "GT panel outline", small, MUTED)

        labels = [("Target", case["target"]), ("Stage-1 top-1", case["stage1"]), ("Reranked top-1", case["reranked"])]
        for idx, (name, signature) in enumerate(labels):
            yy = y0 + 86 + idx * 86
            draw.text((905, yy + 6), name, font=body_bold, fill=INK)
            draw_signature(draw, (1080, yy), signature, body)
            correct = signature == case["target"]
            draw.text((1695, yy + 6), "OK" if correct else "WRONG", font=body_bold, fill=GOOD if correct else BAD)
        draw.text((905, y0 + 355), case["note"], font=body, fill=case["color"])

    draw.text(
        (45, 980),
        "These cases visualize line composition only; length, angle, control points, seams, and complete garment validity are not being scored here.",
        font=small,
        fill=MUTED,
    )
    save_public_png(canvas, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--panel-pretrain-root", type=Path, required=True)
    parser.add_argument("--set-graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root))
    from benchmark.rgb_pattern_ar import codec  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generic = args.set_graph_root / "run_002" / "A_generic_seed17" / "validation_best_records.json"
    panel = args.set_graph_root / "run_002" / "B_panel_seed17" / "validation_best_records.json"

    build_panel_pretraining_figure(
        args.panel_pretrain_root / "validation_initial.png",
        args.panel_pretrain_root / "validation_best.png",
        args.output_dir / "panel-mask-before-after.png",
        args.font_path,
    )
    build_pattern_comparison(
        dataset_root=args.dataset_root,
        codec_module=codec,
        generic_records=generic,
        panel_records=panel,
        sample_id="rand_F1HMCE6HDL",
        verdict="improvement",
        output=args.output_dir / "set-graph-relative-improvement.png",
        font_path=args.font_path,
    )
    build_ordinary_good_cases(
        dataset_root=args.dataset_root,
        codec_module=codec,
        panel_records=panel,
        output=args.output_dir / "ordinary-garment-stronger-cases.png",
        font_path=args.font_path,
    )
    build_pattern_comparison(
        dataset_root=args.dataset_root,
        codec_module=codec,
        generic_records=generic,
        panel_records=panel,
        sample_id="rand_G55CFYW2WX",
        verdict="regression",
        output=args.output_dir / "set-graph-regression.png",
        font_path=args.font_path,
    )
    build_reranker_cases(
        args.dataset_root,
        args.output_dir / "reranker-recovery-and-failure.png",
        args.font_path,
    )


if __name__ == "__main__":
    main()
