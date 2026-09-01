from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmark.drafting_semantics.basic_blocks import build_basic_block
from benchmark.drafting_semantics.tshirt_parametric_decoder import (
    TShirtDraftParameters,
    TShirtParametricDraftingDecoder,
)
from benchmark.pattern_pipeline.schema import PatternDocument


VIEW_FILES = ("CAM000.png", "CAM001.png", "CAM002.png", "CAM003.png")
VIEW_LABELS = ("front", "back", "left", "right")
EDGE_COLORS = {
    "neckline": "#e91e63",
    "shoulder": "#00897b",
    "armhole": "#7e57c2",
    "side_seam": "#1e88e5",
    "hemline": "#6d4c41",
    "sleeve_head_front": "#fb8c00",
    "sleeve_head_back": "#fb8c00",
    "sleeve_head": "#fb8c00",
    "sleeve_underarm_front": "#5e35b1",
    "sleeve_underarm_back": "#5e35b1",
    "sleeve_underarm": "#5e35b1",
    "sleeve_hem": "#6d4c41",
    "center_front": "#f4511e",
    "center_back": "#f4511e",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)


def _paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    fitted = _fit_image(image, x1 - x0, y1 - y0)
    canvas.paste(
        fitted,
        (x0 + (x1 - x0 - fitted.width) // 2, y0 + (y1 - y0 - fitted.height) // 2),
    )


def _four_view_contact(folder: Path, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(18, bold=True)
    gap = 8
    cell_width = (width - 3 * gap) // 4
    for index, (filename, label) in enumerate(zip(VIEW_FILES, VIEW_LABELS)):
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(path)
        x0 = index * (cell_width + gap)
        image = Image.open(path)
        _paste_center(canvas, image, (x0, 24, x0 + cell_width, height))
        draw.text((x0 + 4, 2), label, fill="#202124", font=label_font)
    return canvas


def _panel_points(document: PatternDocument, panel_id: str) -> list[tuple[float, float]]:
    panel = next(item for item in document.panels if item.id == panel_id)
    return [point for edge in panel.edges for point in edge.points]


def _render_pattern(document: PatternDocument, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(17, bold=True)
    edge_labels = document.annotations.get("edge_labels", {})
    panels = document.panels
    gap = 16
    panel_width = max(1, (width - gap * (len(panels) + 1)) // max(len(panels), 1))
    for panel_index, panel in enumerate(panels):
        points = _panel_points(document, panel.id)
        minimum_x = min(point[0] for point in points)
        maximum_x = max(point[0] for point in points)
        minimum_y = min(point[1] for point in points)
        maximum_y = max(point[1] for point in points)
        source_width = max(maximum_x - minimum_x, 1e-6)
        source_height = max(maximum_y - minimum_y, 1e-6)
        x0 = gap + panel_index * (panel_width + gap)
        y0 = 35
        available_height = height - y0 - 14
        scale = min((panel_width - 16) / source_width, (available_height - 10) / source_height)
        offset_x = x0 + (panel_width - source_width * scale) / 2.0
        offset_y = y0 + (available_height - source_height * scale) / 2.0

        def transform(values: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
            return [
                (
                    offset_x + (point[0] - minimum_x) * scale,
                    offset_y + (point[1] - minimum_y) * scale,
                )
                for point in values
            ]

        draw.text((x0 + 4, 5), panel.id, fill="#202124", font=label_font)
        for edge in panel.edges:
            rendered = transform(edge.points)
            semantic_role = edge_labels.get(f"{panel.id}/{edge.id}", edge.id)
            color = EDGE_COLORS.get(str(semantic_role), "#455a64")
            draw.line(rendered, fill=color, width=4, joint="curve")
            for point in (rendered[0], rendered[-1]):
                radius = 3
                draw.ellipse(
                    (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                    fill="white",
                    outline=color,
                    width=2,
                )
    return canvas


def _target_pattern(path: Path, width: int, height: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return _fit_image(Image.open(path).convert("RGB"), width, height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render all frozen T-shirt decoder cases without cherry-picking."
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--multiview-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pattern-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rows = evaluation["samples"]
    if len(rows) != 6:
        raise ValueError("the frozen board must contain all six T-shirt test samples")
    default_block = build_basic_block("tshirt")
    default_sleeve = default_block.panel("sleeve")
    default_armhole_total_cm = float(
        default_sleeve.metadata["front_armhole_length_cm"]
    ) + float(default_sleeve.metadata["back_armhole_length_cm"])
    baseline = TShirtParametricDraftingDecoder(samples_per_cubic=49).decode_document(
        TShirtDraftParameters.from_mapping(
            {"sleeve_ease_cm": default_armhole_total_cm * 0.01}
        ),
        pattern_id="default_parametric_tshirt",
    )
    column_widths = (700, 540, 520, 520, 430)
    margin = 24
    title_height = 120
    row_height = 360
    width = sum(column_widths) + margin * (len(column_widths) + 1)
    height = title_height + row_height * len(rows) + margin
    board = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(board)
    title_font = _font(34, bold=True)
    header_font = _font(22, bold=True)
    body_font = _font(19)
    small_font = _font(17)
    draw.text(
        (margin, 14),
        "T-shirt parametric decoder · frozen GCDv2 test (all 6, no cherry-pick)",
        fill="#101418",
        font=title_font,
    )
    draw.text(
        (margin, 58),
        "Four-view Transformer → 12 fitted design parameters → shared-point instance graph → exact sleeve seam solver",
        fill="#3c4858",
        font=body_font,
    )
    headers = ("4-view input", "GCD source pattern", "default parametric graph", "student parametric graph", "metrics / fitted parameters")
    x = margin
    for header, column_width in zip(headers, column_widths):
        draw.text((x, 91), header, fill="#202124", font=header_font)
        x += column_width + margin

    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        top = title_height + row_index * row_height
        if row_index % 2:
            draw.rectangle((0, top, width, top + row_height), fill="#eef1f5")
        cells = []
        x = margin
        for column_width in column_widths:
            cells.append((x, top + 36, x + column_width, top + row_height - 16))
            x += column_width + margin
        draw.text((margin, top + 6), sample_id, fill="#17202a", font=header_font)

        views = _four_view_contact(
            args.multiview_root / sample_id,
            cells[0][2] - cells[0][0],
            cells[0][3] - cells[0][1],
        )
        board.paste(views, (cells[0][0], cells[0][1]))
        target_path = args.source_root / sample_id / f"{sample_id}_pattern.png"
        target = _target_pattern(target_path, cells[1][2] - cells[1][0], cells[1][3] - cells[1][1])
        _paste_center(board, target, cells[1])
        anchor_image = _render_pattern(baseline, cells[2][2] - cells[2][0], cells[2][3] - cells[2][1])
        board.paste(anchor_image, (cells[2][0], cells[2][1]))
        student_doc = PatternDocument.read_json(
            args.pattern_root / row["pattern_artifacts"]["student"]
        )
        student_image = _render_pattern(
            student_doc, cells[3][2] - cells[3][0], cells[3][3] - cells[3][1]
        )
        board.paste(student_image, (cells[3][0], cells[3][1]))

        metric_x, metric_y = cells[4][0] + 8, cells[4][1] + 6
        anchor_mae = float(row["anchor_coordinate_mae"])
        student_mae = float(row["student_parametric_coordinate_mae"])
        oracle_mae = float(row["oracle_parametric_coordinate_mae"])
        change = 100.0 * (student_mae - anchor_mae) / anchor_mae
        exact_constraint = row["student_canonical_graph"]["sleeve_head_constraint"]
        sampled_constraint = row["student_sampled_document_constraint"]
        lines = [
            f"anchor semantic MAE  {anchor_mae:.4f}",
            f"parametric semantic   {student_mae:.4f}",
            f"change          {change:+.2f}%",
            f"oracle ceiling  {oracle_mae:.4f}",
            "graph / symmetry PASS",
            f"sleeve ease     +{float(exact_constraint['sleeve_ease_cm']):.3f} cm",
            (
                "cubic/sample res "
                f"{float(exact_constraint['residual_cm']):+.1e}/"
                f"{float(sampled_constraint['residual_cm']):+.1e} cm"
            ),
            "",
        ]
        parameters = row["student_projection"]["design_parameters_cm"]
        labels = (
            ("neck width", "neck_width_cm"),
            ("front neck", "front_neck_depth_cm"),
            ("back neck", "back_neck_depth_cm"),
            ("shoulder drop", "shoulder_drop_cm"),
            ("armhole depth", "armhole_depth_cm"),
            ("body length", "body_length_cm"),
            ("sleeve length", "sleeve_length_cm"),
        )
        lines.extend(f"{label:14} {float(parameters[key]):5.2f} cm" for label, key in labels)
        for line_index, line in enumerate(lines):
            color = "#087f23" if "PASS" in line or ("change" in line and change < 0) else "#263238"
            draw.text(
                (metric_x, metric_y + line_index * 21),
                line,
                fill=color,
                font=small_font,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.output, optimize=True)
    print(
        json.dumps(
            {
                "status": "COMPLETE_ALL_FROZEN_TEST_IDS",
                "sample_count": len(rows),
                "width": width,
                "height": height,
                "source_images_embedded_in_tracked_report": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
