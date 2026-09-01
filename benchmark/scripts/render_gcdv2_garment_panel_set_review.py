from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _relative_control(start: Sequence[float], end: Sequence[float], relative: Sequence[float]) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return start[0] + relative[0] * dx - relative[1] * dy, start[1] + relative[0] * dy + relative[1] * dx


def _bezier(points: Sequence[Sequence[float]], samples: int = 50) -> list[tuple[float, float]]:
    result = []
    for index in range(samples + 1):
        t = index / samples
        if len(points) == 3:
            weights = ((1-t)**2, 2*(1-t)*t, t*t)
        else:
            weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
        result.append((sum(weight * point[0] for weight, point in zip(weights, points)), sum(weight * point[1] for weight, point in zip(weights, points))))
    return result


def _predicted_points(panel: dict[str, Any], edge_index: int) -> list[tuple[float, float]]:
    vertices = panel["predicted_vertices_uv"]
    start = vertices[edge_index]
    end = vertices[(edge_index + 1) % len(vertices)]
    kind = panel["predicted_curve_types"][edge_index]
    controls = panel["predicted_relative_controls"][edge_index]
    if kind == "quadratic_bezier":
        return _bezier((start, _relative_control(start, end, controls[:2]), end))
    if kind == "cubic_bezier":
        return _bezier((start, _relative_control(start, end, controls[:2]), _relative_control(start, end, controls[2:4]), end))
    # Arc rendering is intentionally a chord here: radius is evaluated
    # numerically, while the review focuses on vertex/incidence recovery.
    return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]


def _panel_card(row: dict[str, Any], panel: dict[str, Any], size: int = 320) -> Image.Image:
    source = Image.open(row["panel_image_path"]).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(source)
    truth = panel["target_vertices_uv"]
    prediction = panel["predicted_vertices_uv"]
    for edge_index in range(len(prediction)):
        points = [(x * size, y * size) for x, y in _predicted_points(panel, edge_index)]
        draw.line(points, fill="#ff493d", width=4, joint="curve")
    for index, (x, y) in enumerate(truth):
        x, y = x * size, y * size
        draw.ellipse((x-5, y-5, x+5, y+5), fill="#00d991", outline="#07130f", width=1)
        draw.text((x+5, y-12), str(index), font=_font(11, bold=True), fill="#00a26d")
    for index, (x, y) in enumerate(prediction):
        x, y = x * size, y * size
        draw.ellipse((x-4, y-4, x+4, y+4), fill="#ff493d", outline="#2d0805", width=1)
    return source


def _render_garment(prediction: dict[str, Any], rows: dict[str, dict[str, Any]], destination: Path) -> Path:
    columns, card_w, card_h = 4, 350, 430
    panel_count = len(prediction["panels"])
    rows_count = math.ceil(panel_count / columns)
    board = Image.new("RGB", (columns * card_w + 60, rows_count * card_h + 170), "#f3efe7")
    draw = ImageDraw.Draw(board)
    draw.text((30, 25), f"Unseen garment: {prediction['sample_id']}", font=_font(34, bold=True), fill="#17181b")
    category_ok = prediction["target_category"] == prediction["predicted_category"]
    draw.text((30, 75), f"category GT={prediction['target_category']}  PRED={prediction['predicted_category']}  {'PASS' if category_ok else 'FAIL'}", font=_font(23, bold=True), fill="#078261" if category_ok else "#c53127")
    draw.text((30, 112), "Green = target vertex · red = predicted graph · white fill = input panel.png", font=_font(20), fill="#4b4d52")
    for index, panel in enumerate(prediction["panels"]):
        row = rows[panel["panel_uid"]]
        x = 30 + (index % columns) * card_w
        y = 155 + (index // columns) * card_h
        board.paste(_panel_card(row, panel), (x, y))
        source_ok = panel["target_source_panel_id"] == panel["predicted_source_panel_id"]
        draw.text((x, y + 325), f"GT {panel['target_source_panel_id']}", font=_font(16, bold=True), fill="#107a5f")
        draw.text((x, y + 348), f"P  {panel['predicted_source_panel_id']}", font=_font(16, bold=True), fill="#a32b24" if not source_ok else "#107a5f")
        draw.text((x, y + 372), f"role {panel['target_part']}→{panel['predicted_part']}  edges {panel['target_count']}→{panel['predicted_count']}", font=_font(15), fill="#33353a")
        draw.text((x, y + 396), f"surface {panel['target_surface']}→{panel['predicted_surface']} · side {panel['target_side']}→{panel['predicted_side']}", font=_font(14), fill="#55575c")
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, optimize=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render unseen garment-set graph predictions.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/test_predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_garment_panel_set/review_seed_20260829"))
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    rows = {row["panel_uid"]: row for line in args.index.read_text(encoding="utf-8").splitlines() if line for row in [json.loads(line)]}
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(args.seed)
    selected = []
    for category in ("top", "skirt", "pants"):
        candidates = [row for row in predictions if row["target_category"] == category]
        rng.shuffle(candidates)
        selected.extend(candidates[:2])
    boards = []
    for index, prediction in enumerate(selected, 1):
        boards.append(_render_garment(prediction, rows, args.output / f"{index:02d}_{prediction['target_category']}_{prediction['sample_id']}.png"))
    thumb_w, thumb_h = 850, 700
    contact = Image.new("RGB", (thumb_w * 2, thumb_h * 3), "#d9d4cb")
    for index, path in enumerate(boards):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (thumb_w, thumb_h), "#d9d4cb")
        cell.paste(image, ((thumb_w-image.width)//2, (thumb_h-image.height)//2))
        contact.paste(cell, ((index%2)*thumb_w, (index//2)*thumb_h))
    contact_path = args.output / "contact_sheet_6_unseen_garments.png"
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    contact.save(contact_path, optimize=True)
    result = {"status": "PASS", "seed": args.seed, "contact_sheet": contact_path.as_posix(), "boards": [path.as_posix() for path in boards]}
    (args.output / "review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
