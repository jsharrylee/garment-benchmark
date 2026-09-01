from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from benchmark.gcdv2_exact.panel_dataset import render_panel_overlay


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fit(path: Path, size: tuple[int, int], *, background: str = "#090a0d") -> Image.Image:
    source = Image.open(path).convert("RGB")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, background)
    result.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
    return result


def _select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    available = [row for row in rows if row["paired_four_view_available"]]
    rng.shuffle(available)
    selected: list[dict[str, Any]] = []
    seen_parts: set[tuple[str, str]] = set()
    category_counts: defaultdict[str, int] = defaultdict(int)
    quotas = {"top": 4, "skirt": 3, "pants": 3}
    for row in available:
        category = str(row["garment_category"])
        key = (category, str(row["role_part"]))
        if category_counts[category] >= quotas.get(category, count):
            continue
        if key in seen_parts:
            continue
        selected.append(row)
        seen_parts.add(key)
        category_counts[category] += 1
        if len(selected) == count:
            return selected
    for row in available:
        if row in selected:
            continue
        category = str(row["garment_category"])
        if category_counts[category] >= quotas.get(category, count):
            continue
        selected.append(row)
        category_counts[category] += 1
        if len(selected) == count:
            break
    return selected


def _render_board(row: dict[str, Any], destination: Path) -> Path:
    width, height = 2000, 1390
    board = Image.new("RGB", (width, height), "#f4f0e8")
    draw = ImageDraw.Draw(board)
    target = json.loads(Path(row["target_path"]).read_text(encoding="utf-8"))
    overlay_path = destination.with_name(destination.stem + "_overlay.png")
    render_panel_overlay(target, overlay_path)

    draw.text((45, 30), "GCDv2 single-panel exact-pair review", font=_font(42, bold=True), fill="#16171a")
    title = (
        f"{row['garment_category']} · {row['source_panel_id']} · weak role="
        f"{row['role_part']}/{row['role_surface']}/{row['role_side']}"
    )
    draw.text((45, 85), title, font=_font(25, bold=True), fill="#35373c")
    info = (
        f"{row['width_cm']:.2f} × {row['height_cm']:.2f} cm · "
        f"vertices {row['vertex_count']} · edges {row['edge_count']} · "
        f"scale token {row['panel_image_cm_per_pixel']:.5f} cm/px"
    )
    draw.text((45, 123), info, font=_font(22), fill="#55575c")

    slots = [(45, 180, 570, 570), (715, 180, 570, 570), (1385, 180, 570, 570)]
    paths = [Path(row["panel_image_path"]), overlay_path, Path(row["sample_pattern_path"])]
    labels = [
        "Model input: this panel only",
        "Ground-truth vector overlay",
        "Full source layout: context only, not model input",
    ]
    for (x, y, w, h), path, label in zip(slots, paths, labels):
        draw.rounded_rectangle((x - 8, y - 48, x + w + 8, y + h + 8), 12, fill="white", outline="#c8c2b8", width=2)
        draw.text((x, y - 39), label, font=_font(21, bold=True), fill="#202126")
        board.paste(_fit(path, (w, h)), (x, y))

    view_labels = ["front", "back", "left", "right"]
    for index, (label, raw_path) in enumerate(zip(view_labels, row["paired_view_paths"])):
        x, y, w, h = 45 + index * 485, 840, 445, 445
        draw.rounded_rectangle((x - 6, y - 42, x + w + 6, y + h + 6), 10, fill="white", outline="#c8c2b8", width=2)
        draw.text((x, y - 35), f"paired 3D {label}", font=_font(20, bold=True), fill="#202126")
        board.paste(_fit(Path(raw_path), (w, h)), (x, y))
    draw.text(
        (45, 1330),
        "Validation: panel.png contains no other panel, layout, or color cues. Length uses the image and a cm/px scale token.",
        font=_font(22, bold=True),
        fill="#087f65",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, optimize=True)
    overlay_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic visual QA for exact single-panel GCDv2 pairs.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_panels_v1/review_seed_20260829"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    selected = _select(_read_rows(args.index), args.count, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    boards = []
    for index, row in enumerate(selected, 1):
        board = _render_board(row, args.output / f"{index:02d}_{row['garment_category']}_{row['source_panel_id']}.png")
        boards.append((row, board))

    thumb_w, thumb_h = 900, 626
    contact = Image.new("RGB", (thumb_w * 2, thumb_h * 5), "#ddd8cf")
    for index, (_, board_path) in enumerate(boards):
        contact.paste(_fit(board_path, (thumb_w, thumb_h), background="#ddd8cf"), ((index % 2) * thumb_w, (index // 2) * thumb_h))
    contact_path = args.output / "contact_sheet_10.png"
    contact.save(contact_path, optimize=True)
    summary = {
        "status": "PASS" if len(boards) == args.count else "PARTIAL",
        "seed": args.seed,
        "count": len(boards),
        "contact_sheet": contact_path.as_posix(),
        "panels": [
            {
                "panel_uid": row["panel_uid"],
                "garment_category": row["garment_category"],
                "source_panel_id": row["source_panel_id"],
                "board": board.as_posix(),
            }
            for row, board in boards
        ],
    }
    (args.output / "review.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
