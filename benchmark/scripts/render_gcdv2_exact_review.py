from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from benchmark.gcdv2_exact.geometry import CURVE_COLORS, load_exact_label, render_exact_overlay


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    root = Path("C:/Windows/Fonts")
    path = root / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, *, bold: bool = False, fill="#202326", anchor=None) -> None:
    draw.text(xy, value, font=_font(size, bold), fill=fill, anchor=anchor, spacing=4)


def _contain(path: Path, size: tuple[int, int], background="#111216") -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    output = Image.new("RGB", size, background)
    output.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return output


def _board(record: dict, destination: Path) -> dict:
    label = load_exact_label(Path(record["label_path"]))
    overlay_path = destination.with_name(destination.stem + "_overlay.png")
    render_exact_overlay(Path(record["label_path"]), overlay_path, size=1600)
    width, height = 2400, 1920
    canvas = Image.new("RGB", (width, height), "#f5f1eb")
    draw = ImageDraw.Draw(canvas)
    _text(draw, (55, 40), f"GCDv2 exact paired review · {record['category'].upper()} · {record['sample_id']}", 43, bold=True)
    counts = Counter(edge["curve"]["type"] for panel in label["panels"] for edge in panel["edges"])
    _text(
        draw,
        (55, 100),
        f"{len(label['panels'])} panels · {sum(counts.values())} edges · "
        + " · ".join(f"{key.replace('_bezier','').replace('circular_arc','arc')} {counts[key]}" for key in counts),
        24,
        fill="#5c574f",
    )
    cell_w, cell_h = 1135, 770
    cells = [(55, 150), (1210, 150), (55, 980), (630, 980), (1205, 980), (1780, 980)]
    names = ["Non-overlapping clean 2D pattern", "Exact label overlay", "front (CAM001)", "back (CAM000)", "left", "right"]
    image_paths = [Path(record["pattern_path"]), overlay_path, *(Path(value) for value in record["view_paths"])]
    for index, ((x, y), name, path) in enumerate(zip(cells, names, image_paths)):
        w = cell_w if index < 2 else 535
        h = cell_h if index < 2 else 640
        draw.rounded_rectangle((x, y, x + w, y + h), 18, fill="#ffffff", outline="#d2cabf", width=3)
        shown = _contain(path, (w - 22, h - 65), background="#0b0c0f")
        canvas.paste(shown, (x + 11, y + 45))
        _text(draw, (x + w // 2, y + 10), name, 24, bold=True, anchor="ma")
    legend_y = 1750
    _text(draw, (55, legend_y), "edge type:", 23, bold=True)
    cursor = 190
    for kind, color in CURVE_COLORS.items():
        draw.line((cursor, legend_y + 15, cursor + 50, legend_y + 15), fill=color, width=7)
        _text(draw, (cursor + 62, legend_y), kind, 20, bold=True)
        cursor += 310
    _text(draw, (55, 1810), "Overlay notation: v# = source vertex index · e# = source edge index · each edge shows length (cm) and chord direction (°)", 22, fill="#5c574f")
    _text(draw, (55, 1850), f"Automatic validation: {label['validation']['status']} · packed overlap={not label['validation']['packed_non_overlap']} · four views={label['validation']['all_views_present']}", 22, bold=True, fill="#187a5b")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return {
        "sample_id": record["sample_id"],
        "category": record["category"],
        "board": destination.as_posix(),
        "overlay": overlay_path.as_posix(),
        "panel_count": len(label["panels"]),
        "edge_count": sum(counts.values()),
        "curve_type_counts": dict(counts),
        "validation": label["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a deterministic stratified ten-sample visual audit of exact GCDv2 pairs.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/review_seed_20260829"))
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    rng = random.Random(args.seed)
    selected = []
    for category, count in (("top", 4), ("skirt", 3), ("pants", 3)):
        selected.extend(rng.sample(sorted(by_category[category], key=lambda value: value["sample_id"]), count))
    rng.shuffle(selected)
    args.output.mkdir(parents=True, exist_ok=True)
    reviews = []
    for index, row in enumerate(selected, start=1):
        destination = args.output / f"{index:02d}_{row['category']}_{row['sample_id']}.png"
        reviews.append(_board(row, destination))
        print(json.dumps({"rendered": index, "total": len(selected), "sample_id": row["sample_id"]}), flush=True)

    thumb_w, thumb_h = 1060, 860
    sheet = Image.new("RGB", (2200, 4490), "#efeae2")
    draw = ImageDraw.Draw(sheet)
    _text(draw, (1100, 35), f"GCDv2 exact-pair review: 10 random samples · seed {args.seed}", 43, bold=True, anchor="ma")
    _text(draw, (1100, 92), "Each thumbnail: clean pattern · exact overlay · semantic front/back/left/right", 25, fill="#5c574f", anchor="ma")
    for index, row in enumerate(reviews):
        board = _contain(Path(row["board"]), (thumb_w, thumb_h), background="#f5f1eb")
        x = 30 + (index % 2) * 1080
        y = 140 + (index // 2) * 860
        sheet.paste(board, (x, y))
    contact = args.output / "contact_sheet_10.png"
    sheet.save(contact, optimize=True)
    manifest = {
        "schema_version": "gcdv2-exact-review-1.0",
        "seed": args.seed,
        "selection": "stratified random: top=4, skirt=3, pants=3",
        "contact_sheet": contact.as_posix(),
        "reviews": reviews,
        "all_validation_pass": all(row["validation"]["status"] == "PASS" for row in reviews),
    }
    (args.output / "review_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
