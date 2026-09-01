from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, "#090a0d")
    result.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return result


def _sdf_image(sdf: np.ndarray, size: tuple[int, int]) -> Image.Image:
    limit = max(float(np.percentile(np.abs(sdf), 95)), 1e-6)
    normalized = np.clip(sdf / limit, -1, 1)
    rgb = np.zeros((*sdf.shape, 3), np.uint8)
    rgb[..., 0] = np.where(normalized > 0, normalized * 255, 0).astype(np.uint8)
    rgb[..., 2] = np.where(normalized < 0, -normalized * 255, 0).astype(np.uint8)
    rgb[..., 1] = (255 * (1 - np.abs(normalized)) * 0.25).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)


def _render(row: dict, destination: Path) -> Path:
    graph = json.loads(Path(row["formal_graph_path"]).read_text(encoding="utf-8"))
    with np.load(row["visual_truth_path"]) as visual:
        contour = visual["dense_contour_uv_f32"].copy()
        sdf = visual["sdf_cm_f16"].astype(np.float32)
    board = Image.new("RGB", (1900, 960), "#f3efe7")
    draw = ImageDraw.Draw(board)
    draw.text((45, 28), "GCDv2 neuro-symbolic panel training pair", font=_font(39, bold=True), fill="#17181b")
    draw.text((45, 78), f"{row['split']} · {row['garment_category']} · {row['source_panel_id']} · {row['panel_uid']}", font=_font(23, bold=True), fill="#3d3f44")
    slots = ((45, 165, "INPUT: panel.png + cm_per_pixel"), (655, 165, "VISUAL TRUTH: contour + visible/latent junction"), (1265, 165, "VISUAL TRUTH: signed distance (cm)"))
    input_image = _fit(Path(row["input_panel_image"]), (540, 540))
    overlay = input_image.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.line([(float(u)*540, float(v)*540) for u, v in contour] + [(float(contour[0,0])*540, float(contour[0,1])*540)], fill="#20d7ff", width=4, joint="curve")
    visible = latent = 0
    for point in graph["points"]:
        u, v = point["uv"]
        x, y = u*540, v*540
        if point["visual_supervision_eligible"]:
            color, radius = "#00e38e", 7
            visible += 1
        else:
            color, radius = "#ff42c6", 5
            latent += 1
        overlay_draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="#101216")
        overlay_draw.text((x+6, y-10), point["point_id"], font=_font(13, bold=True), fill=color)
    images = (input_image, overlay, _sdf_image(sdf, (540, 540)))
    for (x, y, label), image in zip(slots, images):
        draw.rounded_rectangle((x-10, y-48, x+550, y+550), 12, fill="white", outline="#c7c1b7", width=2)
        draw.text((x, y-40), label, font=_font(20, bold=True), fill="#202126")
        board.paste(image, (x, y))
    draw.text((45, 740), f"scale={row['input_scale_cm_per_pixel']:.6f} cm/px · points={len(graph['points'])} · visible={visible} · latent smooth subdivisions={latent}", font=_font(22, bold=True), fill="#135f50")
    primitive_counts = {}
    for curve in graph["curves"]:
        primitive_counts[curve["primitive"]] = primitive_counts.get(curve["primitive"], 0) + 1
    draw.text((45, 782), f"FORMAL OUTPUT · primitives={primitive_counts}", font=_font(21, bold=True), fill="#35373c")
    operations = graph["operation_tokens"][:6]
    token_text = "   ".join(f"{value['op']}({','.join(map(str,value['args']))})" for value in operations)
    draw.text((45, 822), token_text[:180], font=_font(17), fill="#44464b")
    draw.text((45, 860), "Relations: NEXT · SHARED_ENDPOINT · CLOSED_CYCLE · DEGREE=2   |   cyclic rotations represent the same shape", font=_font(19, bold=True), fill="#6f2e8a")
    draw.text((45, 902), "Green = visible corner · magenta = smooth source-graph split · cyan = dense contour", font=_font(20, bold=True), fill="#2c2e33")
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, optimize=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render visual/formal supervision review boards.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/panel_index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gcdv2_neurosymbolic_v1/review_seed_20260829"))
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(args.seed)
    selected = []
    quotas = {"top": 4, "skirt": 3, "pants": 3}
    for category, count in quotas.items():
        candidates = [row for row in rows if row["split"] == "test" and row["garment_category"] == category]
        rng.shuffle(candidates)
        selected.extend(candidates[:count])
    boards = [_render(row, args.output / f"{index:02d}_{row['garment_category']}_{row['source_panel_id']}.png") for index, row in enumerate(selected, 1)]
    thumb_w, thumb_h = 950, 480
    contact = Image.new("RGB", (thumb_w*2, thumb_h*5), "#d9d4cb")
    for index, path in enumerate(boards):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (thumb_w, thumb_h), "#d9d4cb")
        cell.paste(image, ((thumb_w-image.width)//2, (thumb_h-image.height)//2))
        contact.paste(cell, ((index%2)*thumb_w, (index//2)*thumb_h))
    contact_path = args.output / "contact_sheet_10.png"
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    contact.save(contact_path, optimize=True)
    result = {"status": "PASS", "contact_sheet": contact_path.as_posix(), "boards": [path.as_posix() for path in boards]}
    (args.output / "review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
