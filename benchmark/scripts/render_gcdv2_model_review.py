from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.gcdv2_exact.geometry import load_exact_label, sample_curve
from benchmark.gcdv2_exact.residual_learning import topology_hash


COLORS = {"target": "#45d483", "anchor": "#ffad42", "edited": "#f052c8"}


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform(
    point: Sequence[float], transform: Mapping[str, Any], size: int
) -> tuple[float, float]:
    original_size = float(transform["canvas_size_px"][0])
    x = (
        (float(point[0]) + float(transform["translation_cm"][0]))
        * float(transform["scale_px_per_cm"])
        + float(transform["canvas_offset_px"][0])
    )
    y = float(transform["canvas_size_px"][1]) - (
        (float(point[1]) + float(transform["translation_cm"][1]))
        * float(transform["scale_px_per_cm"])
        + float(transform["canvas_offset_px"][1])
    )
    scale = size / original_size
    return x * scale, y * scale


def _ordered_panels(label: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(label["panels"], key=lambda value: int(value["source_order_index"]))


def _ordered_edges(panel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(panel["edges"], key=lambda value: int(value["edge_index"]))


def _panel_lookup(label: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(panel["panel_id"]): panel for panel in label["panels"]}


def _render_geometry(
    geometry: Mapping[str, Any],
    *,
    packing_source: Mapping[str, Any],
    color_source: Mapping[str, Any],
    size: int = 1024,
) -> Image.Image:
    """Render arbitrary local geometry in one explicit evaluation packing."""

    image = Image.new("RGB", (size, size), "#0b0c0f")
    draw = ImageDraw.Draw(image)
    colors = {
        str(panel["panel_id"]): tuple(panel.get("render_color_rgb", (92, 97, 170)))
        for panel in color_source["panels"]
    }
    for panel in _ordered_panels(geometry):
        panel_id = str(panel["panel_id"])
        transform = packing_source["packing"][panel_id]
        transformed = []
        for edge in _ordered_edges(panel):
            points = sample_curve(edge["start_cm"], edge["end_cm"], edge["curve"], samples=65)
            transformed.append([_transform(point, transform, size) for point in points])
        boundary = [point for edge in transformed for point in edge[:-1]]
        if boundary:
            draw.polygon(boundary, fill=colors[panel_id], outline="#050507")
        for points in transformed:
            draw.line(points, fill="#09090b", width=3, joint="curve")
    return image


def _draw_geometry_overlay(
    target: Mapping[str, Any],
    anchor: Mapping[str, Any],
    prediction: Mapping[str, Any],
    size: int = 1024,
) -> Image.Image:
    image = Image.new("RGB", (size, size), "#0b0c0f")
    draw = ImageDraw.Draw(image)
    anchor_panels = _panel_lookup(anchor)
    prediction_panels = _panel_lookup(prediction)
    for target_panel in _ordered_panels(target):
        panel_id = str(target_panel["panel_id"])
        transform = target["packing"][panel_id]
        layers = (
            ("target", target_panel, 6),
            ("anchor", anchor_panels[panel_id], 4),
            ("edited", prediction_panels[panel_id], 3),
        )
        for name, panel, width in layers:
            for edge in _ordered_edges(panel):
                points = sample_curve(edge["start_cm"], edge["end_cm"], edge["curve"], samples=65)
                draw.line(
                    [_transform(point, transform, size) for point in points],
                    fill=COLORS[name],
                    width=width,
                    joint="curve",
                )
        for vertex in prediction_panels[panel_id]["vertices_cm"]:
            x, y = _transform(vertex, transform, size)
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=COLORS["edited"])
    draw.rectangle((12, 12, 548, 70), fill="#15171d", outline="#545967")
    draw.text(
        (24, 24),
        "GT green | retrieved orange | edited magenta",
        font=_font(18, bold=True),
        fill="white",
    )
    return image


def _thumbnail(path: Path, box: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "#101117")
    canvas.paste(image, ((box[0] - image.width) // 2, (box[1] - image.height) // 2))
    return canvas


def _fit_image(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    value = image.copy()
    value.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "#101117")
    canvas.paste(value, ((box[0] - value.width) // 2, (box[1] - value.height) // 2))
    return canvas


def _flatten_vertices(label: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [point for panel in _ordered_panels(label) for point in panel["vertices_cm"]],
        dtype=float,
    )


def _curve_values(edge: Mapping[str, Any]) -> np.ndarray:
    curve = edge["curve"]
    kind = str(curve["type"])
    if kind == "line":
        return np.empty(0, dtype=float)
    if kind == "quadratic_bezier":
        return np.asarray(curve["controls_cm"][0], dtype=float)
    if kind == "cubic_bezier":
        return np.asarray(curve["controls_cm"][:2], dtype=float).reshape(-1)
    if kind == "circular_arc":
        return np.asarray([curve["arc"]["radius_cm"]], dtype=float)
    raise ValueError(f"unsupported curve type: {kind}")


def _angle_error(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _sample_metrics(
    target: Mapping[str, Any], anchor: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    target_vertices = _flatten_vertices(target)
    anchor_vertices = _flatten_vertices(anchor)
    edited_vertices = _flatten_vertices(prediction)
    if target_vertices.shape != anchor_vertices.shape or target_vertices.shape != edited_vertices.shape:
        raise ValueError("vertex slots are not aligned despite exact-topology gate")
    vertex_before = math.sqrt(float(np.square(anchor_vertices - target_vertices).sum(axis=1).mean()))
    vertex_after = math.sqrt(float(np.square(edited_vertices - target_vertices).sum(axis=1).mean()))
    anchor_panels = _panel_lookup(anchor)
    edited_panels = _panel_lookup(prediction)
    curve_before_sq: list[float] = []
    curve_after_sq: list[float] = []
    length_before: list[float] = []
    length_after: list[float] = []
    direction_before: list[float] = []
    direction_after: list[float] = []
    for target_panel in _ordered_panels(target):
        panel_id = str(target_panel["panel_id"])
        target_edges = _ordered_edges(target_panel)
        anchor_edges = _ordered_edges(anchor_panels[panel_id])
        edited_edges = _ordered_edges(edited_panels[panel_id])
        for target_edge, anchor_edge, edited_edge in zip(target_edges, anchor_edges, edited_edges):
            types = {
                str(target_edge["curve"]["type"]),
                str(anchor_edge["curve"]["type"]),
                str(edited_edge["curve"]["type"]),
            }
            if len(types) != 1:
                raise ValueError("curve type changed under a frozen-topology prediction")
            target_values = _curve_values(target_edge)
            anchor_values = _curve_values(anchor_edge)
            edited_values = _curve_values(edited_edge)
            curve_before_sq.extend(np.square(anchor_values - target_values).tolist())
            curve_after_sq.extend(np.square(edited_values - target_values).tolist())
            length_before.append(abs(float(anchor_edge["length_cm"]) - float(target_edge["length_cm"])))
            length_after.append(abs(float(edited_edge["length_cm"]) - float(target_edge["length_cm"])))
            direction_before.append(
                _angle_error(anchor_edge["chord_direction_deg"], target_edge["chord_direction_deg"])
            )
            direction_after.append(
                _angle_error(edited_edge["chord_direction_deg"], target_edge["chord_direction_deg"])
            )
    curve_before = math.sqrt(float(np.mean(curve_before_sq))) if curve_before_sq else None
    curve_after = math.sqrt(float(np.mean(curve_after_sq))) if curve_after_sq else None
    return {
        "vertex_rmse_cm": {"before": vertex_before, "after": vertex_after, "delta": vertex_after - vertex_before},
        "curve_parameter_rmse_cm": {
            "before": curve_before,
            "after": curve_after,
            "delta": curve_after - curve_before if curve_before is not None and curve_after is not None else None,
            "support": len(curve_before_sq),
        },
        "edge_length_mae_cm": {
            "before": float(np.mean(length_before)),
            "after": float(np.mean(length_after)),
            "delta": float(np.mean(length_after)) - float(np.mean(length_before)),
        },
        "edge_direction_mae_deg": {
            "before": float(np.mean(direction_before)),
            "after": float(np.mean(direction_after)),
            "delta": float(np.mean(direction_after)) - float(np.mean(direction_before)),
        },
    }


def _prediction_endpoint_consistency(prediction: Mapping[str, Any], tolerance: float = 1e-5) -> bool:
    for panel in prediction["panels"]:
        vertices = panel["vertices_cm"]
        for edge in panel["edges"]:
            start, end = (int(value) for value in edge["endpoints"])
            if math.dist(edge["start_cm"], vertices[start]) > tolerance:
                return False
            if math.dist(edge["end_cm"], vertices[end]) > tolerance:
                return False
    return True


def _metric_line(name: str, unit: str, values: Mapping[str, Any]) -> str:
    before, after = values["before"], values["after"]
    if before is None or after is None:
        return f"{name:<24} no curved-edge support"
    arrow = "improved" if after < before else "worse" if after > before else "unchanged"
    return f"{name:<24} {before:7.3f} -> {after:7.3f} {unit}  ({after-before:+7.3f}, {arrow})"


def _board(
    row: Mapping[str, Any], destination: Path
) -> tuple[Path, dict[str, Any]]:
    target = load_exact_label(Path(row["target_label_path"]))
    anchor = load_exact_label(Path(row["anchor_label_path"]))
    target_hash, anchor_hash = topology_hash(target), topology_hash(anchor)
    if row["sample_id"] == row["anchor_id"]:
        raise ValueError("review row selected itself as anchor")
    if row.get("_retrieval_pair_verified") is not True:
        raise ValueError("prediction anchor was not verified against retrieval_pairs.jsonl")
    if target_hash != anchor_hash or target_hash != row["topology_hash"]:
        raise ValueError("review row does not satisfy exact-topology compatibility")
    if len(target["views"]) != 4 or not all(Path(view["path"]).is_file() for view in target["views"]):
        raise ValueError("review target lacks a complete four-view bundle")
    metrics = _sample_metrics(target, anchor, row)
    endpoint_consistency = _prediction_endpoint_consistency(row)
    if not endpoint_consistency:
        raise ValueError("edited prediction violates shared-vertex endpoint consistency")

    # A/B are the exact paired/retrieved source artifacts in their own complete
    # display packings.  C/D use the target frame; D supplies the scale-aligned
    # comparison that would be lost if B were rescaled or clipped into it.
    target_pattern = _thumbnail(Path(target["pattern_image"]), (690, 690))
    anchor_pattern = _thumbnail(Path(anchor["pattern_image"]), (690, 690))
    edited_pattern = _fit_image(
        _render_geometry(row, packing_source=target, color_source=target), (690, 690)
    )
    overlay = _fit_image(_draw_geometry_overlay(target, anchor, row), (690, 690))

    width, height = 3000, 1880
    board = Image.new("RGB", (width, height), "#f4f1eb")
    draw = ImageDraw.Draw(board)
    title = f"Stage 3 best-geometry held-out review · {row['category'].upper()} · {row['sample_id']}"
    draw.text((45, 25), title, font=_font(34, bold=True), fill="#15171c")
    draw.text(
        (45, 72),
        f"actual selected train anchor {row['anchor_id']} · topology {target_hash[:12]} · visual cosine {float(row['visual_cosine_similarity']):.4f}",
        font=_font(19),
        fill="#444852",
    )
    view_box = (650, 375)
    for index, view in enumerate(target["views"]):
        image = _thumbnail(Path(view["path"]), view_box)
        x, y = 45 + index * 735, 120
        board.paste(image, (x, y))
        draw.text(
            (x + 8, y + view_box[1] + 7),
            f"paired input {view['view_label']} · {Path(view['path']).name}",
            font=_font(17, bold=True),
            fill="#22252b",
        )
    panes = (
        ("A. Paired truth exact pattern", target_pattern),
        ("B. Selected anchor (exact source PNG)", anchor_pattern),
        ("C. Edited prediction", edited_pattern),
        ("D. Same-frame overlay", overlay),
    )
    pane_y = 590
    for index, (caption, image) in enumerate(panes):
        x = 45 + index * 735
        draw.text((x, pane_y), caption, font=_font(21, bold=True), fill="#1c1f25")
        board.paste(image, (x, pane_y + 38))

    metrics_y = 1345
    draw.rounded_rectangle((45, metrics_y, 2955, 1815), radius=20, fill="#ffffff", outline="#c8c3ba", width=2)
    draw.text((75, metrics_y + 24), "Per-sample errors against paired exact truth", font=_font(25, bold=True), fill="#191b20")
    lines = (
        _metric_line("shared vertex RMSE", "cm", metrics["vertex_rmse_cm"]),
        _metric_line("native curve-param RMSE", "cm", metrics["curve_parameter_rmse_cm"]),
        _metric_line("edge length MAE", "cm", metrics["edge_length_mae_cm"]),
        _metric_line("edge direction MAE", "deg", metrics["edge_direction_mae_deg"]),
    )
    y = metrics_y + 72
    for line in lines:
        color = "#9f2d38" if "worse" in line else "#17683b" if "improved" in line else "#333740"
        draw.text((75, y), line, font=_font(21, bold=True), fill=color)
        y += 43
    draw.text(
        (75, y + 8),
        "Evaluation condition: ORACLE EXACT-TOPOLOGY COMPATIBILITY GATE. The gate reads target topology metadata; it is not deployable from RGB alone.",
        font=_font(19, bold=True),
        fill="#8b3b13",
    )
    draw.text(
        (75, y + 47),
        "Retrieval ranking uses four-view FPN cosine only. Target geometry is not used for ranking. Curve types/stitches/incidence are frozen; endpoint consistency = PASS.",
        font=_font(18),
        fill="#41454d",
    )
    draw.text(
        (75, y + 84),
        "A/B are complete exact source PNGs in their own packings. C/D use the target frame; D is the scale-aligned comparison.",
        font=_font(18),
        fill="#41454d",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, optimize=True)
    return destination, {
        "sample_id": row["sample_id"],
        "anchor_id": row["anchor_id"],
        "category": row["category"],
        "split": row["split"],
        "topology_hash": target_hash,
        "visual_cosine_similarity": row["visual_cosine_similarity"],
        "metrics": metrics,
        "integrity": {
            "target_not_anchor": row["sample_id"] != row["anchor_id"],
            "exact_topology_equal": target_hash == anchor_hash == row["topology_hash"],
            "four_views_complete": True,
            "shared_vertex_endpoint_consistency": endpoint_consistency,
            "matches_retrieval_pair_record": True,
        },
        "board": str(destination.as_posix()),
    }


def _select(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Mapping[str, Any]]:
    """Deterministic category-stratified random audit, without quality cherry-picking."""

    rng = random.Random(seed)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    for values in by_category.values():
        rng.shuffle(values)
    selected = []
    categories = [name for name in ("top", "skirt", "pants") if by_category[name]]
    while len(selected) < count and categories:
        next_categories = []
        for category in categories:
            if by_category[category] and len(selected) < count:
                selected.append(by_category[category].pop())
            if by_category[category]:
                next_categories.append(category)
        categories = next_categories
    return selected


def _aggregate_review(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in (
        "vertex_rmse_cm",
        "curve_parameter_rmse_cm",
        "edge_length_mae_cm",
        "edge_direction_mae_deg",
    ):
        before = [row["metrics"][name]["before"] for row in rows if row["metrics"][name]["before"] is not None]
        after = [row["metrics"][name]["after"] for row in rows if row["metrics"][name]["after"] is not None]
        result[name] = {
            "sample_macro_before": float(np.mean(before)) if before else None,
            "sample_macro_after": float(np.mean(after)) if after else None,
            "improved_sample_count": sum(
                row["metrics"][name]["after"] < row["metrics"][name]["before"]
                for row in rows
                if row["metrics"][name]["before"] is not None
            ),
            "support_samples": len(before),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render held-out four-view/truth/retrieved/corrected exact-geometry review boards."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/gcdv2_exact_models/retrieved_residual/heldout_predictions.jsonl"),
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path("artifacts/gcdv2_exact_models/retrieved_residual/training_metrics.json"),
    )
    parser.add_argument(
        "--retrieval-pairs",
        type=Path,
        default=Path("artifacts/gcdv2_exact_models/retrieved_residual/retrieval_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_exact_models/retrieved_residual/review_best_geometry_seed_20260829"),
    )
    parser.add_argument("--split", choices=("validation", "test", "all"), default="test")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]
    selected = _select(rows, args.count, args.seed)
    if len(selected) != min(args.count, len(rows)):
        raise ValueError("stratified selector did not return the requested number of rows")
    pair_rows = [
        json.loads(line)
        for line in args.retrieval_pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_by_target = {str(row["target_id"]): row for row in pair_rows}
    verified_selected = []
    for prediction in selected:
        pair = pair_by_target.get(str(prediction["sample_id"]))
        if pair is None:
            raise ValueError(f"no retrieval-pair record for {prediction['sample_id']}")
        if pair["anchor_id"] != prediction["anchor_id"] or pair["topology_hash"] != prediction["topology_hash"]:
            raise ValueError(f"prediction/retrieval anchor mismatch for {prediction['sample_id']}")
        if pair.get("anchor_is_training_bank") is not True or pair.get("target_self_excluded") is not True:
            raise ValueError(f"invalid retrieval provenance for {prediction['sample_id']}")
        current = dict(prediction)
        current["_retrieval_pair_verified"] = True
        verified_selected.append(current)
    selected = verified_selected
    reviews = []
    for index, row in enumerate(selected, 1):
        path = args.output / f"{index:02d}_{row['category']}_{row['sample_id']}.png"
        _, review = _board(row, path)
        reviews.append(review)

    # Contact sheet is an index; full boards retain readable measurements.
    thumbs = [_thumbnail(Path(review["board"]), (930, 590)) for review in reviews]
    columns = 2
    rows_count = math.ceil(len(thumbs) / columns)
    contact = Image.new("RGB", (columns * 950, rows_count * 610), "#e7e3dc")
    for index, thumb in enumerate(thumbs):
        contact.paste(thumb, ((index % columns) * 950 + 10, (index // columns) * 610 + 10))
    contact_path = args.output / "contact_sheet.png"
    contact.save(contact_path, optimize=True)

    training = (
        json.loads(args.training_metrics.read_text(encoding="utf-8"))
        if args.training_metrics.is_file()
        else None
    )
    integrity_fields = (
        "target_not_anchor",
        "exact_topology_equal",
        "four_views_complete",
        "shared_vertex_endpoint_consistency",
        "matches_retrieval_pair_record",
    )
    summary = {
        "schema_version": "gcdv2-retrieved-residual-best-geometry-review-2.0",
        "review_scope": "Stage 3 default/best-geometry held-out predictions",
        "selection": {
            "method": "category-stratified deterministic random selection; no metric cherry-picking",
            "seed": args.seed,
            "split": args.split,
            "count": len(reviews),
            "category_counts": dict(Counter(review["category"] for review in reviews)),
        },
        "evaluation_condition": {
            "oracle_exact_topology_compatibility_gate": True,
            "target_geometry_used_for_retrieval_ranking": False,
            "ranking_signal": "four-view mean frozen FPN cosine within oracle-compatible topology",
            "deployment_warning": "RGB-only deployment requires a preceding topology prediction/compatibility model",
        },
        "source": {
            "predictions": str(args.predictions.as_posix()),
            "predictions_sha256": _sha256(args.predictions),
            "training_metrics": str(args.training_metrics.as_posix()),
            "retrieval_pairs": str(args.retrieval_pairs.as_posix()),
            "retrieval_pairs_sha256": _sha256(args.retrieval_pairs),
            "model_best_epoch": training.get("best_epoch") if training else None,
            "full_test_metrics": training.get("test") if training else None,
        },
        "selected_review_macro": _aggregate_review(reviews),
        "integrity": {
            name: all(review["integrity"][name] for review in reviews)
            for name in integrity_fields
        },
        "reviews": reviews,
        "contact_sheet": str(contact_path.as_posix()),
        "legend": COLORS,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path.as_posix()),
                "contact_sheet": str(contact_path.as_posix()),
                "category_counts": summary["selection"]["category_counts"],
                "integrity": summary["integrity"],
                "selected_review_macro": summary["selected_review_macro"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
