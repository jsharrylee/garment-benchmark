from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIEW_NAMES = ("front", "back", "left", "right")
DEFAULT_QUOTAS = {"top": 4, "skirt": 3, "pants": 3}
BOARD_SIZE = (2400, 1630)


def _font(size: int, *, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> list[dict[str, Any]]:
    """Select a deterministic category-stratified sample independent of scores."""

    generator = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category, count in quotas.items():
        candidates = sorted(
            (dict(row) for row in predictions if row["target_category"] == category),
            key=lambda row: row["sample_id"],
        )
        if len(candidates) < count:
            raise ValueError(f"need {count} {category} predictions, found {len(candidates)}")
        selected.extend(generator.sample(candidates, count))
    # Do not group the final boards by category.
    generator.shuffle(selected)
    return selected


def recompute_retrieval_contract(
    prediction: Mapping[str, Any],
    *,
    embedding_lookup: Mapping[str, int],
    image_embeddings: np.ndarray,
    pattern_embeddings: np.ndarray,
    split_lookup: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute raw global ranks without category/topology candidate filtering."""

    sample_id = str(prediction["sample_id"])
    query_index = int(embedding_lookup[sample_id])
    test_ids = sorted(sample for sample, split in split_lookup.items() if split == "test")
    train_ids = sorted(sample for sample, split in split_lookup.items() if split == "train")
    test_indices = np.asarray([embedding_lookup[sample] for sample in test_ids], dtype=np.int64)
    train_indices = np.asarray([embedding_lookup[sample] for sample in train_ids], dtype=np.int64)
    query = image_embeddings[query_index]

    paired_similarity = query @ pattern_embeddings[test_indices].T
    paired_order = np.argsort(-paired_similarity, kind="stable")
    target_position = test_ids.index(sample_id)
    paired_rank = int(np.flatnonzero(paired_order == target_position)[0]) + 1
    paired_target_similarity = float(paired_similarity[target_position])

    bank_similarity = query @ pattern_embeddings[train_indices].T
    bank_order = np.argsort(-bank_similarity, kind="stable")
    raw_winner = train_ids[int(bank_order[0])]
    raw_winner_similarity = float(bank_similarity[int(bank_order[0])])
    return {
        "paired_gallery_size": len(test_ids),
        "paired_target_rank_recomputed": paired_rank,
        "paired_target_similarity_recomputed": paired_target_similarity,
        "paired_rank_matches_saved": paired_rank
        == int(prediction["paired_gallery_target_rank"]),
        "train_bank_size": len(train_ids),
        "raw_global_top1_recomputed": raw_winner,
        "raw_global_top1_similarity_recomputed": raw_winner_similarity,
        "raw_global_top1_matches_saved": raw_winner
        == str(prediction["retrieved_sample_id"]),
        "raw_global_top1_similarity_matches_saved": abs(
            raw_winner_similarity - float(prediction["similarity"])
        )
        < 1e-5,
        "category_filter": False,
        "topology_filter": False,
        "reranking": False,
    }


def _contain(path: Path, size: tuple[int, int], background: str = "#0b0c10") -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    image_path: Path | None = None,
    title_color: str = "#152238",
    border: str = "#c8d1dc",
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline=border, width=4)
    draw.text(
        ((box[0] + box[2]) // 2, box[1] + 22),
        title,
        anchor="ma",
        font=_font(25, bold=True),
        fill=title_color,
    )
    if image_path is not None:
        width = box[2] - box[0] - 36
        height = box[3] - box[1] - 78
        image = _contain(image_path, (width, height))
        canvas.paste(image, (box[0] + 18, box[1] + 62))


def _status(value: bool) -> tuple[str, str]:
    return ("YES", "#177245") if value else ("NO", "#c7352c")


def render_board(
    destination: Path,
    *,
    board_index: int,
    board_count: int,
    prediction: Mapping[str, Any],
    verification: Mapping[str, Any],
    target_record: Mapping[str, Any],
    retrieved_record: Mapping[str, Any],
) -> Path:
    canvas = Image.new("RGB", BOARD_SIZE, "#f2f5f8")
    draw = ImageDraw.Draw(canvas)
    sample_id = str(prediction["sample_id"])
    target_category = str(prediction["target_category"])
    retrieved_id = str(prediction["retrieved_sample_id"])
    retrieved_category = str(prediction["retrieved_category"])
    draw.text(
        (56, 38),
        f"PURE CROSS-MODAL RETRIEVAL REVIEW  {board_index:02d}/{board_count:02d}",
        font=_font(46, bold=True),
        fill="#10233d",
    )
    draw.text(
        (56, 102),
        f"TEST query  {sample_id}  |  ground-truth category: {target_category}",
        font=_font(31, bold=True),
        fill="#31465e",
    )
    draw.text(
        (56, 153),
        (
            f"Paired exact target: rank #{verification['paired_target_rank_recomputed']} / "
            f"{verification['paired_gallery_size']}  |  cosine {verification['paired_target_similarity_recomputed']:.4f}"
        ),
        font=_font(25),
        fill="#425a70",
    )
    draw.text(
        (1190, 153),
        (
            f"Actual train-bank top-1: {retrieved_id} ({retrieved_category})  |  "
            f"cosine {verification['raw_global_top1_similarity_recomputed']:.4f}"
        ),
        font=_font(25, bold=True),
        fill="#425a70",
    )

    left, right, gap = 48, 2352, 18
    tile_width = (right - left - 3 * gap) // 4
    y0, y1 = 222, 650
    for view_index, (name, raw_path) in enumerate(zip(VIEW_NAMES, target_record["view_paths"])):
        x0 = left + view_index * (tile_width + gap)
        _panel(
            canvas,
            (x0, y0, x0 + tile_width, y1),
            title=f"4-view input · {name}",
            image_path=Path(raw_path),
        )

    pattern_y0, pattern_y1 = 690, 1560
    target_box = (48, pattern_y0, 795, pattern_y1)
    retrieved_box = (822, pattern_y0, 1569, pattern_y1)
    metrics_box = (1596, pattern_y0, 2352, pattern_y1)
    _panel(
        canvas,
        target_box,
        title=f"PAIRED EXACT TARGET · {sample_id} · {target_category}",
        image_path=Path(target_record["pattern_path"]),
        border="#1b8b62",
    )
    _panel(
        canvas,
        retrieved_box,
        title=f"RAW TRAIN-BANK TOP-1 · {retrieved_id} · {retrieved_category}",
        image_path=Path(retrieved_record["pattern_path"]),
        border="#a06cc0",
    )
    draw.rounded_rectangle(metrics_box, radius=18, fill="#ffffff", outline="#c8d1dc", width=4)
    draw.text(
        (metrics_box[0] + 36, metrics_box[1] + 38),
        "Retrieval verification",
        font=_font(34, bold=True),
        fill="#142943",
    )
    category_text, category_color = _status(bool(prediction["category_match"]))
    topology_text, topology_color = _status(bool(prediction["topology_compatible"]))
    global_text, global_color = _status(bool(verification["raw_global_top1_matches_saved"]))
    rows = [
        ("Train-bank rank", "#1", "#142943"),
        ("Category match", category_text, category_color),
        ("Exact topology match", topology_text, topology_color),
        (
            "Normalized geometry distance",
            f"{float(prediction['normalized_geometry_distance']):.4f}",
            "#142943",
        ),
        ("Saved top-1 = recomputed", global_text, global_color),
        ("Candidate bank", f"all {verification['train_bank_size']} train patterns", "#142943"),
        ("Category filter", "OFF", "#c7352c"),
        ("Topology filter", "OFF", "#c7352c"),
        ("Reranking", "OFF", "#c7352c"),
    ]
    y = metrics_box[1] + 115
    for label, value, color in rows:
        draw.text((metrics_box[0] + 38, y), label, font=_font(24), fill="#526579")
        draw.text(
            (metrics_box[2] - 38, y),
            value,
            anchor="ra",
            font=_font(25, bold=True),
            fill=color,
        )
        draw.line((metrics_box[0] + 38, y + 39, metrics_box[2] - 38, y + 39), fill="#e4e9ee", width=2)
        y += 68
    draw.multiline_text(
        (metrics_box[0] + 38, y + 15),
        (
            "Green frame = paired ground truth\n"
            "Purple frame = model-selected anchor\n\n"
            "Selection shown here is the unmodified\n"
            "global argmax from the train bank."
        ),
        font=_font(23),
        fill="#445a70",
        spacing=10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return destination


def make_contact_sheet(board_paths: Sequence[Path], destination: Path) -> Path:
    thumb_size = (1200, 815)
    columns = 2
    rows = (len(board_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb_size[0] * columns, thumb_size[1] * rows), "#dfe5eb")
    for index, path in enumerate(board_paths):
        with Image.open(path) as source:
            board = source.convert("RGB").resize(thumb_size, Image.Resampling.LANCZOS)
        sheet.paste(board, ((index % columns) * thumb_size[0], (index // columns) * thumb_size[1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 10-sample visual audit of pure Stage-2 train-bank retrieval."
    )
    parser.add_argument(
        "--exact-index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/test_predictions.jsonl"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/embeddings.npz"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/split_assignments.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcdv2_exact_crossmodal_retrieval/review_seed_20260829"),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    index_rows = _read_jsonl(args.exact_index)
    index_lookup = {str(row["sample_id"]): row for row in index_rows}
    predictions = _read_jsonl(args.predictions)
    split_lookup = json.loads(args.splits.read_text(encoding="utf-8"))
    embeddings = np.load(args.embeddings, allow_pickle=False)
    sample_ids = [str(value) for value in embeddings["sample_ids"]]
    embedding_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    image_embeddings = embeddings["image_embeddings"].astype(np.float32)
    pattern_embeddings = embeddings["pattern_embeddings"].astype(np.float32)
    selected = select_stratified_predictions(predictions, seed=args.seed)
    board_paths = []
    summary_rows = []
    failures = []
    for board_index, prediction in enumerate(selected, start=1):
        sample_id = str(prediction["sample_id"])
        retrieved_id = str(prediction["retrieved_sample_id"])
        if split_lookup.get(sample_id) != "test":
            failures.append(f"query_not_test:{sample_id}")
        if split_lookup.get(retrieved_id) != "train":
            failures.append(f"retrieved_not_train:{retrieved_id}")
        if prediction["top_train_bank"][0]["sample_id"] != retrieved_id:
            failures.append(f"saved_top1_inconsistent:{sample_id}")
        verification = recompute_retrieval_contract(
            prediction,
            embedding_lookup=embedding_lookup,
            image_embeddings=image_embeddings,
            pattern_embeddings=pattern_embeddings,
            split_lookup=split_lookup,
        )
        if not verification["paired_rank_matches_saved"]:
            failures.append(f"paired_rank_recompute_mismatch:{sample_id}")
        if not verification["raw_global_top1_matches_saved"]:
            failures.append(f"raw_top1_recompute_mismatch:{sample_id}")
        if not verification["raw_global_top1_similarity_matches_saved"]:
            failures.append(f"raw_top1_similarity_mismatch:{sample_id}")
        target_record = index_lookup[sample_id]
        retrieved_record = index_lookup[retrieved_id]
        required_paths = [
            *(Path(value) for value in target_record["view_paths"]),
            Path(target_record["pattern_path"]),
            Path(retrieved_record["pattern_path"]),
        ]
        missing = [str(path) for path in required_paths if not path.is_file()]
        failures.extend(f"missing_asset:{value}" for value in missing)
        board_path = args.output / f"{board_index:02d}_{prediction['target_category']}_{sample_id}.png"
        render_board(
            board_path,
            board_index=board_index,
            board_count=len(selected),
            prediction=prediction,
            verification=verification,
            target_record=target_record,
            retrieved_record=retrieved_record,
        )
        board_paths.append(board_path)
        summary_rows.append(
            {
                "board_index": board_index,
                "board_path": str(board_path.as_posix()),
                "sample_id": sample_id,
                "target_category": prediction["target_category"],
                "retrieved_sample_id": retrieved_id,
                "retrieved_category": prediction["retrieved_category"],
                "category_match": bool(prediction["category_match"]),
                "topology_compatible": bool(prediction["topology_compatible"]),
                "normalized_geometry_distance": float(prediction["normalized_geometry_distance"]),
                **verification,
            }
        )
    contact_sheet = make_contact_sheet(board_paths, args.output / "contact_sheet_10.png")
    if len(board_paths) != 10:
        failures.append(f"board_count:{len(board_paths)}")
    category_counts = Counter(row["target_category"] for row in summary_rows)
    if category_counts != Counter(DEFAULT_QUOTAS):
        failures.append(f"category_mix:{dict(category_counts)}")
    for path in [*board_paths, contact_sheet]:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing_render:{path}")
    summary = {
        "schema_version": "gcdv2-exact-crossmodal-retrieval-visual-review-1.0",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "seed": args.seed,
        "selection_policy": (
            "deterministic stratified random sample independent of retrieval score/result: "
            "top=4, skirt=3, pants=3"
        ),
        "pure_retrieval_contract": {
            "candidate_bank": "every sample assigned to train",
            "ground_truth_category_filter": False,
            "ground_truth_topology_filter": False,
            "reranking": False,
            "selection": "global cosine-similarity argmax",
            "saved_top1_recomputed_from_embeddings": all(
                row["raw_global_top1_matches_saved"] for row in summary_rows
            ),
        },
        "counts": {
            "boards": len(board_paths),
            "categories": dict(category_counts),
            "category_matches": sum(row["category_match"] for row in summary_rows),
            "topology_matches": sum(row["topology_compatible"] for row in summary_rows),
        },
        "inputs": {
            "exact_index": str(args.exact_index.as_posix()),
            "exact_index_sha256": _sha256(args.exact_index),
            "predictions": str(args.predictions.as_posix()),
            "predictions_sha256": _sha256(args.predictions),
            "embeddings": str(args.embeddings.as_posix()),
            "embeddings_sha256": _sha256(args.embeddings),
            "splits": str(args.splits.as_posix()),
            "splits_sha256": _sha256(args.splits),
        },
        "contact_sheet": str(contact_sheet.as_posix()),
        "rows": summary_rows,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "failures": failures,
                "counts": summary["counts"],
                "contact_sheet": str(contact_sheet),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

