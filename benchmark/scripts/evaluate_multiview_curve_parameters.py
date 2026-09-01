from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.drafting_semantics.curve_evaluation import (
    curve_pair_metrics,
    evaluate_frozen_curve_predictions,
)
from benchmark.drafting_semantics.multiview_curve_parameters import (
    CONTROL_SLICE,
    CURVE_PARAMETER_NAMES,
    CURVE_QUERY_NAMES,
    CURVE_TRUTH_DENSE_APPROXIMATION,
    sample_two_cubic_formula,
)
from benchmark.drafting_semantics.multiview_pattern_semantics import VIEW_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_contract(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("query_names", ())) != CURVE_QUERY_NAMES:
        raise ValueError("checkpoint curve query order does not match evaluator contract")
    if tuple(checkpoint.get("parameter_names", ())) != CURVE_PARAMETER_NAMES:
        raise ValueError("checkpoint curve parameter order does not match evaluator contract")
    standardizer = checkpoint.get("standardizer", {})
    means = np.asarray(standardizer.get("means"), dtype=np.float32)
    deviations = np.asarray(standardizer.get("standard_deviations"), dtype=np.float32)
    expected = (len(CURVE_QUERY_NAMES), len(CURVE_PARAMETER_NAMES))
    if means.shape != expected or deviations.shape != expected:
        raise ValueError("checkpoint standardizer does not match evaluator contract")
    return means, deviations, checkpoint


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    required = {
        "sample_ids",
        "predicted_curve_parameters",
        "target_curve_parameters",
        "target_role_mask",
        "predicted_presence_probability",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"prediction archive is missing keys: {sorted(missing)}")
        output = {key: np.asarray(archive[key]) for key in required}
    sample_ids = [str(value) for value in output["sample_ids"].tolist()]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prediction archive contains duplicate sample ids")
    if output["predicted_curve_parameters"].shape[0] != len(sample_ids):
        raise ValueError("prediction arrays and sample_ids have different lengths")
    return output


def _validate_ablation_alignment(
    reference: dict[str, np.ndarray], ablation: dict[str, np.ndarray]
) -> None:
    """Fail closed unless two frozen archives describe the exact same targets.

    Reordering by ID would make an accidental split mismatch harder to notice,
    so the evaluator requires byte-order-equivalent sample IDs as well as
    identical masks and numerically identical target vectors.
    """

    reference_ids = [str(value) for value in reference["sample_ids"].tolist()]
    ablation_ids = [str(value) for value in ablation["sample_ids"].tolist()]
    if reference_ids != ablation_ids:
        raise ValueError(
            "global-token ablation sample_ids are not in the exact frozen-test order"
        )
    if not np.array_equal(
        reference["target_role_mask"], ablation["target_role_mask"]
    ):
        raise ValueError("global-token ablation target role masks do not match")
    if not np.allclose(
        reference["target_curve_parameters"],
        ablation["target_curve_parameters"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("global-token ablation target curve parameters do not match")


def _train_presence_rates(training_metrics_path: Path | None) -> np.ndarray | None:
    if training_metrics_path is None:
        return None
    payload = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    train_count = int(payload["split_counts"]["train"])
    per_query = payload["train"]["presence"]["per_query"]
    return np.asarray(
        [float(per_query[name]["support"]) / max(train_count, 1) for name in CURVE_QUERY_NAMES],
        dtype=np.float32,
    )


def _view_lookup(index_path: Path) -> dict[str, tuple[str, ...]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    output = {}
    for row in payload["records"]:
        sample_id = str(row["sample_id"])
        views = tuple(str(value) for value in row["source_views"])
        if len(views) != len(VIEW_NAMES):
            raise ValueError(f"{sample_id} does not have exactly four views")
        output[sample_id] = views
    return output


def _sample_curve_error(
    predicted: np.ndarray, expected: np.ndarray, mask: np.ndarray
) -> float:
    values = [
        curve_pair_metrics(predicted[index], expected[index])["symmetric_chamfer_over_chord"]
        for index in np.flatnonzero(mask)
    ]
    return float(np.mean(values)) if values else float("inf")


def _select_representative_indices(
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    sample_ids: list[str],
    count: int,
) -> list[tuple[int, float, float]]:
    if count <= 0:
        raise ValueError("board row count must be positive")
    role_counts = role_mask.sum(axis=1)
    maximum_roles = int(role_counts.max())
    candidates = [
        (index, _sample_curve_error(predicted[index], expected[index], role_mask[index]))
        for index in range(len(sample_ids))
        if int(role_counts[index]) == maximum_roles
    ]
    candidates.sort(key=lambda item: (item[1], sample_ids[item[0]]))
    count = min(count, len(candidates))
    if count == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.2, 0.8, count).tolist()
    selected: list[tuple[int, float, float]] = []
    used: set[int] = set()
    for quantile in positions:
        raw = int(round(quantile * (len(candidates) - 1)))
        available = sorted(
            (index for index in range(len(candidates)) if index not in used),
            key=lambda index: (abs(index - raw), index),
        )
        chosen = available[0]
        used.add(chosen)
        sample_index, error = candidates[chosen]
        selected.append((sample_index, float(quantile), float(error)))
    return selected


def _control_polygons(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters)[CONTROL_SLICE]
    knot = values[0:2]
    return (
        np.stack(((0.0, 0.0), values[2:4], values[4:6], knot)),
        np.stack((knot, values[6:8], values[8:10], (1.0, 0.0))),
    )


def _render_board(
    output_path: Path,
    manifest_path: Path,
    *,
    sample_ids: list[str],
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    presence: np.ndarray,
    views_by_id: dict[str, tuple[str, ...]],
    row_count: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = _select_representative_indices(
        predicted, expected, role_mask, sample_ids, row_count
    )
    fig = plt.figure(figsize=(27, 3.55 * len(selected) + 0.8), constrained_layout=True)
    grid = fig.add_gridspec(
        len(selected),
        len(VIEW_NAMES) + len(CURVE_QUERY_NAMES),
        width_ratios=(1.05, 1.05, 1.05, 1.05, 1.18, 1.18, 1.18, 1.18, 1.18),
    )
    manifest_rows = []
    for board_row, (sample_index, quantile, sample_error) in enumerate(selected):
        sample_id = sample_ids[sample_index]
        view_paths = views_by_id.get(sample_id)
        if view_paths is None:
            raise KeyError(f"index has no four-view record for frozen-test sample {sample_id}")
        for column, (view_name, raw_path) in enumerate(zip(VIEW_NAMES, view_paths)):
            axis = fig.add_subplot(grid[board_row, column])
            path = Path(raw_path)
            if path.is_file():
                with Image.open(path) as image:
                    axis.imshow(image.convert("RGB"))
            else:
                axis.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center")
            axis.set_title(f"{view_name} RGB input", fontsize=8.5)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"{sample_id}\nerror q={quantile:.1f}\nmean Chamfer={sample_error:.3f}",
                    rotation=90,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )

        role_payload = {}
        for query_index, query_name in enumerate(CURVE_QUERY_NAMES):
            axis = fig.add_subplot(grid[board_row, len(VIEW_NAMES) + query_index])
            if not role_mask[sample_index, query_index]:
                axis.text(0.5, 0.5, "target absent", ha="center", va="center", fontsize=8)
                axis.set_title(
                    f"{query_name}\np(present)={presence[sample_index, query_index]:.2f}",
                    fontsize=7.7,
                )
                axis.axis("off")
                role_payload[query_name] = {
                    "target_present": False,
                    "predicted_presence_probability": float(presence[sample_index, query_index]),
                }
                continue
            truth = expected[sample_index, query_index]
            estimate = predicted[sample_index, query_index]
            truth_curve = sample_two_cubic_formula(truth[CONTROL_SLICE], 65)
            estimate_curve = sample_two_cubic_formula(estimate[CONTROL_SLICE], 65)
            truth_polygons = _control_polygons(truth)
            estimate_polygons = _control_polygons(estimate)
            for polygon in truth_polygons:
                axis.plot(polygon[:, 0], polygon[:, 1], color="#777777", alpha=0.45, lw=0.7, ls=":")
            for polygon in estimate_polygons:
                axis.plot(polygon[:, 0], polygon[:, 1], color="#d81b60", alpha=0.30, lw=0.7, ls=":")
            axis.plot(truth_curve[:, 0], truth_curve[:, 1], color="#111111", lw=2.0, label="GT")
            axis.plot(
                estimate_curve[:, 0],
                estimate_curve[:, 1],
                color="#d81b60",
                lw=1.8,
                ls="--",
                label="prediction",
            )
            axis.scatter((0.0, 1.0), (0.0, 0.0), s=11, color="#1976d2", zorder=4)
            values = curve_pair_metrics(estimate, truth)
            all_points = np.concatenate((truth_curve, estimate_curve), axis=0)
            minimum = all_points.min(axis=0)
            maximum = all_points.max(axis=0)
            center = 0.5 * (minimum + maximum)
            extent = max(float(np.max(maximum - minimum)), 1.0) * 0.62
            axis.set_xlim(center[0] - extent, center[0] + extent)
            axis.set_ylim(center[1] - extent, center[1] + extent)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(color="#dddddd", lw=0.45)
            axis.tick_params(labelsize=6)
            axis.set_title(
                f"{query_name}\nC={values['symmetric_chamfer_over_chord']:.3f} "
                f"H={values['hausdorff_over_chord']:.3f} "
                f"T={values['endpoint_tangent_angle_error_degrees']:.1f}°\n"
                f"p(present)={presence[sample_index, query_index]:.2f}",
                fontsize=7.2,
            )
            if board_row == 0 and query_index == 0:
                axis.legend(loc="best", fontsize=6.5, frameon=True)
            role_payload[query_name] = {
                "target_present": True,
                "predicted_presence_probability": float(presence[sample_index, query_index]),
                **values,
            }
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "selection_error_quantile": quantile,
                "mean_symmetric_chamfer_over_observed_roles": sample_error,
                "view_paths": list(view_paths),
                "target_provenance": CURVE_TRUTH_DENSE_APPROXIMATION,
                "roles": role_payload,
            }
        )

    fig.suptitle(
        "Frozen-test actual 4-view RGB → spatial FPN role-query Transformer → normalized two-cubic curves\n"
        "GT = black, prediction = magenta dashed, dotted = control polygons | "
        f"TARGET PROVENANCE: {CURVE_TRUTH_DENSE_APPROXIMATION} (not original generator controls)",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)
    manifest = {
        "schema_version": "multiview-spatial-curve-visual-board-1.0",
        "selection": (
            "deterministic 20%-to-80% Chamfer error quantiles among frozen-test samples "
            "having the maximum observed curve-role count"
        ),
        "target_provenance": CURVE_TRUTH_DENSE_APPROXIMATION,
        "warning": (
            "GT control points are a least-squares two-cubic approximation of the dense "
            "canonical curve; they are not original authoring controls."
        ),
        "output": str(output_path),
        "records": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _plot_mobile_curve(
    axis,
    *,
    query_name: str,
    truth: np.ndarray | None,
    spatial_prediction: np.ndarray | None,
    spatial_presence: float,
    global_prediction: np.ndarray | None,
    global_presence: float | None,
    show_legend: bool,
) -> dict[str, Any]:
    if truth is None or spatial_prediction is None:
        global_label = (
            f" / global={global_presence:.2f}" if global_presence is not None else ""
        )
        axis.text(0.5, 0.5, "target absent", ha="center", va="center", fontsize=12)
        axis.set_title(
            f"{query_name}\npresent probability: FPN={spatial_presence:.2f}{global_label}",
            fontsize=10,
        )
        axis.axis("off")
        return {
            "target_present": False,
            "spatial_presence_probability": float(spatial_presence),
            "global_presence_probability": (
                float(global_presence) if global_presence is not None else None
            ),
        }

    truth_curve = sample_two_cubic_formula(truth[CONTROL_SLICE], 65)
    spatial_curve = sample_two_cubic_formula(spatial_prediction[CONTROL_SLICE], 65)
    plotted = [truth_curve, spatial_curve]
    axis.plot(truth_curve[:, 0], truth_curve[:, 1], color="#111111", lw=3.0, label="GT target")
    axis.plot(
        spatial_curve[:, 0],
        spatial_curve[:, 1],
        color="#d81b60",
        lw=2.6,
        ls="--",
        label="spatial FPN",
    )
    spatial_metrics = curve_pair_metrics(spatial_prediction, truth)
    global_metrics = None
    if global_prediction is not None:
        global_curve = sample_two_cubic_formula(global_prediction[CONTROL_SLICE], 65)
        plotted.append(global_curve)
        axis.plot(
            global_curve[:, 0],
            global_curve[:, 1],
            color="#1565c0",
            lw=2.3,
            ls=(0, (5, 2, 1.3, 2)),
            label="global-token ablation",
        )
        global_metrics = curve_pair_metrics(global_prediction, truth)
    axis.scatter((0.0, 1.0), (0.0, 0.0), s=24, color="#666666", zorder=4)
    all_points = np.concatenate(plotted, axis=0)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = max(float(np.max(maximum - minimum)), 1.0) * 0.61
    axis.set_xlim(center[0] - extent, center[0] + extent)
    axis.set_ylim(center[1] - extent, center[1] + extent)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#d8d8d8", lw=0.65)
    axis.tick_params(labelsize=8)
    spatial_line = (
        f"spatial: C={spatial_metrics['symmetric_chamfer_over_chord']:.3f} "
        f"H={spatial_metrics['hausdorff_over_chord']:.3f} "
        f"T={spatial_metrics['endpoint_tangent_angle_error_degrees']:.1f}°"
    )
    if global_metrics is None:
        global_line = "global-token ablation unavailable"
    else:
        global_line = (
            f"global:  C={global_metrics['symmetric_chamfer_over_chord']:.3f} "
            f"H={global_metrics['hausdorff_over_chord']:.3f} "
            f"T={global_metrics['endpoint_tangent_angle_error_degrees']:.1f}°"
        )
    axis.set_title(
        f"{query_name}\n{spatial_line}\n{global_line}",
        fontsize=9.4,
    )
    if show_legend:
        axis.legend(loc="best", fontsize=8.5, frameon=True)
    return {
        "target_present": True,
        "spatial_presence_probability": float(spatial_presence),
        "global_presence_probability": (
            float(global_presence) if global_presence is not None else None
        ),
        "spatial_metrics": spatial_metrics,
        "global_metrics": global_metrics,
    }


def _render_mobile_boards(
    output_directory: Path,
    manifest_path: Path,
    *,
    sample_ids: list[str],
    predicted: np.ndarray,
    expected: np.ndarray,
    role_mask: np.ndarray,
    presence: np.ndarray,
    views_by_id: dict[str, tuple[str, ...]],
    row_count: int,
    global_arrays: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Render one readable portrait comparison image per selected sample."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = _select_representative_indices(
        predicted, expected, role_mask, sample_ids, row_count
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_index, quantile, spatial_sample_error in selected:
        sample_id = sample_ids[sample_index]
        view_paths = views_by_id.get(sample_id)
        if view_paths is None:
            raise KeyError(f"index has no four-view record for frozen-test sample {sample_id}")
        global_prediction = (
            global_arrays["predicted_curve_parameters"][sample_index]
            if global_arrays is not None
            else None
        )
        global_presence = (
            global_arrays["predicted_presence_probability"][sample_index]
            if global_arrays is not None
            else None
        )
        global_sample_error = (
            _sample_curve_error(
                global_prediction,
                expected[sample_index],
                role_mask[sample_index],
            )
            if global_prediction is not None
            else None
        )
        # 10 inches * 180 dpi = 1800 pixels wide: large enough to read on a
        # phone after the app scales the image to its viewport.
        fig = plt.figure(figsize=(10, 15.2), constrained_layout=True)
        grid = fig.add_gridspec(
            5,
            2,
            height_ratios=(1.13, 1.13, 0.82, 0.82, 0.86),
        )
        for view_index, (view_name, raw_path) in enumerate(zip(VIEW_NAMES, view_paths)):
            axis = fig.add_subplot(grid[view_index // 2, view_index % 2])
            path = Path(raw_path)
            if path.is_file():
                with Image.open(path) as image:
                    axis.imshow(image.convert("RGB"))
            else:
                axis.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center")
            axis.set_title(f"{view_name.upper()} — actual RGB input", fontsize=12)
            axis.axis("off")

        role_payload = {}
        curve_locations = ((2, 0), (2, 1), (3, 0), (3, 1), (4, slice(None)))
        legend_pending = True
        for query_index, (query_name, location) in enumerate(
            zip(CURVE_QUERY_NAMES, curve_locations)
        ):
            row, column = location
            axis = fig.add_subplot(grid[row, column])
            present = bool(role_mask[sample_index, query_index])
            payload = _plot_mobile_curve(
                axis,
                query_name=query_name,
                truth=expected[sample_index, query_index] if present else None,
                spatial_prediction=predicted[sample_index, query_index] if present else None,
                spatial_presence=float(presence[sample_index, query_index]),
                global_prediction=(
                    global_prediction[query_index]
                    if global_prediction is not None and present
                    else None
                ),
                global_presence=(
                    float(global_presence[query_index]) if global_presence is not None else None
                ),
                show_legend=legend_pending and present,
            )
            if present:
                legend_pending = False
            role_payload[query_name] = payload

        global_summary = (
            f" | global mean Chamfer={global_sample_error:.3f}"
            if global_sample_error is not None
            else ""
        )
        fig.suptitle(
            f"{sample_id} — frozen-test curve reconstruction\n"
            f"selection q={quantile:.1f} | spatial mean Chamfer={spatial_sample_error:.3f}"
            f"{global_summary}\n"
            "GT black | spatial FPN magenta dashed | global-token ablation blue dash-dot\n"
            f"TARGET PROVENANCE: {CURVE_TRUTH_DENSE_APPROXIMATION}\n"
            "Dense canonical curve fitted to two cubics; NOT original generator control points.",
            fontsize=13,
        )
        output_path = output_directory / f"{sample_id}_mobile_curve_comparison.png"
        fig.savefig(output_path, dpi=180, facecolor="white")
        plt.close(fig)
        with Image.open(output_path) as rendered:
            dimensions = [int(rendered.width), int(rendered.height)]
        if dimensions[0] < 1600:
            raise RuntimeError(f"mobile board is narrower than 1600 pixels: {output_path}")
        records.append(
            {
                "sample_id": sample_id,
                "selection_error_quantile": quantile,
                "spatial_mean_symmetric_chamfer": spatial_sample_error,
                "global_mean_symmetric_chamfer": global_sample_error,
                "output": str(output_path),
                "dimensions_px": dimensions,
                "view_layout": "2x2 front/back/left/right above five curve panels",
                "view_paths": list(view_paths),
                "target_provenance": CURVE_TRUTH_DENSE_APPROXIMATION,
                "roles": role_payload,
            }
        )
    manifest = {
        "schema_version": "multiview-spatial-curve-mobile-boards-1.0",
        "selection": (
            "same deterministic frozen-test 20%-to-80% spatial Chamfer quantiles as the "
            "wide board, restricted to maximum curve-role support"
        ),
        "target_provenance": CURVE_TRUTH_DENSE_APPROXIMATION,
        "warning": (
            "GT controls are a two-cubic fit to dense canonical curves, not original "
            "generator authoring controls."
        ),
        "global_token_ablation_included": global_arrays is not None,
        "minimum_width_px": 1600,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen spatial curve predictions against train-mean and render an "
            "actual-four-view GT/predicted two-cubic board."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/drafting_semantics/multiview_curve_parameters_fpn.pt"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/test_predictions.npz"),
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/training_metrics.json"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_multiview_index.json"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/frozen_test_evaluation.json"),
    )
    parser.add_argument(
        "--output-board",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/frozen_test_curve_board.png"),
    )
    parser.add_argument(
        "--board-manifest",
        type=Path,
        default=Path("artifacts/drafting_semantics/multiview_curve_parameters/frozen_test_curve_board.json"),
    )
    parser.add_argument(
        "--global-predictions",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_curve_parameters/global_test_predictions.npz"
        ),
    )
    parser.add_argument(
        "--mobile-output-directory",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_curve_parameters/mobile_curve_boards"
        ),
    )
    parser.add_argument(
        "--mobile-manifest",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_curve_parameters/mobile_curve_boards.json"
        ),
    )
    parser.add_argument("--board-rows", type=int, default=3)
    parser.add_argument("--samples-per-segment", type=int, default=65)
    parser.add_argument(
        "--mobile-only",
        action="store_true",
        help="Generate mobile boards without rewriting the existing metrics or wide board.",
    )
    parser.add_argument(
        "--skip-mobile-boards",
        action="store_true",
        help="Keep the pre-mobile evaluator behavior.",
    )
    args = parser.parse_args()

    means, deviations, checkpoint = _load_checkpoint_contract(args.checkpoint)
    arrays = _load_predictions(args.predictions)
    global_arrays = None
    if args.global_predictions.is_file():
        global_arrays = _load_predictions(args.global_predictions)
        _validate_ablation_alignment(arrays, global_arrays)
    train_presence = _train_presence_rates(
        args.training_metrics if args.training_metrics.is_file() else None
    )
    evaluation = evaluate_frozen_curve_predictions(
        arrays["predicted_curve_parameters"],
        arrays["target_curve_parameters"],
        arrays["target_role_mask"],
        arrays["predicted_presence_probability"],
        means,
        deviations,
        train_presence_rates=train_presence,
        samples_per_segment=args.samples_per_segment,
    )
    evaluation["target_contract"] = {
        "provenance": CURVE_TRUTH_DENSE_APPROXIMATION,
        "meaning": (
            "least-squares normalized two-cubic approximation of the paired dense canonical "
            "semantic path, not the generator's original Bezier controls"
        ),
    }
    evaluation["frozen_checkpoint"] = {
        "path": str(args.checkpoint),
        "sha256": _sha256(args.checkpoint),
        "best_epoch": checkpoint.get("best_epoch"),
    }
    evaluation["prediction_archive"] = {
        "path": str(args.predictions),
        "sha256": _sha256(args.predictions),
    }
    evaluation["training_presence_baseline_available"] = train_presence is not None
    sample_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    views_by_id = _view_lookup(args.index)
    board = None
    if not args.mobile_only:
        args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.output_metrics.write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        board = _render_board(
            args.output_board,
            args.board_manifest,
            sample_ids=sample_ids,
            predicted=arrays["predicted_curve_parameters"],
            expected=arrays["target_curve_parameters"],
            role_mask=arrays["target_role_mask"],
            presence=arrays["predicted_presence_probability"],
            views_by_id=views_by_id,
            row_count=args.board_rows,
        )
    mobile = None
    if not args.skip_mobile_boards:
        mobile = _render_mobile_boards(
            args.mobile_output_directory,
            args.mobile_manifest,
            sample_ids=sample_ids,
            predicted=arrays["predicted_curve_parameters"],
            expected=arrays["target_curve_parameters"],
            role_mask=arrays["target_role_mask"],
            presence=arrays["predicted_presence_probability"],
            views_by_id=views_by_id,
            row_count=args.board_rows,
            global_arrays=global_arrays,
        )
    print(
        json.dumps(
            {
                "metrics": str(args.output_metrics),
                "board": str(args.output_board),
                "board_records": (
                    [row["sample_id"] for row in board["records"]] if board else None
                ),
                "mobile_boards": (
                    [row["output"] for row in mobile["records"]] if mobile else None
                ),
                "global_token_ablation_included": global_arrays is not None,
                "sample_count": evaluation["sample_count"],
                "model_gain_over_train_mean": evaluation["model_gain_over_train_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
