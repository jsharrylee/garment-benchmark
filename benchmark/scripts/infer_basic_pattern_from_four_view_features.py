"""Run the fail-closed four-view semantic-to-basic-pattern bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.drafting_semantics.semantic_teacher_student import CATEGORY_NAMES
from benchmark.pattern_pipeline.four_view_semantic_inference import (
    CANONICAL_VIEW_ORDER,
    infer_provisional_basic_pattern,
    load_four_view_student_checkpoint,
    load_precomputed_four_view_features,
)
from benchmark.pattern_pipeline.parametric_drafting_inference import (
    infer_parametric_tshirt_pattern,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Infer bounded semantic residuals from exactly ordered front/back/left/right "
            "feature tensors and apply them to a provisional category BasicBlock."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORY_NAMES, required=True)
    parser.add_argument(
        "--decoder",
        choices=("residual", "parametric"),
        default="residual",
        help=(
            "`residual` preserves the legacy topology-local editor. `parametric` "
            "jointly redrafts a T-shirt and re-solves its sleeve/armhole constraint."
        ),
    )
    parser.add_argument(
        "--sample-id",
        help="Required only when the feature archive contains more than one sample.",
    )
    parser.add_argument(
        "--generic-feature-kind",
        choices=("global", "spatial"),
        help="Required for legacy archives whose tensor key is simply `features`.",
    )
    parser.add_argument(
        "--view-order",
        nargs=4,
        metavar=("FRONT", "BACK", "LEFT", "RIGHT"),
        help=(
            "Explicit order attestation for archives without embedded view_names. "
            "The only accepted value is: front back left right."
        ),
    )
    parser.add_argument("--output-pattern", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--curve-samples", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--uncalibrated-confidence",
        type=float,
        help=(
            "Explicit uncalibrated reliability fallback in [0,1]. Omit to fail closed "
            "with zero edit confidence when checkpoint validation calibration is absent."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.view_order is not None and tuple(value.lower() for value in args.view_order) != CANONICAL_VIEW_ORDER:
        raise SystemExit(
            "--view-order must be exactly: " + " ".join(CANONICAL_VIEW_ORDER)
        )
    features = load_precomputed_four_view_features(
        args.features,
        sample_id=args.sample_id,
        generic_feature_kind=args.generic_feature_kind,
        declared_view_order=args.view_order,
    )
    loaded = load_four_view_student_checkpoint(
        args.checkpoint,
        device=args.device,
        uncalibrated_confidence=args.uncalibrated_confidence,
    )
    if args.decoder == "parametric":
        if args.category != "tshirt":
            raise SystemExit("--decoder parametric currently supports --category tshirt only")
        result = infer_parametric_tshirt_pattern(
            loaded,
            features,
            output_id=(
                f"tshirt_four_view_parametric_{args.sample_id}"
                if args.sample_id
                else "tshirt_four_view_parametric"
            ),
        )
    else:
        result = infer_provisional_basic_pattern(
            loaded,
            features,
            category=args.category,
            curve_samples=args.curve_samples,
        )
    result.save(args.output_pattern, args.receipt)
    # Deliberately do not echo checkpoint, feature archive, sample, or image
    # paths.  The structured receipt is also path-free.
    print(
        json.dumps(
            {
                "status": result.receipt["status"],
                "category": args.category,
                "decoder": args.decoder,
                "pattern_id": result.document.pattern_id,
                "view_order": list(CANONICAL_VIEW_ORDER),
                "source_paths_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
