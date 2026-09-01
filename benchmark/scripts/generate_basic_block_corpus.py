from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmark.drafting_semantics.basic_blocks import SCHEMA_VERSION, generate_corpus, write_corpus_json
from benchmark.pattern_pipeline.validation import validate_pattern


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic PROVISIONAL_EXPERT_REVIEW T-shirt, pants, "
            "and skirt basic-block variations."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tshirt-count", type=int, default=100)
    parser.add_argument("--pants-count", type=int, default=100)
    parser.add_argument("--skirt-count", type=int, default=100)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--curve-samples", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "tshirt": args.tshirt_count,
        "pants": args.pants_count,
        "skirt": args.skirt_count,
    }
    corpus = generate_corpus(counts, seed=args.seed)
    destination = write_corpus_json(corpus, args.output)
    validation_errors: Counter[str] = Counter()
    accepted = 0
    seam_ratios: dict[str, list[float]] = {"front": [], "back": []}
    for record in corpus.records:
        result = validate_pattern(record.to_pattern_document(curve_samples=args.curve_samples))
        if result.accepted:
            accepted += 1
        else:
            validation_errors.update(item.code for item in result.errors)
        if record.category == "tshirt":
            audit = record.panel("sleeve").metadata
            seam_ratios["front"].append(
                float(audit["front_cap_length_cm"]) / float(audit["front_armhole_length_cm"])
            )
            seam_ratios["back"].append(
                float(audit["back_cap_length_cm"]) / float(audit["back_armhole_length_cm"])
            )
    if args.manifest:
        payload = {
            "schema_version": "basic-garment-block-manifest/v3",
            "block_schema_version": SCHEMA_VERSION,
            "status": corpus.provenance_status,
            "expert_review": "PENDING",
            "industrial_pattern_truth": False,
            "generator": "benchmark.drafting_semantics.basic_blocks",
            "generator_method": "bounded_correlated_parametric_basic_block_v3",
            "seed": corpus.seed,
            "record_count": len(corpus.records),
            "category_counts": dict(Counter(item.category for item in corpus.records)),
            "scope": [
                "basic short-sleeve T-shirt",
                "straight-leg pants",
                "two-panel pencil skirt with a provisional back vent",
            ],
            "semantic_content": [
                "named landmarks and construction levels",
                "explicit front/back centre-hip landmarks on lower-body centre paths",
                "named boundary paths and cubic control geometry",
                "reference-line geometry exposed as non-boundary teacher tokens",
                "separate front/back lower-body waist darts",
                "panel symmetry and checked seam compatibility relations",
            ],
            "correlated_variation_policy": {
                "tshirt": "waist <= bust + 4 cm and hip >= waist + 4 cm",
                "pants_skirt": "waist <= hip - 12 cm",
                "family_disjoint": False,
            },
            "tshirt_cap_armhole_ratio": {
                side: {
                    "minimum": min(values),
                    "maximum": max(values),
                    "outside_declared_12_percent": sum(not 0.88 <= value <= 1.12 for value in values),
                }
                for side, values in seam_ratios.items()
            },
            "unsupported_production_claims": [
                "expert-approved industrial fit",
                "zipper or closure geometry without source evidence",
                "notches, seam allowance, or approved grain direction",
            ],
            "artifact": {
                "relative_path": destination.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "tracked": False,
            },
            "geometric_validation": {
                "curve_samples_per_cubic": args.curve_samples,
                "accepted": accepted,
                "rejected": len(corpus.records) - accepted,
                "error_counts": dict(validation_errors),
            },
            "review_contract": (
                "Use as provisional synthetic supervision and edit anchors only until a "
                "pattern expert accepts or revises the ranges and formulas."
            ),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        {
            "output": str(destination.resolve()),
            "record_count": len(corpus.records),
            "seed": corpus.seed,
            "provenance_status": corpus.provenance_status,
            "accepted": accepted,
            "manifest": str(args.manifest.resolve()) if args.manifest else None,
        }
    )


if __name__ == "__main__":
    main()
