from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from benchmark.drafting_semantics.freesewing_split import (
    TEST_BODY_MODELS,
    TRAIN_BODY_MODELS,
    VALIDATION_BODY_MODELS,
)
from benchmark.drafting_semantics.freesewing_teagan import read_teagan_extractor_json


# All values stay within Teagan's published option ranges.  The matrix varies
# fit, length, neckline and sleeve blocks.  Teagan topology remains fixed at
# three panels / 22 primitives, so this corpus tests geometric variation but
# cannot rule out a recipe/topology fingerprint.
TRAIN_VARIANTS = (
    ("default", {}),
    ("fitted_short", {"chestEase": 0.06, "waistEase": 0.12, "hipsEase": 0.10, "lengthBonus": 0.02, "sleeveLength": 0.20}),
    ("relaxed_short", {"chestEase": 0.20, "waistEase": 0.30, "hipsEase": 0.26, "lengthBonus": 0.05, "sleeveLength": 0.24}),
    ("fitted_long_sleeve", {"chestEase": 0.08, "waistEase": 0.14, "hipsEase": 0.12, "lengthBonus": 0.12, "sleeveLength": 0.92}),
    ("relaxed_long_sleeve", {"chestEase": 0.22, "waistEase": 0.34, "hipsEase": 0.28, "lengthBonus": 0.20, "sleeveLength": 0.84}),
    ("wide_shallow_neck", {"necklineWidth": 0.44, "necklineDepth": 0.21, "backNeckCutout": 0.05, "sleeveLength": 0.34}),
    ("narrow_scoop", {"necklineWidth": 0.18, "necklineDepth": 0.36, "backNeckCutout": 0.07, "necklineBend": 0.50}),
    ("long_body_mid_sleeve", {"lengthBonus": 0.48, "sleeveLength": 0.56, "chestEase": 0.16, "hipsEase": 0.22}),
)
VALIDATION_VARIANTS = (
    ("validation_boat", {"necklineWidth": 0.49, "necklineDepth": 0.24, "backNeckCutout": 0.10, "sleeveLength": 0.40}),
    ("validation_compact", {"chestEase": 0.07, "waistEase": 0.10, "hipsEase": 0.09, "lengthBonus": -0.12, "sleeveLength": 0.30}),
)
TEST_VARIANTS = (
    ("test_deep_long", {"necklineWidth": 0.32, "necklineDepth": 0.40, "backNeckCutout": 0.12, "lengthBonus": 0.38, "sleeveLength": 0.74}),
    ("test_wide_relaxed", {"necklineWidth": 0.50, "necklineDepth": 0.31, "chestEase": 0.25, "waistEase": 0.40, "hipsEase": 0.30, "sleeveLength": 0.48}),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group(value: str, train: tuple[str, ...], validation: tuple[str, ...], test: tuple[str, ...]) -> str:
    if value in train:
        return "train"
    if value in validation:
        return "validation"
    if value in test:
        return "test"
    raise ValueError(f"unassigned split value: {value}")


def _split(model: str, variant: str) -> str:
    body = _group(model, TRAIN_BODY_MODELS, VALIDATION_BODY_MODELS, TEST_BODY_MODELS)
    design = _group(
        variant,
        tuple(name for name, _ in TRAIN_VARIANTS),
        tuple(name for name, _ in VALIDATION_VARIANTS),
        tuple(name for name, _ in TEST_VARIANTS),
    )
    if body == design == "train":
        return "train"
    return f"{body}_{design}" if body != design else f"{body}_double"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a diverse, split-safe FreeSewing Teagan corpus.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/teagan_diverse.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/freesewing_teagan_diverse.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    variants = (*TRAIN_VARIANTS, *VALIDATION_VARIANTS, *TEST_VARIANTS)
    models = (*TRAIN_BODY_MODELS, *VALIDATION_BODY_MODELS, *TEST_BODY_MODELS)
    plan = [(model, variant, options) for model in models for variant, options in variants]
    if args.max_samples:
        plan = plan[: args.max_samples]
    extractor = Path("benchmark/scripts/extract_freesewing_teagan.mjs").resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="teagan-diverse-") as temp_value:
        temp = Path(temp_value)

        def extract(item: tuple[int, tuple[str, str, dict]]) -> tuple[int, object]:
            index, (model, variant, options) = item
            raw_path = temp / f"{index:04d}.json"
            subprocess.run(
                [
                    "node",
                    str(extractor),
                    "--model",
                    model,
                    "--sa",
                    "10",
                    "--options",
                    json.dumps(options, separators=(",", ":")),
                    "--output",
                    str(raw_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            sample_id = f"freesewing_teagan_diverse__{model}__{variant}"
            record = read_teagan_extractor_json(raw_path, sample_id=sample_id)
            return index, replace(
                record,
                split=_split(model, variant),
                metadata={
                    **record.metadata,
                    "cross_source_zero_shot": False,
                    "training_usage": "integrated_multigarment_training",
                },
            )

        records: list[object] = [None] * len(plan)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for completed, (index, record) in enumerate(executor.map(extract, enumerate(plan)), start=1):
                records[index] = record
                if completed % 40 == 0:
                    print(json.dumps({"completed": completed, "total": len(plan)}), flush=True)

        with gzip.open(args.output, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
            for record in records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")

    split_counts = Counter(record.split for record in records)
    canonical_split_counts = Counter(
        "train" if record.split == "train" else ("test" if "test" in record.split else "validation")
        for record in records
    )
    requested_option_keys = sorted(
        {key for _, options in variants for key in options}
    )
    manifest = {
        "schema_version": "freesewing-teagan-diverse-1.0",
        "source": "official @freesewing/teagan@4.10.1 and @freesewing/models@4.10.1 packages",
        "license": "MIT",
        "record_count": len(records),
        "body_model_count": len(models),
        "design_variant_count": len(variants),
        "split_counts": dict(sorted(split_counts.items())),
        "canonical_split_counts": dict(sorted(canonical_split_counts.items())),
        "jointly_unseen_body_and_option_count": split_counts.get("test_double", 0),
        "train_variants": [name for name, _ in TRAIN_VARIANTS],
        "validation_variants": [name for name, _ in VALIDATION_VARIANTS],
        "test_variants": [name for name, _ in TEST_VARIANTS],
        "implementation_family_id": "freesewing-brian-library-sleeve-family",
        "requested_option_keys": requested_option_keys,
        "topology_note": "all records use the same Teagan recipe with 3 panels and 22 primitives",
        "option_note": "waistEase is requested in some presets but is inactive while fitWaist remains false",
        "artifact_sha256": _sha256(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "claim_boundary": "unseen body/options within one implementation family; not unseen recipe",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
