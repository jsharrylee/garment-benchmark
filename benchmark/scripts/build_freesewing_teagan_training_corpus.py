from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from benchmark.drafting_semantics.freesewing_split import (
    EXPECTED_SPLIT_COUNTS,
    TEST_BODY_MODELS,
    TEST_DESIGNS,
    TRAIN_BODY_MODELS,
    TRAIN_DESIGNS,
    VALIDATION_BODY_MODELS,
    VALIDATION_DESIGNS,
    repartition_teagan_records,
)
from benchmark.drafting_semantics.tshirt_learning import read_tshirt_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-resistant FreeSewing Teagan training matrix.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/drafting_semantics/teagan_holdout.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/teagan_training.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/freesewing_teagan_training.json"),
    )
    args = parser.parse_args()

    original = read_tshirt_records(args.input)
    records = repartition_teagan_records(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "freesewing-teagan-training-split-1.0",
        "source": "official @freesewing/teagan@4.10.1 package",
        "purpose": "SOURCE_SPECIFIC_BASIC_TSHIRT_SEMANTIC_BASELINE",
        "input_artifact_sha256": _sha256(args.input),
        "output_artifact_sha256": _sha256(args.output),
        "record_count": len(records),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "train_body_models": TRAIN_BODY_MODELS,
        "validation_body_models": VALIDATION_BODY_MODELS,
        "test_body_models": TEST_BODY_MODELS,
        "train_designs": TRAIN_DESIGNS,
        "validation_designs": VALIDATION_DESIGNS,
        "test_designs": TEST_DESIGNS,
        "leakage_contract": "body identities and design variants used for frozen tests are absent from training",
        "original_zero_shot_result": "PRESERVED; this derivative corpus does not rewrite the original holdout",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
