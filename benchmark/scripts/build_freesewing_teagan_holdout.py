from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from benchmark.drafting_semantics.freesewing_teagan import read_teagan_extractor_json


MODELS = tuple(
    [f"cisFemaleAdult{size}" for size in range(28, 48, 2)]
    + [f"cisMaleAdult{size}" for size in range(32, 52, 2)]
)
OPTION_VARIANTS = (
    ("default", {}),
    ("fitted_short", {"chestEase": 0.06, "waistEase": 0.12, "hipsEase": 0.10, "lengthBonus": 0.02, "sleeveLength": 0.2}),
    ("loose_long", {"chestEase": 0.20, "waistEase": 0.34, "hipsEase": 0.28, "lengthBonus": 0.26, "sleeveLength": 0.52}),
    ("wide_deep_neck", {"necklineWidth": 0.48, "necklineDepth": 0.42, "backNeckCutout": 0.14, "sleeveLength": 0.36}),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an official FreeSewing Teagan strict source holdout.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/teagan_holdout.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/freesewing_teagan_holdout.json"))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    plan = [(model, variant_id, options) for model in MODELS for variant_id, options in OPTION_VARIANTS]
    if args.max_samples is not None:
        plan = plan[: max(0, args.max_samples)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extractor = Path("benchmark/scripts/extract_freesewing_teagan.mjs").resolve()
    with tempfile.TemporaryDirectory(prefix="teagan-holdout-") as temp_value:
        temp = Path(temp_value)
        with gzip.open(args.output, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
            for index, (model, variant_id, options) in enumerate(plan):
                raw_path = temp / f"{index:03d}.json"
                command = [
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
                ]
                subprocess.run(command, check=True)
                sample_id = f"freesewing_teagan__{model}__{variant_id}"
                record = read_teagan_extractor_json(raw_path, sample_id=sample_id)
                stream.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "freesewing-teagan-holdout-1.0",
        "source": "official @freesewing/teagan@4.10.1 package",
        "sample_count": len(plan),
        "full_plan_count": len(MODELS) * len(OPTION_VARIANTS),
        "model_count": len(set(model for model, _, _ in plan)),
        "design_variant_count": len(set(variant for _, variant, _ in plan)),
        "split": "unseen_source",
        "training_use": False,
        "artifact_sha256": _sha256(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "landmark_scope": ["FNP", "BNP", "SNP", "SP"],
        "BP": "NOT_DEFINED_BY_RECIPE",
        "darts": "NOT_APPLICABLE",
        "production_annotations": ["4 armhole notches", "grainline", "cut-on-fold", "seam allowance"],
        "creation_time_DAG": "NOT_AVAILABLE; named output domain only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
