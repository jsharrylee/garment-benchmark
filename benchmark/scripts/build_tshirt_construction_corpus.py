from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.drafting_semantics.tshirt_corpus import build_sample_plan, plan_digest, split_counts
from benchmark.drafting_semantics.tshirt_garmentcode import generate_garmentcode_tshirt_trace


def _generate(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    sample, root = payload
    try:
        record = generate_garmentcode_tshirt_trace(
            sample_id=sample["sample_id"],
            split=sample["split"],
            body_values=sample["body_values"],
            design_values=sample["design_values"],
            garmentcode_root=Path(root),
            body_id=sample["body_id"],
            design_id=sample["design_id"],
        )
    except Exception as error:  # A bad parametric sample must not erase the rest of the corpus.
        return {
            "ok": False,
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "body_id": sample["body_id"],
            "design_id": sample["design_id"],
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "ok": True,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "creation_contract": record.metadata["creation_semantic_contract"],
        "row": json.dumps(
            record.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build creation-time truth for GarmentCode basic T-shirts.")
    parser.add_argument("--garmentcode-root", type=Path, default=Path("external/GarmentCode"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drafting_semantics/tshirt_traces.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/tshirt_construction_truth.json"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--splits", help="Optional comma-separated split labels for a bounded diagnostic build.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunksize", type=int, default=2)
    args = parser.parse_args()

    root = args.garmentcode_root.resolve()
    full_plan = build_sample_plan(root)
    requested_splits = None if not args.splits else {value.strip() for value in args.splits.split(",") if value.strip()}
    eligible = full_plan if requested_splits is None else tuple(sample for sample in full_plan if sample.split in requested_splits)
    selected = eligible if args.max_samples is None else eligible[: max(0, args.max_samples)]
    payloads = [
        (
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "body_id": sample.body.id,
                "design_id": sample.design.id,
                "body_values": sample.body.values,
                "design_values": sample.design.values,
            },
            str(root),
        )
        for sample in selected
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    context = mp.get_context("spawn")
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with gzip.open(args.output, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        if args.workers == 1:
            rows = map(_generate, payloads)
            for result in rows:
                if result["ok"]:
                    stream.write(result["row"] + "\n")
                    generated.append(result)
                else:
                    failures.append(result)
        else:
            with context.Pool(processes=args.workers) as pool:
                for result in pool.imap(_generate, payloads, chunksize=max(1, args.chunksize)):
                    if result["ok"]:
                        stream.write(result["row"] + "\n")
                        generated.append(result)
                    else:
                        failures.append(result)
    elapsed = time.perf_counter() - started
    contract_counts = Counter(
        json.dumps(result["creation_contract"], sort_keys=True, separators=(",", ":"))
        for result in generated
    )
    manifest = {
        "schema_version": "tshirt-construction-corpus-1.0",
        "source": "maria-korosteleva/GarmentCode",
        "recipe": "Shirt(fitted=False), symmetric CircleNeckHalf + ArmholeCurve + two half-sleeves per side",
        "truth_policy": "creation-time runtime operations and live pre-assembly geometry",
        "full_plan_count": len(full_plan),
        "attempted_count": len(selected),
        "requested_splits": None if requested_splits is None else sorted(requested_splits),
        "generated_count": len(generated),
        "failure_count": len(failures),
        "is_full_corpus": len(selected) == len(full_plan) and not failures,
        "split_counts": dict(sorted(Counter(result["split"] for result in generated).items())),
        "failure_split_counts": dict(sorted(Counter(result["split"] for result in failures).items())),
        "failures": failures,
        "creation_contract_distribution": [
            {"count": count, "contract": json.loads(contract)}
            for contract, count in sorted(contract_counts.items())
        ],
        "full_plan_split_counts": split_counts(full_plan),
        "plan_sha256": plan_digest(full_plan),
        "artifact_sha256": _sha256(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "elapsed_seconds": round(elapsed, 3),
        "workers": args.workers,
        "limitations": {
            "BP": "NOT_DEFINED_BY_RECIPE",
            "darts": "NOT_APPLICABLE_TO_BASIC_TSHIRT",
            "notches_grainline_seam_allowance": "NOT_CREATED_BY_GARMENTCODE_RECIPE",
            "reference_lines": "separate anthropometric_adapter domain",
            "expert_validation": "PENDING",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
