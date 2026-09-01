"""Build exact T-shirt 2D-element -> GCDv2 simulated-surface supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.drafting_semantics.gcdv2_surface_correspondence import (
    ELEMENT_NAMES,
    SCHEMA_VERSION,
    TSHIRT_PARAMETER_NAMES,
    build_tshirt_surface_example,
    is_simple_tshirt_record,
    read_jsonl,
)


SPLIT_TO_INDEX = {"training": 0, "validation": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/gcdv2_exact_pairs_v1/index.jsonl"))
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("artifacts/drafting_semantics/gcdv2_multigarment/records.jsonl"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/processed/garmentcode_v2/batch_0_full"),
    )
    parser.add_argument(
        "--fpn-cache",
        type=Path,
        default=Path("artifacts/gcdv2_exact_pairs_v1/multiview_fpn_tokens.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/drafting_semantics/tshirt_visual_causality/surface_correspondence.npz"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/drafting_semantics/tshirt_visual_causality/surface_correspondence_audit.json"),
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def main() -> None:
    args = parse_args()
    index_rows = read_jsonl(args.index)
    index_by_id = {str(row["sample_id"]): row for row in index_rows}
    fpn = np.load(args.fpn_cache, allow_pickle=False, mmap_mode="r")
    feature_lookup = {str(value): index for index, value in enumerate(fpn["sample_ids"])}

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in read_jsonl(args.records):
        sample_id = str(record["sample_id"])
        index = index_by_id.get(sample_id)
        if index is None or sample_id not in feature_lookup:
            continue
        if record.get("split") not in SPLIT_TO_INDEX:
            continue
        if not is_simple_tshirt_record(record, str(index["category"])):
            continue
        candidates.append((index, record))
    candidates.sort(key=lambda value: str(value[0]["sample_id"]))
    if args.limit is not None:
        candidates = candidates[: args.limit]

    examples = []
    failures = []
    for count, (index, record) in enumerate(candidates, start=1):
        try:
            example = build_tshirt_surface_example(
                index_row=index,
                semantic_record=record,
                raw_root=args.raw_root,
            )
            if not bool(np.all(example.element_valid)):
                missing = [
                    name for name, valid in zip(ELEMENT_NAMES, example.element_valid) if not valid
                ]
                raise ValueError(f"missing projected elements: {missing}")
            if not bool(np.all(example.parameter_valid)):
                missing = [
                    name
                    for name, valid in zip(TSHIRT_PARAMETER_NAMES, example.parameter_valid)
                    if not valid
                ]
                raise ValueError(f"missing physical parameters: {missing}")
            examples.append(example)
        except Exception as error:
            failures.append(
                {
                    "sample_id": str(index["sample_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if count == 1 or count % 10 == 0 or count == len(candidates):
            print(
                json.dumps(
                    {
                        "processed": count,
                        "candidate_count": len(candidates),
                        "accepted": len(examples),
                        "failed": len(failures),
                    }
                ),
                flush=True,
            )
    if not examples:
        raise RuntimeError("no T-shirt correspondence examples passed")

    sample_ids = np.asarray([value.sample_id for value in examples])
    splits = np.asarray([SPLIT_TO_INDEX[value.split] for value in examples], dtype=np.int8)
    feature_indices = np.asarray([feature_lookup[value] for value in sample_ids], dtype=np.int32)
    parameters = np.stack([value.parameter_values for value in examples])
    parameter_valid = np.stack([value.parameter_valid for value in examples])
    element_heatmaps = np.stack([value.element_heatmaps for value in examples]).astype(np.float16)
    element_valid = np.stack([value.element_valid for value in examples])
    element_vertex_counts = np.stack([value.element_vertex_counts for value in examples])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=sample_ids,
        splits=splits,
        feature_indices=feature_indices,
        parameter_values=parameters,
        parameter_valid=parameter_valid,
        element_heatmaps=element_heatmaps,
        element_valid=element_valid,
        element_vertex_counts=element_vertex_counts,
        parameter_names=np.asarray(TSHIRT_PARAMETER_NAMES),
        element_names=np.asarray(ELEMENT_NAMES),
        schema_version=np.asarray(SCHEMA_VERSION),
    )
    split_names = ("train", "validation", "test")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "PASS_WITH_QUARANTINE",
        "claim_boundary": (
            "Existing official GCDv2 samples provide exact shared-topology 2D-to-3D "
            "correspondence. This corpus is observational and contains no edited-pattern "
            "counterfactual render pairs."
        ),
        "source": {
            "dataset": "GarmentCodeData v2",
            "license": "CC BY 4.0",
            "index_sha256": sha256(args.index),
            "semantic_records_sha256": sha256(args.records),
            "fpn_cache_sha256": sha256(args.fpn_cache),
        },
        "candidate_count": len(candidates),
        "accepted_count": len(examples),
        "quarantined_count": len(failures),
        "split_counts": {
            split_names[index]: int(np.sum(splits == index)) for index in range(len(split_names))
        },
        "element_names": list(ELEMENT_NAMES),
        "parameter_names": list(TSHIRT_PARAMETER_NAMES),
        "all_accepted_elements_present": bool(np.all(element_valid)),
        "all_accepted_parameters_present": bool(np.all(parameter_valid)),
        "element_vertex_count_summary": {
            name: {
                "minimum": int(element_vertex_counts[:, index].min()),
                "median": float(np.median(element_vertex_counts[:, index])),
                "maximum": int(element_vertex_counts[:, index].max()),
            }
            for index, name in enumerate(ELEMENT_NAMES)
        },
        "parameter_support": {
            name: _stats(parameters[:, index])
            for index, name in enumerate(TSHIRT_PARAMETER_NAMES)
        },
        "mapping_evidence_counts": dict(
            Counter(
                key
                for value in examples
                for key, count in value.audit["element_mapping"]["evidence_counts"].items()
                for _ in range(int(count))
            )
        ),
        "quarantine": failures,
        "true_counterfactual_four_view_pairs": 0,
        "causal_visual_metrics_enabled": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
