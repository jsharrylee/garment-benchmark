from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from benchmark.retrieval.corpus import infer_garment_category


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BATCH_0_BYTES = 5_121_727_032
SOURCE_URL = (
    "https://libdrive.ethz.ch/public.php/webdav/"
    "GarmentCodeData_v2/garments_5000_0/default_body/data.tar.gz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(member_name: str) -> str | None:
    return next((part for part in PurePosixPath(member_name).parts if part.startswith("rand_")), None)


def safe_relative_path(member_name: str) -> Path:
    value = PurePosixPath(member_name)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"unsafe archive member: {member_name}")
    return Path(*value.parts)


def split_lookup(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        PurePosixPath(item).name: split
        for split, items in raw.items()
        for item in items
        if "/garments_5000_0/default_body/" in f"/{item}"
    }


def render_metrics(payload: bytes) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    colored = (maximum - minimum > 18) & (maximum > 35) & (minimum < 245)
    ys, xs = np.nonzero(colored)
    if not len(xs):
        return {"colored_fraction": 0.0, "centroid_y": None, "bbox": None}
    height, width = colored.shape
    return {
        "colored_fraction": float(colored.mean()),
        "centroid_y": float(ys.mean() / height),
        "bbox": [float(xs.min() / width), float(ys.min() / height), float((xs.max() + 1) / width), float((ys.max() + 1) / height)],
    }


def render_quality(category: str, metrics: dict | None) -> str:
    if not metrics or metrics["centroid_y"] is None or metrics["colored_fraction"] < 0.002:
        return "REJECT_EMPTY_OR_TINY"
    ranges = {
        "top": (0.20, 0.66),
        "pants": (0.42, 0.86),
        "skirt": (0.38, 0.84),
        "dress": (0.25, 0.82),
        "jumpsuit": (0.25, 0.86),
    }
    lower, upper = ranges.get(category, (0.15, 0.9))
    return "PASS" if lower <= metrics["centroid_y"] <= upper else "REJECT_IMPLAUSIBLE_VERTICAL_PLACEMENT"


def inspect(archive: Path, split_path: Path) -> tuple[dict, list[dict]]:
    splits = split_lookup(split_path)
    suffixes: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    members = 0
    uncompressed_bytes = 0
    samples: set[str] = set()
    records: list[dict] = []
    front_renders: dict[str, dict] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            safe_relative_path(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"links are not accepted in dataset archive: {member.name}")
            members += 1
            uncompressed_bytes += int(member.size)
            identifier = sample_id(member.name)
            if identifier:
                samples.add(identifier)
            if member.isfile():
                suffixes["".join(PurePosixPath(member.name).suffixes).lower() or "<none>"] += 1
            if member.isfile() and member.name.endswith("_specification.json"):
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read {member.name}")
                payload = stream.read()
                raw = json.loads(payload)
                panels = raw["pattern"]["panels"]
                panel_names = tuple(panels)
                category = infer_garment_category(panel_names)
                categories[category] += 1
                records.append(
                    {
                        "sample_id": identifier,
                        "category": category,
                        "split": splits.get(identifier or "", "not_in_official_paired_split"),
                        "panel_count": len(panels),
                        "edge_count": sum(len(panel["edges"]) for panel in panels.values()),
                        "stitch_count": len(raw["pattern"].get("stitches", [])),
                        "panel_names": panel_names,
                        "specification_member": member.name,
                        "specification_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            elif member.isfile() and member.name.endswith("_render_front.png") and identifier:
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read {member.name}")
                front_renders[identifier] = render_metrics(stream.read())
    for record in records:
        metrics = front_renders.get(record["sample_id"] or "")
        record["front_render_metrics"] = metrics
        record["render_quality"] = render_quality(record["category"], metrics)
    summary = {
        "archive_member_count": members,
        "archive_uncompressed_bytes": uncompressed_bytes,
        "sample_directory_count": len(samples),
        "specification_count": len(records),
        "category_counts": dict(sorted(categories.items())),
        "suffix_counts": dict(sorted(suffixes.items())),
        "official_split_counts": dict(sorted(Counter(record["split"] for record in records).items())),
        "render_quality_counts": dict(sorted(Counter(record["render_quality"] for record in records).items())),
    }
    return summary, records


def select_balanced(records: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    selected = []
    for category in sorted({record["category"] for record in records} - {"unknown"}):
        candidates = sorted(
            (record for record in records if record["category"] == category and record.get("render_quality", "PASS") == "PASS"),
            key=lambda record: (
                record["split"] == "not_in_official_paired_split",
                record["panel_count"],
                record["stitch_count"],
                record["sample_id"] or "",
            ),
        )
        selected.extend(candidates[:count])
    return selected


def extract_selected(archive: Path, output: Path, selected: list[dict]) -> tuple[int, int]:
    identifiers = {record["sample_id"] for record in selected}
    file_count = 0
    total_bytes = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if sample_id(member.name) not in identifiers:
                continue
            relative = safe_relative_path(member.name)
            target = (output / relative).resolve()
            if output.resolve() not in target.parents and target != output.resolve():
                raise ValueError(f"archive member escaped output root: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot extract {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            file_count += 1
            total_bytes += member.size
    return file_count, total_bytes


def extract_all(archive: Path, output: Path) -> tuple[int, int]:
    """Safely extract all regular files and write a completion receipt."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    file_count = 0
    total_bytes = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            relative = safe_relative_path(member.name)
            target = (output / relative).resolve()
            if target != output and output not in target.parents:
                raise ValueError(f"archive member escaped output root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not accepted in dataset archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member type: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot extract {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            file_count += 1
            total_bytes += member.size
    receipt = {
        "schema_version": "1.0",
        "status": "PASS_COMPLETE_SAFE_EXTRACTION",
        "archive_sha256": sha256(archive),
        "file_count": file_count,
        "extracted_bytes": total_bytes,
    }
    (output / "_extraction_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return file_count, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one official GarmentCodeData v2 archive and extract a bounded balanced subset.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/garmentcode_v2/garments_5000_0/default_body/data.tar.gz"),
    )
    parser.add_argument("--split", type=Path, default=Path("data/raw/garmentcode_v2/metadata/official_split.json"))
    parser.add_argument("--extract-per-category", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_balanced_subset"))
    parser.add_argument("--extract-all", action="store_true")
    parser.add_argument("--full-output", type=Path, default=Path("data/processed/garmentcode_v2/batch_0_full"))
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_v2_batch_0_catalog.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/garmentcode_v2_batch_0.json"))
    args = parser.parse_args()

    if args.archive.stat().st_size != EXPECTED_BATCH_0_BYTES:
        raise SystemExit(f"archive is incomplete: {args.archive.stat().st_size} != {EXPECTED_BATCH_0_BYTES}")
    summary, records = inspect(args.archive, args.split)
    selected = select_balanced(records, args.extract_per_category)
    extracted_files, extracted_bytes = (0, 0)
    if selected:
        extracted_files, extracted_bytes = extract_selected(args.archive, args.output, selected)

    full_extracted_files, full_extracted_bytes = (0, 0)
    if args.extract_all:
        full_extracted_files, full_extracted_bytes = extract_all(args.archive, args.full_output)

    archive_digest = sha256(args.archive)
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(
        json.dumps(
            {
                "dataset": "GarmentCodeData v2",
                "doi": "https://doi.org/10.3929/ethz-b-000690432",
                "license": "CC BY 4.0",
                "batch": "garments_5000_0/default_body",
                "summary": summary,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "1.0",
        "dataset": "GarmentCodeData v2",
        "doi": "https://doi.org/10.3929/ethz-b-000690432",
        "official_source": SOURCE_URL,
        "license": "CC BY 4.0",
        "batch": "garments_5000_0/default_body",
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_digest,
        "integrity": "PASS_FULL_TAR_GZIP_SCAN",
        **summary,
        "balanced_subset_per_category": args.extract_per_category,
        "selected_records": [{key: value for key, value in record.items() if key != "specification_member"} for record in selected],
        "extracted_file_count": extracted_files,
        "extracted_bytes": extracted_bytes,
        "full_extraction_performed": bool(args.extract_all),
        "full_extracted_file_count": full_extracted_files,
        "full_extracted_bytes": full_extracted_bytes,
        "large_download_performed": True,
        "additional_batch_downloaded": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "selected": len(selected),
                "extracted_bytes": extracted_bytes,
                "full_extracted_files": full_extracted_files,
                "full_extracted_bytes": full_extracted_bytes,
                "sha256": archive_digest,
            }
        )
    )


if __name__ == "__main__":
    main()
