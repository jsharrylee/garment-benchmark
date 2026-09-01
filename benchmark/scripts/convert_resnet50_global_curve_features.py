from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_global_feature_archive(
    source: Path,
    output: Path,
    manifest: Path,
    *,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    """Wrap four global ResNet vectors as one token per view.

    The conversion is deliberately lossless: no projection, pooling, or image
    recomputation occurs here.  The existing ``[N, 4, 2048]`` float16 array is
    exposed to the shared curve trainer as ``[N, 4, 1, 2048]``.  This keeps the
    target builder, split, decoder, losses, and optimization path identical to
    the spatial-FPN experiment while removing within-view spatial tokens.
    """

    source = Path(source)
    output = Path(output)
    manifest = Path(manifest)
    if not source.is_file():
        raise FileNotFoundError(f"global ResNet-50 feature archive not found: {source}")

    with np.load(source, allow_pickle=False) as archive:
        if "sample_ids" not in archive.files or "features" not in archive.files:
            raise ValueError("source archive must contain sample_ids and features")
        sample_ids = np.asarray(archive["sample_ids"])
        features = np.asarray(archive["features"])

    if features.ndim != 3 or features.shape[1] != 4:
        raise ValueError(
            "source global features must have shape [records, 4, channels]; "
            f"got {features.shape}"
        )
    if len(sample_ids) != len(features):
        raise ValueError(
            f"sample_ids/features length mismatch: {len(sample_ids)} != {len(features)}"
        )
    if len(set(str(value) for value in sample_ids.tolist())) != len(sample_ids):
        raise ValueError("sample_ids must be unique")

    # ``reshape`` inserts a singleton token axis without changing any value.
    token_features = features.reshape(features.shape[0], 4, 1, features.shape[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, sample_ids=sample_ids, features=token_features)

    payload: dict[str, Any] = {
        "schema_version": "gcdv2-four-view-resnet50-global-token-features-1.0",
        "record_count": int(features.shape[0]),
        "view_count": 4,
        "tokens_per_view": 1,
        "feature_dimension": int(features.shape[2]),
        "source_feature_shape": list(features.shape),
        "feature_shape": list(token_features.shape),
        "feature_dtype": str(token_features.dtype),
        "transformation": "features.reshape(N, 4, 1, C)",
        "numerical_values_changed": False,
        "images_recomputed": False,
        "network_download": False,
        "source_archive_sha256": _sha256(source),
        "output_sha256": _sha256(output),
    }
    if source_manifest is not None:
        source_manifest = Path(source_manifest)
        if not source_manifest.is_file():
            raise FileNotFoundError(f"source feature manifest not found: {source_manifest}")
        payload["source_manifest_sha256"] = _sha256(source_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Losslessly wrap existing four-view global ResNet-50 vectors as "
            "one token per view for the curve-formula ablation."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_pattern_semantics/"
            "resnet50_features.npz"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_pattern_semantics/"
            "resnet50_features_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_curve_parameters/"
            "resnet50_global_tokens.npz"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/drafting_semantics/multiview_curve_parameters/"
            "resnet50_global_tokens_manifest.json"
        ),
    )
    args = parser.parse_args()
    payload = convert_global_feature_archive(
        args.source,
        args.output,
        args.manifest,
        source_manifest=args.source_manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
