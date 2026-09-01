from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_value(design: dict, keys: tuple[str, ...], value: object) -> None:
    current = design
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]]["v"] = value


def _variants(default_design: dict, tshirt_design: dict) -> dict[str, tuple[str, dict]]:
    tshirt = copy.deepcopy(tshirt_design)
    sleeveless = copy.deepcopy(tshirt_design)
    _set_value(sleeveless, ("sleeve", "sleeveless"), True)

    pants = copy.deepcopy(default_design)
    _set_value(pants, ("meta", "upper"), None)
    _set_value(pants, ("meta", "wb"), "StraightWB")
    _set_value(pants, ("meta", "bottom"), "Pants")
    _set_value(pants, ("pants", "length"), 0.9)
    _set_value(pants, ("pants", "width"), 1.0)
    _set_value(pants, ("pants", "flare"), 1.0)

    shorts = copy.deepcopy(pants)
    _set_value(shorts, ("pants", "length"), 0.3)

    skirt = copy.deepcopy(default_design)
    _set_value(skirt, ("meta", "upper"), None)
    _set_value(skirt, ("meta", "wb"), "StraightWB")
    _set_value(skirt, ("meta", "bottom"), "Skirt2")
    _set_value(skirt, ("skirt", "length"), 0.6)

    return {
        "tshirt_short_sleeve": ("top", tshirt),
        "top_sleeveless": ("top", sleeveless),
        "pants_straight": ("pants", pants),
        "shorts_straight": ("shorts", shorts),
        "skirt_two_panel": ("skirt", skirt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny official GarmentCode structured anchor bank.")
    parser.add_argument("--repo", type=Path, default=Path("external/GarmentCode"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_v2/garmentcode_anchors"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/garmentcode_anchor_bank.json"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs.meta_garment import MetaGarment

    default_design = yaml.safe_load((repo / "assets/design_params/default.yaml").read_text(encoding="utf-8"))["design"]
    tshirt_design = yaml.safe_load((repo / "assets/design_params/t-shirt.yaml").read_text(encoding="utf-8"))["design"]
    body = BodyParameters(repo / "assets/bodies/mean_all.yaml")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for anchor_id, (category, design) in _variants(default_design, tshirt_design).items():
        anchor_dir = args.output / anchor_id
        anchor_dir.mkdir(parents=True, exist_ok=True)
        piece = MetaGarment(anchor_id, body, design)
        piece.assert_non_empty()
        pattern = piece.assembly()
        self_intersecting = bool(piece.is_self_intersecting())
        pattern.serialize(
            anchor_dir,
            to_subfolder=False,
            with_3d=False,
            with_text=False,
            view_ids=False,
            with_printable=False,
        )
        design_path = anchor_dir / "design.yaml"
        design_path.write_text(yaml.safe_dump({"design": design}, sort_keys=False), encoding="utf-8")
        specification = next(anchor_dir.glob("*_specification.json"))
        raw = json.loads(specification.read_text(encoding="utf-8"))
        records.append(
            {
                "anchor_id": anchor_id,
                "category": category,
                "panel_count": len(raw["pattern"]["panels"]),
                "stitch_count": len(raw["pattern"]["stitches"]),
                "self_intersecting": self_intersecting,
                "specification_sha256": _sha256(specification),
                "source_design_sha256": _sha256(design_path),
            }
        )

    manifest = {
        "schema_version": "2.0",
        "mode": "retrieval_anchored_v2",
        "source": "maria-korosteleva/GarmentCode",
        "source_commit": commit,
        "source_code_license": "MIT",
        "generated_anchor_scope": "local_retrieval_and_simulation_evaluation",
        "record_count": len(records),
        "downloaded_source_bytes": sum(path.stat().st_size for path in repo.rglob("*") if path.is_file()),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
