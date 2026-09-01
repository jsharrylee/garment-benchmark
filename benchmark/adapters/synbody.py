from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


CAMERA_SUFFIXES = ("", "_CAM001", "_CAM002", "_CAM003")


@dataclass(frozen=True)
class SynBodyBundle:
    scene: str
    sequence: str
    frame: str
    views: tuple[Path, Path, Path, Path]

    @property
    def job_id(self) -> str:
        return f"synbody:{self.scene}:{self.sequence}:{self.frame}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_bundles(root: Path) -> list[SynBodyBundle]:
    """Find synchronized frames present in all four SynBody camera folders."""
    bundles: list[SynBodyBundle] = []
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        names = {path.name for path in scene.iterdir() if path.is_dir()}
        bases = sorted(name for name in names if not any(name.endswith(suffix) for suffix in CAMERA_SUFFIXES[1:]))
        for base in bases:
            camera_dirs = [scene / f"{base}{suffix}" / "rgb" for suffix in CAMERA_SUFFIXES]
            if not all(path.is_dir() for path in camera_dirs):
                continue
            per_camera = [{path.name: path for path in directory.glob("*.jpeg")} for directory in camera_dirs]
            common = sorted(set.intersection(*(set(mapping) for mapping in per_camera)))
            for frame in common:
                bundles.append(SynBodyBundle(scene.name, base, frame, tuple(mapping[frame] for mapping in per_camera)))
    return bundles


def validate_bundle(bundle: SynBodyBundle) -> dict:
    dimensions: list[list[int]] = []
    hashes: list[str] = []
    for path in bundle.views:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dimensions.append([image.width, image.height])
        hashes.append(sha256(path))
    return {
        "valid": len(set(hashes)) == 4 and len(set(map(tuple, dimensions))) == 1,
        "dimensions": dimensions,
        "sha256": hashes,
        "distinct_views": len(set(hashes)) == 4,
    }
