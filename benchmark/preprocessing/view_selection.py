from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelectedView:
    camera: str
    source: Path
    bbox_xyxy: tuple[int, int, int, int]


def resolve_selected_views(root: Path, sample: dict) -> list[SelectedView]:
    views: list[SelectedView] = []
    for item in sample["views"]:
        sequence = f'{sample["sequence"]}{item["sequence_suffix"]}'
        source = root / sample["scene"] / sequence / "rgb" / sample["frame"]
        views.append(SelectedView(item["camera"], source, tuple(item["bbox_xyxy"])))
    return views


def validate_selected_views(views: list[SelectedView]) -> None:
    cameras = [view.camera for view in views]
    if cameras != ["CAM000", "CAM001", "CAM002", "CAM003"]:
        raise ValueError(f"Unexpected camera order: {cameras}")
    missing = [str(view.source) for view in views if not view.source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing selected SynBody views: {missing}")
