"""Lossless GarmentCodeData v2 geometry labels and learning utilities.

Geometry rendering depends on ``svgpathtools`` while the learned raster and
intrinsic-graph stages do not.  Lazy exports keep those independent stages
usable in the CUDA/OpenCV environments without importing an unrelated SVG
dependency at package-import time.
"""

__all__ = ["build_exact_sample", "load_exact_label", "render_exact_overlay"]


def __getattr__(name: str):
    if name in __all__:
        from .geometry import build_exact_sample, load_exact_label, render_exact_overlay

        return {
            "build_exact_sample": build_exact_sample,
            "load_exact_label": load_exact_label,
            "render_exact_overlay": render_exact_overlay,
        }[name]
    raise AttributeError(name)
