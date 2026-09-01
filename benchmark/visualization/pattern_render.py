from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.adapters.reweaver import order_panel_edges


def render_reweaver_patterns(npz_path: Path, output_path: Path, *, title: str) -> dict:
    archive = np.load(npz_path, allow_pickle=True)
    panels = archive["flatten_pred"].item()
    columns = min(5, max(1, int(np.ceil(np.sqrt(len(panels))))))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    colors = plt.get_cmap("tab20")
    total_edges = 0
    for axis in axes.flat:
        axis.set_axis_off()
        axis.set_aspect("equal")
    for slot, (panel_id, panel) in enumerate(sorted(panels.items())):
        axis = axes.flat[slot]
        edges, gaps = order_panel_edges(np.asarray(panel["edge_points"], dtype=np.float64))
        total_edges += len(edges)
        for index, edge in enumerate(edges):
            axis.plot(edge[:, 0], edge[:, 1], color=colors(index % 20), linewidth=2)
            axis.scatter(edge[0, 0], edge[0, 1], color="black", s=5)
        all_points = edges.reshape(-1, 2)
        span = np.ptp(all_points, axis=0)
        margin = max(float(span.max()) * 0.08, 0.02)
        axis.set_xlim(float(all_points[:, 0].min() - margin), float(all_points[:, 0].max() + margin))
        axis.set_ylim(float(all_points[:, 1].min() - margin), float(all_points[:, 1].max() + margin))
        axis.set_title(f"panel {panel_id} · {len(edges)} edges · gap {gaps.mean():.3f}", fontsize=9)
        axis.set_axis_on()
        axis.grid(alpha=0.15)
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, facecolor="white")
    plt.close(figure)
    return {"panel_count": len(panels), "edge_count": total_edges, "output": str(output_path)}
