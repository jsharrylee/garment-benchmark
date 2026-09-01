from __future__ import annotations

from dataclasses import replace

import numpy as np

from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument
from benchmark.pattern_pipeline.validation import validate_pattern

from .data import loop_features, normalize_loop


def panel_unique_nodes(panel: Panel) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    nodes: list[tuple[float, float]] = []
    spans = []
    for edge in panel.edges:
        start = len(nodes)
        nodes.extend(edge.points[:-1])
        spans.append((start, len(nodes)))
    return np.asarray(nodes, dtype=np.float32), tuple(spans)


def rebuild_panel(panel: Panel, nodes: np.ndarray, spans: tuple[tuple[int, int], ...]) -> Panel:
    edges = []
    for edge_index, (edge, (start, end)) in enumerate(zip(panel.edges, spans, strict=True)):
        following_start = spans[(edge_index + 1) % len(spans)][0]
        points = np.concatenate((nodes[start:end], nodes[following_start : following_start + 1]), axis=0)
        edges.append(replace(edge, points=tuple((float(x), float(y)) for x, y in points)))
    return replace(panel, edges=tuple(edges))


def predict_panel(model, panel: Panel, device: str) -> tuple[np.ndarray, dict]:
    import torch

    nodes, spans = panel_unique_nodes(panel)
    normalized, center, scale = normalize_loop(nodes)
    maximum_nodes = int(model.repair_config["maximum_nodes"])
    if len(nodes) > maximum_nodes:
        raise ValueError(f"panel {panel.id} has {len(nodes)} nodes; model limit is {maximum_nodes}")
    features = np.zeros((1, maximum_nodes, 10), dtype=np.float32)
    valid = np.zeros((1, maximum_nodes), dtype=bool)
    features[0, : len(nodes)] = loop_features(normalized)
    valid[0, : len(nodes)] = True
    with torch.inference_mode():
        predicted = model(
            torch.from_numpy(features).to(device),
            torch.from_numpy(valid).to(device),
        )[0, : len(nodes)].float().cpu().numpy()
    restored = predicted * scale + center
    displacement = np.linalg.norm(restored - nodes, axis=1)
    diagonal = max(float(np.linalg.norm(np.ptp(nodes, axis=0))), 1e-6)
    return restored, {
        "node_count": len(nodes),
        "mean_displacement_ratio": float(displacement.mean() / diagonal),
        "max_displacement_ratio": float(displacement.max() / diagonal),
        "spans": spans,
    }


def _validation_rank(document: PatternDocument, displacement: float) -> tuple[float, ...]:
    report = validate_pattern(document)
    intersection_count = sum(
        int(issue.value or 0) for issue in report.issues if issue.code == "SELF_INTERSECTION"
    )
    return (
        float(report.metrics["error_count"]),
        float(intersection_count),
        float(report.metrics["max_closure_gap_cm"]),
        float(displacement),
    )


def repair_target_panels(document: PatternDocument) -> set[str]:
    report = validate_pattern(document)
    panel_ids = {panel.id for panel in document.panels}
    edge_to_panel = {edge.id: panel.id for panel in document.panels for edge in panel.edges}
    targets: set[str] = set()
    for issue in report.issues:
        if issue.severity != "error":
            continue
        if issue.subject in panel_ids:
            targets.add(issue.subject)
        for edge_id, panel_id in edge_to_panel.items():
            if edge_id in issue.subject:
                targets.add(panel_id)
    return targets


def repair_document(
    model,
    document: PatternDocument,
    device: str,
    strengths: tuple[float, ...],
    maximum_passes: int = 3,
) -> tuple[PatternDocument, dict]:
    if maximum_passes < 1:
        raise ValueError("maximum_passes must be positive")
    original_report = validate_pattern(document)
    original_rank = _validation_rank(document, 0.0)
    selected, displacement, selected_rank = document, 0.0, original_rank
    candidate_records = []
    accepted_passes = 0
    for pass_index in range(1, maximum_passes + 1):
        target_panels = repair_target_panels(selected)
        if not target_panels:
            break
        panel_predictions = {}
        panel_metadata = {}
        for panel in selected.panels:
            if panel.id not in target_panels:
                continue
            predicted, metadata = predict_panel(model, panel, device)
            panel_predictions[panel.id] = predicted
            panel_metadata[panel.id] = metadata
        candidates = []
        for strength in strengths:
            panels = []
            incremental_displacements = []
            for panel in selected.panels:
                nodes, spans = panel_unique_nodes(panel)
                if panel.id not in target_panels:
                    panels.append(panel)
                    continue
                predicted = panel_predictions[panel.id]
                blended = nodes + float(strength) * (predicted - nodes)
                panels.append(rebuild_panel(panel, blended, spans))
                incremental_displacements.append(panel_metadata[panel.id]["max_displacement_ratio"] * float(strength))
            cumulative_displacement = displacement + max(incremental_displacements, default=0.0)
            candidate = replace(
                selected,
                pattern_id=f"{document.pattern_id}_repair_p{pass_index}_{strength:.2f}",
                generator=f"{document.generator} + PatternRepairNet",
                panels=tuple(panels),
                annotations={
                    **document.annotations,
                    "learned_pattern_repair": True,
                    "topology_preserved": True,
                    "template_retrieval": False,
                    "repair_pass": pass_index,
                    "repair_strength": float(strength),
                },
            )
            rank = _validation_rank(candidate, cumulative_displacement)
            candidates.append((candidate, cumulative_displacement, rank))
            candidate_records.append(
                {
                    "pass": pass_index,
                    "strength": float(strength),
                    "target_panels": sorted(target_panels),
                    "rank": list(rank),
                }
            )
        pass_selected, pass_displacement, pass_rank = min(candidates, key=lambda item: item[2])
        if pass_rank >= selected_rank:
            break
        selected, displacement, selected_rank = pass_selected, pass_displacement, pass_rank
        accepted_passes += 1
        if selected_rank[0] == 0:
            break
    improved = selected_rank < original_rank
    receipt = {
        "model": "PatternRepairNet",
        "input_pattern_id": document.pattern_id,
        "output_pattern_id": selected.pattern_id,
        "topology_preserved": True,
        "panel_count_before": len(document.panels),
        "panel_count_after": len(selected.panels),
        "edge_count_before": sum(len(panel.edges) for panel in document.panels),
        "edge_count_after": sum(len(panel.edges) for panel in selected.panels),
        "stitch_count_before": len(document.stitches),
        "stitch_count_after": len(selected.stitches),
        "template_retrieval": False,
        "nearest_pattern_selection": False,
        "improved": improved,
        "original_validation": original_report.to_dict(),
        "selected_validation": validate_pattern(selected).to_dict(),
        "original_rank": list(original_rank),
        "selected_rank": list(selected_rank),
        "max_displacement_ratio": displacement,
        "maximum_passes": maximum_passes,
        "accepted_passes": accepted_passes,
        "candidate_ranks": candidate_records,
    }
    return selected, receipt
