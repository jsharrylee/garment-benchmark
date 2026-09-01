"""Compile representative frozen-test garments to Pattern DSL review boards.

This command never trains.  It reads GCD formal graphs and garment stitch
records, then either an optional panel/edge role mapping or a trained Pattern
DSL proposer checkpoint.  Checkpoint propositions are projected through the
symbolic grammar before they are attached to the exact source edges.  Original
panel images and four-view renders are never consumed.

Example::

    python -m benchmark.scripts.render_gcdv2_pattern_dsl_review \
      --count 3 \
      --roles-json artifacts/local_role_predictions.json

The output directory is under ``artifacts/`` by default and is therefore an
ignored local review artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.schema import EDGE_ROLES
from benchmark.gcdv2_exact.pattern_dsl import (
    CloseCommand,
    CurveCommand,
    LandmarkCommand,
    MoveCommand,
    NextCommand,
    PanelCommand,
    PatternProgram,
    RoleCommand,
    SewnToCommand,
    SharedEndpointCommand,
    compile_garment_record,
)
from benchmark.gcdv2_exact.pattern_dsl_learning import (
    build_pattern_dsl_model,
    validate_edge_feature_schema,
)
from benchmark.gcdv2_exact.pattern_dsl_solver import (
    SymbolicProjectionReport,
    symbolic_project_and_verify,
)


DEFAULT_INDEX = Path("artifacts/gcdv2_neurosymbolic_v1/garment_index.jsonl")
DEFAULT_OUTPUT = Path("artifacts/gcdv2_pattern_dsl_review")
DEFAULT_DSL_DATASET = Path("artifacts/gcdv2_pattern_dsl_v1/programs.npz")
DEFAULT_DSL_METADATA = Path("artifacts/gcdv2_pattern_dsl_v1/metadata.jsonl")
OP_COLORS = {
    "L": "#28a9b7",
    "Q": "#ec9b27",
    "C": "#e64b92",
    "A": "#64ad55",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _complexity(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row.get("panel_count", 0)), int(row.get("stitch_count", 0))


def select_representative_test_rows(
    rows: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select category-covering median-complexity test garments deterministically."""

    if count <= 0:
        raise ValueError("count must be positive")
    candidates = [dict(value) for value in rows if str(value.get("split")) == "test"]
    if not candidates:
        raise ValueError("garment index has no frozen-test records")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row.get("garment_category", "unspecified"))].append(row)
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        ordered = sorted(
            grouped[category],
            key=lambda value: (*_complexity(value), str(value["sample_id"])),
        )
        # Lower median is stable and always names an observed garment.
        selected.append(ordered[(len(ordered) - 1) // 2])
        if len(selected) == count:
            return selected
    selected_ids = {str(value["sample_id"]) for value in selected}
    remainder = sorted(
        (value for value in candidates if str(value["sample_id"]) not in selected_ids),
        key=lambda value: (
            sum(_complexity(value)),
            *_complexity(value),
            str(value.get("garment_category", "")),
            str(value["sample_id"]),
        ),
    )
    selected.extend(remainder[: max(0, count - len(selected))])
    return selected[:count]


def _role_payload(raw: Any, sample_id: str) -> dict[str, dict[str, str]]:
    """Normalize optional role labels/predictions to panel_uid -> edge_id -> role."""

    if raw is None:
        return {}
    if isinstance(raw, Mapping) and "samples" in raw:
        raw = raw["samples"].get(sample_id, {})
    elif isinstance(raw, Mapping) and sample_id in raw:
        raw = raw[sample_id]
    if isinstance(raw, Mapping) and "panels" in raw:
        raw = raw["panels"]
    if isinstance(raw, Mapping) and "edge_roles" in raw:
        raw = raw["edge_roles"]
    if not isinstance(raw, Mapping):
        raise ValueError("role payload must resolve to a panel mapping")
    result: dict[str, dict[str, str]] = {}
    for panel_id, values in raw.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"roles for panel {panel_id!r} must be a mapping")
        result[str(panel_id)] = {str(edge): str(role) for edge, role in values.items()}
    return result


def load_roles(path: Path | None) -> Any:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("roles JSON must contain an object")
    return raw


def _resolve_input_path(path: str | Path, *, relative_to: Path | None = None) -> Path:
    """Resolve a recorded artifact path without silently changing its meaning."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if relative_to is not None:
        relative = Path(relative_to) / candidate
        if relative.is_file():
            return relative
    raise FileNotFoundError(candidate)


def map_projected_edge_roles(
    metadata_row: Mapping[str, Any],
    projected_roles: np.ndarray,
    edge_valid: np.ndarray,
    *,
    role_names: Sequence[str] = EDGE_ROLES,
    metadata_directory: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Map padded model slots back to exact formal-graph panel/edge IDs.

    The packed neural arrays intentionally use integer panel and edge slots.
    Those slots are not stable CAD identifiers.  The dataset metadata fixes the
    panel order, while each referenced formal graph fixes its ordered ``edge_id``
    sequence.  Both are checked here before a semantic ROLE is allowed into a
    serialized Pattern DSL program.
    """

    roles = np.asarray(projected_roles, dtype=np.int64)
    valid = np.asarray(edge_valid, dtype=bool)
    if roles.shape != valid.shape or roles.ndim != 2:
        raise ValueError(
            "projected_roles and edge_valid must have identical [panels, edges] shapes"
        )
    names = tuple(str(value) for value in role_names)
    panels = metadata_row.get("panels", ())
    if not isinstance(panels, Sequence) or isinstance(panels, (str, bytes)):
        raise ValueError("DSL metadata row must contain an ordered panels array")
    if len(panels) > roles.shape[0]:
        raise ValueError("DSL metadata contains more panels than the model output")

    mapped: dict[str, dict[str, str]] = {}
    for panel_slot, panel in enumerate(panels):
        if not isinstance(panel, Mapping):
            raise ValueError(f"panel metadata slot {panel_slot} is not an object")
        graph_path = _resolve_input_path(
            str(panel["formal_graph_path"]), relative_to=metadata_directory
        )
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        curves = graph.get("curves", ())
        if not isinstance(curves, Sequence) or isinstance(curves, (str, bytes)):
            raise ValueError(f"formal graph curves must be an array: {graph_path}")
        panel_uid = str(panel["panel_uid"])
        if str(graph.get("panel_uid")) != panel_uid:
            raise ValueError(
                f"metadata/formal-graph panel mismatch: {panel_uid!r} != "
                f"{graph.get('panel_uid')!r}"
            )
        declared = int(panel.get("edge_count", len(curves)))
        valid_slots = np.flatnonzero(valid[panel_slot]).tolist()
        expected_slots = list(range(len(curves)))
        if declared != len(curves) or valid_slots != expected_slots:
            raise ValueError(
                f"metadata/formal-graph/model edge order mismatch for {panel_uid}: "
                f"declared={declared}, graph={len(curves)}, valid_slots={valid_slots}"
            )
        edge_roles: dict[str, str] = {}
        for edge_slot, curve in enumerate(curves):
            role_id = int(roles[panel_slot, edge_slot])
            if not 0 <= role_id < len(names):
                raise ValueError(
                    f"projected role {role_id} is invalid for {panel_uid}/slot {edge_slot}"
                )
            edge_id = str(curve["edge_id"])
            if edge_id in edge_roles:
                raise ValueError(f"duplicate formal-graph edge id: {panel_uid}/{edge_id}")
            edge_roles[edge_id] = names[role_id]
        mapped[panel_uid] = edge_roles

    padded_valid = valid[len(panels) :]
    if padded_valid.any():
        raise ValueError("model edge_valid contains panels absent from DSL metadata")
    return mapped


def map_projected_seams(
    metadata_row: Mapping[str, Any],
    report: SymbolicProjectionReport,
    edge_valid: np.ndarray,
    *,
    metadata_directory: Path | None = None,
) -> list[dict[str, Any]]:
    """Map solver seam slots to exact panel/edge IDs with provenance intact."""

    valid = np.asarray(edge_valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("edge_valid must have [panels, edges] shape")
    panels = metadata_row.get("panels", ())
    if not isinstance(panels, Sequence) or isinstance(panels, (str, bytes)):
        raise ValueError("DSL metadata row must contain an ordered panels array")
    graph_edges: list[tuple[str, tuple[str, ...]]] = []
    for panel_slot, panel in enumerate(panels):
        if not isinstance(panel, Mapping):
            raise ValueError(f"panel metadata slot {panel_slot} is not an object")
        graph_path = _resolve_input_path(
            str(panel["formal_graph_path"]), relative_to=metadata_directory
        )
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        panel_uid = str(panel["panel_uid"])
        if str(graph.get("panel_uid")) != panel_uid:
            raise ValueError(
                f"metadata/formal-graph panel mismatch: {panel_uid!r} != "
                f"{graph.get('panel_uid')!r}"
            )
        edges = tuple(str(value["edge_id"]) for value in graph.get("curves", ()))
        valid_slots = np.flatnonzero(valid[panel_slot]).tolist()
        if valid_slots != list(range(len(edges))):
            raise ValueError(f"model/formal-graph edge order mismatch for {panel_uid}")
        graph_edges.append((panel_uid, edges))

    output: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    edge_capacity = valid.shape[1]
    for seam_index, pair in enumerate(report.seams.pairs):
        endpoints: list[tuple[str, str]] = []
        for reference in (pair.first, pair.second):
            panel_index = int(reference.panel_index)
            edge_index = int(reference.edge_index)
            if not 0 <= panel_index < len(graph_edges):
                raise ValueError(f"predicted seam references unknown panel slot {panel_index}")
            panel_uid, edges = graph_edges[panel_index]
            if not 0 <= edge_index < len(edges) or not valid[panel_index, edge_index]:
                raise ValueError(
                    f"predicted seam references unknown edge slot p{panel_index}:e{edge_index}"
                )
            expected_flat = panel_index * edge_capacity + edge_index
            if int(reference.flat_index) != expected_flat:
                raise ValueError(
                    f"predicted seam flat-index mismatch: {reference.flat_index} != {expected_flat}"
                )
            endpoint = (panel_uid, edges[edge_index])
            if endpoint in seen_edges:
                raise ValueError(f"predicted seam reuses an edge: {endpoint[0]}/{endpoint[1]}")
            endpoints.append(endpoint)
        seen_edges.update(endpoints)
        output.append(
            {
                "seam_id": f"predicted_seam_{seam_index:03d}",
                "first_panel_id": endpoints[0][0],
                "first_edge_id": endpoints[0][1],
                "second_panel_id": endpoints[1][0],
                "second_edge_id": endpoints[1][1],
                "score": float(pair.score),
                "provenance": "CHECKPOINT_SYMBOLIC_PROJECTION",
            }
        )
    return output


def replace_source_seams_with_predictions(
    program: PatternProgram,
    mapped_seams: Sequence[Mapping[str, Any]],
) -> PatternProgram:
    """Return a program containing predicted seams and no source seam facts."""

    panels = {value.panel_id: value for value in program.panels}
    curves = {
        (value.panel_id, value.edge_id): value
        for value in program.commands
        if isinstance(value, CurveCommand)
    }
    commands = [
        value for value in program.commands if not isinstance(value, SewnToCommand)
    ]
    seen_edges: set[tuple[str, str]] = set()
    for seam in mapped_seams:
        first = (str(seam["first_panel_id"]), str(seam["first_edge_id"]))
        second = (str(seam["second_panel_id"]), str(seam["second_edge_id"]))
        for endpoint in (first, second):
            if endpoint not in curves or endpoint[0] not in panels:
                raise ValueError(
                    f"mapped predicted seam references unknown edge: {endpoint[0]}/{endpoint[1]}"
                )
            if endpoint in seen_edges:
                raise ValueError(
                    f"mapped predicted seam reuses edge: {endpoint[0]}/{endpoint[1]}"
                )
        seen_edges.update((first, second))
        first_length = curves[first].length_ratio * panels[first[0]].panel_scale_cm
        second_length = curves[second].length_ratio * panels[second[0]].panel_scale_cm
        score = float(seam["score"])
        commands.append(
            SewnToCommand(
                seam_id=str(seam["seam_id"]),
                first_panel_id=first[0],
                first_edge_id=first[1],
                second_panel_id=second[0],
                second_edge_id=second[1],
                length_ratio_a_over_b=first_length / second_length,
                source_annotations=(
                    str(seam.get("provenance", "CHECKPOINT_SYMBOLIC_PROJECTION")),
                    f"score={score:.8g}",
                ),
            )
        )
    projected = PatternProgram(tuple(commands), program.schema_version).with_derived_landmarks()
    verification = projected.verify()
    if not verification.valid:
        codes = ", ".join(
            value.code for value in verification.issues if value.severity == "error"
        )
        raise ValueError(f"predicted-seam Pattern DSL verification failed: {codes}")
    return projected


def project_proposer_outputs(
    metadata_row: Mapping[str, Any],
    role_logits: np.ndarray,
    seam_scores: np.ndarray,
    edge_valid: np.ndarray,
    allowed_transitions: np.ndarray,
    *,
    role_names: Sequence[str] = EDGE_ROLES,
    metadata_directory: Path | None = None,
    seam_threshold: float = 0.5,
    seam_top_k_per_edge: int | None = 16,
) -> tuple[dict[str, dict[str, str]], SymbolicProjectionReport]:
    """Project one proposer's scores and recover exact source edge references."""

    report = symbolic_project_and_verify(
        role_logits,
        seam_scores,
        edge_valid,
        allowed_transitions,
        role_names=role_names,
        seam_threshold=seam_threshold,
        seam_top_k_per_edge=seam_top_k_per_edge,
    )
    if not report.valid:
        codes = ", ".join(value.code for value in report.issues) or "UNKNOWN"
        raise ValueError(f"symbolic Pattern DSL projection failed: {codes}")
    mapped = map_projected_edge_roles(
        metadata_row,
        report.roles.projected_roles,
        edge_valid,
        role_names=role_names,
        metadata_directory=metadata_directory,
    )
    return mapped, report


def predict_pattern_dsl_roles(
    sample_ids: Sequence[str],
    *,
    checkpoint_path: Path,
    dataset_path: Path = DEFAULT_DSL_DATASET,
    metadata_path: Path = DEFAULT_DSL_METADATA,
    seam_top_k_per_edge: int | None = 16,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, Any]]]:
    """Run the trained vector-only proposer for named samples.

    Returned ROLE mappings have already passed symbolic projection and have
    been translated from padded neural slots to formal-graph identifiers.
    """

    import torch

    requested = [str(value) for value in sample_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("sample_ids must be unique")
    metadata_rows = read_jsonl(metadata_path)
    metadata_lookup: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(metadata_rows):
        sample_id = str(row["sample_id"])
        if sample_id in metadata_lookup:
            raise ValueError(f"duplicate sample in DSL metadata: {sample_id}")
        metadata_lookup[sample_id] = (index, row)
    missing = sorted(set(requested) - set(metadata_lookup))
    if missing:
        raise KeyError(f"selected samples absent from DSL metadata: {missing}")

    with np.load(dataset_path) as archive:
        required = ("edge_features", "edge_commands", "edge_valid", "panel_valid")
        absent = [key for key in required if key not in archive.files]
        if absent:
            raise ValueError(f"DSL dataset is missing arrays: {absent}")
        arrays = {key: archive[key] for key in required}
        if "edge_feature_schema" in archive.files:
            arrays["edge_feature_schema"] = archive["edge_feature_schema"]
    if len(metadata_rows) != len(arrays["edge_features"]):
        raise ValueError("DSL dataset and metadata row counts differ")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_edge_feature_schema(arrays, checkpoint)
    role_names = tuple(str(value) for value in checkpoint.get("edge_roles", EDGE_ROLES))
    if len(role_names) != len(EDGE_ROLES):
        raise ValueError("checkpoint edge-role head is incompatible with the current model")
    allowed = np.asarray(checkpoint["allowed_transitions"], dtype=bool)
    seam_threshold = float(checkpoint["seam_threshold"])
    model = build_pattern_dsl_model(width=int(checkpoint["width"]))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    role_payload: dict[str, dict[str, dict[str, str]]] = {}
    projection_payload: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for sample_id in requested:
            source, metadata_row = metadata_lookup[sample_id]
            features = torch.from_numpy(
                arrays["edge_features"][source].astype(np.float32)
            )[None].to(device)
            commands = torch.from_numpy(
                arrays["edge_commands"][source].astype(np.int64)
            )[None].to(device)
            edge_valid_tensor = torch.from_numpy(arrays["edge_valid"][source])[None].to(device)
            panel_valid = torch.from_numpy(arrays["panel_valid"][source])[None].to(device)
            prediction = model(features, commands, edge_valid_tensor, panel_valid)
            role_logits = prediction["edge_role_logits"][0].float().cpu().numpy()
            seam_scores = prediction["seam_logits"][0].sigmoid().float().cpu().numpy()
            mapped, report = project_proposer_outputs(
                metadata_row,
                role_logits,
                seam_scores,
                arrays["edge_valid"][source],
                allowed,
                role_names=role_names,
                metadata_directory=Path(metadata_path).parent,
                seam_threshold=seam_threshold,
                seam_top_k_per_edge=seam_top_k_per_edge,
            )
            role_payload[sample_id] = mapped
            projection_payload[sample_id] = {
                "source_index": int(source),
                "device": str(device),
                "checkpoint": Path(checkpoint_path).as_posix(),
                "mapped_seams": map_projected_seams(
                    metadata_row,
                    report,
                    arrays["edge_valid"][source],
                    metadata_directory=Path(metadata_path).parent,
                ),
                **report.to_dict(),
            }
    return role_payload, projection_payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "pattern"


def _program_maps(program: PatternProgram):
    panels = {value.panel_id: value for value in program.commands if isinstance(value, PanelCommand)}
    curves: defaultdict[str, dict[str, CurveCommand]] = defaultdict(dict)
    roles: dict[tuple[str, str], str] = {}
    moves: dict[str, MoveCommand] = {}
    next_relations: dict[tuple[str, str], str] = {}
    shared: dict[tuple[str, str, str], str] = {}
    landmarks: defaultdict[str, list[LandmarkCommand]] = defaultdict(list)
    seams: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for command in program.commands:
        if isinstance(command, CurveCommand):
            curves[command.panel_id][command.edge_id] = command
        elif isinstance(command, RoleCommand):
            roles[(command.panel_id, command.edge_id)] = command.role
        elif isinstance(command, MoveCommand):
            moves[command.panel_id] = command
        elif isinstance(command, NextCommand):
            next_relations[(command.panel_id, command.first_edge_id)] = command.second_edge_id
        elif isinstance(command, SharedEndpointCommand):
            shared[(command.panel_id, command.first_edge_id, command.second_edge_id)] = command.point_id
        elif isinstance(command, LandmarkCommand):
            landmarks[command.panel_id].append(command)
        elif isinstance(command, SewnToCommand):
            seams[(command.first_panel_id, command.first_edge_id)].append(command.seam_id)
            seams[(command.second_panel_id, command.second_edge_id)].append(command.seam_id)
        elif isinstance(command, CloseCommand):
            pass
    return panels, curves, roles, moves, next_relations, shared, landmarks, seams


def _edge_midpoint(points: np.ndarray) -> np.ndarray:
    if len(points) == 2:
        return points.mean(axis=0)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = cumulative[-1] * 0.5
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(points) - 2)
    fraction = (target - cumulative[index]) / max(lengths[index], 1e-12)
    return points[index] * (1.0 - fraction) + points[index + 1] * fraction


def render_program_review(
    program: PatternProgram,
    destination: Path,
    *,
    samples_per_curve: int = 65,
    seam_label: str = "SOURCE SEWN",
) -> Path:
    """Render one exact analytic DSL program to PNG or SVG by suffix."""

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "gcdv2-pattern-dsl-review-v1"
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    destination = Path(destination)
    if destination.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("review destination must end in .png or .svg")
    document = program.to_pattern_document(samples_per_curve=samples_per_curve)
    report = program.verify()
    panels, curves, roles, moves, next_relations, shared, landmarks, seams = _program_maps(program)
    document_panels = {value.id: value for value in document.panels}
    columns = min(4, max(1, len(panels)))
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.8 * columns, 4.8 * rows + 1.0),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        f"Pattern DSL · {program.pattern_id} · {program.category}\n"
        f"{report.metrics['panel_count']} panels · {report.metrics['edge_count']} edges · "
        f"{report.metrics['seam_count']} seams · symbolic {'PASS' if report.valid else 'FAIL'}",
        fontsize=14,
        fontweight="bold",
    )
    for panel_index, panel_id in enumerate(panels):
        axis = axes.flat[panel_index]
        panel = panels[panel_id]
        document_panel = document_panels[panel_id]
        edge_geometry = {value.id: np.asarray(value.points, dtype=float) for value in document_panel.edges}
        point_coordinates: dict[str, np.ndarray] = {}
        for edge_id, curve in curves[panel_id].items():
            values = edge_geometry[edge_id]
            point_coordinates[curve.start_point_id] = values[0]
            point_coordinates[curve.end_point_id] = values[-1]
            axis.plot(
                values[:, 0],
                values[:, 1],
                color=OP_COLORS[curve.op],
                linewidth=2.4,
                solid_capstyle="round",
            )
            midpoint = _edge_midpoint(values)
            labels = [f"{edge_id} {curve.op}"]
            role = roles.get((panel_id, edge_id))
            if role:
                labels.append(f"ROLE {role}")
            if seams.get((panel_id, edge_id)):
                labels.append(
                    f"{seam_label} " + ",".join(sorted(seams[(panel_id, edge_id)]))
                )
            axis.annotate(
                "\n".join(labels),
                midpoint,
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="#151515",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": OP_COLORS[curve.op], "alpha": 0.88},
            )
        move = moves[panel_id]
        move_xy = point_coordinates[move.point_id]
        axis.scatter(*move_xy, marker="s", s=48, color="#d62728", edgecolor="white", linewidth=0.7, zorder=6)
        axis.annotate(f"M {move.point_id}", move_xy, xytext=(5, -12), textcoords="offset points", fontsize=7, fontweight="bold", color="#b2182b")
        for (owner, first), second in sorted(next_relations.items()):
            if owner != panel_id:
                continue
            point_id = shared.get((panel_id, first, second))
            if point_id is None or point_id not in point_coordinates:
                continue
            point = point_coordinates[point_id]
            axis.scatter(*point, s=17, color="#202020", edgecolor="white", linewidth=0.45, zorder=5)
            axis.annotate(
                f"{point_id}\nNEXT {first}→{second}\nSHARED",
                point,
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.2,
                color="#333333",
            )
        for landmark in landmarks.get(panel_id, ()):
            if landmark.point_id not in point_coordinates:
                continue
            point = point_coordinates[landmark.point_id]
            axis.scatter(*point, marker="*", s=115, color="#ffe066", edgecolor="#111111", linewidth=0.8, zorder=7)
            axis.annotate(
                f"{landmark.name} ({landmark.point_id})",
                point,
                xytext=(6, -17),
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold",
                color="#7a4b00",
                bbox={"boxstyle": "round,pad=0.15", "fc": "#fff7cc", "ec": "#d6aa00", "alpha": 0.92},
            )
        axis.set_title(
            f"{panel_id}\n{panel.part} · {panel.surface} · {panel.side} · perimeter {panel.panel_scale_cm:.2f} cm",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.set_aspect("equal")
        axis.margins(0.13)
        axis.axis("off")
    for index in range(len(panels), rows * columns):
        axes.flat[index].axis("off")
    legend = "   ".join(f"{op}={name}" for op, name in (("L", "line"), ("Q", "quadratic"), ("C", "cubic"), ("A", "arc")))
    if report.metrics["landmark_count"]:
        semantic_note = "yellow star = derived FNP/BNP/SNP/SP"
    elif roles:
        semantic_note = "ROLE supplied; no applicable FNP/BNP/SNP/SP junction"
    else:
        semantic_note = "semantic ROLE input absent: no FNP/BNP/SNP/SP claimed"
    figure.text(0.5, 0.004, f"{legend}   ·   red square=M   ·   {semantic_note}", ha="center", fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Creator": "game-garment-benchmark Pattern DSL review"}
    figure.savefig(
        destination,
        dpi=160,
        facecolor="white",
        metadata=metadata,
    )
    plt.close(figure)
    return destination


def build_reviews(
    garment_index: Path,
    output_directory: Path,
    *,
    count: int = 3,
    roles: Any = None,
    projection_reports: Mapping[str, Mapping[str, Any]] | None = None,
    semantic_role_source: str | None = None,
    samples_per_curve: int = 65,
) -> dict[str, Any]:
    rows = read_jsonl(garment_index)
    selected = select_representative_test_rows(rows, count)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for row in selected:
        sample_id = str(row["sample_id"])
        garment_path = Path(str(row["garment_record_path"]))
        if not garment_path.is_file():
            garment_path = Path(garment_index).parent / garment_path
        record = json.loads(garment_path.read_text(encoding="utf-8"))
        supplied_roles = _role_payload(roles, sample_id)
        projection = (
            dict(projection_reports[sample_id])
            if projection_reports is not None and sample_id in projection_reports
            else None
        )
        program = compile_garment_record(record, edge_roles=supplied_roles)
        source_seam_count = sum(
            isinstance(value, SewnToCommand) for value in program.commands
        )
        predicted_seams = (
            projection.get("mapped_seams") if projection is not None else None
        )
        checkpoint_roles = semantic_role_source == "checkpoint_symbolic_projection"
        if checkpoint_roles and predicted_seams is None:
            raise ValueError(
                f"{sample_id}: checkpoint review lacks mapped predicted seams; "
                "refusing to display source seams as checkpoint output"
            )
        if predicted_seams is not None:
            if not isinstance(predicted_seams, Sequence) or isinstance(
                predicted_seams, (str, bytes)
            ):
                raise ValueError(f"{sample_id}: mapped_seams must be an array")
            program = replace_source_seams_with_predictions(program, predicted_seams)
            seam_label = "PREDICTED SEWN"
            seam_source = "checkpoint_symbolic_projection"
        else:
            seam_label = "SOURCE SEWN"
            seam_source = "garment_record"
        stem = _safe_name(sample_id)
        dsl_path = output_directory / f"{stem}.pattern.dsl"
        svg_path = output_directory / f"{stem}.review.svg"
        png_path = output_directory / f"{stem}.review.png"
        dsl_path.write_text(program.serialize(), encoding="utf-8")
        render_program_review(
            program,
            svg_path,
            samples_per_curve=samples_per_curve,
            seam_label=seam_label,
        )
        render_program_review(
            program,
            png_path,
            samples_per_curve=samples_per_curve,
            seam_label=seam_label,
        )
        verification = program.verify()
        artifacts: dict[str, Any] = {
            "dsl": {"file": dsl_path.name, "sha256": _sha256(dsl_path)},
            "svg": {"file": svg_path.name, "sha256": _sha256(svg_path)},
            "png": {"file": png_path.name, "sha256": _sha256(png_path)},
        }
        if projection is not None:
            projection_path = output_directory / f"{stem}.symbolic_projection.json"
            projection_path.write_text(
                json.dumps(projection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts["symbolic_projection"] = {
                "file": projection_path.name,
                "sha256": _sha256(projection_path),
            }
        results.append(
            {
                "sample_id": sample_id,
                "category": str(row.get("garment_category")),
                "selection_complexity": {
                    "panel_count": int(row.get("panel_count", 0)),
                    "stitch_count": int(row.get("stitch_count", 0)),
                },
                "semantic_roles_supplied": bool(supplied_roles),
                "semantic_role_source": (
                    semantic_role_source
                    or ("roles_json" if supplied_roles else "none")
                ),
                "seam_rendering": {
                    "source": seam_source,
                    "source_seam_count": int(source_seam_count),
                    "rendered_seam_count": int(verification.metrics["seam_count"]),
                    "source_seams_replaced": predicted_seams is not None,
                },
                "derived_landmark_count": int(verification.metrics["landmark_count"]),
                "symbolic_validation": verification.to_dict(),
                "proposal_projection": (
                    {
                        "valid": bool(projection["valid"]),
                        "metrics": projection.get("metrics", {}),
                    }
                    if projection is not None
                    else None
                ),
                "artifacts": artifacts,
            }
        )
    manifest = {
        "schema_version": "gcdv2-pattern-dsl-review/v1",
        "selection": "frozen-test category coverage, then median panel/stitch complexity; deterministic lower-median tie policy",
        "input_modality": (
            "formal graph invariant arrays plus checkpoint; source garment stitches "
            "excluded from rendered DSL; no panel PNG or 4-view input"
            if projection_reports is not None
            else "formal graph and source stitch constraints only; no panel PNG or 4-view input"
        ),
        "requested_count": int(count),
        "selected_count": len(results),
        "records": results,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile representative GCDv2 frozen-test formal graphs to coordinate-free Pattern DSL reviews."
    )
    parser.add_argument("--garment-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--roles-json",
        type=Path,
        help=(
            "Optional JSON mapping sample/panel_uid/edge_id to semantic role. "
            "Without it, no semantic landmarks are claimed."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Optional trained Pattern DSL proposer. Its ROLE/SEWN_TO scores are "
            "symbolically projected before the review is rendered."
        ),
    )
    parser.add_argument(
        "--dsl-dataset",
        type=Path,
        default=DEFAULT_DSL_DATASET,
        help="Packed invariant Pattern DSL arrays used by the checkpoint.",
    )
    parser.add_argument(
        "--dsl-metadata",
        type=Path,
        default=DEFAULT_DSL_METADATA,
        help="Metadata mapping neural panel/edge slots to exact formal graphs.",
    )
    parser.add_argument(
        "--seam-top-k",
        type=int,
        default=16,
        help="Maximum seam candidates retained per edge during symbolic projection.",
    )
    parser.add_argument("--samples-per-curve", type=int, default=65)
    args = parser.parse_args()
    if args.roles_json is not None and args.checkpoint is not None:
        parser.error("--roles-json and --checkpoint are mutually exclusive role sources")
    roles = load_roles(args.roles_json)
    projection_reports = None
    semantic_role_source = None
    if args.checkpoint is not None:
        rows = read_jsonl(args.garment_index)
        selected = select_representative_test_rows(rows, args.count)
        roles, projection_reports = predict_pattern_dsl_roles(
            [str(value["sample_id"]) for value in selected],
            checkpoint_path=args.checkpoint,
            dataset_path=args.dsl_dataset,
            metadata_path=args.dsl_metadata,
            seam_top_k_per_edge=args.seam_top_k,
        )
        semantic_role_source = "checkpoint_symbolic_projection"
    manifest = build_reviews(
        args.garment_index,
        args.output,
        count=args.count,
        roles=roles,
        projection_reports=projection_reports,
        semantic_role_source=semantic_role_source,
        samples_per_curve=args.samples_per_curve,
    )
    print(
        json.dumps(
            {
                "selected_count": manifest["selected_count"],
                "sample_ids": [value["sample_id"] for value in manifest["records"]],
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
