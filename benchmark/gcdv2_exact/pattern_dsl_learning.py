from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.drafting_semantics.dataset import read_records
from benchmark.drafting_semantics.schema import EDGE_ROLES, PANEL_ROLES


CATEGORIES = ("top", "pants", "skirt")
CURVE_COMMANDS = ("L", "Q", "C", "A")
LANDMARK_NAMES = ("FNP", "BNP", "SNP", "SP")
MAXIMUM_PANELS = 22
MAXIMUM_EDGES = 36
MAXIMUM_STITCHES = 62
EDGE_FEATURE_DIMENSION = 18
MASK_COMMAND = len(CURVE_COMMANDS)
# The trained v1 checkpoint was built directly from formal-graph records. Its
# last angular pair is the discontinuity between the previous end tangent and
# the current start tangent. Keep that contract named instead of silently
# changing the meaning of an existing checkpoint feature.
EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1 = "gcdv2-intrinsic-tangent-gap/v1"
# New command models should use the actual Pattern DSL payload. In that
# representation the last angular pair is the current chord's turn relative
# to the previous chord, exactly as serialized by CurveCommand.
EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2 = "gcd-pattern-dsl-chord-turn/v2"


def validate_edge_feature_schema(
    arrays: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> str:
    """Fail closed when a tensor corpus and checkpoint use different angle contracts."""

    array_value = arrays.get(
        "edge_feature_schema", np.asarray(EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1)
    )
    array_schema = str(np.asarray(array_value).item())
    checkpoint_schema = str(
        checkpoint.get("edge_feature_schema", EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1)
    )
    if array_schema != checkpoint_schema:
        raise ValueError(
            "Pattern DSL feature/checkpoint schema mismatch: "
            f"{array_schema!r} != {checkpoint_schema!r}"
        )
    return array_schema


def _angle_features(degrees: float) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    return math.cos(radians), math.sin(radians)


def invariant_edge_features(graph: Mapping[str, Any], index: int) -> np.ndarray:
    """Encode one analytic SVG-like command without absolute x/y coordinates."""
    curve = graph["curves"][index]
    points = graph["points"]
    start = np.asarray(points[index]["xy_cm"], np.float64)
    end = np.asarray(points[(index + 1) % len(points)]["xy_cm"], np.float64)
    chord = end - start
    chord_length = max(float(np.linalg.norm(chord)), 1e-8)
    perimeter = max(sum(float(value["length_cm"]) for value in graph["curves"]), 1e-8)
    direction = float(curve["chord_direction_deg_y_up"])
    start_relative = float(curve["start_tangent_deg_y_up"]) - direction
    end_relative = float(curve["end_tangent_deg_y_up"]) - direction
    previous = graph["curves"][(index - 1) % len(graph["curves"])]
    turn = float(curve["start_tangent_deg_y_up"]) - float(previous["end_tangent_deg_y_up"])
    controls = curve.get("parameters", {}).get("relative_controls_chord_frame", [])
    control_values = np.zeros(4, np.float32)
    for control_index, control in enumerate(controls[:2]):
        control_values[2 * control_index : 2 * control_index + 2] = control
    parameters = curve.get("parameters", {})
    radius_ratio = float(parameters.get("radius_cm", 0.0)) / chord_length
    start_cos, start_sin = _angle_features(start_relative)
    end_cos, end_sin = _angle_features(end_relative)
    turn_cos, turn_sin = _angle_features(turn)
    return np.asarray(
        [
            float(curve["length_cm"]) / perimeter,
            chord_length / perimeter,
            min(float(curve["length_cm"]) / chord_length, 20.0) / 20.0,
            start_cos, start_sin, end_cos, end_sin,
            *control_values.tolist(),
            min(math.log1p(max(radius_ratio, 0.0)) / math.log(101.0), 1.0),
            float(bool(parameters.get("large_arc", False))),
            1.0 if parameters.get("sweep_y_up", False) else -1.0,
            turn_cos, turn_sin,
            math.log1p(float(curve["length_cm"])) / 5.0,
            math.log1p(perimeter) / 7.0,
        ],
        np.float32,
    )


def _primitive_index(name: str) -> int:
    lookup = {"line": 0, "quadratic_bezier": 1, "cubic_bezier": 2, "circular_arc": 3}
    return lookup[str(name)]


def invariant_curve_command_features(command, panel_scale_cm: float) -> np.ndarray:
    """Tensorize one CurveCommand without consulting source geometry.

    This is the canonical DSL-v2 input path. Every value is read from the
    coordinate-free command payload plus the panel's separate physical scale;
    source ``xy_cm``, source identifiers, semantic labels, and stitch facts are
    neither accepted nor returned.

    The layout intentionally matches the existing 18-value model input so a
    future v2 corpus can reuse the architecture. It does not claim checkpoint
    compatibility with the tangent-gap v1 corpus because slots 14:16 have a
    different, now DSL-consistent, definition.
    """

    from benchmark.gcdv2_exact.pattern_dsl import CurveCommand

    if not isinstance(command, CurveCommand):
        raise TypeError("command must be a CurveCommand")
    scale = float(panel_scale_cm)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("panel_scale_cm must be finite and positive")
    control_values = np.zeros(4, np.float32)
    for control_index, control in enumerate(command.controls_chord_frame[:2]):
        control_values[2 * control_index : 2 * control_index + 2] = control
    radius_ratio = float(command.arc_radius_over_chord or 0.0)
    length_cm = float(command.length_ratio) * scale
    return np.asarray(
        [
            float(command.length_ratio),
            float(command.chord_ratio),
            min(float(command.length_ratio) / max(float(command.chord_ratio), 1e-8), 20.0)
            / 20.0,
            float(command.start_tangent_cos),
            float(command.start_tangent_sin),
            float(command.end_tangent_cos),
            float(command.end_tangent_sin),
            *control_values.tolist(),
            min(math.log1p(max(radius_ratio, 0.0)) / math.log(101.0), 1.0),
            float(bool(command.large_arc)),
            1.0 if bool(command.sweep_y_up) else -1.0,
            float(command.turn_cos),
            float(command.turn_sin),
            math.log1p(length_cm) / 5.0,
            math.log1p(scale) / 7.0,
        ],
        np.float32,
    )


def tensorize_pattern_program(program) -> dict[str, np.ndarray]:
    """Convert a verified Pattern DSL program to leakage-safe neural arrays.

    ``NEXT`` and ``M`` determine each panel's cyclic command order. The
    returned dictionary deliberately contains only the four tensors accepted
    by PatternDSLTransformer; IDs and proof/target facts stay outside the
    inference boundary.
    """

    from benchmark.gcdv2_exact.pattern_dsl import (
        CurveCommand,
        MoveCommand,
        NextCommand,
        PanelCommand,
    )

    report = program.verify()
    if not report.valid:
        raise ValueError("cannot tensorize an invalid Pattern DSL program")
    panels = [value for value in program.commands if isinstance(value, PanelCommand)]
    if not panels:
        raise ValueError("Pattern DSL program has no panels")
    curves: dict[str, dict[str, Any]] = {panel.panel_id: {} for panel in panels}
    moves: dict[str, Any] = {}
    next_edges: dict[tuple[str, str], str] = {}
    for command in program.commands:
        if isinstance(command, CurveCommand):
            curves[command.panel_id][command.edge_id] = command
        elif isinstance(command, MoveCommand):
            moves[command.panel_id] = command
        elif isinstance(command, NextCommand):
            next_edges[(command.panel_id, command.first_edge_id)] = command.second_edge_id

    ordered_panels: list[list[Any]] = []
    for panel in panels:
        panel_curves = curves[panel.panel_id]
        move = moves[panel.panel_id]
        candidates = [
            value for value in panel_curves.values() if value.start_point_id == move.point_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"{panel.panel_id}: cannot resolve one cycle start")
        ordered: list[Any] = []
        current = candidates[0]
        while current.edge_id not in {value.edge_id for value in ordered}:
            ordered.append(current)
            current = panel_curves[next_edges[(panel.panel_id, current.edge_id)]]
        if current.edge_id != ordered[0].edge_id or len(ordered) != len(panel_curves):
            raise ValueError(f"{panel.panel_id}: NEXT is not one closed cycle")
        ordered_panels.append(ordered)

    maximum_edges = max(len(value) for value in ordered_panels)
    edge_features = np.zeros(
        (len(panels), maximum_edges, EDGE_FEATURE_DIMENSION), dtype=np.float32
    )
    edge_commands = np.full(
        (len(panels), maximum_edges), MASK_COMMAND, dtype=np.int64
    )
    edge_valid = np.zeros((len(panels), maximum_edges), dtype=bool)
    for panel_index, (panel, ordered) in enumerate(zip(panels, ordered_panels, strict=True)):
        for edge_index, command in enumerate(ordered):
            edge_valid[panel_index, edge_index] = True
            edge_commands[panel_index, edge_index] = CURVE_COMMANDS.index(command.op)
            edge_features[panel_index, edge_index] = invariant_curve_command_features(
                command, panel.panel_scale_cm
            )
    return {
        "edge_features": edge_features,
        "edge_commands": edge_commands,
        "edge_valid": edge_valid,
        "panel_valid": np.ones(len(panels), dtype=bool),
    }


def build_program_arrays(
    garment_rows: Sequence[Mapping[str, Any]],
    records_path: Path,
    *,
    feature_schema: str = EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if feature_schema not in {
        EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1,
        EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2,
    }:
        raise ValueError(f"unsupported edge feature schema: {feature_schema}")
    records = {record.sample_id: record for record in read_records(records_path)}
    count = len(garment_rows)
    arrays: dict[str, np.ndarray] = {
        "categories": np.zeros(count, np.int8),
        "splits": np.zeros(count, np.int8),
        "panel_valid": np.zeros((count, MAXIMUM_PANELS), bool),
        "panel_roles": np.full((count, MAXIMUM_PANELS), -100, np.int8),
        "edge_valid": np.zeros((count, MAXIMUM_PANELS, MAXIMUM_EDGES), bool),
        "edge_commands": np.full((count, MAXIMUM_PANELS, MAXIMUM_EDGES), MASK_COMMAND, np.int8),
        "edge_features": np.zeros((count, MAXIMUM_PANELS, MAXIMUM_EDGES, EDGE_FEATURE_DIMENSION), np.float16),
        "edge_roles": np.full((count, MAXIMUM_PANELS, MAXIMUM_EDGES), -100, np.int8),
        "landmarks": np.full((count, MAXIMUM_PANELS, MAXIMUM_EDGES), -1, np.int8),
        "stitch_pairs": np.full((count, MAXIMUM_STITCHES, 4), -1, np.int16),
        "stitch_valid": np.zeros((count, MAXIMUM_STITCHES), bool),
        "edge_feature_schema": np.asarray(feature_schema),
    }
    split_lookup = {"train": 0, "validation": 1, "test": 2}
    metadata: list[dict[str, Any]] = []
    for garment_index, row in enumerate(garment_rows):
        garment = json.loads(Path(row["garment_record_path"]).read_text(encoding="utf-8"))
        sample_id = str(row["sample_id"])
        record = records.get(sample_id)
        # Geometry and stitches remain valid DSL facts even when the optional
        # derived semantic annotation corpus has no matching record.
        record_panels = {panel.id: panel for panel in record.panels} if record is not None else {}
        arrays["categories"][garment_index] = CATEGORIES.index(str(row["garment_category"]))
        arrays["splits"][garment_index] = split_lookup[str(row["split"])]
        panel_slots: dict[str, int] = {}
        panel_metadata = []
        for panel_slot, panel_row in enumerate(garment["panels"]):
            if panel_slot >= MAXIMUM_PANELS:
                raise ValueError(f"{sample_id} exceeds panel capacity")
            panel_id = str(panel_row["source_panel_id"])
            semantic_panel = record_panels.get(panel_id)
            graph = json.loads(Path(panel_row["formal_graph_path"]).read_text(encoding="utf-8"))
            if len(graph["curves"]) > MAXIMUM_EDGES:
                raise ValueError(f"{sample_id}/{panel_id} exceeds edge capacity")
            canonical = None
            if feature_schema == EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2:
                from benchmark.gcdv2_exact.pattern_dsl import compile_formal_graph

                program = compile_formal_graph(graph, pattern_id=sample_id)
                canonical = tensorize_pattern_program(program)
                if canonical["edge_features"].shape[0] != 1:
                    raise ValueError(f"{sample_id}/{panel_id}: expected one compiled panel")
                if int(canonical["edge_valid"][0].sum()) != len(graph["curves"]):
                    raise ValueError(f"{sample_id}/{panel_id}: DSL/graph edge counts differ")
            arrays["panel_valid"][garment_index, panel_slot] = True
            if semantic_panel is not None:
                arrays["panel_roles"][garment_index, panel_slot] = PANEL_ROLES.index(semantic_panel.role)
            semantic_edges = {edge.id: edge for edge in semantic_panel.edges} if semantic_panel is not None else {}
            panel_slots[str(panel_row["panel_uid"])] = panel_slot
            source_vertex_to_local = {
                int(point["source_vertex_index"]): local for local, point in enumerate(graph["points"])
            }
            for edge_slot, curve in enumerate(graph["curves"]):
                arrays["edge_valid"][garment_index, panel_slot, edge_slot] = True
                if canonical is None:
                    command = _primitive_index(curve["primitive"])
                    features = invariant_edge_features(graph, edge_slot)
                else:
                    command = int(canonical["edge_commands"][0, edge_slot])
                    features = canonical["edge_features"][0, edge_slot]
                    if command != _primitive_index(curve["primitive"]):
                        raise ValueError(
                            f"{sample_id}/{panel_id}/e{edge_slot}: DSL/graph primitive mismatch"
                        )
                arrays["edge_commands"][garment_index, panel_slot, edge_slot] = command
                arrays["edge_features"][garment_index, panel_slot, edge_slot] = features
                if semantic_panel is not None:
                    semantic_edge = semantic_edges.get(str(curve["source_edge_id"]))
                    if semantic_edge is not None:
                        arrays["edge_roles"][garment_index, panel_slot, edge_slot] = EDGE_ROLES.index(
                            semantic_edge.role
                        )
            for landmark in semantic_panel.landmarks if semantic_panel is not None else ():
                if (
                    landmark.name in LANDMARK_NAMES
                    and landmark.training_eligible
                    and landmark.vertex_index is not None
                    and int(landmark.vertex_index) in source_vertex_to_local
                ):
                    vertex = source_vertex_to_local[int(landmark.vertex_index)]
                    arrays["landmarks"][garment_index, panel_slot, vertex] = LANDMARK_NAMES.index(landmark.name)
            panel_metadata.append(
                {
                    "panel_id": panel_id,
                    "panel_uid": panel_row["panel_uid"],
                    "formal_graph_path": panel_row["formal_graph_path"],
                    "edge_count": len(graph["curves"]),
                    "semantic_supervision_available": semantic_panel is not None,
                }
            )
        for stitch_slot, stitch in enumerate(garment["stitch_constraints"]):
            if stitch_slot >= MAXIMUM_STITCHES:
                raise ValueError(f"{sample_id} exceeds stitch capacity")
            sides = stitch["sides"]
            first_panel, second_panel = panel_slots[sides[0]["panel_uid"]], panel_slots[sides[1]["panel_uid"]]
            first_edge, second_edge = int(str(sides[0]["edge_id"])[1:]), int(str(sides[1]["edge_id"])[1:])
            arrays["stitch_pairs"][garment_index, stitch_slot] = (
                first_panel, first_edge, second_panel, second_edge
            )
            arrays["stitch_valid"][garment_index, stitch_slot] = True
        metadata.append(
            {
                "sample_id": sample_id,
                "category": row["garment_category"],
                "split": row["split"],
                "garment_record_path": row["garment_record_path"],
                "edge_feature_schema": feature_schema,
                "panels": panel_metadata,
            }
        )
    return arrays, metadata


class PatternDSLArrayDataset:
    def __init__(self, arrays: Mapping[str, np.ndarray], indices: Sequence[int], *, mask_commands: bool) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, np.int64)
        self.mask_commands = mask_commands

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = int(self.indices[index])
        commands = self.arrays["edge_commands"][source].astype(np.int64)
        command_input = commands.copy()
        edge_valid = self.arrays["edge_valid"][source]
        command_targets = commands.copy()
        # Padded edge slots carry MASK_COMMAND in the packed arrays so they are
        # safe embedding inputs.  They are not, however, primitive labels.
        # Keeping the two meanings separate prevents padding from depressing
        # masked-command accuracy or entering a future unmasked loss.
        command_targets[~edge_valid] = -100
        command_mask = np.zeros_like(edge_valid)
        if self.mask_commands:
            command_mask = (np.random.random(edge_valid.shape) < 0.2) & edge_valid
        command_input[command_mask] = MASK_COMMAND
        return {
            "category": int(self.arrays["categories"][source]),
            "panel_valid": self.arrays["panel_valid"][source],
            "panel_roles": self.arrays["panel_roles"][source].astype(np.int64),
            "edge_valid": edge_valid,
            "commands": command_input,
            "command_targets": command_targets,
            "command_mask": command_mask,
            "features": self.arrays["edge_features"][source].astype(np.float32),
            "edge_roles": self.arrays["edge_roles"][source].astype(np.int64),
            "landmarks": self.arrays["landmarks"][source].astype(np.int64),
            "stitch_pairs": self.arrays["stitch_pairs"][source].astype(np.int64),
            "stitch_valid": self.arrays["stitch_valid"][source],
            "source": source,
        }


def build_pattern_dsl_model(width: int = 128, heads: int = 4, edge_layers: int = 2, garment_layers: int = 3):
    import torch

    class PatternDSLTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.command = torch.nn.Embedding(len(CURVE_COMMANDS) + 1, width)
            self.feature = torch.nn.Sequential(
                torch.nn.Linear(EDGE_FEATURE_DIMENSION, width), torch.nn.GELU(), torch.nn.Linear(width, width)
            )
            self.neighbour = torch.nn.Sequential(
                torch.nn.Linear(width * 3, width), torch.nn.GELU(), torch.nn.Linear(width, width)
            )
            edge_layer = torch.nn.TransformerEncoderLayer(
                width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu"
            )
            self.edge_encoder = torch.nn.TransformerEncoder(edge_layer, edge_layers, enable_nested_tensor=False)
            self.panel_seed = torch.nn.Parameter(torch.zeros(width))
            panel_layer = torch.nn.TransformerEncoderLayer(
                width, heads, width * 3, 0.1, batch_first=True, norm_first=True, activation="gelu"
            )
            self.garment_encoder = torch.nn.TransformerEncoder(panel_layer, garment_layers, enable_nested_tensor=False)
            self.category_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(CATEGORIES)))
            self.panel_role_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(PANEL_ROLES)))
            self.edge_role_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(EDGE_ROLES)))
            self.command_head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, len(CURVE_COMMANDS)))
            self.seam_query = torch.nn.Linear(width, width, bias=False)
            self.seam_key = torch.nn.Linear(width, width, bias=False)

        @staticmethod
        def _cyclic_neighbours(hidden, valid):
            batch, panels, edges, width = hidden.shape
            counts = valid.sum(-1).clamp_min(1)
            axis = torch.arange(edges, device=hidden.device)[None, None].expand(batch, panels, -1)
            previous = torch.remainder(axis - 1, counts[..., None])
            following = torch.remainder(axis + 1, counts[..., None])
            previous_hidden = torch.gather(hidden, 2, previous[..., None].expand(-1, -1, -1, width))
            following_hidden = torch.gather(hidden, 2, following[..., None].expand(-1, -1, -1, width))
            return previous_hidden, following_hidden

        def forward(self, features, commands, edge_valid, panel_valid):
            batch, panels, edges = commands.shape
            hidden = self.feature(features) + self.command(commands)
            previous, following = self._cyclic_neighbours(hidden, edge_valid)
            hidden = hidden + self.neighbour(torch.cat((previous, hidden, following), dim=-1))
            safe_edge_valid = edge_valid.clone()
            # PyTorch attention produces NaNs when every token in a sequence is
            # masked.  Give every empty/padded panel one private dummy token;
            # original edge_valid still excludes it from pooling and all losses.
            safe_edge_valid[:, :, 0] |= ~safe_edge_valid.any(-1)
            encoded = self.edge_encoder(
                hidden.reshape(batch * panels, edges, -1),
                src_key_padding_mask=~safe_edge_valid.reshape(batch * panels, edges),
            ).reshape(batch, panels, edges, -1)
            valid_float = edge_valid[..., None].to(encoded.dtype)
            panel = (encoded * valid_float).sum(2) / valid_float.sum(2).clamp_min(1)
            panel = panel + self.panel_seed
            safe_panel_valid = panel_valid.clone()
            safe_panel_valid[:, 0] |= ~safe_panel_valid.any(-1)
            panel = self.garment_encoder(panel, src_key_padding_mask=~safe_panel_valid)
            panel_float = panel_valid[..., None].to(panel.dtype)
            garment = (panel * panel_float).sum(1) / panel_float.sum(1).clamp_min(1)
            edge = encoded + panel[:, :, None] + garment[:, None, None]
            flat = edge.reshape(batch, panels * edges, -1)
            seam = torch.matmul(self.seam_query(flat), self.seam_key(flat).transpose(1, 2)) / math.sqrt(flat.shape[-1])
            seam = 0.5 * (seam + seam.transpose(1, 2))
            return {
                "category_logits": self.category_head(garment),
                "panel_role_logits": self.panel_role_head(panel),
                "edge_role_logits": self.edge_role_head(edge),
                "command_logits": self.command_head(edge),
                "seam_logits": seam,
                "edge_hidden": edge,
            }

    return PatternDSLTransformer()


def grammar_transition_matrix(edge_roles: np.ndarray, edge_valid: np.ndarray, splits: np.ndarray) -> np.ndarray:
    allowed = np.eye(len(EDGE_ROLES), dtype=bool)
    allowed[EDGE_ROLES.index("other"), :] = True
    allowed[:, EDGE_ROLES.index("other")] = True
    for garment in np.flatnonzero(splits == 0):
        for panel in range(edge_roles.shape[1]):
            count = int(edge_valid[garment, panel].sum())
            if count < 2:
                continue
            values = edge_roles[garment, panel, :count]
            for first, second in zip(values, np.roll(values, -1)):
                if first >= 0 and second >= 0:
                    allowed[int(first), int(second)] = True
                    allowed[int(second), int(first)] = True
    return allowed


__all__ = [
    "CATEGORIES", "CURVE_COMMANDS", "EDGE_FEATURE_DIMENSION",
    "EDGE_FEATURE_SCHEMA_DSL_CHORD_TURN_V2", "EDGE_FEATURE_SCHEMA_TANGENT_GAP_V1",
    "LANDMARK_NAMES", "MASK_COMMAND",
    "MAXIMUM_EDGES", "MAXIMUM_PANELS", "MAXIMUM_STITCHES", "PatternDSLArrayDataset",
    "build_pattern_dsl_model", "build_program_arrays", "grammar_transition_matrix",
    "invariant_curve_command_features", "invariant_edge_features", "tensorize_pattern_program",
    "validate_edge_feature_schema",
]
