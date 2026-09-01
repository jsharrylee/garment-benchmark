"""A coordinate-free symbolic DSL for GCD sewing-pattern graphs.

The raster lanes in this repository are useful for perception, but they are a
poor interchange format for exact geometry.  This module compiles the formal
GCD graph into an AlphaGeometry-style program whose operands are point, edge,
panel, semantic, and sewing relations.  No screen or panel-space absolute
``(x, y)`` coordinate is serialized.

Geometry is expressed in invariant frames:

* edge lengths and chords are ratios of panel perimeter;
* every chord direction is a turn relative to the preceding chord;
* tangents are relative to their own chord;
* Bezier controls are in the unit-chord frame; and
* an arc radius is divided by its chord length.

``panel_scale_cm`` is retained as a separate dimensional scale so a verified
program can still be materialized as a centimetre ``PatternDocument``.  It is
translation/rotation invariant and never identifies where the source panel
was packed in an image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, TypeAlias


SCHEMA_VERSION = "gcd-pattern-dsl/v1"
CURVE_OPS = ("L", "Q", "C", "A")
SEMANTIC_JUNCTIONS: dict[str, frozenset[str]] = {
    "FNP": frozenset(("center_front", "neckline")),
    "BNP": frozenset(("center_back", "neckline")),
    "SNP": frozenset(("neckline", "shoulder")),
    "SP": frozenset(("shoulder", "armhole")),
}


class PatternDSLError(ValueError):
    """Base class for parse, compile, and symbolic-verification failures."""


class PatternDSLParseError(PatternDSLError):
    pass


class PatternDSLCompileError(PatternDSLError):
    pass


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PatternDSLError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PatternDSLError(f"{label} must be a finite number")
    return result


def _text(value: Any, label: str) -> str:
    result = str(value)
    if not result.strip():
        raise PatternDSLError(f"{label} is required")
    return result


def _unit_pair(first: Any, second: Any, label: str) -> tuple[float, float]:
    sine, cosine = _finite(first, f"{label}.sin"), _finite(second, f"{label}.cos")
    norm = math.hypot(sine, cosine)
    if norm <= 1e-9:
        raise PatternDSLError(f"{label} direction is degenerate")
    return sine / norm, cosine / norm


@dataclass(frozen=True)
class PanelCommand:
    panel_id: str
    pattern_id: str
    category: str
    panel_scale_cm: float
    part: str = "unspecified"
    surface: str = "unspecified"
    side: str = "unspecified"
    source_panel_id: str | None = None
    op: Literal["PANEL"] = field(default="PANEL", init=False)

    def __post_init__(self) -> None:
        _text(self.panel_id, "panel_id")
        _text(self.pattern_id, "pattern_id")
        _text(self.category, "category")
        if _finite(self.panel_scale_cm, "panel_scale_cm") <= 0.0:
            raise PatternDSLError("panel_scale_cm must be positive")


@dataclass(frozen=True)
class MoveCommand:
    panel_id: str
    point_id: str
    op: Literal["M"] = field(default="M", init=False)


@dataclass(frozen=True)
class CurveCommand:
    """One analytic edge represented entirely in panel/chord-local values."""

    op: Literal["L", "Q", "C", "A"]
    panel_id: str
    edge_id: str
    start_point_id: str
    end_point_id: str
    length_ratio: float
    chord_ratio: float
    turn_sin: float
    turn_cos: float
    start_tangent_sin: float
    start_tangent_cos: float
    end_tangent_sin: float
    end_tangent_cos: float
    controls_chord_frame: tuple[tuple[float, float], ...] = ()
    arc_radius_over_chord: float | None = None
    large_arc: bool | None = None
    sweep_y_up: bool | None = None
    source_edge_index: int | None = None

    def __post_init__(self) -> None:
        if self.op not in CURVE_OPS:
            raise PatternDSLError(f"unsupported curve opcode: {self.op!r}")
        for label in ("panel_id", "edge_id", "start_point_id", "end_point_id"):
            _text(getattr(self, label), label)
        if _finite(self.length_ratio, "length_ratio") <= 0.0:
            raise PatternDSLError("length_ratio must be positive")
        if _finite(self.chord_ratio, "chord_ratio") <= 0.0:
            raise PatternDSLError("chord_ratio must be positive")
        turn = _unit_pair(self.turn_sin, self.turn_cos, "turn")
        object.__setattr__(self, "turn_sin", turn[0])
        object.__setattr__(self, "turn_cos", turn[1])
        start = _unit_pair(self.start_tangent_sin, self.start_tangent_cos, "start_tangent")
        end = _unit_pair(self.end_tangent_sin, self.end_tangent_cos, "end_tangent")
        object.__setattr__(self, "start_tangent_sin", start[0])
        object.__setattr__(self, "start_tangent_cos", start[1])
        object.__setattr__(self, "end_tangent_sin", end[0])
        object.__setattr__(self, "end_tangent_cos", end[1])
        controls = tuple(
            (_finite(value[0], "control.u"), _finite(value[1], "control.v"))
            for value in self.controls_chord_frame
        )
        object.__setattr__(self, "controls_chord_frame", controls)
        expected = {"L": 0, "Q": 1, "C": 2, "A": 0}[self.op]
        if len(controls) != expected:
            raise PatternDSLError(f"{self.op} requires {expected} chord-frame controls")
        if self.op == "A":
            if self.arc_radius_over_chord is None or _finite(
                self.arc_radius_over_chord, "arc_radius_over_chord"
            ) < 0.5:
                raise PatternDSLError("A requires radius/chord >= 0.5")
            if self.large_arc is None or self.sweep_y_up is None:
                raise PatternDSLError("A requires large_arc and sweep_y_up flags")
        elif any(value is not None for value in (self.arc_radius_over_chord, self.large_arc, self.sweep_y_up)):
            raise PatternDSLError(f"{self.op} cannot carry circular-arc parameters")

    @property
    def primitive(self) -> str:
        return {
            "L": "line",
            "Q": "quadratic_bezier",
            "C": "cubic_bezier",
            "A": "circular_arc",
        }[self.op]


@dataclass(frozen=True)
class CloseCommand:
    panel_id: str
    op: Literal["Z"] = field(default="Z", init=False)


@dataclass(frozen=True)
class RoleCommand:
    panel_id: str
    edge_id: str
    role: str
    op: Literal["ROLE"] = field(default="ROLE", init=False)


@dataclass(frozen=True)
class NextCommand:
    panel_id: str
    first_edge_id: str
    second_edge_id: str
    op: Literal["NEXT"] = field(default="NEXT", init=False)


@dataclass(frozen=True)
class SharedEndpointCommand:
    panel_id: str
    first_edge_id: str
    second_edge_id: str
    point_id: str
    op: Literal["SHARED_ENDPOINT"] = field(default="SHARED_ENDPOINT", init=False)


@dataclass(frozen=True)
class SewnToCommand:
    seam_id: str
    first_panel_id: str
    first_edge_id: str
    second_panel_id: str
    second_edge_id: str
    length_ratio_a_over_b: float
    source_annotations: tuple[Any, ...] = ()
    op: Literal["SEWN_TO"] = field(default="SEWN_TO", init=False)

    def __post_init__(self) -> None:
        if _finite(self.length_ratio_a_over_b, "length_ratio_a_over_b") <= 0.0:
            raise PatternDSLError("seam length ratio must be positive")


@dataclass(frozen=True)
class LandmarkCommand:
    panel_id: str
    name: str
    point_id: str
    derived: bool = True
    op: Literal["LANDMARK"] = field(default="LANDMARK", init=False)

    @property
    def base_name(self) -> str:
        return self.name.split("#", 1)[0]


Command: TypeAlias = (
    PanelCommand
    | MoveCommand
    | CurveCommand
    | CloseCommand
    | RoleCommand
    | NextCommand
    | SharedEndpointCommand
    | SewnToCommand
    | LandmarkCommand
)


_COMMAND_TYPES: dict[str, type[Command]] = {
    "PANEL": PanelCommand,
    "M": MoveCommand,
    "L": CurveCommand,
    "Q": CurveCommand,
    "C": CurveCommand,
    "A": CurveCommand,
    "Z": CloseCommand,
    "ROLE": RoleCommand,
    "NEXT": NextCommand,
    "SHARED_ENDPOINT": SharedEndpointCommand,
    "SEWN_TO": SewnToCommand,
    "LANDMARK": LandmarkCommand,
}


def _payload(command: Command) -> dict[str, Any]:
    result = asdict(command)
    result.pop("op", None)
    return result


@dataclass(frozen=True)
class VerificationIssue:
    severity: Literal["error", "warning"]
    code: str
    subject: str
    message: str


@dataclass(frozen=True)
class SymbolicVerificationReport:
    valid: bool
    issues: tuple[VerificationIssue, ...]
    derived_landmarks: tuple[LandmarkCommand, ...]
    metrics: Mapping[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [asdict(value) for value in self.issues],
            "derived_landmarks": [asdict(value) for value in self.derived_landmarks],
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class PatternProgram:
    commands: tuple[Command, ...]
    schema_version: str = SCHEMA_VERSION

    @property
    def panels(self) -> tuple[PanelCommand, ...]:
        return tuple(value for value in self.commands if isinstance(value, PanelCommand))

    @property
    def pattern_id(self) -> str:
        values = {value.pattern_id for value in self.panels}
        return next(iter(values)) if len(values) == 1 else "mixed"

    @property
    def category(self) -> str:
        values = {value.category for value in self.panels}
        return next(iter(values)) if len(values) == 1 else "mixed"

    def serialize(self) -> str:
        lines = [f"# {self.schema_version}"]
        for command in self.commands:
            lines.append(
                f"{command.op} "
                + json.dumps(_payload(command), sort_keys=True, separators=(",", ":"))
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def parse(cls, text: str) -> "PatternProgram":
        return parse_pattern_dsl(text)

    def verify(self) -> SymbolicVerificationReport:
        return verify_pattern_dsl(self)

    def with_derived_landmarks(self) -> "PatternProgram":
        existing = {
            (value.panel_id, value.name, value.point_id)
            for value in self.commands
            if isinstance(value, LandmarkCommand)
        }
        additions = tuple(
            value
            for value in self.verify().derived_landmarks
            if (value.panel_id, value.name, value.point_id) not in existing
        )
        return PatternProgram(self.commands + additions, self.schema_version)

    def to_pattern_document(self, *, samples_per_curve: int = 33):
        return materialize_pattern_document(self, samples_per_curve=samples_per_curve)


def parse_pattern_dsl(text: str) -> PatternProgram:
    commands: list[Command] = []
    schema_version = SCHEMA_VERSION
    for line_number, raw in enumerate(str(text).splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = line[1:].strip()
            if candidate:
                schema_version = candidate
            continue
        try:
            op, payload_text = line.split(None, 1)
        except ValueError as error:
            raise PatternDSLParseError(f"line {line_number}: command payload is missing") from error
        if op not in _COMMAND_TYPES:
            raise PatternDSLParseError(f"line {line_number}: unknown command {op!r}")
        try:
            payload = json.loads(payload_text)
            if not isinstance(payload, Mapping):
                raise TypeError("payload is not an object")
            values = dict(payload)
            if op in CURVE_OPS:
                values["op"] = op
                if "controls_chord_frame" in values:
                    values["controls_chord_frame"] = tuple(
                        tuple(item) for item in values["controls_chord_frame"]
                    )
            if op == "SEWN_TO" and "source_annotations" in values:
                values["source_annotations"] = tuple(values["source_annotations"])
            commands.append(_COMMAND_TYPES[op](**values))
        except (TypeError, ValueError, KeyError) as error:
            raise PatternDSLParseError(f"line {line_number}: invalid {op} payload: {error}") from error
    return PatternProgram(tuple(commands), schema_version)


def _angle_vector(degrees: float) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    return math.sin(radians), math.cos(radians)


def _angle_from_points(start: Sequence[float], end: Sequence[float]) -> float:
    return math.degrees(
        math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
    )


def _relative_controls(edge: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    parameters = edge.get("parameters", {})
    raw = parameters.get("relative_controls_chord_frame", ())
    return tuple((float(value[0]), float(value[1])) for value in raw)


def _role_for_edge(
    graph: Mapping[str, Any], edge: Mapping[str, Any], edge_roles: Mapping[str, str] | None
) -> str | None:
    edge_id = str(edge["edge_id"])
    source_id = str(edge.get("source_edge_id", ""))
    if edge_roles:
        for key in (edge_id, source_id, str(edge.get("source_edge_index", ""))):
            if key in edge_roles:
                return str(edge_roles[key])
    for key in ("role", "semantic_role", "edge_role"):
        if edge.get(key):
            return str(edge[key])
    raw_roles = graph.get("edge_roles", {})
    if isinstance(raw_roles, Mapping):
        for key in (edge_id, source_id):
            if key in raw_roles:
                return str(raw_roles[key])
    return None


def _compile_panel_commands(
    graph: Mapping[str, Any],
    *,
    pattern_id: str,
    category: str | None = None,
    edge_roles: Mapping[str, str] | None = None,
) -> list[Command]:
    panel_id = _text(graph.get("panel_uid") or graph.get("source_panel_id"), "panel id")
    curves = list(graph.get("curves", ()))
    points = {str(value["point_id"]): value for value in graph.get("points", ())}
    if len(curves) < 3:
        raise PatternDSLCompileError(f"{panel_id}: at least three curves are required")
    if not points:
        raise PatternDSLCompileError(f"{panel_id}: points are missing")
    lengths = [_finite(value["length_cm"], "length_cm") for value in curves]
    perimeter = sum(lengths)
    if perimeter <= 0.0:
        raise PatternDSLCompileError(f"{panel_id}: perimeter is degenerate")
    chords: list[float] = []
    directions: list[float] = []
    for edge in curves:
        try:
            start = points[str(edge["start_point_id"])]["xy_cm"]
            end = points[str(edge["end_point_id"])]["xy_cm"]
        except KeyError as error:
            raise PatternDSLCompileError(f"{panel_id}: curve has an unknown endpoint") from error
        chord = math.dist(start, end)
        if chord <= 1e-9:
            raise PatternDSLCompileError(f"{panel_id}/{edge['edge_id']}: zero chord")
        chords.append(chord)
        directions.append(_angle_from_points(start, end))
    weak = graph.get("weak_role", {}) if isinstance(graph.get("weak_role", {}), Mapping) else {}
    commands: list[Command] = [
        PanelCommand(
            panel_id=panel_id,
            pattern_id=str(pattern_id),
            category=str(category or graph.get("garment_category", "unspecified")),
            panel_scale_cm=perimeter,
            part=str(weak.get("part", "unspecified")),
            surface=str(weak.get("surface", "unspecified")),
            side=str(weak.get("side", "unspecified")),
            source_panel_id=str(graph.get("source_panel_id", panel_id)),
        ),
        MoveCommand(panel_id, str(curves[0]["start_point_id"])),
    ]
    for index, (edge, length, chord, direction) in enumerate(
        zip(curves, lengths, chords, directions, strict=True)
    ):
        primitive = str(edge["primitive"])
        try:
            op = {
                "line": "L",
                "quadratic_bezier": "Q",
                "cubic_bezier": "C",
                "circular_arc": "A",
            }[primitive]
        except KeyError as error:
            raise PatternDSLCompileError(f"unsupported primitive {primitive!r}") from error
        previous_direction = directions[(index - 1) % len(directions)]
        turn = _angle_vector(direction - previous_direction)
        start_tangent = _angle_vector(
            float(edge.get("start_tangent_deg_y_up", direction)) - direction
        )
        end_tangent = _angle_vector(
            float(edge.get("end_tangent_deg_y_up", direction)) - direction
        )
        parameters = edge.get("parameters", {})
        radius_ratio = None
        large_arc = sweep = None
        if op == "A":
            radius_ratio = float(parameters["radius_cm"]) / chord
            large_arc = bool(parameters["large_arc"])
            sweep = bool(parameters.get("sweep_y_up", parameters.get("right", False)))
        commands.append(
            CurveCommand(
                op=op,
                panel_id=panel_id,
                edge_id=str(edge["edge_id"]),
                start_point_id=str(edge["start_point_id"]),
                end_point_id=str(edge["end_point_id"]),
                length_ratio=length / perimeter,
                chord_ratio=chord / perimeter,
                turn_sin=turn[0],
                turn_cos=turn[1],
                start_tangent_sin=start_tangent[0],
                start_tangent_cos=start_tangent[1],
                end_tangent_sin=end_tangent[0],
                end_tangent_cos=end_tangent[1],
                controls_chord_frame=_relative_controls(edge),
                arc_radius_over_chord=radius_ratio,
                large_arc=large_arc,
                sweep_y_up=sweep,
                source_edge_index=(
                    int(edge["source_edge_index"])
                    if edge.get("source_edge_index") is not None
                    else None
                ),
            )
        )
        role = _role_for_edge(graph, edge, edge_roles)
        if role:
            commands.append(RoleCommand(panel_id, str(edge["edge_id"]), role))
    relations = list(graph.get("relations", ()))
    if relations:
        for relation in relations:
            predicate = str(relation.get("predicate"))
            arguments = [str(value) for value in relation.get("arguments", ())]
            if predicate == "NEXT" and len(arguments) == 2:
                commands.append(NextCommand(panel_id, arguments[0], arguments[1]))
            elif predicate == "SHARED_ENDPOINT" and len(arguments) == 3:
                commands.append(
                    SharedEndpointCommand(panel_id, arguments[0], arguments[1], arguments[2])
                )
    else:
        for first, second in zip(curves, curves[1:] + curves[:1]):
            commands.append(
                NextCommand(panel_id, str(first["edge_id"]), str(second["edge_id"]))
            )
            commands.append(
                SharedEndpointCommand(
                    panel_id,
                    str(first["edge_id"]),
                    str(second["edge_id"]),
                    str(first["end_point_id"]),
                )
            )
    commands.append(CloseCommand(panel_id))
    return commands


def compile_formal_graph(
    graph: Mapping[str, Any],
    *,
    pattern_id: str | None = None,
    edge_roles: Mapping[str, str] | None = None,
) -> PatternProgram:
    resolved_id = str(pattern_id or str(graph.get("panel_uid", "pattern")).split(":", 1)[0])
    program = PatternProgram(
        tuple(
            _compile_panel_commands(
                graph,
                pattern_id=resolved_id,
                edge_roles=edge_roles,
            )
        )
    ).with_derived_landmarks()
    report = program.verify()
    if not report.valid:
        codes = ", ".join(issue.code for issue in report.issues if issue.severity == "error")
        raise PatternDSLCompileError(f"compiled graph failed symbolic verification: {codes}")
    return program


def _load_graph(
    panel: Mapping[str, Any],
    *,
    base_dir: Path | None,
    graph_loader: Callable[[str], Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    inline = panel.get("formal_graph")
    if isinstance(inline, Mapping):
        return inline
    path_text = str(panel["formal_graph_path"])
    if graph_loader is not None:
        return graph_loader(path_text)
    path = Path(path_text)
    if not path.is_file() and base_dir is not None:
        path = Path(base_dir) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PatternDSLCompileError(f"formal graph must be an object: {path}")
    return raw


def compile_garment_record(
    record: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    graph_loader: Callable[[str], Mapping[str, Any]] | None = None,
    edge_roles: Mapping[str, Mapping[str, str]] | None = None,
) -> PatternProgram:
    pattern_id = _text(record.get("sample_id"), "sample_id")
    category = str(record.get("garment_category", "unspecified"))
    commands: list[Command] = []
    for panel in record.get("panels", ()):
        graph = _load_graph(
            panel,
            base_dir=Path(base_dir) if base_dir is not None else None,
            graph_loader=graph_loader,
        )
        panel_id = str(graph.get("panel_uid") or graph.get("source_panel_id"))
        supplied = edge_roles.get(panel_id) if edge_roles else None
        commands.extend(
            _compile_panel_commands(
                graph,
                pattern_id=pattern_id,
                category=category,
                edge_roles=supplied,
            )
        )
    for seam in record.get("stitch_constraints", ()):
        sides = list(seam.get("sides", ()))
        if len(sides) != 2:
            raise PatternDSLCompileError(
                f"{seam.get('constraint_id', 'seam')}: SEWN_TO needs two sides"
            )
        first, second = sides
        length_a = _finite(first.get("length_cm", 1.0), "first seam length")
        length_b = _finite(second.get("length_cm", 1.0), "second seam length")
        if min(length_a, length_b) <= 0.0:
            raise PatternDSLCompileError("seam lengths must be positive")
        commands.append(
            SewnToCommand(
                seam_id=str(seam.get("constraint_id", f"stitch_{len(commands)}")),
                first_panel_id=str(first["panel_uid"]),
                first_edge_id=str(first["edge_id"]),
                second_panel_id=str(second["panel_uid"]),
                second_edge_id=str(second["edge_id"]),
                length_ratio_a_over_b=length_a / length_b,
                source_annotations=tuple(seam.get("source_annotations", ())),
            )
        )
    program = PatternProgram(tuple(commands)).with_derived_landmarks()
    report = program.verify()
    if not report.valid:
        codes = ", ".join(issue.code for issue in report.issues if issue.severity == "error")
        raise PatternDSLCompileError(f"compiled garment failed symbolic verification: {codes}")
    return program


def _panel_maps(program: PatternProgram):
    panels = {value.panel_id: value for value in program.commands if isinstance(value, PanelCommand)}
    curves: dict[str, dict[str, CurveCommand]] = {key: {} for key in panels}
    roles: dict[tuple[str, str], str] = {}
    next_map: dict[tuple[str, str], str] = {}
    shared: list[SharedEndpointCommand] = []
    for command in program.commands:
        if isinstance(command, CurveCommand):
            curves.setdefault(command.panel_id, {})[command.edge_id] = command
        elif isinstance(command, RoleCommand):
            roles[(command.panel_id, command.edge_id)] = command.role
        elif isinstance(command, NextCommand):
            next_map[(command.panel_id, command.first_edge_id)] = command.second_edge_id
        elif isinstance(command, SharedEndpointCommand):
            shared.append(command)
    return panels, curves, roles, next_map, shared


def _derive_landmarks(
    curves: Mapping[str, Mapping[str, CurveCommand]],
    roles: Mapping[tuple[str, str], str],
) -> tuple[LandmarkCommand, ...]:
    output: list[LandmarkCommand] = []
    for panel_id, edges in curves.items():
        incident: dict[str, list[str]] = {}
        for edge in edges.values():
            incident.setdefault(edge.start_point_id, []).append(edge.edge_id)
            incident.setdefault(edge.end_point_id, []).append(edge.edge_id)
        found: dict[str, list[str]] = {name: [] for name in SEMANTIC_JUNCTIONS}
        for point_id, edge_ids in incident.items():
            point_roles = {
                roles[(panel_id, edge_id)]
                for edge_id in edge_ids
                if (panel_id, edge_id) in roles
            }
            for name, required in SEMANTIC_JUNCTIONS.items():
                if required.issubset(point_roles):
                    found[name].append(point_id)
        for name, point_ids in found.items():
            for index, point_id in enumerate(sorted(point_ids)):
                resolved_name = name if len(point_ids) == 1 else f"{name}#{index}"
                output.append(LandmarkCommand(panel_id, resolved_name, point_id, True))
    return tuple(output)


def verify_pattern_dsl(program: PatternProgram) -> SymbolicVerificationReport:
    issues: list[VerificationIssue] = []

    def error(code: str, subject: str, message: str) -> None:
        issues.append(VerificationIssue("error", code, subject, message))

    def warning(code: str, subject: str, message: str) -> None:
        issues.append(VerificationIssue("warning", code, subject, message))

    panel_commands = [value for value in program.commands if isinstance(value, PanelCommand)]
    panels: dict[str, PanelCommand] = {}
    for panel in panel_commands:
        if panel.panel_id in panels:
            error("DUPLICATE_PANEL", panel.panel_id, "panel id appears more than once")
        panels[panel.panel_id] = panel
    if not panels:
        error("NO_PANELS", "program", "at least one PANEL is required")
    pattern_ids = {panel.pattern_id for panel in panel_commands}
    categories = {panel.category for panel in panel_commands}
    if len(pattern_ids) > 1:
        error("MIXED_PATTERN_IDS", "program", "all panels must belong to one pattern")
    if len(categories) > 1:
        error("MIXED_CATEGORIES", "program", "all panels must have one category")

    curves: dict[str, dict[str, CurveCommand]] = {key: {} for key in panels}
    moves: dict[str, list[MoveCommand]] = {key: [] for key in panels}
    closes: dict[str, int] = {key: 0 for key in panels}
    roles: dict[tuple[str, str], str] = {}
    next_values: dict[tuple[str, str], list[str]] = {}
    shared_values: list[SharedEndpointCommand] = []
    declared_landmarks: list[LandmarkCommand] = []
    seams: list[SewnToCommand] = []
    for command in program.commands:
        panel_id = getattr(command, "panel_id", None)
        if panel_id is not None and not isinstance(command, PanelCommand) and panel_id not in panels:
            error("UNKNOWN_PANEL_REF", str(panel_id), f"{command.op} references an unknown panel")
            continue
        if isinstance(command, CurveCommand):
            if command.edge_id in curves[command.panel_id]:
                error("DUPLICATE_EDGE", f"{command.panel_id}/{command.edge_id}", "edge id is duplicated")
            curves[command.panel_id][command.edge_id] = command
        elif isinstance(command, MoveCommand):
            moves[command.panel_id].append(command)
        elif isinstance(command, CloseCommand):
            closes[command.panel_id] += 1
        elif isinstance(command, RoleCommand):
            key = (command.panel_id, command.edge_id)
            if key in roles and roles[key] != command.role:
                error("CONFLICTING_ROLE", "/".join(key), "one edge has conflicting roles")
            roles[key] = command.role
        elif isinstance(command, NextCommand):
            next_values.setdefault((command.panel_id, command.first_edge_id), []).append(command.second_edge_id)
        elif isinstance(command, SharedEndpointCommand):
            shared_values.append(command)
        elif isinstance(command, LandmarkCommand):
            declared_landmarks.append(command)
        elif isinstance(command, SewnToCommand):
            seams.append(command)

    closed_count = degree_two_count = 0
    all_points: dict[str, set[str]] = {}
    for panel_id, edges in curves.items():
        if len(edges) < 3:
            error("TOO_FEW_EDGES", panel_id, "a closed panel needs at least three edges")
        points = {
            value
            for edge in edges.values()
            for value in (edge.start_point_id, edge.end_point_id)
        }
        all_points[panel_id] = points
        degree = {point: 0 for point in points}
        for edge in edges.values():
            degree[edge.start_point_id] += 1
            degree[edge.end_point_id] += 1
        bad_degree = {point: value for point, value in degree.items() if value != 2}
        if bad_degree:
            error("DEGREE_NOT_TWO", panel_id, f"boundary point degrees are {bad_degree}")
        else:
            degree_two_count += 1
        if len(moves.get(panel_id, ())) != 1:
            error("MOVE_COUNT", panel_id, "a panel requires exactly one M command")
        elif moves[panel_id][0].point_id not in points:
            error("UNKNOWN_MOVE_POINT", panel_id, "M references an unknown point")
        if closes.get(panel_id, 0) != 1:
            error("CLOSE_COUNT", panel_id, "a panel requires exactly one Z command")
        for (next_panel, first), seconds in next_values.items():
            if next_panel != panel_id:
                continue
            if first not in edges or any(second not in edges for second in seconds):
                error("UNKNOWN_NEXT_EDGE", f"{panel_id}/{first}", "NEXT references an unknown edge")
            if len(seconds) != 1:
                error("NEXT_CARDINALITY", f"{panel_id}/{first}", "every edge needs exactly one successor")
        if set(first for (owner, first) in next_values if owner == panel_id) != set(edges):
            error("INCOMPLETE_NEXT", panel_id, "NEXT must define one successor for every edge")
        if edges and len(moves.get(panel_id, ())) == 1:
            start_point = moves[panel_id][0].point_id
            candidates = [edge.edge_id for edge in edges.values() if edge.start_point_id == start_point]
            if len(candidates) != 1:
                error("AMBIGUOUS_CYCLE_START", panel_id, "M point must start exactly one directed edge")
            else:
                visited: list[str] = []
                current = candidates[0]
                while current not in visited and current in edges:
                    visited.append(current)
                    successors = next_values.get((panel_id, current), ())
                    if len(successors) != 1:
                        break
                    following = successors[0]
                    if following in edges and edges[current].end_point_id != edges[following].start_point_id:
                        error("NEXT_ENDPOINT_MISMATCH", f"{panel_id}/{current}", "NEXT edges do not share directed endpoints")
                        break
                    current = following
                if current == candidates[0] and len(visited) == len(edges):
                    closed_count += 1
                else:
                    error("NOT_ONE_CLOSED_CYCLE", panel_id, "NEXT does not traverse every edge once and close")
        for (owner, edge_id), _role in roles.items():
            if owner == panel_id and edge_id not in edges:
                error("UNKNOWN_ROLE_EDGE", f"{owner}/{edge_id}", "ROLE references an unknown edge")

    shared_pairs: set[tuple[str, str, str]] = set()
    for relation in shared_values:
        relation_key = (
            relation.panel_id,
            relation.first_edge_id,
            relation.second_edge_id,
        )
        if relation_key in shared_pairs:
            error(
                "DUPLICATE_SHARED_ENDPOINT",
                "/".join(relation_key),
                "edge pair has more than one SHARED_ENDPOINT declaration",
            )
        shared_pairs.add(relation_key)
        panel_edges = curves.get(relation.panel_id, {})
        first = panel_edges.get(relation.first_edge_id)
        second = panel_edges.get(relation.second_edge_id)
        if first is None or second is None:
            error("UNKNOWN_SHARED_EDGE", relation.panel_id, "SHARED_ENDPOINT references an unknown edge")
        elif not (
            relation.point_id in {first.start_point_id, first.end_point_id}
            and relation.point_id in {second.start_point_id, second.end_point_id}
        ):
            error("BAD_SHARED_ENDPOINT", relation.point_id, "declared point is not shared by both edges")
    for (panel_id, first_edge_id), second_edge_ids in next_values.items():
        if len(second_edge_ids) == 1 and (
            panel_id,
            first_edge_id,
            second_edge_ids[0],
        ) not in shared_pairs:
            error(
                "MISSING_SHARED_ENDPOINT",
                f"{panel_id}/{first_edge_id}->{second_edge_ids[0]}",
                "every NEXT relation requires an explicit SHARED_ENDPOINT fact",
            )

    seam_ids: set[str] = set()
    for seam in seams:
        if seam.seam_id in seam_ids:
            error("DUPLICATE_SEAM", seam.seam_id, "seam id appears more than once")
        seam_ids.add(seam.seam_id)
        for panel_id, edge_id in (
            (seam.first_panel_id, seam.first_edge_id),
            (seam.second_panel_id, seam.second_edge_id),
        ):
            if panel_id not in curves:
                error("UNKNOWN_SEAM_PANEL", seam.seam_id, f"unknown seam panel {panel_id}")
            elif edge_id not in curves[panel_id]:
                error("UNKNOWN_SEAM_EDGE", seam.seam_id, f"unknown seam edge {panel_id}/{edge_id}")
        if max(seam.length_ratio_a_over_b, 1.0 / seam.length_ratio_a_over_b) > 1.25:
            warning("SEAM_LENGTH_MISMATCH", seam.seam_id, "seam length ratio exceeds 25%")

    derived = _derive_landmarks(curves, roles)
    derived_by_key = {(value.panel_id, value.base_name, value.point_id) for value in derived}
    for landmark in declared_landmarks:
        if landmark.panel_id not in all_points or landmark.point_id not in all_points[landmark.panel_id]:
            error("UNKNOWN_LANDMARK_POINT", f"{landmark.panel_id}/{landmark.name}", "landmark references an unknown point")
            continue
        if landmark.base_name in SEMANTIC_JUNCTIONS and (
            landmark.panel_id,
            landmark.base_name,
            landmark.point_id,
        ) not in derived_by_key:
            error(
                "SEMANTIC_JUNCTION_MISMATCH",
                f"{landmark.panel_id}/{landmark.name}",
                "landmark is not the required pair of incident semantic edges",
            )

    errors = sum(value.severity == "error" for value in issues)
    return SymbolicVerificationReport(
        valid=errors == 0,
        issues=tuple(issues),
        derived_landmarks=derived,
        metrics={
            "panel_count": len(panels),
            "edge_count": sum(len(value) for value in curves.values()),
            "point_count": sum(len(value) for value in all_points.values()),
            "closed_cycle_count": closed_count,
            "degree_two_panel_count": degree_two_count,
            "seam_count": len(seams),
            "landmark_count": len(derived),
            "error_count": errors,
            "warning_count": len(issues) - errors,
        },
    )


def _ordered_edges(
    panel_id: str,
    curves: Mapping[str, CurveCommand],
    next_map: Mapping[tuple[str, str], str],
    move: MoveCommand,
) -> list[CurveCommand]:
    candidates = [value for value in curves.values() if value.start_point_id == move.point_id]
    if len(candidates) != 1:
        raise PatternDSLError(f"{panel_id}: cannot resolve canonical cycle start")
    ordered = []
    current = candidates[0]
    while current.edge_id not in {value.edge_id for value in ordered}:
        ordered.append(current)
        current = curves[next_map[(panel_id, current.edge_id)]]
    if current.edge_id != ordered[0].edge_id or len(ordered) != len(curves):
        raise PatternDSLError(f"{panel_id}: edge graph is not one cycle")
    return ordered


def _relative_control(
    start: tuple[float, float], end: tuple[float, float], control: tuple[float, float]
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return (
        start[0] + control[0] * dx - control[1] * dy,
        start[1] + control[0] * dy + control[1] * dx,
    )


def materialize_pattern_document(program: PatternProgram, *, samples_per_curve: int = 33):
    """Materialize a verified DSL program while retaining analytic payloads.

    The canonical frame starts each panel at the origin with its first chord on
    +X.  Consequently the result is geometrically equivalent to the source up
    to a rigid transform, while no source packing coordinate is required.
    """

    if samples_per_curve < 4:
        raise ValueError("samples_per_curve must be at least four")
    report = program.verify()
    if not report.valid:
        raise PatternDSLError("cannot materialize a symbolically invalid program")
    from benchmark.gcdv2_exact.geometry import sample_curve
    from benchmark.pattern_pipeline.schema import Edge, Panel, PatternDocument, Stitch, StitchSide

    panels, curves_by_panel, roles, next_map, _shared = _panel_maps(program)
    moves = {
        value.panel_id: value
        for value in program.commands
        if isinstance(value, MoveCommand)
    }
    document_panels = []
    analytic: dict[str, Any] = {}
    semantic_paths: dict[str, list[dict[str, Any]]] = {}
    for panel_id, panel in panels.items():
        ordered = _ordered_edges(panel_id, curves_by_panel[panel_id], next_map, moves[panel_id])
        vertex_coordinates: dict[str, tuple[float, float]] = {ordered[0].start_point_id: (0.0, 0.0)}
        direction = 0.0
        reconstructed: list[tuple[CurveCommand, tuple[float, float], tuple[float, float]]] = []
        for index, edge in enumerate(ordered):
            if index:
                direction += math.atan2(edge.turn_sin, edge.turn_cos)
            start = vertex_coordinates[edge.start_point_id]
            chord_cm = edge.chord_ratio * panel.panel_scale_cm
            computed_end = (
                start[0] + chord_cm * math.cos(direction),
                start[1] + chord_cm * math.sin(direction),
            )
            end = vertex_coordinates.get(edge.end_point_id, computed_end)
            vertex_coordinates.setdefault(edge.end_point_id, computed_end)
            reconstructed.append((edge, start, end))
        document_edges = []
        for edge, start, end in reconstructed:
            curve: dict[str, Any] = {
                "type": edge.primitive,
                "controls_cm": [
                    list(_relative_control(start, end, value))
                    for value in edge.controls_chord_frame
                ],
            }
            if edge.op == "A":
                curve["arc"] = {
                    "radius_cm": float(edge.arc_radius_over_chord) * math.dist(start, end),
                    "large_arc": bool(edge.large_arc),
                    "sweep_y_up": bool(edge.sweep_y_up),
                    "right": bool(edge.sweep_y_up),
                }
            points = sample_curve(start, end, curve, samples=samples_per_curve)
            document_edges.append(
                Edge(
                    id=edge.edge_id,
                    points=tuple((float(x), float(y)) for x, y in points),
                    source_curve_id=edge.source_edge_index,
                    confidence=1.0,
                )
            )
            key = f"{panel_id}/{edge.edge_id}"
            analytic[key] = {
                "panel_id": panel_id,
                "edge_id": edge.edge_id,
                "start_point_id": edge.start_point_id,
                "end_point_id": edge.end_point_id,
                "curve": curve,
                "invariant_payload": _payload(edge),
                "length_cm": edge.length_ratio * panel.panel_scale_cm,
            }
            role = roles.get((panel_id, edge.edge_id))
            if role:
                semantic_paths.setdefault(role, []).append(
                    {"panel_id": panel_id, "edge_ids": [edge.edge_id]}
                )
        document_panels.append(
            Panel(
                id=panel_id,
                edges=tuple(document_edges),
                source_panel_id=None,
                confidence=1.0,
            )
        )
    stitches = tuple(
        Stitch(
            value.seam_id,
            StitchSide(value.first_panel_id, value.first_edge_id),
            StitchSide(value.second_panel_id, value.second_edge_id),
            confidence=1.0,
        )
        for value in program.commands
        if isinstance(value, SewnToCommand)
    )
    landmark_entries: dict[str, list[dict[str, Any]]] = {}
    for landmark in report.derived_landmarks:
        incident = [
            edge
            for edge in curves_by_panel[landmark.panel_id].values()
            if landmark.point_id in (edge.start_point_id, edge.end_point_id)
        ]
        if not incident:
            continue
        edge = incident[0]
        point_index = 0 if edge.start_point_id == landmark.point_id else -1
        landmark_entries.setdefault(landmark.name, []).append(
            {
                "panel_id": landmark.panel_id,
                "edge_id": edge.edge_id,
                "point_index": point_index,
                "point_id": landmark.point_id,
                "derived_from_semantic_junction": True,
            }
        )
    return PatternDocument(
        pattern_id=program.pattern_id,
        generator="coordinate-free GCD pattern DSL",
        panels=tuple(document_panels),
        stitches=stitches,
        provenance={
            "schema_version": program.schema_version,
            "source_modality": "formal_graph_without_images",
            "absolute_source_coordinates_serialized": False,
        },
        annotations={
            "pattern_dsl": program.serialize(),
            "analytic_edge_geometry": analytic,
            "semantic_paths": semantic_paths,
            "semantic_landmarks": landmark_entries,
            "edge_labels": {
                f"{panel_id}/{edge_id}": role
                for (panel_id, edge_id), role in roles.items()
            },
            "symbolic_verification": report.to_dict(),
            "stitch_orientation": "unresolved_by_formal_graph_dsl",
            "coordinate_contract": {
                "panel_origin": "canonical_zero",
                "first_chord": "canonical_positive_x",
                "serialized_geometry": "perimeter/chord ratios, relative turns/tangents, chord-frame controls",
                "units": "cm after applying PANEL.panel_scale_cm",
            },
        },
    )


__all__ = [
    "CURVE_OPS",
    "SCHEMA_VERSION",
    "SEMANTIC_JUNCTIONS",
    "CloseCommand",
    "CurveCommand",
    "LandmarkCommand",
    "MoveCommand",
    "NextCommand",
    "PanelCommand",
    "PatternDSLCompileError",
    "PatternDSLError",
    "PatternDSLParseError",
    "PatternProgram",
    "RoleCommand",
    "SewnToCommand",
    "SharedEndpointCommand",
    "SymbolicVerificationReport",
    "VerificationIssue",
    "compile_formal_graph",
    "compile_garment_record",
    "materialize_pattern_document",
    "parse_pattern_dsl",
    "verify_pattern_dsl",
]
