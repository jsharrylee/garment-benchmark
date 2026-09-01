"""Contracts for controlled drafting counterfactual pairs.

The important distinction in this module is between an *input intervention*
and its downstream geometry.  A valid pair changes exactly one author-facing
drafting input while body, material, simulation, and camera state stay fixed.
Derived points, curves, panels, and rendered pixels are expected to change and
are therefore not included in the one-change assertion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping


class CounterfactualContractError(ValueError):
    """Raised when a pair is not a controlled one-parameter intervention."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_garmentcode_values(design: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return only GarmentCode parameter leaves (mappings containing ``v``)."""

    output: dict[str, Any] = {}
    for key in sorted(design):
        value = design[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and "v" in value:
            output[path] = copy.deepcopy(value["v"])
        elif isinstance(value, Mapping):
            output.update(flatten_garmentcode_values(value, path))
    return output


def set_garmentcode_value(design: MutableMapping[str, Any], parameter: str, value: Any) -> None:
    keys = parameter.split(".")
    current: MutableMapping[str, Any] = design
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            raise KeyError(f"unknown GarmentCode parameter path: {parameter}")
        current = child
    leaf = current.get(keys[-1])
    if not isinstance(leaf, MutableMapping) or "v" not in leaf:
        raise KeyError(f"unknown GarmentCode parameter leaf: {parameter}")
    leaf["v"] = copy.deepcopy(value)


def input_differences(baseline: Mapping[str, Any], intervention: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    keys = sorted(set(baseline) | set(intervention))
    missing = object()
    differences: dict[str, dict[str, Any]] = {}
    for key in keys:
        before = baseline.get(key, missing)
        after = intervention.get(key, missing)
        if before != after:
            differences[key] = {
                "baseline": None if before is missing else copy.deepcopy(before),
                "intervention": None if after is missing else copy.deepcopy(after),
            }
    return differences


def assert_single_intervention(
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    expected_parameter: str,
) -> dict[str, Any]:
    """Assert exactly one requested drafting input changed and return its delta."""

    differences = input_differences(baseline, intervention)
    if set(differences) != {expected_parameter}:
        raise CounterfactualContractError(
            "expected exactly one changed parameter "
            f"{expected_parameter!r}; observed {sorted(differences)!r}"
        )
    return differences[expected_parameter]


def unchanged_input_fingerprint(inputs: Mapping[str, Any], intervention_parameter: str) -> str:
    """Canonical byte hash of every recipe input except the intervention.

    This is stronger evidence than a count of changed fields: both pair
    members must serialize to the same canonical JSON bytes after removing the
    one explicitly named intervention key.
    """

    if intervention_parameter not in inputs:
        raise CounterfactualContractError(
            f"intervention parameter is absent from recipe inputs: {intervention_parameter}"
        )
    return canonical_json_sha256(
        {key: value for key, value in inputs.items() if key != intervention_parameter}
    )


def fixed_state_fingerprint(
    *,
    body: Mapping[str, Any],
    material: Mapping[str, Any],
    simulator: Mapping[str, Any],
    cameras: Mapping[str, Any],
) -> str:
    """Hash every state component that must remain invariant inside a pair."""

    return canonical_json_sha256(
        {
            "body": body,
            "material": material,
            "simulator": simulator,
            "cameras": cameras,
        }
    )


def garmentcode_topology_signature(specification: Mapping[str, Any]) -> dict[str, Any]:
    pattern = specification["pattern"]
    panels = pattern["panels"]
    return {
        "panel_count": len(panels),
        "panel_edge_counts": {name: len(panel["edges"]) for name, panel in sorted(panels.items())},
        "stitch_count": len(pattern.get("stitches", [])),
    }


def freesewing_topology_signature(extractor_output: Mapping[str, Any]) -> dict[str, Any]:
    visible: dict[str, int] = {}
    for name, part in sorted(extractor_output["parts"].items()):
        if part.get("hidden"):
            continue
        seam = part.get("paths", {}).get("seam", {})
        visible[name] = sum(
            operation.get("type") in {"line", "curve"}
            for operation in seam.get("operations", [])
        )
    return {"panel_count": len(visible), "panel_edge_counts": visible}


def validate_shared_state(baseline_fingerprint: str, intervention_fingerprint: str) -> None:
    if baseline_fingerprint != intervention_fingerprint:
        raise CounterfactualContractError(
            "body/material/simulator/camera state differs between pair members"
        )


def pair_contract(
    *,
    pair_id: str,
    source: str,
    expected_parameter: str,
    baseline_inputs: Mapping[str, Any],
    intervention_inputs: Mapping[str, Any],
    baseline_state_fingerprint: str,
    intervention_state_fingerprint: str,
    baseline_topology: Mapping[str, Any],
    intervention_topology: Mapping[str, Any],
) -> dict[str, Any]:
    delta = assert_single_intervention(baseline_inputs, intervention_inputs, expected_parameter)
    validate_shared_state(baseline_state_fingerprint, intervention_state_fingerprint)
    baseline_unchanged_hash = unchanged_input_fingerprint(
        baseline_inputs, expected_parameter
    )
    intervention_unchanged_hash = unchanged_input_fingerprint(
        intervention_inputs, expected_parameter
    )
    if baseline_unchanged_hash != intervention_unchanged_hash:
        raise CounterfactualContractError(
            "non-intervened recipe input bytes differ between pair members"
        )
    return {
        "pair_id": pair_id,
        "source": source,
        "intervention_parameter": expected_parameter,
        "baseline_value": delta["baseline"],
        "intervention_value": delta["intervention"],
        "changed_input_count": 1,
        "baseline_recipe_inputs_sha256": canonical_json_sha256(baseline_inputs),
        "intervention_recipe_inputs_sha256": canonical_json_sha256(intervention_inputs),
        "unchanged_recipe_inputs_sha256": baseline_unchanged_hash,
        "unchanged_recipe_inputs_canonical_byte_equality": True,
        "unchanged_state_fingerprint": baseline_state_fingerprint,
        "topology_stable": baseline_topology == intervention_topology,
        "baseline_topology": dict(baseline_topology),
        "intervention_topology": dict(intervention_topology),
        "contract_validation": "PASS",
    }


def _rounded(value: float) -> float:
    result = round(float(value), 8)
    return 0.0 if result == -0.0 else result


def _geometry_group(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "primitive_count": len(values),
        "total_length_cm": _rounded(sum(item["length_cm"] for item in values)),
        "total_chord_cm": _rounded(sum(item["chord_cm"] for item in values)),
        "curvature_types": sorted(str(item["curvature_type"]) for item in values),
        "primitive_geometry": values,
    }


def semantic_snapshot_from_drafting_record(
    record: Any, source_specification: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Extract landmark and labeled-curve truth from a DraftingSemanticRecord."""

    landmarks: dict[str, list[list[float]]] = {}
    grouped_edges: dict[str, list[dict[str, Any]]] = {}
    source_panels = (
        source_specification.get("pattern", {}).get("panels", {})
        if source_specification is not None
        else {}
    )
    for panel in record.panels:
        for landmark in panel.landmarks:
            key = f"{panel.role}/{panel.id}/{landmark.name}"
            landmarks.setdefault(key, []).append(
                [_rounded(landmark.xy_cm[0]), _rounded(landmark.xy_cm[1])]
            )
        for edge in panel.edges:
            if edge.role == "other":
                continue
            key = f"{panel.role}/{panel.id}/{edge.role}"
            chord = math.dist(edge.start_cm, edge.end_cm)
            geometry = {
                "edge_id": edge.id,
                "length_cm": _rounded(edge.length_cm),
                "chord_cm": _rounded(chord),
                "curvature_type": edge.curvature_type,
                "start_cm": [_rounded(value) for value in edge.start_cm],
                "end_cm": [_rounded(value) for value in edge.end_cm],
            }
            source_edges = source_panels.get(panel.id, {}).get("edges", [])
            if 0 <= int(edge.index) < len(source_edges):
                # GarmentCode stores quadratic/cubic controls in a chord-local
                # frame.  Retaining these exact source parameters catches
                # curve-shape changes that preserve endpoints and even total
                # arc length (for example x vs 1-x in a symmetric neckline).
                geometry["source_curvature"] = copy.deepcopy(
                    source_edges[int(edge.index)].get("curvature")
                )
            grouped_edges.setdefault(key, []).append(geometry)
    return {
        "landmarks": {key: sorted(value) for key, value in sorted(landmarks.items())},
        "curves": {
            key: _geometry_group(value) for key, value in sorted(grouped_edges.items())
        },
    }


def semantic_snapshot_from_trace(record: Any) -> dict[str, Any]:
    """Extract canonical points and exact named edge geometry from a T-shirt trace."""

    from .tshirt_learning import edge_length_cm

    landmarks: dict[str, list[list[float]]] = {}
    grouped_edges: dict[str, list[dict[str, Any]]] = {}
    for panel in record.panels:
        for point in panel.points:
            if not point.canonical_name:
                continue
            key = f"{panel.semantic_role}/{panel.id}/{point.canonical_name}"
            landmarks.setdefault(key, []).append(
                [_rounded(point.xy_cm[0]), _rounded(point.xy_cm[1])]
            )
        for edge in panel.edges:
            if edge.semantic_role == "other":
                continue
            geometry = edge.geometry
            key = f"{panel.semantic_role}/{panel.id}/{edge.semantic_role}"
            grouped_edges.setdefault(key, []).append(
                {
                    "edge_id": edge.id,
                    "length_cm": _rounded(edge_length_cm(edge)),
                    "chord_cm": _rounded(math.dist(geometry.start_cm, geometry.end_cm)),
                    "curvature_type": geometry.kind,
                    "start_cm": [_rounded(value) for value in geometry.start_cm],
                    "end_cm": [_rounded(value) for value in geometry.end_cm],
                    "control_points_cm": [
                        [_rounded(value) for value in point]
                        for point in geometry.control_points_cm
                    ],
                }
            )
    return {
        "landmarks": {key: sorted(value) for key, value in sorted(landmarks.items())},
        "curves": {
            key: _geometry_group(value) for key, value in sorted(grouped_edges.items())
        },
    }


def semantic_ground_truth_delta(
    baseline: Mapping[str, Any], intervention: Mapping[str, Any]
) -> dict[str, Any]:
    """Return only source-evidenced landmarks/curves that actually changed."""

    output: dict[str, Any] = {}
    for group in ("landmarks", "curves"):
        changes: dict[str, Any] = {}
        baseline_values = baseline.get(group, {})
        intervention_values = intervention.get(group, {})
        for key in sorted(set(baseline_values) | set(intervention_values)):
            before = baseline_values.get(key)
            after = intervention_values.get(key)
            if before == after:
                continue
            item: dict[str, Any] = {"baseline": before, "intervention": after}
            if group == "curves" and before and after:
                item["delta_total_length_cm"] = _rounded(
                    after["total_length_cm"] - before["total_length_cm"]
                )
                item["delta_total_chord_cm"] = _rounded(
                    after["total_chord_cm"] - before["total_chord_cm"]
                )
                before_controls = [
                    (
                        primitive.get("source_curvature"),
                        primitive.get("control_points_cm"),
                    )
                    for primitive in before.get("primitive_geometry", [])
                ]
                after_controls = [
                    (
                        primitive.get("source_curvature"),
                        primitive.get("control_points_cm"),
                    )
                    for primitive in after.get("primitive_geometry", [])
                ]
                item["control_geometry_changed"] = before_controls != after_controls
            changes[key] = item
        output[group] = changes
    output["changed_landmark_group_count"] = len(output["landmarks"])
    output["changed_curve_group_count"] = len(output["curves"])
    output["changed_control_geometry_group_count"] = sum(
        bool(item.get("control_geometry_changed")) for item in output["curves"].values()
    )
    return output


def semantic_delta_coverage(
    delta: Mapping[str, Any], expected_elements: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Diagnose whether expected semantic effects are visible in exact truth.

    Coverage is intentionally separate from the input contract.  A pair can
    be a valid exact-one-input intervention while an expected label is absent
    from the available semantic adapter; that case is reported as PARTIAL or
    NONE instead of being hidden behind the contract PASS.
    """

    observed: set[str] = set()
    for key in delta.get("landmarks", {}):
        observed.add(key.rsplit("/", 1)[-1])
    for key, item in delta.get("curves", {}).items():
        parts = key.split("/")
        panel_role, curve_role = parts[0], parts[-1]
        observed.add(curve_role)
        if panel_role in {"front", "front_bodice"}:
            observed.add("front_bodice_width")
        if panel_role in {"back", "back_bodice"}:
            observed.add("back_bodice_width")
        if item.get("control_geometry_changed"):
            observed.add(f"{curve_role}_control_point")
            observed.add(f"{curve_role}_control_points")

    def normalized(value: str) -> str:
        return str(value).strip().lower().replace("-", "_")

    observed_by_normalized = {normalized(value): value for value in observed}
    matched: list[str] = []
    missing: list[str] = []
    for expected in expected_elements:
        key = normalized(expected)
        aliases = {key}
        if key.endswith("_control_point"):
            aliases.add(key + "s")
        if key.endswith("_control_points"):
            aliases.add(key[:-1])
        if aliases & set(observed_by_normalized):
            matched.append(str(expected))
        else:
            missing.append(str(expected))
    status = "FULL" if not missing else ("PARTIAL" if matched else "NONE")
    return {
        "status": status,
        "expected_elements": list(expected_elements),
        "matched_expected_elements": matched,
        "missing_expected_elements": missing,
        "observed_changed_semantics": sorted(observed),
        "diagnostic": (
            "all configured expected effects are visible in exact semantic geometry"
            if status == "FULL"
            else "input contract is valid, but one or more expected effects are not visible in the current semantic adapter"
        ),
    }


def validate_four_view_receipt(
    pair_record: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Fail-closed validation for a future true four-view pair receipt."""

    if receipt.get("pair_id") != pair_record.get("pair_id"):
        raise CounterfactualContractError("render receipt pair_id mismatch")
    if receipt.get("unchanged_state_fingerprint") != pair_record.get(
        "unchanged_state_fingerprint"
    ):
        raise CounterfactualContractError("render receipt state fingerprint mismatch")
    required_views = {"front", "back", "left", "right"}
    verified: dict[str, Any] = {}
    dimensions_by_view: dict[str, tuple[int, int]] = {}
    paths: set[Path] = set()
    for member in ("baseline", "intervention"):
        member_value = receipt.get("members", {}).get(member, {})
        views = member_value.get("views", {})
        if set(views) != required_views:
            raise CounterfactualContractError(
                f"{member} receipt must contain exactly four orthogonal views"
            )
        verified[member] = {}
        for view in sorted(required_views):
            item = views[view]
            path = (Path(root) / item["path"]).resolve()
            if not path.is_file():
                raise CounterfactualContractError(f"missing rendered view: {path}")
            if path in paths:
                raise CounterfactualContractError("pair members reuse the same rendered file")
            paths.add(path)
            observed_sha = file_sha256(path)
            if observed_sha != item.get("sha256"):
                raise CounterfactualContractError(f"rendered view SHA-256 mismatch: {path}")
            size = tuple(int(value) for value in item.get("image_size", ()))
            if len(size) != 2 or min(size) <= 0:
                raise CounterfactualContractError(f"invalid image_size for {member}/{view}")
            if view in dimensions_by_view and dimensions_by_view[view] != size:
                raise CounterfactualContractError(
                    f"baseline/intervention image dimensions differ for {view}"
                )
            dimensions_by_view[view] = size
            verified[member][view] = {
                "path": item["path"],
                "sha256": observed_sha,
                "image_size": list(size),
            }
    return {
        "pair_id": pair_record["pair_id"],
        "status": "PASS_VALIDATED_FOUR_VIEW_RECEIPT",
        "pattern_only": False,
        "render_status": "VALIDATED",
        "unchanged_state_fingerprint": pair_record["unchanged_state_fingerprint"],
        "members": verified,
    }


TRAINING_ELIGIBILITY_RULE = (
    "contract_validation == PASS && pattern_geometry_changed == true && "
    "topology_stable == true && semantic_delta_coverage.status == FULL"
)


def counterfactual_training_eligibility(record: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fail-closed pattern-only counterfactual training contract."""

    reasons: list[str] = []
    if record.get("contract_validation") != "PASS":
        reasons.append("INPUT_CONTRACT_NOT_PASS")
    if record.get("pattern_geometry_changed") is not True:
        reasons.append("PATTERN_GEOMETRY_UNCHANGED")
    if record.get("topology_stable") is not True:
        reasons.append("TOPOLOGY_CHANGED")
    if record.get("semantic_delta_coverage", {}).get("status") != "FULL":
        reasons.append("SEMANTIC_DELTA_COVERAGE_NOT_FULL")
    return {
        "training_eligible": not reasons,
        "training_eligibility_rule": TRAINING_ELIGIBILITY_RULE,
        "quarantine_reasons": reasons,
    }
