from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from benchmark.drafting_semantics.counterfactual_pairs import (
    assert_single_intervention,
    canonical_json_sha256,
    file_sha256,
    fixed_state_fingerprint,
    flatten_garmentcode_values,
    freesewing_topology_signature,
    garmentcode_topology_signature,
    pair_contract,
    semantic_ground_truth_delta,
    semantic_delta_coverage,
    semantic_snapshot_from_drafting_record,
    semantic_snapshot_from_trace,
    set_garmentcode_value,
)
from benchmark.drafting_semantics.freesewing_teagan import read_teagan_extractor_json
from benchmark.drafting_semantics.garmentcode import annotate_garmentcode_sample
from benchmark.pattern_pipeline.validation import validate_pattern
from benchmark.retrieval.garmentcode import convert_garmentcode_specification


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "drafting_counterfactual_pairs.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "drafting_semantics" / "counterfactual_pairs" / "smoke"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "drafting_counterfactual_pairs.json"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _shared_state(config: dict[str, Any], body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    state = {
        "body": body,
        "material": config["shared_render_state"]["material"],
        "simulator": config["shared_render_state"]["simulator"],
        "cameras": config["shared_render_state"]["cameras"],
    }
    return state, fixed_state_fingerprint(**state)


def _generate_garmentcode_pattern(
    *,
    name: str,
    design: dict[str, Any],
    body: Any,
    output: Path,
) -> tuple[Path, bool]:
    from assets.garment_programs.meta_garment import MetaGarment

    output.mkdir(parents=True, exist_ok=True)
    # GarmentCode's assembly process may normalize/mirror nested design
    # dictionaries in-place.  The counterfactual input is therefore frozen
    # before generation, and the generator receives a disposable deep copy.
    frozen_input_design = copy.deepcopy(design)
    piece = MetaGarment(name, body, copy.deepcopy(frozen_input_design))
    piece.assert_non_empty()
    pattern = piece.assembly()
    self_intersecting = bool(piece.is_self_intersecting())
    pattern.serialize(
        output,
        to_subfolder=False,
        with_3d=False,
        with_text=False,
        view_ids=False,
        with_printable=False,
    )
    design_path = output / "design.yaml"
    design_path.write_text(
        yaml.safe_dump({"design": frozen_input_design}, sort_keys=False), encoding="utf-8"
    )
    specifications = sorted(output.glob("*_specification.json"))
    if len(specifications) != 1:
        raise RuntimeError(f"expected one GarmentCode specification in {output}, got {len(specifications)}")
    return specifications[0], self_intersecting


def build_garmentcode(
    config: dict[str, Any], output: Path, limit: int, pair_suffix: str = ""
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source = config["garmentcode"]
    repo = ROOT / "external" / "GarmentCode"
    if not repo.is_dir():
        raise RuntimeError(f"ignored GarmentCode checkout is missing: {repo}")
    sys.path.insert(0, str(repo))
    from assets.bodies.body_params import BodyParameters

    design_path = ROOT / source["design_file"]
    body_path = ROOT / source["body_file"]
    base_design = yaml.safe_load(design_path.read_text(encoding="utf-8"))["design"]
    for parameter, value in source.get("base_overrides", {}).items():
        set_garmentcode_value(base_design, parameter, value)
    baseline_inputs = flatten_garmentcode_values(base_design)
    state, state_fingerprint = _shared_state(
        config,
        {
            "source": _relative(body_path),
            "sha256": file_sha256(body_path),
            "measurement_policy": "same exact BodyParameters object inputs for both members",
        },
    )
    source_output = output / "garmentcode"
    baseline_output = source_output / "baseline"
    baseline_id = f"gcd_cf_baseline{pair_suffix}"
    baseline_spec_path, baseline_self_intersecting = _generate_garmentcode_pattern(
        name=baseline_id,
        design=base_design,
        body=BodyParameters(body_path),
        output=baseline_output,
    )
    baseline_design_path = baseline_output / "design.yaml"
    baseline_raw = json.loads(baseline_spec_path.read_text(encoding="utf-8"))
    baseline_topology = garmentcode_topology_signature(baseline_raw)
    baseline_pattern_hash = canonical_json_sha256(baseline_raw["pattern"])
    baseline_document = convert_garmentcode_specification(
        baseline_spec_path,
        anchor_id=baseline_id,
        source_license="MIT",
    )
    baseline_document = replace(
        baseline_document,
        generator="GarmentCode controlled counterfactual",
        annotations={
            **baseline_document.annotations,
            "template_retrieval": False,
            "counterfactual_member": "baseline",
            "fixed_state_fingerprint": state_fingerprint,
        },
    )
    baseline_canonical = baseline_output / "canonical_pattern.json"
    baseline_document.write_json(baseline_canonical)
    baseline_validation = validate_pattern(baseline_document).to_dict()
    baseline_semantics = annotate_garmentcode_sample(
        baseline_spec_path,
        body_path,
        baseline_design_path,
        split="counterfactual_smoke",
        source_license="MIT",
    )
    baseline_semantics_path = baseline_output / "semantic_record.json"
    baseline_semantics.write_json(baseline_semantics_path)
    baseline_semantic_snapshot = semantic_snapshot_from_drafting_record(
        baseline_semantics, baseline_raw
    )

    interventions = source["interventions"][: limit or None]
    records: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []
    for intervention in interventions:
        pair_id = f"gcd__{intervention['id']}{pair_suffix}"
        variant_design = copy.deepcopy(base_design)
        set_garmentcode_value(
            variant_design, intervention["parameter"], intervention["value"]
        )
        variant_inputs = flatten_garmentcode_values(variant_design)
        # Fail before invoking the generator if the config accidentally alters
        # zero or multiple author-facing inputs.
        assert_single_intervention(
            baseline_inputs, variant_inputs, intervention["parameter"]
        )
        variant_output = source_output / pair_id / "intervention"
        variant_spec_path, variant_self_intersecting = _generate_garmentcode_pattern(
            name=pair_id,
            design=variant_design,
            body=BodyParameters(body_path),
            output=variant_output,
        )
        variant_design_path = variant_output / "design.yaml"
        variant_raw = json.loads(variant_spec_path.read_text(encoding="utf-8"))
        variant_topology = garmentcode_topology_signature(variant_raw)
        variant_pattern_hash = canonical_json_sha256(variant_raw["pattern"])
        if variant_pattern_hash == baseline_pattern_hash:
            raise RuntimeError(
                f"configured GarmentCode intervention is inactive: {intervention['parameter']}"
            )
        variant_document = convert_garmentcode_specification(
            variant_spec_path, anchor_id=pair_id, source_license="MIT"
        )
        variant_document = replace(
            variant_document,
            generator="GarmentCode controlled counterfactual",
            annotations={
                **variant_document.annotations,
                "template_retrieval": False,
                "counterfactual_member": "intervention",
                "counterfactual_parameter": intervention["parameter"],
                "fixed_state_fingerprint": state_fingerprint,
            },
        )
        variant_canonical = variant_output / "canonical_pattern.json"
        variant_document.write_json(variant_canonical)
        variant_validation = validate_pattern(variant_document).to_dict()
        variant_semantics = annotate_garmentcode_sample(
            variant_spec_path,
            body_path,
            variant_design_path,
            split="counterfactual_smoke",
            source_license="MIT",
        )
        variant_semantics_path = variant_output / "semantic_record.json"
        variant_semantics.write_json(variant_semantics_path)
        ground_truth_delta = semantic_ground_truth_delta(
            baseline_semantic_snapshot,
            semantic_snapshot_from_drafting_record(variant_semantics, variant_raw),
        )
        coverage = semantic_delta_coverage(
            ground_truth_delta, intervention["expected_elements"]
        )
        contract = pair_contract(
            pair_id=pair_id,
            source="GarmentCode",
            expected_parameter=intervention["parameter"],
            baseline_inputs=baseline_inputs,
            intervention_inputs=variant_inputs,
            baseline_state_fingerprint=state_fingerprint,
            intervention_state_fingerprint=state_fingerprint,
            baseline_topology=baseline_topology,
            intervention_topology=variant_topology,
        )
        contract.update(
            {
                "expected_affected_elements": intervention["expected_elements"],
                "ground_truth_semantic_delta": ground_truth_delta,
                "semantic_delta_coverage": coverage,
                "pattern_geometry_changed": True,
                "pattern_only": True,
                "render_status": "PENDING_VALIDATED_SIMULATOR",
                "body_source_file_byte_sha256_equal": True,
                "baseline_self_intersecting": baseline_self_intersecting,
                "intervention_self_intersecting": variant_self_intersecting,
                "baseline_structural_validation": baseline_validation["accepted"],
                "intervention_structural_validation": variant_validation["accepted"],
                "baseline_specification": _relative(baseline_spec_path),
                "intervention_specification": _relative(variant_spec_path),
                "baseline_canonical_pattern": _relative(baseline_canonical),
                "intervention_canonical_pattern": _relative(variant_canonical),
                "baseline_semantic_record": _relative(baseline_semantics_path),
                "intervention_semantic_record": _relative(variant_semantics_path),
                "baseline_pattern_sha256": baseline_pattern_hash,
                "intervention_pattern_sha256": variant_pattern_hash,
            }
        )
        records.append(contract)
        render_jobs.append(
            {
                "pair_id": pair_id,
                "source": "GarmentCode",
                "baseline_pattern": _relative(baseline_canonical),
                "intervention_pattern": _relative(variant_canonical),
                "unchanged_state_fingerprint": state_fingerprint,
                "pattern_only": True,
                "render_status": "PENDING_VALIDATED_SIMULATOR",
                "reason": "The available Blender sewing proxy failed the prior GCD visual-fidelity checkpoint; using it as 3D ground truth would mislabel the pair.",
            }
        )
    summary = {
        "source": "maria-korosteleva/GarmentCode",
        "source_commit": _source_commit(repo),
        "source_code_license": "MIT",
        "body_sha256": state["body"]["sha256"],
        "state_fingerprint": state_fingerprint,
        "pair_count": len(records),
    }
    return records, summary, render_jobs


def _run_freesewing_extract(
    *, extractor: Path, model: str, sa_mm: float, options: dict[str, Any], output: Path
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "node",
            str(extractor),
            "--model",
            model,
            "--sa",
            str(sa_mm),
            "--options",
            json.dumps(options, sort_keys=True, separators=(",", ":")),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def build_freesewing(
    config: dict[str, Any], output: Path, limit: int, pair_suffix: str = ""
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source = config["freesewing_teagan"]
    extractor = ROOT / source["extractor"]
    source_output = output / "freesewing_teagan"
    base_options = copy.deepcopy(source["base_options"])
    baseline_raw_path = source_output / "baseline" / "extractor_output.json"
    baseline_raw = _run_freesewing_extract(
        extractor=extractor,
        model=source["body_model"],
        sa_mm=source["seam_allowance_mm"],
        options=base_options,
        output=baseline_raw_path,
    )
    baseline_trace = read_teagan_extractor_json(
        baseline_raw_path, sample_id=f"freesewing_cf_baseline{pair_suffix}"
    )
    baseline_trace_path = source_output / "baseline" / "trace_record.json"
    _write_json(baseline_trace_path, baseline_trace.to_dict())
    baseline_semantic_snapshot = semantic_snapshot_from_trace(baseline_trace)
    baseline_topology = freesewing_topology_signature(baseline_raw)
    baseline_geometry_hash = canonical_json_sha256(baseline_raw["parts"])
    baseline_resolved_options = baseline_raw["input"]["resolved_options"]
    resolved_measurements_hash = canonical_json_sha256(
        baseline_raw["input"]["resolved_measurements_mm"]
    )
    state, state_fingerprint = _shared_state(
        config,
        {
            "source": f"@freesewing/models@{baseline_raw['source']['models_version']}",
            "model": source["body_model"],
            "resolved_measurements_sha256": resolved_measurements_hash,
            "measurement_policy": "same exact resolved measurement dictionary for both members",
        },
    )

    interventions = source["interventions"][: limit or None]
    records: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []
    for intervention in interventions:
        pair_id = f"freesewing_teagan__{intervention['id']}{pair_suffix}"
        variant_options = copy.deepcopy(base_options)
        variant_options[intervention["parameter"]] = intervention["value"]
        assert_single_intervention(
            base_options, variant_options, intervention["parameter"]
        )
        variant_raw_path = source_output / pair_id / "intervention" / "extractor_output.json"
        variant_raw = _run_freesewing_extract(
            extractor=extractor,
            model=source["body_model"],
            sa_mm=source["seam_allowance_mm"],
            options=variant_options,
            output=variant_raw_path,
        )
        variant_measurements_hash = canonical_json_sha256(
            variant_raw["input"]["resolved_measurements_mm"]
        )
        if variant_measurements_hash != resolved_measurements_hash:
            raise RuntimeError(f"FreeSewing body measurements drifted for {pair_id}")
        resolved_delta = assert_single_intervention(
            baseline_resolved_options,
            variant_raw["input"]["resolved_options"],
            intervention["parameter"],
        )
        variant_trace = read_teagan_extractor_json(
            variant_raw_path, sample_id=pair_id
        )
        variant_trace_path = variant_raw_path.parent / "trace_record.json"
        _write_json(variant_trace_path, variant_trace.to_dict())
        variant_topology = freesewing_topology_signature(variant_raw)
        variant_geometry_hash = canonical_json_sha256(variant_raw["parts"])
        if variant_geometry_hash == baseline_geometry_hash:
            raise RuntimeError(
                f"configured FreeSewing intervention is inactive: {intervention['parameter']}"
            )
        ground_truth_delta = semantic_ground_truth_delta(
            baseline_semantic_snapshot,
            semantic_snapshot_from_trace(variant_trace),
        )
        coverage = semantic_delta_coverage(
            ground_truth_delta, intervention["expected_elements"]
        )
        contract = pair_contract(
            pair_id=pair_id,
            source="FreeSewing Teagan",
            expected_parameter=intervention["parameter"],
            baseline_inputs=base_options,
            intervention_inputs=variant_options,
            baseline_state_fingerprint=state_fingerprint,
            intervention_state_fingerprint=state_fingerprint,
            baseline_topology=baseline_topology,
            intervention_topology=variant_topology,
        )
        contract.update(
            {
                "expected_affected_elements": intervention["expected_elements"],
                "ground_truth_semantic_delta": ground_truth_delta,
                "semantic_delta_coverage": coverage,
                "pattern_geometry_changed": True,
                "pattern_only": True,
                "render_status": "PENDING_VALIDATED_SIMULATOR",
                "resolved_body_measurements_equal": True,
                "resolved_body_measurements_canonical_byte_sha256_equal": True,
                "resolved_recipe_inputs_exact_one_change": True,
                "resolved_recipe_input_delta": resolved_delta,
                "baseline_resolved_recipe_inputs_sha256": canonical_json_sha256(
                    baseline_resolved_options
                ),
                "intervention_resolved_recipe_inputs_sha256": canonical_json_sha256(
                    variant_raw["input"]["resolved_options"]
                ),
                "baseline_extractor_output": _relative(baseline_raw_path),
                "intervention_extractor_output": _relative(variant_raw_path),
                "baseline_trace_record": _relative(baseline_trace_path),
                "intervention_trace_record": _relative(variant_trace_path),
                "baseline_pattern_sha256": baseline_geometry_hash,
                "intervention_pattern_sha256": variant_geometry_hash,
            }
        )
        records.append(contract)
        render_jobs.append(
            {
                "pair_id": pair_id,
                "source": "FreeSewing Teagan",
                "baseline_pattern": _relative(baseline_trace_path),
                "intervention_pattern": _relative(variant_trace_path),
                "unchanged_state_fingerprint": state_fingerprint,
                "pattern_only": True,
                "render_status": "PENDING_VALIDATED_SIMULATOR",
                "reason": "Teagan provides deterministic 2D named pattern output but this repository has no validated Teagan-to-fixed-body 3D reference simulator adapter.",
            }
        )
    summary = {
        "source": f"@freesewing/teagan@{baseline_raw['source']['design_version']}",
        "source_code_license": baseline_raw["source"]["source_code_license_spdx"],
        "body_model": source["body_model"],
        "resolved_measurements_sha256": resolved_measurements_hash,
        "state_fingerprint": state_fingerprint,
        "pair_count": len(records),
    }
    return records, summary, render_jobs


def _reflected_value(value: float, lower: float, upper: float, *, integer: bool) -> float | int:
    candidate = lower + upper - value
    if abs(candidate - value) < 0.15 * (upper - lower):
        candidate = lower if value > (lower + upper) * 0.5 else upper
    if integer:
        candidate = int(round(candidate))
        if candidate == int(round(value)):
            candidate = int(lower) if int(round(value)) != int(lower) else int(upper)
        return candidate
    return round(float(candidate), 6)


def _sample_replicate_config(
    config: dict[str, Any], replicate: int
) -> dict[str, Any]:
    """Create deterministic pair-between diversity without pair-within drift."""

    if replicate == 0:
        return copy.deepcopy(config)
    sampled = copy.deepcopy(config)
    sampler = config["seeded_baseline_sampler"]

    gcd_rng = random.Random(int(config["seed"]) + 104729 * replicate + 11)
    gcd_source = sampled["garmentcode"]
    body_files = sampler["garmentcode_body_files"]
    gcd_source["body_file"] = body_files[gcd_rng.randrange(len(body_files))]
    baseline_values: dict[str, float | int] = {}
    for parameter, bounds in sorted(sampler["garmentcode_ranges"].items()):
        lower, upper = float(bounds[0]), float(bounds[1])
        integer = parameter == "sleeve.sleeve_angle"
        raw = gcd_rng.uniform(lower, upper)
        value: float | int = int(round(raw)) if integer else round(raw, 6)
        baseline_values[parameter] = value
        gcd_source.setdefault("base_overrides", {})[parameter] = value
    for intervention in gcd_source["interventions"]:
        parameter = intervention["parameter"]
        lower, upper = map(float, sampler["garmentcode_ranges"][parameter])
        intervention["value"] = _reflected_value(
            float(baseline_values[parameter]),
            lower,
            upper,
            integer=parameter == "sleeve.sleeve_angle",
        )

    fs_rng = random.Random(int(config["seed"]) + 130363 * replicate + 29)
    fs_source = sampled["freesewing_teagan"]
    from benchmark.drafting_semantics.freesewing_split import (
        TEST_BODY_MODELS,
        TRAIN_BODY_MODELS,
        VALIDATION_BODY_MODELS,
    )

    body_models = (*TRAIN_BODY_MODELS, *VALIDATION_BODY_MODELS, *TEST_BODY_MODELS)
    fs_source["body_model"] = body_models[fs_rng.randrange(len(body_models))]
    fs_baseline: dict[str, float] = {}
    for parameter, bounds in sorted(sampler["freesewing_ranges"].items()):
        lower, upper = float(bounds[0]), float(bounds[1])
        fs_baseline[parameter] = round(fs_rng.uniform(lower, upper), 6)
    fs_source["base_options"] = fs_baseline
    for intervention in fs_source["interventions"]:
        parameter = intervention["parameter"]
        lower, upper = map(float, sampler["freesewing_ranges"][parameter])
        intervention["value"] = _reflected_value(
            fs_baseline[parameter], lower, upper, integer=False
        )
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic one-drafting-parameter counterfactual smoke pairs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source", choices=("all", "garmentcode", "freesewing"), default="all"
    )
    parser.add_argument(
        "--limit-per-source", type=int, default=0, help="0 uses every bounded config intervention"
    )
    parser.add_argument(
        "--pairs-per-parameter",
        type=int,
        default=1,
        help="Deterministic pair-between baseline/body diversity; 32 is the recommended training plan",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.pairs_per_parameter < 1:
        raise SystemExit("--pairs-per-parameter must be at least 1")
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []
    for replicate in range(args.pairs_per_parameter):
        replicate_config = _sample_replicate_config(config, replicate)
        suffix = "" if args.pairs_per_parameter == 1 else f"__r{replicate:03d}"
        replicate_output = (
            output
            if args.pairs_per_parameter == 1
            else output / f"replicate_{replicate:03d}"
        )
        if args.source in {"all", "garmentcode"}:
            items, summary, jobs = build_garmentcode(
                replicate_config,
                replicate_output,
                args.limit_per_source,
                suffix,
            )
            summary["replicate"] = replicate
            records.extend(items)
            source_summaries.append(summary)
            render_jobs.extend(jobs)
        if args.source in {"all", "freesewing"}:
            items, summary, jobs = build_freesewing(
                replicate_config,
                replicate_output,
                args.limit_per_source,
                suffix,
            )
            summary["replicate"] = replicate
            records.extend(items)
            source_summaries.append(summary)
            render_jobs.extend(jobs)

    render_contract = {
        "schema_version": "drafting-counterfactual-render-contract/v1",
        "pattern_only": True,
        "render_status": "PENDING_VALIDATED_SIMULATOR",
        "shared_render_state": config["shared_render_state"],
        "required_views": ["front", "back", "left", "right"],
        "requirements": [
            "Render both members with the state fingerprint recorded by the pair.",
            "Use the same body mesh and pose, material, simulator settings, and random seed.",
            "Compute one orthographic camera/crop from the union of both members and reuse it.",
            "Do not substitute existing random GCDv2 renders or the unvalidated Blender sewing proxy as ground truth.",
            "A completion receipt must include the pair ID, unchanged state fingerprint, four view paths per member, dimensions, and SHA-256 hashes.",
        ],
        "existing_data_audit": {
            "garmentcode_v2": "2,937 independently randomized accepted samples; their four views are valid observations but are not exact-one-parameter pairs.",
            "freesewing_teagan_diverse": "240 deterministic 2D records from a body-by-preset matrix; presets commonly change multiple options and no 3D reference renders exist.",
            "blender_sewing_proxy": "Available, but its prior best observed GCD reference mean four-view IoU was 0.2368 and did not pass visual fidelity.",
        },
        "jobs": render_jobs,
    }
    render_contract_path = output / "render_contract.json"
    _write_json(render_contract_path, render_contract)

    coverage_counts = dict(
        sorted(Counter(row["semantic_delta_coverage"]["status"] for row in records).items())
    )
    all_coverage_full = coverage_counts.get("FULL", 0) == len(records)
    manifest = {
        "schema_version": "drafting-counterfactual-pairs/v1",
        "status": (
            "PASS_PATTERN_ONLY_CONTROLLED_GROUND_TRUTH"
            if all_coverage_full
            else "PATTERN_ONLY_INPUT_CONTRACT_PASS__SEMANTIC_DELTA_COVERAGE_PARTIAL"
        ),
        "pattern_only": True,
        "render_status": "PENDING_VALIDATED_SIMULATOR",
        "config": _relative(config_path),
        "config_sha256": file_sha256(config_path),
        "seed": config["seed"],
        "claim_boundary": config["claim_boundary"],
        "pair_count": len(records),
        "pairs_per_parameter": args.pairs_per_parameter,
        "recommended_pairs_per_parameter": config["recommended_pairs_per_parameter"],
        "seeded_pair_between_diversity": args.pairs_per_parameter > 1,
        "source_counts": dict(sorted(Counter(row["source"] for row in records).items())),
        "changed_input_count_histogram": dict(
            sorted(Counter(str(row["changed_input_count"]) for row in records).items())
        ),
        "all_pattern_geometry_changed": all(row["pattern_geometry_changed"] for row in records),
        "all_contracts_passed": all(row["contract_validation"] == "PASS" for row in records),
        "semantic_delta_coverage_counts": coverage_counts,
        "all_semantic_delta_coverage_full": all_coverage_full,
        "topology_stable_count": sum(row["topology_stable"] for row in records),
        "true_four_view_pair_count": 0,
        "source_summaries": source_summaries,
        "render_contract": _relative(render_contract_path),
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "pairs": len(records),
                "sources": manifest["source_counts"],
                "topology_stable": manifest["topology_stable_count"],
                "true_four_view_pairs": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
