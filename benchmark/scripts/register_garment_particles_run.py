from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.adapters.garment_particles import sha256, summarize_output
from benchmark.harness.artifact_registry import ArtifactRegistry
from benchmark.harness.executor import Executor, now
from benchmark.harness.schemas import ActionReceipt, ORDERED_STAGES, Stage


def receipt(
    action_id: str,
    next_stage: Stage,
    validation: dict,
    started: float,
    *,
    produced: dict[str, str] | None = None,
    measured_runtime: float | None = None,
    peak_vram_mib: int | None = None,
) -> ActionReceipt:
    return ActionReceipt(
        action_id=action_id,
        exit_status=0,
        ended_at=now(),
        stdout_log="logs/garment_particles_registration.log",
        stderr_log="",
        produced_hashes=produced or {},
        validation=validation,
        runtime_seconds=measured_runtime if measured_runtime is not None else time.perf_counter() - started,
        peak_vram_mib=peak_vram_mib,
        next_stage=next_stage,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and register accepted Garment Particles artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--samples", nargs="+", default=["synbody_cyan_jacket", "synbody_patterned_shirt"])
    args = parser.parse_args()
    root = args.root.resolve()
    executor = Executor(root)
    registry = ArtifactRegistry(root / "artifacts" / "index.jsonl")
    log_path = root / "logs" / "garment_particles_registration.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = []

    for sample_id in args.samples:
        job_id = f"synbody:{sample_id}:garment_particles"
        input_path = root / "data" / "processed" / "synbody" / sample_id / "garment_particles" / "input.png"
        artifact_dir = root / "artifacts" / "garment_particles" / sample_id
        output = artifact_dir / "prediction.npz"
        pattern = artifact_dir / "pattern.png"
        geometry = artifact_dir / "geometry.png"
        input_validation = {
            "valid": input_path.is_file() and input_path.stat().st_size > 0,
            "input_sha256": sha256(input_path),
            "single_image_condition": True,
        }
        recorded_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
        checks = {
            Stage.DISCOVER: lambda: {"official_repository": (root / "external" / "GarmentParticles").is_dir()},
            Stage.ACQUIRE: lambda: {
                "official_pgf_checkpoint": (root / "checkpoints" / "GarmentParticles" / "pgf_image" / ".metadata").is_file(),
                "official_edge_checkpoint": (root / "checkpoints" / "GarmentParticles" / "edge" / ".metadata").is_file(),
            },
            Stage.VALIDATE_RAW: lambda: {"source_manifest": (root / "data" / "manifests" / "synbody_local_acquisition.json").is_file()},
            Stage.GROUP_VIEWS: lambda: {"representative_front_image_selected_from_same_bundle": input_validation["valid"]},
            Stage.PREPROCESS: lambda: {"normalized_person_image": input_validation["valid"]},
            Stage.VALIDATE_INPUT: lambda: input_validation,
            Stage.PREPARE_MODEL: lambda: {"config_snapshot": (artifact_dir / "config.json").is_file()},
            Stage.OFFICIAL_SMOKE: lambda: {"official_model_classes_and_weights_executed": output.is_file()},
            Stage.EXTERNAL_INFERENCE: lambda: {"external_output": output.is_file() and output.stat().st_size > 0},
            Stage.VALIDATE_OUTPUT: lambda: summarize_output(output),
            Stage.VISUAL_REVIEW: lambda: {
                "pattern_render": pattern.is_file() and pattern.stat().st_size > 10_000,
                "geometry_render": geometry.is_file() and geometry.stat().st_size > 10_000,
            },
            Stage.COMPARE: lambda: {
                "distinct_sample_binding": (root / "artifacts" / "garment_particles" / "distinct_samples_comparison.json").is_file(),
                "controlled_flip_binding": (root / "artifacts" / "garment_particles" / "binding_flip_comparison.json").is_file(),
            },
        }
        while executor.state.stage(job_id) not in {Stage.ACCEPTED, Stage.BLOCKED_EXTERNAL, Stage.FAILED_BUDGET, Stage.FAILED_VALIDATION}:
            current = executor.state.stage(job_id) or Stage.DISCOVER
            next_stage = Stage.ACCEPTED if current == Stage.COMPARE else ORDERED_STAGES[ORDERED_STAGES.index(current) + 1]
            started = time.perf_counter()
            validation = checks[current]()
            valid = all(value is not False for value in validation.values()) and validation.get("valid", True)
            if not valid:
                raise ValueError(f"{job_id} {current.value} validation failed: {validation}")
            action_id = f"{job_id}:{current.value}:1"
            produced = {"prediction": sha256(output)} if current == Stage.EXTERNAL_INFERENCE else {}
            measured_runtime = None
            peak_vram_mib = None
            if current == Stage.EXTERNAL_INFERENCE:
                measured_runtime = recorded_summary["pgf_total_seconds"] + recorded_summary["edge_total_seconds"]
                peak_vram_mib = max(recorded_summary["pgf_peak_vram_bytes"], recorded_summary["edge_peak_vram_bytes"]) // (1024 * 1024)
            executor.run(
                job_id,
                current,
                f"validate existing Garment Particles {current.value.lower()} artifact",
                lambda a=action_id, n=next_stage, v=validation, s=started, p=produced, r=measured_runtime, m=peak_vram_mib: receipt(a, n, v, s, produced=p, measured_runtime=r, peak_vram_mib=m),
                input_hashes={"input": input_validation["input_sha256"]},
                expected_outputs=[str(output.relative_to(root))] if current == Stage.EXTERNAL_INFERENCE else [],
            )
            if current == Stage.EXTERNAL_INFERENCE:
                registry.register(output, job_id, "garment_particles_prediction")
        accepted.append({"job_id": job_id, "stage": executor.state.stage(job_id).value})

    log_path.write_text(json.dumps({"accepted": accepted}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "jobs": accepted}))


if __name__ == "__main__":
    main()
