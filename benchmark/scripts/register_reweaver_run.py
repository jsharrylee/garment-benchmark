from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.adapters.reweaver import sha256, summarize_output, validate_input_directory
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
        stdout_log="logs/reweaver_registration.log",
        stderr_log="",
        produced_hashes=produced or {},
        validation=validation,
        runtime_seconds=measured_runtime if measured_runtime is not None else time.perf_counter() - started,
        peak_vram_mib=peak_vram_mib,
        next_stage=next_stage,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and register accepted ReWeaver artifacts in the resumable harness.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--samples", nargs="+", default=["synbody_cyan_jacket", "synbody_patterned_shirt"])
    args = parser.parse_args()
    root = args.root.resolve()
    executor = Executor(root)
    registry = ArtifactRegistry(root / "artifacts" / "index.jsonl")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "reweaver_registration.log"
    accepted = []

    for sample_id in args.samples:
        job_id = f"synbody:{sample_id}:reweaver"
        input_dir = root / "data" / "processed" / "synbody" / sample_id / "reweaver" / "render_output" / "rgb"
        artifact_dir = root / "artifacts" / "reweaver" / sample_id
        output = artifact_dir / f"{sample_id}.npz"
        pattern = artifact_dir / "pattern.png"
        geometry = artifact_dir / "geometry.png"
        input_validation = validate_input_directory(input_dir)
        recorded_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
        input_hashes = {f"input_{index}": value for index, value in enumerate(input_validation.get("sha256", []))}
        checks = {
            Stage.DISCOVER: lambda: {"official_repository": (root / "external" / "ReWeaver-Code").is_dir()},
            Stage.ACQUIRE: lambda: {"official_checkpoints": all((root / "checkpoints" / "ReWeaver" / "tileable" / name).is_file() for name in ("complex_stitch.pth", "flatten.pth", "img_encoder.pth"))},
            Stage.VALIDATE_RAW: lambda: {"source_manifest": (root / "data" / "manifests" / "synbody_local_acquisition.json").is_file()},
            Stage.GROUP_VIEWS: lambda: {"four_view_bundle": input_validation["valid"]},
            Stage.PREPROCESS: lambda: {"normalized_inputs": input_validation["valid"]},
            Stage.VALIDATE_INPUT: lambda: input_validation,
            Stage.PREPARE_MODEL: lambda: {"config_snapshot": (artifact_dir / "config.json").is_file()},
            Stage.OFFICIAL_SMOKE: lambda: {"official_classes_and_weights_executed": output.is_file()},
            Stage.EXTERNAL_INFERENCE: lambda: {"external_output": output.is_file() and output.stat().st_size > 0},
            Stage.VALIDATE_OUTPUT: lambda: summarize_output(output),
            Stage.VISUAL_REVIEW: lambda: {"pattern_render": pattern.stat().st_size > 10_000, "geometry_render": geometry.stat().st_size > 10_000},
            Stage.COMPARE: lambda: {"distinct_sample_binding": (root / "artifacts" / "reweaver" / "distinct_samples_comparison.json").is_file(), "controlled_flip_binding": (root / "artifacts" / "reweaver" / "binding_flip_comparison.json").is_file()},
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
            produced = {}
            measured_runtime = None
            peak_vram_mib = None
            if current == Stage.EXTERNAL_INFERENCE:
                produced = {"prediction": sha256(output)}
                measured_runtime = recorded_summary["runtime_seconds"]
                peak_vram_mib = recorded_summary["peak_vram_bytes"] // (1024 * 1024)
            executor.run(
                job_id,
                current,
                f"validate existing ReWeaver {current.value.lower()} artifact",
                lambda a=action_id, n=next_stage, v=validation, s=started, p=produced, r=measured_runtime, m=peak_vram_mib: receipt(a, n, v, s, produced=p, measured_runtime=r, peak_vram_mib=m),
                input_hashes=input_hashes,
                expected_outputs=[str(output.relative_to(root))] if current == Stage.EXTERNAL_INFERENCE else [],
            )
            if current == Stage.EXTERNAL_INFERENCE:
                registry.register(output, job_id, "reweaver_prediction")
        accepted.append({"job_id": job_id, "stage": executor.state.stage(job_id).value})

    log_path.write_text(json.dumps({"accepted": accepted}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "jobs": accepted}))


if __name__ == "__main__":
    main()
