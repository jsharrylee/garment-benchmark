from __future__ import annotations

import json
from pathlib import Path

from benchmark.adapters.garment_particles import summarize_output as summarize_particles
from benchmark.adapters.reweaver import summarize_output as summarize_reweaver
from benchmark.harness.artifact_registry import ArtifactRegistry
from benchmark.harness.schemas import Stage
from benchmark.harness.state import RunState


EXPECTED_JOBS = {
    "synbody:synbody_cyan_jacket:reweaver",
    "synbody:synbody_patterned_shirt:reweaver",
    "synbody:synbody_cyan_jacket:garment_particles",
    "synbody:synbody_patterned_shirt:garment_particles",
}


def main() -> None:
    root = Path.cwd().resolve()
    state = RunState(root / "state" / "run_state.json")
    accepted = {job for job in EXPECTED_JOBS if state.stage(job) == Stage.ACCEPTED}
    if accepted != EXPECTED_JOBS:
        raise ValueError({"failure": "FAILED_VALIDATION", "accepted": sorted(accepted)})

    index_path = root / "artifacts" / "index.jsonl"
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    records = [record for record in records if record["job_id"] in EXPECTED_JOBS]
    if len(records) != 4 or {record["job_id"] for record in records} != EXPECTED_JOBS or not all(ArtifactRegistry.verify(record) for record in records):
        raise ValueError({"failure": "CHECKSUM_MISMATCH", "artifact_records": len(records)})

    output_validations = []
    for sample in ("synbody_cyan_jacket", "synbody_patterned_shirt"):
        output_validations.append(summarize_reweaver(root / "artifacts" / "reweaver" / sample / f"{sample}.npz"))
        output_validations.append(summarize_particles(root / "artifacts" / "garment_particles" / sample / "prediction.npz"))
    if not all(item["valid"] for item in output_validations):
        raise ValueError({"failure": "FAILED_VALIDATION", "outputs": output_validations})

    binding_paths = [
        root / "artifacts" / model / name
        for model in ("reweaver", "garment_particles")
        for name in ("distinct_samples_comparison.json", "binding_flip_comparison.json")
    ]
    if not all(json.loads(path.read_text(encoding="utf-8"))["valid"] for path in binding_paths):
        raise ValueError({"failure": "OUTPUT_NOT_BOUND_TO_INPUT"})

    boards = [root / "artifacts" / "review_boards" / f"{sample}.jpg" for sample in ("synbody_cyan_jacket", "synbody_patterned_shirt")]
    if not all(path.is_file() and path.stat().st_size > 10_000 for path in boards):
        raise ValueError({"failure": "VISUAL_REVIEW_FAILURE"})

    required_tracked = [
        root / "data" / "manifests" / "final_benchmark.json",
        root / "data" / "manifests" / "reweaver_checkpoint_4.json",
        root / "data" / "manifests" / "garment_particles_checkpoint_5.json",
        root / "reports" / "final_benchmark.md",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_tracked):
        raise ValueError({"failure": "FAILED_VALIDATION", "reason": "missing final evidence"})

    events = (root / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    result = {
        "status": "PASS_WITH_QUALITY_LIMITATIONS",
        "global_acceptance": "PASS",
        "accepted_jobs": len(accepted),
        "verified_artifacts": len(records),
        "validated_outputs": len(output_validations),
        "binding_tests": len(binding_paths),
        "visual_boards": len(boards),
        "event_records": len(events),
        "permission_to_redistribute_dataset": False,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
