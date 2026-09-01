from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .export import export_bundle
from .garment_particles_convert import convert_garment_particles_npz
from .repair import snap_boundary_junctions
from .reweaver_convert import convert_reweaver_npz
from .simulation import simulation_stage_status
from .validation import validate_pattern


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_pattern_pipeline(source: Path, output: Path, converter) -> dict:
    stages: list[dict] = []
    document = converter(source)
    stages.append(
        {
            "stage": "CANONICALIZE_GENERATED_OUTPUT",
            "status": "PASS",
            "panel_count": len(document.panels),
            "stitch_count": len(document.stitches),
            "input_sha256": _sha256(source),
        }
    )
    initial = validate_pattern(document)
    stages.append({"stage": "VALIDATE_STRUCTURE", "status": "PASS" if initial.accepted else "REPAIR_REQUIRED", **initial.to_dict()})
    repair_receipt = None
    if not initial.accepted:
        document, repair_receipt = snap_boundary_junctions(document)
        stages.append({"stage": "REPAIR_STRUCTURE", "status": "PASS" if repair_receipt.accepted else "NO_IMPROVEMENT", **repair_receipt.to_dict()})
    final = validate_pattern(document)
    paths = export_bundle(document, output)
    stages.append(
        {
            "stage": "EXPORT",
            "status": "PASS" if final.accepted else "FAILED_VALIDATION",
            "validation": final.to_dict(),
            "artifacts": {name: {"file": path.name, "sha256": _sha256(path)} for name, path in paths.items()},
            "dxf_scope": "generic R12 closed panel outlines; stitch semantics are in stitches.json",
        }
    )
    simulation = simulation_stage_status(paths["canonical_json"])
    stages.append(simulation)
    receipt = {
        "pipeline_version": 1,
        "pattern_id": document.pattern_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_contract": {
            "variable_topology": True,
            "template_retrieval": False,
            "nearest_pattern_selection": False,
        },
        "structural_export": "PASS" if final.accepted else "FAILED_VALIDATION",
        "full_simulation_benchmark": simulation["status"],
        "stages": stages,
    }
    (output / "pipeline_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def run_reweaver_pattern_pipeline(source: Path, output: Path) -> dict:
    """Canonicalize and validate one variable-topology ReWeaver generation."""
    return _run_pattern_pipeline(source, output, convert_reweaver_npz)


def run_garment_particles_pattern_pipeline(source: Path, output: Path) -> dict:
    """Canonicalize and validate one variable-topology Garment Particles generation."""
    return _run_pattern_pipeline(source, output, convert_garment_particles_npz)
