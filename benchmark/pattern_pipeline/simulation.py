from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendProbe:
    backend: str
    available: bool
    executable: str | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def probe_simulation_backends() -> tuple[BackendProbe, ...]:
    """Report real local backends only; this never substitutes a geometry mock."""
    configured = os.environ.get("GARMENT_SIMULATOR_COMMAND")
    project_root = Path(__file__).resolve().parents[2]
    portable_blender = project_root / "external" / "blender-4.5.12-windows-x64" / "blender.exe"
    blender = shutil.which("blender") or (str(portable_blender) if portable_blender.is_file() else None)
    importer = project_root / "benchmark" / "blender_scripts" / "simulate_canonical_pattern.py"
    integration_available = bool(blender and importer.is_file())
    probes = [
        BackendProbe(
            "configured_garment_simulator",
            bool(configured),
            configured,
            "GARMENT_SIMULATOR_COMMAND is configured" if configured else "GARMENT_SIMULATOR_COMMAND is not set",
        ),
        BackendProbe(
            "blender_sewing",
            integration_available,
            blender,
            "canonical-pattern importer and explicit loose-edge cloth sewing-spring recipe are installed"
            if integration_available
            else "Blender executable or canonical stitch-aware importer is missing",
        ),
    ]
    return tuple(probes)


def simulation_stage_status(pattern_json: Path) -> dict:
    probes = probe_simulation_backends()
    available = next((probe for probe in probes if probe.available), None)
    return {
        "stage": "SIMULATE",
        "status": "READY" if available else "BLOCKED_EXTERNAL_SIMULATOR",
        "input": pattern_json.name,
        "selected_backend": available.backend if available else None,
        "probes": [probe.to_dict() for probe in probes],
        "mock_or_proxy_used": False,
    }
