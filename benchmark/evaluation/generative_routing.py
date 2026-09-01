from __future__ import annotations


def assess_generated_patterns(reweaver_receipt: dict, garment_particles_summary: dict, garment_particles_receipt: dict) -> dict:
    """Route only between generated outputs; never retrieve or select a pattern template."""

    reweaver_structural = reweaver_receipt.get("structural_export") == "PASS"
    particles_valid = garment_particles_summary.get("valid") is True
    particles_structural = garment_particles_receipt.get("structural_export") == "PASS"
    if reweaver_structural:
        primary = "reweaver"
        technical = "DRAFT_PATTERN_AVAILABLE"
    elif particles_valid and particles_structural:
        primary = "garment_particles"
        technical = "DRAFT_PATTERN_AVAILABLE"
    elif particles_valid:
        primary = "garment_particles"
        technical = "DRAFT_REQUIRES_REPAIR"
    else:
        primary = None
        technical = "NO_VALID_GENERATED_DRAFT"

    maximum_gap = garment_particles_summary.get("panel_closure_gap_max")
    closure_review = bool(maximum_gap is not None and float(maximum_gap) > 5.0)
    return {
        "technical_status": technical,
        "primary_generated_draft": primary,
        "reweaver": {
            "structural_status": reweaver_receipt.get("structural_export"),
            "role": "primary_multiview_generator" if reweaver_structural else "rejected_by_structural_gate",
        },
        "garment_particles": {
            "valid_output": particles_valid,
            "structural_status": garment_particles_receipt.get("structural_export"),
            "role": (
                "primary_single_view_generator"
                if primary == "garment_particles" and particles_structural
                else "primary_single_view_draft_requiring_repair"
                if primary == "garment_particles"
                else "rejected_by_structural_gate"
            ),
            "panel_count": garment_particles_summary.get("panel_count"),
            "edge_count": garment_particles_summary.get("edge_count"),
            "stitch_pair_count": garment_particles_summary.get("stitch_pair_count"),
            "closure_review_required": closure_review,
        },
        "manufacturing_status": "BLOCKED_NO_STITCH_AWARE_SIMULATION",
        "visual_fidelity_status": "UNVERIFIED_NO_SIMULATED_RERENDER",
        "generation_contract": {
            "variable_topology": True,
            "template_retrieval": False,
            "nearest_pattern_selection": False,
        },
    }
