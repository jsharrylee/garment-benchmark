"""Four-view inference through a constrained parametric drafting decoder.

This module is intentionally separate from the legacy affine residual editor
so both candidates can be evaluated on the same frozen predictions.  It never
accepts a source pattern at inference time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.drafting_semantics.basic_blocks import PROVENANCE_STATUS
from benchmark.drafting_semantics.tshirt_parametric_projection import (
    TShirtParametricProjection,
    TShirtProjectionConfig,
    fit_tshirt_drafting_parameters,
    materialize_tshirt_projection_graph,
)
from benchmark.drafting_semantics.tshirt_parametric_decoder import (
    CanonicalPatternGraph,
    audit_sampled_tshirt_document_sleeve_constraint,
)
from benchmark.pattern_pipeline.four_view_semantic_inference import (
    FourViewFeatureBundle,
    LoadedFourViewStudent,
    StaticSemanticPrediction,
    predict_static_semantic_queries,
)
from benchmark.pattern_pipeline.schema import PatternDocument
from benchmark.pattern_pipeline.validation import ValidationReport, validate_pattern


PARAMETRIC_INFERENCE_SCHEMA_VERSION = "four-view-parametric-drafting-inference/v1"


@dataclass(frozen=True)
class FourViewParametricDraftingResult:
    document: PatternDocument
    prediction: StaticSemanticPrediction
    projection: TShirtParametricProjection
    graph: CanonicalPatternGraph
    validation: ValidationReport
    receipt: Mapping[str, Any]

    def save(self, pattern_path: str | Path, receipt_path: str | Path) -> None:
        """Write the vector draft and a path-free numeric/text receipt."""

        self.document.write_json(Path(pattern_path))
        resolved = Path(receipt_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def decode_static_tshirt_prediction(
    prediction: StaticSemanticPrediction,
    *,
    config: TShirtProjectionConfig | None = None,
    output_id: str = "tshirt_four_view_parametric",
    _visual_origin_attested: bool = False,
) -> FourViewParametricDraftingResult:
    """Decode a semantic table whose upstream origin may be unattested.

    Callers that only hold ``StaticSemanticPrediction`` cannot prove whether it
    came from images or a target-derived synthetic table.  The end-to-end
    four-view wrapper is the only public path that sets the visual attestation.
    """

    if prediction.category != "tshirt":
        raise ValueError("the current parametric decoder supports T-shirts only")
    projection = fit_tshirt_drafting_parameters(
        prediction.coordinates,
        prediction.presence_probability,
        prediction.coordinate_confidence,
        query_mask=prediction.query_mask,
        config=config,
        sample_id=output_id,
    )
    graph = materialize_tshirt_projection_graph(
        projection,
        pattern_id=output_id,
        samples_per_cubic=49,
    )
    raw = graph.to_pattern_document()
    exact_constraint = graph.sleeve_head_constraint
    sampled_constraint = audit_sampled_tshirt_document_sleeve_constraint(
        raw,
        sleeve_ease_cm=exact_constraint.sleeve_ease_cm,
        samples_per_cubic=graph.samples_per_cubic,
    )
    origin_contract = (
        {
            "modality": "FOUR_PRECOMPUTED_VISUAL_FEATURES_ONLY",
            "visual_origin_attested": True,
            "target_pattern_used_for_fit": False,
        }
        if _visual_origin_attested
        else {
            "modality": "STATIC_SEMANTIC_PREDICTION",
            "visual_origin_attested": False,
            "target_pattern_used_for_fit": "NOT_ATTESTED",
        }
    )
    document = replace(
        raw,
        pattern_id=output_id,
        generator=(
            "four-view semantic Transformer + bounded drafting parameters + "
            "instance graph + seam constraint solver"
        ),
        provenance={
            **raw.provenance,
            "status": PROVENANCE_STATUS,
            "industrial_pattern_claim": False,
            "visual_student_input_only": bool(_visual_origin_attested),
        },
        annotations={
            **raw.annotations,
            "four_view_parametric_inference": {
                "schema_version": PARAMETRIC_INFERENCE_SCHEMA_VERSION,
                "input_contract": origin_contract,
                "body_prior": "fixed category default",
                "post_decoder_affine_residual": False,
                "archetype_constraint_audit": projection.constraint_audit.to_dict(),
                "physical_graph_constraint": asdict(exact_constraint),
                "sampled_document_constraint": sampled_constraint.to_dict(),
            },
        },
    )
    validation = validate_pattern(document)
    if (
        not validation.accepted
        or not projection.constraint_audit.passed
        or not exact_constraint.converged
        or abs(exact_constraint.residual_cm) > exact_constraint.tolerance_cm
        or not sampled_constraint.passed
    ):
        raise RuntimeError("parametric drafting inference failed structural postconditions")
    receipt = {
        "schema_version": PARAMETRIC_INFERENCE_SCHEMA_VERSION,
        "status": "APPLIED_VALIDATED_PARAMETRIC_DRAFT",
        "category": "tshirt",
        "input_contract": {
            **origin_contract,
            "pattern_input": "REJECTED",
        },
        "projection": projection.to_receipt(),
        "physical_pattern_graph": {
            "schema_version": graph.schema_version,
            "parameter_count": len(graph.parameters.to_vector()),
            "panel_count": len(graph.panels),
            "path_count": len(graph.paths),
            "stitch_count": len(graph.stitches),
            "symmetry_relation_count": len(graph.symmetry_relations),
            "all_path_ids_are_instance_aware": all("#" in path.id for path in graph.paths),
            "left_right_geometry": "EXACT_REFLECTION",
            "shared_landmarks": "REFERENCED_BY_ID",
            "analytic_cubic_sleeve_head_constraint": asdict(exact_constraint),
            "sampled_document_sleeve_head_constraint": sampled_constraint.to_dict(),
        },
        "validation": validation.to_dict(),
        "output": {
            "pattern_id": document.pattern_id,
            "panel_count": len(document.panels),
            "stitch_count": len(document.stitches),
            "source_images_embedded": False,
            "source_paths_embedded": False,
        },
        "claim_boundary": (
            ("Attested four-view input; " if _visual_origin_attested else "Input origin unattested; ")
            + "same-generator provisional technical draft; not family-disjoint, "
            "cross-domain, expert-approved, or industrial pattern truth."
        ),
    }
    return FourViewParametricDraftingResult(
        document=document,
        prediction=prediction,
        projection=projection,
        graph=graph,
        validation=validation,
        receipt=receipt,
    )


def infer_parametric_tshirt_pattern(
    loaded: LoadedFourViewStudent,
    bundle: FourViewFeatureBundle,
    *,
    config: TShirtProjectionConfig | None = None,
    output_id: str = "tshirt_four_view_parametric",
    pattern_input: Any | None = None,
) -> FourViewParametricDraftingResult:
    """Run the complete four-view -> semantic -> parameter -> pattern lane."""

    prediction = predict_static_semantic_queries(
        loaded,
        bundle,
        category="tshirt",
        pattern_input=pattern_input,
    )
    return decode_static_tshirt_prediction(
        prediction,
        config=config,
        output_id=output_id,
        _visual_origin_attested=True,
    )


__all__ = [
    "PARAMETRIC_INFERENCE_SCHEMA_VERSION",
    "FourViewParametricDraftingResult",
    "decode_static_tshirt_prediction",
    "infer_parametric_tshirt_pattern",
]
