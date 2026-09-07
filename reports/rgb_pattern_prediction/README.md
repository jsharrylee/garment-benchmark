# Four-view RGB to Structured Sewing Patterns

This report directory is a curated internal research snapshot of predicting editable sewing-pattern structure from neutral four-view garment renders. It extends the pattern-semantics work in the repository but remains a separate component study, not an integrated end-to-end system.

## What was built

- A paired corpus connecting 3,450 local GarmentCodeData v2 garments to 13,800 neutral RGB views and round-trip-checked analytic pattern records.
- Pixel-aligned visible-panel instance masks accepted for 2,995 garments and 11,980 views.
- A ViT-B/16 panel-mask pretraining pipeline, direct autoregressive and set/cycle/graph baselines, a panel bank, a canonical line-configuration classifier, and a top-10 candidate reranker.
- A corrected mask target that supervises inter-panel boundary cells instead of discarding them as a thick ignore band.

On the same 2,027 internal-selection visible panels, the corrected encoder changed line-configuration top-1 from **50.27% to 52.74%** and reranked top-1 from **56.64% to 58.95%**. Target-in-top-10 coverage increased from **92.25% to 93.04%**.

These are **not full-pattern reconstruction results**. The downstream diagnostic first matches a panel query to a GT panel using mask-based Hungarian matching, then classifies only the canonical cyclic sequence of `L/Q/C/A` boundary primitive types. It does not predict coordinates, lengths, curvature, seams, sewing validity, fit, or a drawable pattern.

![Correct internal line-configuration examples](figures/line-configuration-correct-cases.png)

## Read more

- [Full Korean research report](RESEARCH_REPORT_KO.md)
- [Claim boundary](CLAIM_BOUNDARY_KO.md)
- [Preregistered next direction: source-edge-aware pretraining](FUTURE_WORK_LINE_MASK_PRETRAINING_KO.md)
- [Figure provenance](FIGURE_PROVENANCE.md)
- [Aggregate metrics](../../data/manifests/rgb_pattern_prediction/summary_metrics.json)

## Public snapshot scope

Raw GCDv2 archives, full RGB/mask collections, meshes, checkpoints, feature caches, and machine-local records are intentionally excluded. The included derivative figures are attributed under the upstream CC BY 4.0 license. See the repository [third-party notices](../../THIRD_PARTY_NOTICES.md).

```bash
python -m pytest -q benchmark/tests/test_rgb_pattern_signature_utils.py
python -m benchmark.scripts.build_github_release_zip --check-only
```
