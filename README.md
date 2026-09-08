# Four-view RGB to Editable Sewing Patterns

[![Tests](https://github.com/jsharrylee/garment-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/jsharrylee/garment-benchmark/actions/workflows/tests.yml)

This repository studies an inverse-design pipeline that starts from neutral front, back, left, and right garment renders and aims to recover an editable analytic 2D sewing pattern. The intended downstream path is pattern editing, seam assembly, and CLO simulation, but the current public evidence covers only the intermediate stages described below—not completed image-to-CAD reconstruction.

An initial autoregressive model attempted to generate panel topology, continuous geometry, and seams together. It failed to produce a valid closed pattern on all 173 internal validation garments. That negative result motivated a staged formulation:

```text
four-view RGB
    -> visible-panel discovery
    -> discrete panel-structure retrieval
    -> continuous geometry refinement
    -> seam graph
    -> validation and CLO export
```

The implemented public results currently reach the first two stages. Continuous coordinate/control-point refinement, seam-graph recovery, and end-to-end CLO export remain unfinished.

![Ordinary garments: four-view input, visible-panel ground truth, and model prediction](reports/rgb_pattern_prediction/figures/panel-mask-ordinary-strong-cases.png)

## Paired data and analytic target

The study connects 3,450 GarmentCodeData v2 garments to 13,800 neutral orthographic renders: front, back, left, and right. Their 37,031 source panels are represented as ordered cycles of line (`L`), quadratic Bezier (`Q`), cubic Bezier (`C`), and circular-arc (`A`) primitives. The source pattern, analytic representation, and decoded geometry were round-trip checked before model experiments.

Pixel-aligned visible-panel supervision passed quality gates for 2,995 garments and 11,980 views. Raw patterns, RGB/mask collections, meshes, and checkpoints are not redistributed in this public snapshot.

## Why the problem was decomposed

The direct baseline asked one decoder to decide variable panel counts, edge counts and primitive types while also regressing lengths, angles, curve controls, and seam relations. In free generation, early structural errors cascaded into open panels and invalid downstream seams; valid-pattern rate was `0/173` on the internal validation split.

The revised experiments therefore ask smaller diagnostic questions:

1. Can a four-view encoder separate the visible source panels?
2. Given a GT-matched panel query, can it retrieve the panel's rotation/reversal-invariant cyclic `L/Q/C/A` boundary configuration from a train-only bank?
3. Only after those stages are reliable, can a model refine actual length, angle, endpoint, and control-point values and recover seams?

This decomposition makes failures attributable to visual grounding, discrete structure selection, continuous geometry, or sewing constraints instead of collapsing them into one invalid output.

## Results at a glance

These are component studies, not one integrated end-to-end system. The four-view results use an internal selection population and must not be read as full-pattern reconstruction rates.

| Stage | Result | Evidence and scope |
|---|---|---|
| Complete vector pattern to semantic structure | **Bounded pass**: category accuracy `1.000`, panel-role macro-F1 `0.930`, edge-role macro-F1 `0.942`, landmark F1 `0.928` | 198 sample-ID-disjoint test garments from the same GarmentCode generator |
| Seam-mate reconstruction | **Main bottleneck**: raw seam-pair F1 `0.424`, symbolic seam F1 `0.593`, mate recall@1 `0.526` | Same 198-garment test set and parser |
| Four views to semantic coordinates | **Weak positive result**: normalized MAE `0.0424`, versus `0.0461` for a train-only category mean and `0.0417` for the vector-input teacher | 78 sample-ID-unseen garments; same generator, render style, and fixed neutral body |
| Four views to named curve parameterizations | **Partial**: parameter R² `0.259`, versus `0.204` for a matched global-token ablation | 144 same-domain test garments; fitted two-cubic targets, not original drafting formulas |
| Four views to visible-panel decomposition | **Internal target-policy A/B**: matched-panel IoU `0.6003 → 0.6813` and inter-panel boundary IoU `0.4475 → 0.5212`; panel-count MAE worsened `1.5584 → 1.7727` | Same corrected target on 154 internal-selection garments; seed 17; official test untouched |
| GT-matched panel query to boundary primitive cycle | **Partial internal result**: top-1 `52.74%`, target-in-top-10 `93.04%`, reranked top-1 `58.95%` | 2,027 internal-selection visible panels; GT-mask Hungarian matching; canonical `L/Q/C/A` cycle only |
| Simulator-ready export | **Incomplete**: generic R12 outline DXF and separate stitch JSON exist; seam-aware CAD and predicted-pattern OBJ export do not | Implementation boundary, not a benchmark result |

The strongest current image-based result is visible-panel decomposition followed by canonical boundary-primitive selection under GT panel matching. The seam graph, continuous geometry, cross-source transfer, and precise pixel-to-CAD recovery are not solved.

## Four-view RGB to panel-boundary diagnostics

A newer component study connected 3,450 GCDv2 garments to 13,800 neutral four-view renders and round-trip-checked analytic pattern records. It first trained a ViT-B/16 model to decompose visible garment pixels into source panels, then tested whether a GT-matched panel query could select the panel's cyclic boundary primitive configuration.

Correcting the mask target so that panel interfaces were supervised instead of discarded as a thick gray ignore band improved both boundary IoU and downstream `L/Q/C/A` configuration selection on the same internal population. The drawing in the center of the figure below is **GT geometry for interpretation**, not model-generated geometry; the actual model output shown on the right is only a primitive-type sequence.

![Internal line-configuration examples](reports/rgb_pattern_prediction/figures/line-configuration-correct-cases.png)

- [English overview](reports/rgb_pattern_prediction/README.md)
- [Full Korean research report](reports/rgb_pattern_prediction/RESEARCH_REPORT_KO.md)
- [Exact claim boundary](reports/rgb_pattern_prediction/CLAIM_BOUNDARY_KO.md)
- [Next experiment: source-edge-aware pretraining](reports/rgb_pattern_prediction/FUTURE_WORK_LINE_MASK_PRETRAINING_KO.md)
- [Figure provenance](reports/rgb_pattern_prediction/FIGURE_PROVENANCE.md)

This study does **not** demonstrate continuous geometry recovery, seam prediction, a drawable pattern, fit, simulation readiness, external-photo generalization, multi-seed stability, or untouched-test performance.

## Supporting benchmark: semantics of completed vector patterns

Before the image-based study, this repository built a 950,820-parameter Transformer that reads an already completed set of analytic vector panels and predicts garment category, panel and edge roles, role-junction landmarks, and candidate seam mates. This supporting benchmark established the pattern representation and exposed seam matching as a separate bottleneck; it is not the same model as the four-view pipeline.

On 198 sample-ID-disjoint garments from the same GarmentCode generator, it recorded panel-role macro-F1 `0.930`, edge-role macro-F1 `0.942`, role-junction landmark F1 `0.928`, and symbolic seam F1 `0.593` from a raw `0.424`.

![Supporting semantic-parser schematic](reports/figures/pattern_semantic_parser_schematic_en.png)

## Negative results that define the boundary

### Cross-source semantics

An early T-shirt-only model reached edge macro-F1 `0.981` and landmark-existence F1 `0.965` on held-out GarmentCode patterns, but fell to edge macro-F1 `0.183` and landmark-existence F1 `0.000` on FreeSewing. Median landmark error increased from about `2.2 cm` to `24.1 cm`.

This result belongs to the early T-shirt model. The current unified DSL parser has not yet received a proper cross-source evaluation, so cross-source generalization is **unproven**, not automatically failed.

### Direct raster panel image to CAD graph

A separate raster model received one normalized image per panel and attempted to recover its ordered vector graph. On garment-ID-disjoint samples from the same generator it achieved silhouette IoU `0.717`, but exact ordered graph-edge F1 was only `0.004` and ordered-vertex MAE was `5.22 cm`. It recognized coarse shape while failing at CAD-level topology and coordinates.

### Symbolic consistency is not target selection

Symbolic role-cycle projection removed all 168 grammar violations in a retrieved candidate beam, yet exact target-topology top-1 stayed unchanged at `15.66%`. A trained visual-to-DSL retriever did improve rank-10 topology coverage from `36.36%` to `46.46%` over a parameter-free raw-FPN nearest-neighbour baseline, but rank-1 improved by only `1.01` percentage points. The semantic layer helps construct a cleaner candidate set; it does not identify the correct target pattern by itself.

## Claim boundary

The evidence supports same-generator parsing of GCDv2-derived weak semantic labels and limited component-level transfer from synthetic four-view renders. It does not establish:

- expert-approved industrial drafting semantics;
- recipe-family, body, renderer, generator, or real-image generalization;
- end-to-end four-view-to-pattern generation;
- direct recovery of boundary coordinates, lengths, curvature, control points, or source-edge splits from RGB;
- complete seam-graph recovery;
- CAD validity, fit, sewability, manufacturability, or simulation readiness;
- superiority to ReWeaver, Garment Particles, or another image-to-pattern model.

The four-view inputs are orthographic re-renders of GarmentCode meshes with a fixed neutral body and a simple material. They are not game screenshots.

## Portfolio artifacts

- [English technical portfolio — four-view RGB study as the main narrative](output/docx/Jinseob_Lee_CLO_Portfolio_v9_RGB_to_Pattern.docx)
- [Four-view RGB-to-pattern research report](reports/rgb_pattern_prediction/README.md)
- [Full Korean research report](reports/rgb_pattern_prediction/RESEARCH_REPORT_KO.md)
- [Exact claim boundary](reports/rgb_pattern_prediction/CLAIM_BOUNDARY_KO.md)
- [System schematic](reports/figures/pattern_semantic_parser_schematic_en.png)
- [Analytic DSL example](reports/figures/pattern_dsl_semantic_example_en.png)

The figures are project-produced adaptations that include an attributed GarmentCodeData v2 panel contour. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the source, modifications, and CC BY 4.0 attribution.

## Run the public checks

The test suite uses synthetic fixtures and compact manifests. It needs no dataset, checkpoint, GPU, or external model repository; two tests that require local GCDv2 payloads remain explicitly skip-gated.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"
python -m pip install -e ".[test,vision]"
python -m pytest -q -ra benchmark/tests
```

The GitHub Actions matrix runs the suite on Python 3.10 and 3.12 with CPU-only PyTorch and headless OpenCV.

Verify the committed SHA-256 manifest against both Git blob contents and working-tree bytes:

```bash
python verify_release_manifest.py
```

Validate the public-release allowlist, UTF-8/LF policy, document package, and content-safety checks without writing an archive:

```bash
python -m benchmark.scripts.build_github_release_zip --check-only
```

To build the public archive:

```bash
python -m benchmark.scripts.build_github_release_zip
```

The archive contains a `game-garment-benchmark/` root and a non-recursive `RELEASE_MANIFEST.sha256.json` that records every other payload file's byte size and SHA-256 digest.

## Reproducibility boundary

This public repository is a source-and-evidence snapshot. A visitor can run the offline tests, verify release bytes, inspect implementation and frozen manifests, and read the prebuilt portfolio artifacts.

The reported training and evaluation numbers cannot be independently re-derived from this snapshot, and ready-made inference cannot be run, because trained checkpoints, source vector-pattern records, renders, external repositories, and third-party weights are excluded. Full experimental reproduction requires obtaining the upstream data under its own license and retraining the recorded experiments.

The DOCX and figures are prebuilt review artifacts. Their builder is retained for provenance, but a standalone rebuild is not supported by this public snapshot because it omits the original local GCDv2 evidence inputs and uses Windows font paths.

## Repository structure

```text
benchmark/           parser, retrieval, preprocessing, evaluation, and tests
data/manifests/      split contracts, provenance, hashes, metrics, and claim boundaries
reports/figures/     attributed public figures
reports/rgb_pattern_prediction/  four-view component report and attributed diagnostics
output/docx/         prebuilt technical portfolio
```

Datasets, source images, checkpoints, caches, external repositories, local environments, and large runtime artifacts are intentionally excluded.

## Licensing

Project-authored source code and software configuration are released under the [MIT License](LICENSE). This grant does not cover datasets, data-derived manifests, figures, reports, generated documents, model weights, or third-party material. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the exact scope and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and terms.

GarmentCodeData v2 is attributed under CC BY 4.0. Technical benchmark results do not grant permission to redistribute or publicly display source data.
