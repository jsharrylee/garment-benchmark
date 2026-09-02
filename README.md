# Garment Pattern Semantics Benchmark

[![Tests](https://github.com/jsharrylee/garment-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/jsharrylee/garment-benchmark/actions/workflows/tests.yml)

This project tests a specific intermediate step toward inverse garment design: whether a model can read a completed analytic sewing pattern and recover its semantic structure. The motivating application is game-character clothing, but this repository does **not** claim to reconstruct production patterns or 3D assets from game images.

The implemented result is a semantic parser for vector pattern panels. It identifies garment and panel roles, names boundary segments, derives landmarks from shared edge junctions, and proposes seam mates. The image-to-pattern and simulation stages remain separate experiments or future work.

![Semantic 2D pattern parser overview](reports/figures/pattern_semantic_parser_schematic_en.png)

## What the model reads and predicts

One input garment is a set of separate vector panels. Each panel is represented as an ordered cycle of line, quadratic Bezier, cubic Bezier, and circular-arc commands in a panel-local frame. Absolute canvas coordinates and source identifiers are withheld from the network.

The 950,820-parameter Transformer predicts:

- garment category and panel role;
- edge roles such as neckline, shoulder, armhole, side seam, and hem;
- FNP/SNP/SP-style landmarks derived from predicted edge-role junctions;
- candidate seam relations between panel edges.

Training used 1,983 top/skirt/pants patterns from one GarmentCodeData v2 batch, split into 1,587/198/198 garments by sample ID. The split is disjoint, but all three partitions come from the same generator.

In plain language, it reads geometry that already exists and explains what the pieces and boundary segments mean. It does not draw a new pattern.

## Results at a glance

These are component studies, not one integrated end-to-end system.

| Stage | Result | Evidence and scope |
|---|---|---|
| Complete vector pattern to semantic structure | **Bounded pass**: category accuracy `1.000`, panel-role macro-F1 `0.930`, edge-role macro-F1 `0.942`, landmark F1 `0.928` | 198 sample-ID-disjoint test garments from the same GarmentCode generator |
| Seam-mate reconstruction | **Main bottleneck**: raw seam-pair F1 `0.424`, symbolic seam F1 `0.593`, mate recall@1 `0.526` | Same 198-garment test set and parser |
| Four views to semantic coordinates | **Weak positive result**: normalized MAE `0.0424`, versus `0.0461` for a train-only category mean and `0.0417` for the vector-input teacher | 78 sample-ID-unseen garments; same generator, render style, and fixed neutral body |
| Four views to named curve parameterizations | **Partial**: parameter R² `0.259`, versus `0.204` for a matched global-token ablation | 144 same-domain test garments; fitted two-cubic targets, not original drafting formulas |
| Simulator-ready export | **Incomplete**: generic R12 outline DXF and separate stitch JSON exist; seam-aware CAD and predicted-pattern OBJ export do not | Implementation boundary, not a benchmark result |

The strongest current result is semantic interpretation of a complete vector pattern inside one generator domain. The seam graph, cross-source transfer, and precise pixel-to-CAD recovery are not solved.

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
- complete seam-graph recovery;
- CAD validity, fit, sewability, manufacturability, or simulation readiness;
- superiority to ReWeaver, Garment Particles, or another image-to-pattern model.

The four-view inputs are orthographic re-renders of GarmentCode meshes with a fixed neutral body and a simple material. They are not game screenshots.

## Portfolio artifacts

- [English technical portfolio](output/docx/semantic_pattern_bridge_portfolio_en.docx)
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
output/docx/         prebuilt technical portfolio
```

Datasets, source images, checkpoints, caches, external repositories, local environments, and large runtime artifacts are intentionally excluded.

## Licensing

Project-authored source code and software configuration are released under the [MIT License](LICENSE). This grant does not cover datasets, data-derived manifests, figures, reports, generated documents, model weights, or third-party material. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the exact scope and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and terms.

GarmentCodeData v2 is attributed under CC BY 4.0. Technical benchmark results do not grant permission to redistribute or publicly display source data.
