# Representative Figure Provenance

These are deliberately selected qualitative diagnostics, not random samples and
not independent test-set evidence. They show what each metric measures and where
visually similar cases diverge.

## Dataset attribution

The neutral garment renders and source-derived target pattern geometry are
adapted from **GarmentCodeData v2**:

- Maria Korosteleva et al., *GarmentCodeData v2: 115,000+ made-to-measure garments with sewing patterns and simulated drapes*
- Official record: https://doi.org/10.3929/ethz-b-000690432
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Project-authored adaptations include neutral four-view compositing, panel-mask
targets and predictions, analytic edge plotting, labels, resizing, and
multi-panel layouts. Model outputs and figure compositions are project-authored.
No endorsement by the dataset creators is implied.

## Figure inventory

| File | Source sample(s) | Purpose and exact scope |
|---|---|---|
| `panel-mask-target-policy-old-vs-new.png` | `rand_0BYYF4ET0T` | Same RGB with the historical gray ignore-band target and the boundary-inclusive majority-ID target. This is a target-policy illustration, not a prediction. |
| `panel-mask-ordinary-strong-cases.png` | `rand_WDI9CL3VNE`, `rand_9048W5H2M7`, `rand_WHLZHHDMI2`, `rand_AGJAABUCUK`, `rand_SIFH5D3WLC` | Post-hoc internal examples with familiar silhouettes, exact panel count, and relatively strong mask geometry. |
| `panel-mask-error-cases.png` | `rand_2FPTWA78B3`, `rand_B32KOAGHI0`, `rand_B2C5B8V0SO`, `rand_OLVSBMGNZ0`, `rand_BPLV7VAHFS` | Five lowest-IoU internal examples used to inspect small panels, overlap, and over/under-segmentation. |
| `ordinary-garment-stronger-cases.png` | `rand_ZLFD860PWX`, `rand_2BWWUFZYF0`, `rand_NF31FYNRTZ` | Familiar silhouettes with relatively strong historical set/cycle/graph metrics. Not exact reconstruction. |
| `set-graph-relative-improvement.png` | `rand_F1HMCE6HDL` | Historical validation example where a panel-pretrained encoder improved count, geometry, and seam metrics. |
| `set-graph-regression.png` | `rand_G55CFYW2WX` | Historical counterexample where the panel-pretrained encoder overpredicted panels and edges. |
| `line-configuration-correct-cases.png` | `rand_0V45X4ELM8`, `rand_2UT239WX5N`, `rand_632J39T40I`, `rand_0FU48H5R3N`, `rand_166TVX8TI3` | Latest internal cases where final canonical L/Q/C/A cycle classification is correct. The center drawing is GT geometry, not model-generated geometry. |
| `line-configuration-error-cases.png` | `rand_0V45X4ELM8`, `rand_2UT239WX5N`, `rand_1HX9UGCJ83`, `rand_1226AZVFJZ`, `rand_166TVX8TI3` | Latest internal near-confusions. The model output is the displayed primitive sequence only. |

Panel display colors are sample-local. Where predicted mask colors visually match
GT colors, the assignment is applied after Hungarian matching for readability;
GT IDs were not supplied as visual input.

## Rebuilding

`benchmark/scripts/build_rgb_pattern_representative_figures.py` accepts repository, dataset,
experiment, and output roots as command-line arguments and contains no fixed
machine-local path. Rebuilding all figures requires separately obtained
GCDv2-derived artifacts and private experiment records, which are not included
in this snapshot.
