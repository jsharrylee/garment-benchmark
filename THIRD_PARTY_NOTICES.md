# Third-Party Attribution and Public-Release Scope

This notice documents third-party projects and datasets referenced during the
benchmark. The public release does **not** vendor an upstream source tree,
dataset payload, pretrained weight, executable, restricted image, or private
cloud-storage link. Project-authored reports may retain citations and aggregate
experimental findings. Rights in third-party material remain governed by the
upstream terms described below; the repository's lack of a blanket project
license does not replace or narrow those upstream licenses.

## GarmentCodeData v2

- Title: *GarmentCodeData v2: 115,000+ made-to-measure garments with sewing
  patterns and simulated drapes*
- Creators: Maria Korosteleva, Timur Levent Kesdogan, Fabian Kemper, Stephan
  Wenninger, Jasmin Koller, Yuhan Zhang, Mario Botsch, and Olga
  Sorkine-Hornung
- Official record: https://doi.org/10.3929/ethz-b-000690432
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/

Changes made in this project: official vector sewing-pattern specifications
were parsed and normalized into panel-local analytic DSL records; project
semantic labels, split metadata, hashes, and aggregate statistics were then
derived. The portfolio schematics reproduce selected contours from sample
`rand_LKC1OG530J`; contour geometry is preserved while non-overlap placement,
scale, colour, and labels are project adaptations. The public release excludes
the original dataset archives, vector records, renders, meshes, and other source
payloads. No endorsement by the creators is implied.

## GarmentCode

- Official repository: https://github.com/maria-korosteleva/GarmentCode
- Upstream license: MIT License
- Copyright notice in the upstream license: Copyright (c) 2024 Maria
  Korosteleva

No GarmentCode source tree is vendored in this release. The repository contains
project-authored interoperability, analysis, and experiment code. Exact
generated counterfactual pattern geometry is excluded from the public archive.

## FreeSewing

- Archived official GitHub repository: https://github.com/freesewing/freesewing
- Current project home referenced by that repository:
  https://codeberg.org/freesewing/freesewing
- Upstream license: MIT License

No FreeSewing source tree or exact generated pattern geometry is included in the
public archive. Project-authored reports may describe a source-specific
experiment at an aggregate level.

## ReWeaver

- Official code repository: https://github.com/SII-LiMing/ReWeaver-Code
- Official GCD-TS dataset:
  https://huggingface.co/datasets/SII-LiMing/ReWeaver-GCD-TS
- Official pretrained weights: https://huggingface.co/SII-LiMing/ReWeaver
- GCD-TS and pretrained-weight repository license labels: CC BY-NC 4.0,
  https://creativecommons.org/licenses/by-nc/4.0/

No license file was visible in the root of the official ReWeaver code
repository at the time of this review, so no ReWeaver source code is copied or
redistributed here. The public archive also excludes GCD-TS images, annotations,
weights, and granular acquisition/evaluation manifests. Only citation and
aggregate project-authored findings remain.

## Garment Particles

- Official repository: https://github.com/garment-particles/GarmentParticles
- Official pretrained weights:
  https://huggingface.co/georgeNakayama/GarmentParticles
- Official dataset:
  https://huggingface.co/datasets/georgeNakayama/GarmentParticles
- Upstream repositories state the MIT License.

No Garment Particles source tree, dataset payload, or pretrained weight is
included in the public archive. The project-authored compatibility shim at
`benchmark/adapters/flash_attn_compat.py` provides a Windows fallback and
reconstructs the missing `LightningDiTCrossAttnVarlenBlockV2` interface by
adapting the structure of the upstream LightningDiT module. The upstream MIT
copyright and permission notice is preserved in
`THIRD_PARTY_LICENSES/GarmentParticles-MIT.txt`. References are otherwise
retained only for benchmark context.

## SynBody

- Official dataset repository: https://huggingface.co/datasets/caizhongang/SynBody
- License label: CC BY-NC-SA 4.0,
  https://creativecommons.org/licenses/by-nc-sa/4.0/

The public archive excludes SynBody images, annotations, archives, granular
sample configuration, local-acquisition or preprocessing manifests, and
sample-level reports. Project-authored reports may retain citation and
aggregate technical findings only.

## NOVA-Human

NOVA-Human was handled as an archived-official provenance link with an
unresolved reuse license and a private-evaluation-only boundary. The public
archive excludes the dataset, images, masks, metadata payloads, contact sheets,
live cloud-storage links, and restricted acquisition records. This notice does
not claim public-display or redistribution permission.
