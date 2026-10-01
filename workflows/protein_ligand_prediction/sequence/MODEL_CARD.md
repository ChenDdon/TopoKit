# ESM-2 + CPZ model card

## Supported workflow and availability

TopoKit supports frozen **sequence feature extraction** through
`SequenceEncoder(ligand_profile="cpz")` and the repository extraction script.
The [usage guide](README.md) documents installation and model assets. Encoder
weights must be provided as trusted local files; TopoKit does not download them.

The optional selected CPZ affinity ensemble has no bundled inference helper,
fitted pipelines or benchmark data. Its recorded profile documents an external
model and does not make affinity prediction available from a public clone alone.

## Inputs and features

Inputs are explicitly selected full protein-chain sequences and a prepared
canonical isomeric ligand SMILES. ESM-2 `esm2_t33_650M_UR50D` supplies a
1,280-coordinate residue-mean representation, weighted by residue count across
nonoverlapping windows and chains. CPZ `chembl27_pubchem_zinc_512` supplies
512 coordinates from the ligand BOS state. Output is unscaled finite
float32 `[1792]`, protein first and ligand second.

[RECIPE.json](RECIPE.json) pins model revisions, checkpoint hashes, pooling and
token policies. This representation is independent of bound 3D coordinates.
Select the intended protein chains and prepared ligand, supply the pinned
assets, and set `ligand_profile="cpz"` explicitly.

## Optional affinity ensemble

The model is an equal-weight prediction mean of three StandardScaler + GBDT
pipelines, using seeds 0, 1 and 2. Each is trained on the general-v2020R1
membership of 18,498 complexes with all CASF test IDs excluded. Each pipeline
applies its training-fitted scaler once and returns unscaled logKa/pK.
[MODEL_SELECTION.json](MODEL_SELECTION.json) records identities, software
versions and cohort information; [ENSEMBLE_PARAMETERS.json](ENSEMBLE_PARAMETERS.json)
contains the exact estimator settings. [ENSEMBLE_PROTOCOL.md](ENSEMBLE_PROTOCOL.md)
defines aggregation and compatibility requirements.

## Evaluation and limitations

| Test | Count | Ensemble RMSE | Ensemble PCC | Ensemble MAE |
| --- | ---: | ---: | ---: | ---: |
| CASF-2007 | 195 | 1.400593 | 0.825812 | 1.086143 |
| CASF-2013 | 195 | 1.402958 | 0.799008 | 1.118615 |
| CASF-2016 | 285 | 1.202560 | 0.849905 | 0.934954 |

Metrics come from the averaged prediction vector; RMSE/MAE use logKa/pK.
[ENSEMBLE_RESULTS.md](ENSEMBLE_RESULTS.md) retains member metrics, seed variation
and the single-model comparison, with full precision in
[ENSEMBLE_METRICS.json](ENSEMBLE_METRICS.json). Seed variation is not ensemble
uncertainty. The averaging rule is independent of benchmark rankings and does
not improve every reported metric. Three runs on reused CASF benchmarks do
not establish statistical significance or accuracy on new targets. These
reported external evaluations are separate from the package's extraction tests;
a public clone alone cannot reproduce them without the external models and data.
