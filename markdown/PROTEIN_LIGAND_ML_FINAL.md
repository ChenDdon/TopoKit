# Final protein–ligand topology GBDT contract — September 23, 2026

Feature selection and final model fitting are separate stages. Historical
feature ablations retain learning_rate=0.002, min_samples_split=5,
subsample=0.8 and random_state=0, with 10,000 depth-7 trees and sqrt sampling.
Those settings and numerical results are not retroactively replaced.

The fixed FS-AN encoder remains unchanged: recipe
`hpcc-alpha15-cb4760366e92bd2e`, a 15 Å protein crop, physical alpha radii
0.1–5.0 Å, 55 null/element channels and ten L0 summaries, with a float32
`(10,50,55)` tensor flattened in C order to 27,500 features.

The user selected final GBDT hyperparameters learning_rate=0.005,
min_samples_split=2 and subsample=0.4 after the parameter-transfer comparison.
All other estimator settings match the selected ESM-2 + CPZ template, except
that the final topology ensemble explicitly sets random_state=0/1/2 in its
three members. Each has 10,000 trees, depth 7, sqrt sampling (165 candidate
features), squared-error loss, a training-only StandardScaler and unscaled
pK targets. No early stopping or validation holdout is used.

The final prediction is the arithmetic mean of the three predicted pK values.
RMSE/PCC/MAE are evaluated on that prediction vector. Means and sample standard
deviations of individual-seed metrics are reported separately. All three seeds
are retained; there is no best-seed selection or CASF-fitted ensemble weighting.

The canonical artifact folder is
`datasets/protein_ligand_prediction/final/topology_gbdt`. It holds three seeds
for each of refined-2007 (1,105 training rows), refined-2013 (2,764),
refined-2016 (3,772), and official general-v2020R1 (18,498). The matching refined
test is excluded; general excludes the union of all three CASF tests. Test
sizes remain 195/195/285. The general ensemble is the default companion model.

## Bundle and API

`models/{training_set}/ENSEMBLE.json` records the ordered seeds, equal weights,
schema, runtime and SHA256 of each member's `MODEL.json`. Each seed folder
contains a complete `pipeline.joblib` (fitted scaler then estimator), portable
`scaler.npz`, schema, resolved parameters, training manifest and provenance.
Keep these scaler/model pairs intact.

The existing `predict(model_dir, features, manifest, output, batch_size=128)`
interface recognizes either an `ENSEMBLE.json` or a legacy single `MODEL.json`.
Ensemble inference verifies all three members, feature schemas, runtimes,
checksums, training-set/seed identity and consistent hyperparameters. It
passes each raw float32 batch through each full pipeline once, then averages
the results. The output CSV remains `pdb_id,predicted_pK`; the returned summary
also includes `ensemble_size` and `seeds`. Missing or invalid members fail;
prediction never silently falls back to a smaller ensemble. Existing output
files are not overwritten. Optional ML dependencies remain lazily imported.

Model versions are Python3.12.14, NumPy2.5.2, SciPy1.18.1, sklearn1.9.0 and
joblib1.5.3. Trained weights are external data assets, never wheel contents.
Geometry, L0 construction and all higher-dimensional mathematics are unchanged.

## Preservation and retirement

Final selection is activated after all twelve fitted models/scalers, six
consensus outputs and public API inference pass verification. Superseded
FS-AN single-model pipelines may then be removed under the user's explicit
request. Historical parameters, manifests, predictions, scores, receipts and
standalone scaler arrays remain. Removal receipts list exact paths, hashes,
bytes and replacements; dated completion receipts remain historical evidence.
Sequence and DL model retention is outside this topology-model cleanup.
