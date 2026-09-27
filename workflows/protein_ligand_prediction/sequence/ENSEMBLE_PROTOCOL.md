# ESM-2 + CPZ affinity ensemble protocol

This protocol defines the external general-set affinity model. The public
package supports [sequence feature extraction](README.md); the selected CPZ
ensemble's fitted models and inference helper are not bundled.

## Feature and model contract

Use finite unscaled float32 `(N,1792)` embeddings in protein-then-ligand order:
1,280 ESM-2 coordinates followed by 512 CPZ BOS coordinates. The
[feature recipe](RECIPE.json) specifies checkpoint identities, pooling and
input policies. ChEMBL27-only embeddings have the same width but are not
compatible with this model.

The model comprises three StandardScaler + GBDT pipelines trained on the
18,498-member general-v2020R1 set with all CASF test IDs excluded. Each member
has its own training-only fitted scaler. Preserve all model/scaler identities
in [MODEL_SELECTION.json](MODEL_SELECTION.json) and estimator settings in
[ENSEMBLE_PARAMETERS.json](ENSEMBLE_PARAMETERS.json), together with the model's
recorded software versions. Apply scaling once inside each pipeline; do not
refit scalers on prediction inputs.

## Prediction aggregation

For each sample, average predictions from seeds 0, 1 and 2 with weights 1/3:

```text
prediction[i] = (seed0[i] + seed1[i] + seed2[i]) / 3
```

Preserve input IDs in the same order for all members. Compute RMSE, PCC and
MAE from the averaged prediction vector and the corresponding labels. Averaging
member metrics does not give ensemble metrics. Member mean and sample SD
instead describe variation across seeds.

The aggregation is fixed independently of benchmark rankings. Do not select
a seed or fit ensemble weights using CASF. General and refined cohorts have
distinct training memberships and are not interchangeable. Shared averaging
does not make sequence and topology feature bundles compatible.

## Evaluation and use limits

The [model card](MODEL_CARD.md) summarizes the ensemble and its availability.
[ENSEMBLE_RESULTS.md](ENSEMBLE_RESULTS.md) and
[ENSEMBLE_METRICS.json](ENSEMBLE_METRICS.json) retain reported results, including
member variation and a single-model comparison. Three runs on reused benchmarks
do not establish statistical significance.

A usable external distribution must provide all three trusted pipelines,
their fitted scalers and hash metadata, plus a compatible inference helper.
The public package does not provide that distribution. Its installed
`predict_gbdt` supports the separate ChEMBL27-only contract (internal `FS-AQ`)
and rejects the CPZ ensemble (internal `FS-AU`).
