# PPI Topo GBDT model card

**Model:** `ppi-topo-gbdt-2368-3seed-v1`.
**Task:** estimate signed ΔG_bind (kcal/mol) from an explicitly selected
protein–protein complex structure. The output is binding affinity, not an
interaction probability or a mutation-induced ΔΔG.

## Inputs and prediction

The fixed recipe yields a float32 `(10, 14, 36)` tensor, flattened in C order to
5,040 values: ten spectral summaries, 14 alpha radii from 1.0 to 7.5 Å in 0.5 Å
steps, and 36 ordered atom-category channels. It uses a 20 Å interface crop.
The encoder orders protein1/protein2 by descending observed ATOM residue count
before cropping; ties use first chain appearance in the structure. Prediction
uses that canonical feature order with no partner-swap augmentation.
Preprocessing and structure-selection rules are
specified in the [recipe](../README.md).

Each of the three pipelines contains a fitted `StandardScaler` followed by a
`GradientBoostingRegressor`. Each regressor has 10,000 trees, learning rate 0.005,
maximum depth 7, subsample 0.4 and `max_features="sqrt"`. The seeds are 0, 1 and 2.
The output is their arithmetic mean; the target has no sign inversion, scaling
or logarithmic conversion. Each final pipeline was fitted on all 2,368 training
complexes. Exact estimator parameters, file hashes and runtime versions appear
in the installed
[model profile](../../../src/topokit/workflows/protein_protein_prediction/model_profile.json).

## Evaluation of the selected recipe

The selected recipe was evaluated using ten shuffled complex-level folds
(`KFold`, split seed 0). For each held-out complex, the three fold-model
predictions were averaged. Pooled out-of-fold results across 2,368 complexes are:

| Metric | Value |
| --- | ---: |
| Pearson correlation | 0.589684 |
| RMSE (kcal/mol) | 2.024487 |
| MAE (kcal/mol) | 1.578161 |
| R² | 0.339364 |

[MODEL_SELECTION.json](../MODEL_SELECTION.json) records the selected summary.
The same folds informed recipe selection, so these values are not an independent
held-out estimate of the selected model. Related proteins may occur in different
folds; performance on new protein families is not established. The final
all-data pipelines are for inference and were not used to produce these
held-out scores.

## Distribution and limitations

The public package includes the inference code and pinned model identities.
It does not include trained weights, training data, training programs or research
execution records. Predictions require locally supplied trusted weights and the
recorded compatible runtime: Python 3.12 (training used 3.12.11; patch releases
are accepted), with the exact NumPy, SciPy, scikit-learn and joblib versions in
the profile. Feature extraction needs no model weights.

The model depends on prepared structures in a shared coordinate frame and the
canonical partner ordering defined by the feature recipe. It does not dock proteins, select biological
assemblies, establish an interaction from two sequences, or validate an assay.
Extrapolation beyond the training distribution is unvalidated. The model's
outputs and its seed-to-seed spread are research estimates, not calibrated
uncertainties or experimentally measured affinities.
