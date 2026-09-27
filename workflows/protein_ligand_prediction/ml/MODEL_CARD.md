# Topo GBDT model card

## Purpose and availability

This optional regression model predicts unscaled pK from the predefined Topo
protein–ligand features. TopoKit includes guarded inference APIs and a source
checkout command described in the [prediction guide](README.md). It does not
include fitted pipelines, scalers or benchmark datasets. Model distribution and
permissions are separate from the package's MIT code license.

## Inputs and model identity

Inputs are finite raw float32 Topo tensors of shape `(10, 50, 55)`, flattened
in C order into 27,500 coordinates. The feature recipe is
`hpcc-alpha15-cb4760366e92bd2e`; its schema SHA-256 is
`34dde88aff5d7deab170347299f10b4153d2cdf3408a06d4f681b685490c0fa2`.
The loader checks canonical schema bytes and per-sample hashes in a completed
feature store. A matching feature width alone is insufficient.

The model is the equal-weight prediction ensemble of seeds 0, 1 and 2, each a
StandardScaler + GradientBoostingRegressor pipeline. Each scaler is fitted on
training rows only and applied once to raw inputs. The general-v2020R1 training
membership contains 18,498 complexes. The [model recipe](../MODEL_SELECTION.json)
records all estimator parameters, model/scaler identities and the required
software versions; the installed [model profile](../../../src/topokit/workflows/protein_ligand_prediction/model_profile.json)
is the inference identity reference.

## Use and limitations

Use a trusted compatible bundle and its recorded environment. Loading Joblib
models can execute Python code. Passing a different crop, radius grid, channel
order or summary definition violates this model's feature contract. Fit no new
scaler on prediction inputs.

The package's tests verify feature generation, compatibility checks and
inference mechanics; they do not establish accuracy for a new target, chemical
series or experimental assay. An evaluation for the user's scientific question
requires suitable held-out data and an explicit split. Do not select ensemble
members or weights using the evaluation set. Feature extraction itself does not
fit, evaluate or supply an affinity model.
