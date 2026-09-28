# PPI binding affinity with Topo features

The predefined predictor uses the **5,040 PPI topology features** from the
[structure recipe](../README.md). It returns signed binding free energy,
**ΔG_bind in kcal/mol**, by averaging three GBDT predictions. This is a regression
model for a supplied complex structure, not a classifier of whether arbitrary
protein pairs interact.

## Requirements

Feature extraction runs with the standard TopoKit installation. Prediction also
requires the three trusted pretrained pipelines and their recorded runtime:
**Python 3.12, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, joblib 1.5.3**.
Training used Python 3.12.11; Python 3.12 patch releases are accepted.
Use a separate Python 3.12 environment, then run from the repository root:

```bash
python -m pip install -r workflows/protein_protein_prediction/ml/requirements.txt
python -m pip install .
```

Model weights are **not included in the repository or the installed package**,
and this workflow does not download them. Obtain the selected files from the
maintainer or your own trusted copy of the recorded bundle. If those assets are
unavailable, the topology feature extraction workflow remains usable; pretrained
affinity prediction is unavailable.

Place the three files in a directory outside the source tree:

```text
trusted-ppi-models/
  seed_0.joblib
  seed_1.joblib
  seed_2.joblib
```

The expected identities and SHA-256 hashes are pinned in the installed
[`model_profile.json`](../../../src/topokit/workflows/protein_protein_prediction/model_profile.json).
Joblib files can execute code when opened. Supply only trusted assets. All three
hashes, Python major/minor and exact dependency versions are checked before any model is loaded;
editing a bundle manifest cannot change the package's expected hashes.

## Predict a completed feature batch

First generate a feature directory using the [PPI exporter](../README.md).
The exporter canonicalizes protein1/protein2 order by descending observed ATOM
residue count before cropping, with ties resolved by first chain appearance in
the structure. Keep that stored feature order; prediction does not swap partners.

```bash
python workflows/protein_protein_prediction/ml/predict.py \
  --features examples/output/ppi \
  --models /path/to/trusted-ppi-models \
  --output examples/output/ppi_predictions.csv
```

The CSV preserves sample order and contains `sample_id`,
`predicted_dg_bind_kcal_mol`, the three seed predictions and `model_profile_id`.
An existing output file is never overwritten. The three-seed spread is not a
calibrated uncertainty estimate.

The installed Python API is also available:

```python
from topokit.workflows.protein_protein_prediction import predict_gbdt

result = predict_gbdt("features/ppi", "/path/to/trusted-ppi-models")
print(result.sample_ids)
print(result.prediction)  # signed DeltaG_bind, kcal/mol
```

The reader requires a complete batch receipt, matching artifact hashes, the
canonical feature schema and a finite float32 `(N, 5040)` matrix. Each pipeline
contains its already fitted `StandardScaler`; do not fit a new scaler, change
the feature order, average partner orientations, or transform the target.
See the [model card](MODEL_CARD.md) for model scope and evaluation limits.
