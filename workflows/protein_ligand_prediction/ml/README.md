# Optional topology GBDT inference

Feature extraction is available without model weights. Predicting affinity
requires a trusted external FS-AN model bundle with its fitted scalers and
recorded runtime. The examples contain input structures, not pretrained models.

The selected predictor averages seeds **0, 1 and 2** trained on the
**general-v2020R1** membership. Each pipeline uses its own fitted StandardScaler
and a 10,000-tree GBDT: learning rate 0.005, depth 7, min_samples_split 2,
subsample 0.4 and max_features `sqrt`. Predictions are unscaled pK values.
The installed [model profile](../../../src/topokit/workflows/protein_ligand_prediction/model_profile.json)
is the current parameter/identity reference. Historical feature-selection fits
used different settings and retain their own records.

## Extract, then predict

First follow the [topology guide](../README.md) to obtain a completed feature
store. Its exact schema, tensors and records are compatible with this loader.
Activate the model's recorded [runtime](environment.yml), install TopoKit there,
and supply a trusted bundle path:

```bash
python workflows/protein_ligand_prediction/ml/predict.py \
  --model-dir /path/to/trusted/general_v2020R1_bundle \
  --features examples/output/protein_ligand \
  --manifest examples/output/protein_ligand/manifest.csv \
  --output examples/output/predictions.csv
```

Run from the repository root. Do not pass a lone NPY array or a hand-serialized
schema: the loader verifies exact schema bytes, per-sample records and hashes.
The batch exporter's sample IDs need not be real PDB IDs; `pdb_id` is the loader's
manifest column name. No affinity labels are required for inference.

Ensemble bundles contain `ENSEMBLE.json` and member directories; historical
single-model bundles containing `MODEL.json` are also supported. Every pipeline
contains its fitted scaler. Pass raw features: the API scales once inside each
pipeline, then averages predictions. Do not fit a new scaler on test inputs.

The loader rejects mismatched runtimes/schemas, modified model/scaler files,
missing or mislabelled members, failed feature rows and existing output paths.
Joblib bundles must come from a trusted source. No model training, downloading,
validation split, CASF evaluation or scientific recipe change is performed.

The recorded final runtime is Python 3.12.14 / NumPy 2.5.2 / SciPy 1.18.1 /
scikit-learn 1.9.0 / joblib 1.5.3; see [requirements.txt](requirements.txt).
Installing `.[ml]` alone does not guarantee byte/runtime compatibility with an
existing bundle. Model distribution and model-specific permissions are separate
from the MIT package license.
