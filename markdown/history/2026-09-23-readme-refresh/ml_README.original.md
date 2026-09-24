# Final topology GBDT ensemble inference

The default predictor averages seeds **0, 1 and 2** trained on **18,498
general-v2020R1 complexes**, excluding all three CASF tests. The canonical
[final ML folder](../../../../datasets/protein_ligand_prediction/final/topology_gbdt/README.md)
contains all twelve model/scaler pairs and six final ensemble evaluations.
[MODEL_SELECTION.json](../MODEL_SELECTION.json) records the final profile.

Each seed's `pipeline.joblib` contains its fitted StandardScaler followed by
the 10,000-tree GBDT. Its `scaler.npz` is a portable, inspectable copy. Pass raw
float32 FS-AN tensors; the API scales once inside each pipeline and averages
the resulting predicted pK values. Do not average RMSE/PCC or rescale input.

`topokit.workflows.protein_ligand_prediction.predict` and the `predict.py` CLI
accept the ensemble directory containing `ENSEMBLE.json`; legacy complete
single-model bundles containing `MODEL.json` remain supported. Missing members,
unequal weights, duplicate/mislabelled seeds, mismatched runtime/schema or
changed model/scaler files stop inference. Existing outputs are not overwritten.

```bash
python topokit/workflows/protein_ligand_prediction/ml/predict.py \
  --model-dir datasets/protein_ligand_prediction/pretrained/representative_gbdt \
  --features datasets/protein_ligand_prediction/features/representative_alpha15 \
  --manifest datasets/protein_ligand_prediction/labels/final/test_casf_2016.csv \
  --output datasets/protein_ligand_prediction/predictions/final_gbdt_casf2016.csv
```

Use [environment.yml](environment.yml) or [requirements.txt](requirements.txt)
for the recorded Python3.12.14/NumPy2.5.2/SciPy1.18.1/sklearn1.9.0/joblib1.5.3
runtime. The final artifact folder also retains a tested TopoKit wheel for
installing this ensemble-capable API. Model files and scalers remain external
data assets, never wheel contents.

Historical feature-selection ablations retain their original parameters and
scores in [ST-07](../../../../datasets/protein_ligand_prediction/experiments/protocol_ablation/studies/ST-07/README.md).
The final hyperparameters and three-seed prediction average are a subsequent
user-selected modeling stage. Superseded pipelines are retired only after
verified replacement; historical records and standalone scalers are preserved.
