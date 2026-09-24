# Workflow applications

The selected protein–ligand demonstration is
[protein_ligand_prediction](protein_ligand_prediction/README.md): **FS-AN**,
15 Å crop, 50 alpha radii 0.1–5.0 Å, 55 channels, ten summaries and 27,500 features.
Its encoder and guarded GBDT inference interface are installed in
`topokit.workflows.protein_ligand_prediction`, within the sixth package module.
The default predictor is the general-v2020R1 three-seed GBDT ensemble. Every
member retains its own fitted StandardScaler; the interface averages predicted
pK values. [Final parameters, models and results](../../datasets/protein_ligand_prediction/final/topology_gbdt/README.md)
are separate companion assets. Historical feature-selection ablations keep
their original parameters and scores.

All m/r ablations, source history, results and planned TopoFormer hyperparameter
studies are organized in the external
[protocol study catalogue](../../datasets/protein_ligand_prediction/experiments/protocol_ablation/README.md).
The package demonstration contains no alternate feature engine or research launcher.

The separate [atom-deletion](protein_ligand_prediction_version2/README.md) and
[H0 barcode](protein_ligand_persistent_homology/README.md) research receipts remain
in their original locations. They are not installed APIs or alternative defaults
for this selected application, and their jobs/artifacts were not changed.
