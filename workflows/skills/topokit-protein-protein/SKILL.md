---
name: topokit-protein-protein
description: Extract predefined PPI Topo features from a prepared protein–protein complex with explicit chain groups or a manifest, verify the feature store, and run requested affinity inference with compatible external models.
---

# Protein–protein Topo workflow

Read the [PPI guide](../../protein_protein_prediction/README.md) for inputs and
outputs and [default recipe](../../protein_protein_prediction/DEFAULT_RECIPE.json)
for feature identity. Locate the checkout by `pyproject.toml`; use the Python
environment with that copy installed. Base extraction needs only NumPy/SciPy.

## Identify the complex

Obtain one prepared PDB/mmCIF file, in angstroms, and explicit author-chain
lists for the two binding partners. Keep multi-chain partners intact. Resolve
ambiguous partner identity before proceeding; do not silently dock, align,
repair, expand assemblies or translate chain namespaces. The recipe uses first-
model ATOM rows, atom-wise highest-occupancy selection, observed-residue partner
ordering, a strict 20 Å Cα interface crop and exact-coordinate deduplication.

Requested chains must exist by default. Use `allowed_missing_chains` or
`empty_partner` only when explicitly requested for these inputs, and retain the
choice in the manifest and output records. Do not import historical cohort
exceptions. Neither exception establishes a meaningful binding interface.

## Extract the predefined features

For one complex:

```python
import numpy as np
from topokit.workflows.protein_protein_prediction import featurize

tensor = featurize("complex.pdb", ["A"], ["B"])
np.save("ppi_topology_features.npy", tensor.ravel(order="C"), allow_pickle=False)
```

The fixed PPI Topo output is float32 `(10, 14, 36)`, or **5,040 values** in
C order. It uses 14 radii from 1.0 to 7.5 Å in 0.5 Å steps, 36 channels from
`null,C,N,O,S,CA` with disjoint C/CA, and ten complete ordinary L0 summaries.
Keep the canonical recipe identity and ordering. There is no partner-swap
augmentation. Do not alter the crop, radii or spectrum to bypass a resource cap.

Use the batch exporter, including for a one-row manifest, when provenance or
prediction-ready output is needed. The required CSV columns are
`sample_id,structure_file,partner_a_chains,partner_b_chains`; separate chains
within each partner with semicolons. IDs are case-insensitively unique letters/digits/underscores/
hyphens and structure paths are relative to the CSV directory.

```bash
python workflows/protein_protein_prediction/extract_features.py \
  --manifest my_pairs.csv \
  --output examples/output/ppi
```

Run from the checkout root and choose a new output directory. No affinity
labels are needed. The exporter writes the feature schema and selection records.

## Verify and deliver

- Require exit code 0 and `run.json` status `complete`.
- Verify finite float32 `features.npy` shape `(N, 5040)` and `sample_ids.json`
  matching the requested row order. Per-sample tensors have shape `(10,14,36)`.
- Preserve the exporter-written `feature_schema.json`, normalized manifest,
  hashes and sample records. Do not rewrite schema bytes for prediction.
- Report output paths, sample count, Topo feature shape and input-selection
  diagnostics. Report failed samples directly; never discard rows or impute
  failures as zeros. Feature vectors alone are not affinity predictions.

For requested prediction, follow the [GBDT guide](../../protein_protein_prediction/ml/README.md)
with the compatible runtime and three trusted external model bundles. Its CLI
accepts `--features STORE --models MODEL_DIRECTORY --output NEW.csv`; the target
is signed binding free energy in kcal/mol. Do not refit the stored scalers or
train a model during extraction/inference. Consult the
[model card](../../protein_protein_prediction/ml/MODEL_CARD.md) when interpreting
results. This skill does not download weights or publish inputs or outputs.
