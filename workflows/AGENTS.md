# Help users extract protein–ligand features

For a prepared protein–ligand 3D complex, use the predefined **Topo** recipe.
A user supplies a protein structure and ligand pose; TopoKit returns a numerical
descriptor that they can use for comparison or downstream modelling. No TDA
expertise, affinity labels or pretrained weights are needed for extraction.
The output is **27,500 topology features per complex**, stored as a float32
`(10, 50, 55)` tensor or a vector in C order.

## Quick start: one complex

Run from the repository directory containing `pyproject.toml`, with that copy
installed using `python -m pip install .`:

```python
import numpy as np
from topokit.workflows.protein_ligand_prediction import featurize

tensor = featurize("protein.pdb", "ligand.mol2")
features = tensor.ravel(order="C")  # 27,500 values
np.save("topology_features.npy", features, allow_pickle=False)
```

The structures must describe the intended bound or docked pose in the same
coordinate frame, in angstroms. A ligand SDF/MOL containing one molecule is also
supported. The user chooses the protein model, chains, ligand and conformer
before extraction. TopoKit does not perform docking or structure preparation.

## Quick start: a collection

Use a CSV with `sample_id,protein_file,ligand_file`. IDs must be unique; relative
paths are resolved from the CSV's directory. The supplied ten-pair manifest is
a ready-to-run example:

```bash
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

Replace the manifest path for the user's own collection and choose a new output
directory. Read [the topology skill](skills/topokit-protein-ligand/SKILL.md) and
[workflow guide](protein_ligand_prediction/README.md) for input checks and the
output contract. The exporter saves the schema, input hashes and sample records;
a complete batch also has an ordered `features.npy` matrix and `sample_ids.json`.
Require `run.json` status `complete`, finite float32 `(N, 27500)` features and
IDs in the requested order. Report output paths and any input warnings or errors.
Do not silently drop failed rows or replace failed computations with zeros.

## Choose the workflow from the user's inputs

- **Protein–ligand 3D structures:** use the Topo recipe above. Its fixed settings
  are a 15 Å protein crop, 50 alpha radii, 55 channels and ten spectral summaries.
- **Protein sequences and ligand SMILES:** use
  [the sequence skill](skills/topokit-sequence/SKILL.md). It produces 1,792
  ESM-2 + CPZ embedding values and requires the specified local pretrained assets.
- **Affinity prediction:** use completed compatible features and a trusted
  trained bundle; follow the [GBDT guide](protein_ligand_prediction/ml/README.md)
  or the requested model's guide. Extraction does not fit or select a model.

Preserve the user's molecular identity and scientific question. Resolve missing
pose or chain information before proceeding. Do not silently change the crop,
radii, channels, encoder, input conformer, or feature definition. Ordinary valid
manifest runs need no extra approval. Use the frozen schema and recipe IDs as
written; their internal identifiers are compatibility metadata, not display names.

Repository scripts and examples accompany a source checkout. Use the Python
environment in which the intended TopoKit version is installed; inspect
`topokit.__file__` when diagnosing an installation mismatch. Keep generated
features and receipts outside `src/topokit`.
