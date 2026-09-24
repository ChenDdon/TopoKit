---
name: topokit-protein-ligand
description: Extract the fixed TopoKit FS-AN topology features from prepared protein–ligand 3D pairs or a manifest, and verify the resulting feature store. Use for bound or docked complexes; sequence embeddings use the separate topokit-sequence recipe.
---

# Protein–ligand features with TopoKit

Read the [workflow guide](../../protein_ligand_prediction/README.md) for the
input/output contract and [default recipe](../../protein_ligand_prediction/DEFAULT_RECIPE.json)
for feature identity. Locate the repository root by its `pyproject.toml`.
Check that `topokit.__file__` belongs to the intended installation when multiple
development copies exist. Use `python -m pip install .` from that root when an
installation is needed; base extraction requires only NumPy and SciPy.

## Inputs and recipe

Use explicit protein PDB and ligand MOL2 paths in a shared coordinate frame,
in angstroms. Single-record SDF/MOL ligands are also supported by the API. Do
not dock, align, add hydrogens, select alternate conformations, or substitute
chains as an unannounced preprocessing step. Clarify missing molecular identity
or pose information before extracting features. Review alternate-location and
occupancy warnings; the frozen recipe retains raw eligible PDB rows.

The default is FS-AN: 15 Å crop; 50 alpha radii 0.1–5.0 Å; 55 element/null
channels; ten ordinary Hyperdigraph L0 spectral summaries. Output is float32
`(10, 50, 55)`, flattened in C order to 27,500 features. These are successive
ordinary L0 spectra, not two-scale persistent Laplacians. Changing this recipe
requires an explicit scientific choice and a distinct feature identity.

## Execute

For a batch, prepare a CSV with `sample_id,protein_file,ligand_file`. IDs are
unique letters/digits/underscores/hyphens. Resolve relative structure paths
against the manifest's directory. No labels or historical cohort are required.
Run from the repository root with a new output directory:

```bash
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

Replace the manifest/output paths for user data. The [ten-pair collection](../../../examples/protein_ligand/README.md)
is a local smoke example, not a training/evaluation split. For a requested
single array, use `featurize(protein_path, ligand_path)` directly; use the batch
exporter even for one row when a provenance-complete store is needed.

## Verify and deliver

- Require exit code 0 and `run.json` status `complete`. Report failed samples
  and their recorded errors; do not discard them or turn errors into zeros.
- Verify `features.npy` is finite float32 `(N, 27500)` and `sample_ids.json`
  matches the requested row order. Per-sample arrays have shape `(10, 50, 55)`.
- Retain the canonical `feature_schema.json`, sample records, hashes and
  implementation receipt. Do not reserialize a schema for the prediction loader,
  which validates exact bytes. The exporter already writes the correct files.
- Report output paths, sample count, tensor/vector shape, FS-AN identity and
  any input warnings. A descriptor vector alone is not an affinity prediction.

If a resource cap is reached, explain the cap and likely memory impact before
choosing a larger explicit budget; do not shrink the crop, change alpha geometry,
or compute a partial spectrum to make the default recipe appear successful.

For requested affinity inference, follow the [GBDT guide](../../protein_ligand_prediction/ml/README.md)
with a trusted external bundle and its compatible runtime. Do not refit its
scaler or train a model as part of feature extraction. The workflow does not
publish data or download model assets.
