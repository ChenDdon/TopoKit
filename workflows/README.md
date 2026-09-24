# Workflow applications

Start here to turn your own scientific inputs into features with an explicit
recipe. The reusable APIs are installed under `topokit.workflows`; scripts,
recipes, agent guidance and example manifests remain in the source repository.
Run the commands below from the package root after `python -m pip install .`.

## Protein–ligand topology: FS-AN

The [protein–ligand workflow](protein_ligand_prediction/README.md) is the default
non-DL structure-based recipe. It accepts a prepared protein PDB and ligand
MOL2 in the same angstrom coordinate frame and returns 27,500 topology features:
15 Å protein crop, 50 alpha radii, 55 element/null channels and ten ordinary
Hyperdigraph L0 spectral summaries.

```bash
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

The [ten-pair example set](../examples/protein_ligand/README.md) includes a
manifest and source provenance. The exporter writes per-complex tensors and
records plus the schema; a complete successful batch also receives an ordered
`features.npy` matrix and `sample_ids.json`. For your own inputs, follow the
manifest format and preparation checks in the workflow guide.

The installed Python entry point is:

```python
from topokit.workflows.protein_ligand_prediction import featurize

tensor = featurize("protein.pdb", "ligand.mol2")  # float32 (10, 50, 55)
features = tensor.ravel(order="C")              # 27,500 values
```

An [agent skill](skills/topokit-protein-ligand/SKILL.md) describes how an
assistant should prepare a manifest, run the fixed recipe and verify outputs.
For example, ask an assistant: "Follow
`workflows/skills/topokit-protein-ligand/SKILL.md` to extract the default features
from `examples/protein_ligand/manifest.csv` into a new output directory."
The files are included as repository instructions; no global skill installation
is required when the assistant is told to read them.
No trained model is needed to extract topology features. Optional
[GBDT prediction](protein_ligand_prediction/ml/README.md) additionally requires
compatible external model bundles with their training-fitted scalers.

## Protein–ligand sequence embeddings: FS-AU

The [sequence recipe](protein_ligand_prediction/sequence/README.md) combines
ESM-2 `esm2_t33_650M_UR50D` protein embeddings (1,280) and CPZ
`chembl27_pubchem_zinc_512` ligand embeddings (512). Its 1,792-feature output
represents protein sequences and ligand SMILES, independently of a bound
three-dimensional structure.

Install `.[sequence]` and provide the encoder assets described in the recipe.
The [sequence agent skill](skills/topokit-sequence/SKILL.md) covers model
identity, preparation, extraction and validation. Weights remain external;
substituting another checkpoint changes the feature recipe. A downstream
binding-affinity model is a further, separate asset.

## General topology workflows

For other scientific questions, begin with the
[three tutorial notebooks](../README.md#tutorial-notebooks) and compose the
public layers. The installed general workflows include:

| API | Use |
| --- | --- |
| `workflows.analyze` | Persistence and ordinary spectral snapshots of an explicit topology |
| `workflows.analyze_stationary` | Fixed-scale homology, spectra and scalar summaries |
| `workflows.laplacian_series` | Ordinary spectra over a scale schedule, or explicitly requested two-scale persistent operators |
| `workflows.ml` | Optional estimators over caller-supplied feature matrices |

Construction parameters, coordinate units, feature schemas and training
memberships remain application choices. Mathematical algorithms stay in the
builders and core layers.

## Other application records

[Atom-deletion](protein_ligand_prediction_version2/README.md) and
[H0 barcode](protein_ligand_persistent_homology/README.md) directories preserve
separate research applications. They are not alternative defaults for FS-AN
and may depend on historical datasets or external artifacts. The optional
[supervised TopoFormer application](protein_ligand_prediction/dl/README.md)
uses topology features with a separate PyTorch model.

Earlier workflow-index contents are retained in the
[historical snapshot](../markdown/history/2026-09-23-readme-refresh/workflows.README.original.md).
Research asset references in that snapshot are historical and may not resolve
in a standalone clone.
