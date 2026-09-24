---
name: topokit-sequence
description: Extract frozen ESM-2 and CPZ sequence features for an explicit protein-chain and ligand-SMILES pair using TopoKit. Use for the selected sequence recipe; structure topology features use the separate protein-ligand workflow.
---

# TopoKit sequence features

Use this skill from a TopoKit checkout. Locate the repository through its
`pyproject.toml` and `src/topokit`; resolve the links below relative to this
skill. Read the [sequence guide](../../protein_ligand_prediction/sequence/README.md)
and [RECIPE.json](../../protein_ligand_prediction/sequence/RECIPE.json) before
extraction. They define the exact models and required local asset hashes.

The intended result is one unscaled finite `float32[1792]` vector: ESM-2 protein
1,280 values followed by CPZ ligand BOS 512 values. The feature ID is
`sequence-esm2-t33-cpz-bos-v1`, used by the selected FS-AU model. These are learned
sequence embeddings, not the 27,500-dimensional topology representation.

## Follow the selected recipe

1. Obtain a nonempty list of the intended protein-chain sequences and a prepared
   canonical isomeric SMILES. Use user-provided target/chain choices. Structure
   files alone do not specify full sequence, biological assembly or ligand
   protonation; resolve material missing input before extraction. Keep repeated
   biological chains. Do not infer chains from the example set automatically.
2. Use an environment with `python -m pip install -e '.[sequence]'` from the
   checkout. Obtain explicit trusted local directories for ESM-2
   `esm2_t33_650M_UR50D` and CPZ `chembl27_pubchem_zinc_512`. Weights are not
   bundled. A feature-extraction request does not itself call for downloading
   large model assets or training models. If assets are missing, prepare the
   input and runnable command and report precisely which assets are needed.
3. Save a JSON object with `protein_chains`, `smiles`, and optionally `id`.
   Invoke the repository `extract_features.py` with `--input`, `--esm-dir`,
   `--cpz-dir`, and a new `--output-dir`. Use `--validate-only` to validate asset
   hashes and tokenization without model inference when appropriate. Default
   device is CPU; respect a user's device choice.
4. Inspect `RECEIPT.json`, load `features.npy` with `allow_pickle=False`, and check
   finite float32 shape `(1792,)`, profile `cpz`, recipe ID and saved feature
   hash. Report the artifact paths and any token omissions or UNK symbols.

For Python use, the supported constructor is:

```python
from topokit.workflows.protein_ligand_prediction.sequence import SequenceEncoder
encoder = SequenceEncoder(esm_dir=esm_dir, chembl_dir=cpz_dir,
                          ligand_profile="cpz", device="cpu")
features, token_receipt = encoder.encode(protein_chains, smiles)
```

`chembl_dir` is the shared argument name even for CPZ. Omitting `ligand_profile`
selects historical ChEMBL27-only embeddings. A matching output width is not proof
of recipe compatibility. Keep nonoverlapping 1,022-residue windows,
residue-weighted protein pooling and CPZ BOS pooling unchanged.

Unsupported tokens and overlength ligand strings fail by default. Do not enable
`--allow-unknown` or `--allow-truncation` without an explicit choice to accept the
corresponding representation loss. Do not silently modify the SMILES, chains,
asset hashes or dimensionality to make a case pass.

This skill extracts features. Do not call `fit_gbdt`, tune a model, create a
train/test split, or claim an affinity prediction as part of this recipe.
Historical `predict_gbdt` is bound to FS-AQ and is not an FS-AU predictor.
