# Protein–ligand sequence features: ESM-2 + CPZ

This optional workflow turns **explicit protein sequences and a prepared ligand
SMILES** into one unscaled `float32[1792]` vector. It uses the frozen ESM-2
`esm2_t33_650M_UR50D` protein encoder and the
`chembl27_pubchem_zinc_512` (CPZ) ligand encoder. These are the features used by
the selected **FS-AU** sequence model. They are learned sequence embeddings;
the separate [structure workflow](../README.md) computes topology features.

The supported extraction route is `SequenceEncoder(ligand_profile="cpz")`.
The [machine-readable recipe](RECIPE.json) pins both models, required files,
pooling, feature order and hashes. The default constructor remains the historical
ChEMBL27-only recipe for compatibility; always select `cpz` explicitly here.

## Install and supply the model assets

From a cloned TopoKit repository:

```bash
python -m pip install -e '.[sequence]'
```

Model weights are external and are never downloaded by TopoKit. Supply trusted
local copies of these exact assets:

| Directory argument | Required files | Source |
|---|---|---|
| `esm_dir` / `--esm-dir` | `model.safetensors`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt` | [ESM-2 snapshot, revision 08e4846…](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/08e4846e537177426273712802403f7ba8261b6c) |
| `chembl_dir` / `--cpz-dir` | `checkpoint_best.pt`, `dict.txt` from `chembl27_pubchem_zinc_512` | [WeilabMSU/PretrainModels](https://github.com/WeilabMSU/PretrainModels) |

The complete SHA256 map is in [RECIPE.json](RECIPE.json). The CPZ checkpoint
starts with `c4788751…`; the ChEMBL27-only checkpoint starts with `8f4b94db…` and
is a different representation despite having the same width. The loader rejects
mismatched checkpoint and dictionary bytes before loading the ligand model.
The CPZ hashes identify the retained source-study assets; an upstream archive
must be checked against them after extraction. Check the upstream terms when
obtaining or distributing weights. The legacy `.pt` loader can deserialize
Python objects, so supply weights from a trusted source, not an arbitrary upload.

ESM-2 has 650 million parameters; its weights alone are about 2.6 GB. Allow
additional inference memory. CPU is the portable default; `mps` and `cuda:0`
can be requested when available. Model/device support depends on the installed
PyTorch runtime. TopoKit imports the optional libraries only when needed.

## Extract one feature vector

```python
import numpy as np
from topokit.workflows.protein_ligand_prediction.sequence import SequenceEncoder

encoder = SequenceEncoder(
    esm_dir="/path/to/esm2_t33_650M_UR50D",
    chembl_dir="/path/to/chembl27_pubchem_zinc_512",
    ligand_profile="cpz",
    device="cpu",
)
features, token_receipt = encoder.encode(
    protein_chains=["MKTAYIAKQRQISFVKSHFSRQ"],  # replace with the selected chains
    smiles="CC(=O)Oc1ccccc1C(=O)O",             # illustrative input, not a complex claim
)
assert encoder.recipe_id == "sequence-esm2-t33-cpz-bos-v1"
assert features.shape == (1792,)
np.save("sequence_features.npy", features, allow_pickle=False)
print(token_receipt)
```

For a saved recipe and input receipt, create an input JSON such as:

```json
{
  "id": "my-complex",
  "protein_chains": ["MKTAYIAKQRQISFVKSHFSRQ"],
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

Then run from the repository root:

```bash
python workflows/protein_ligand_prediction/sequence/extract_features.py \
  --input my-complex.json \
  --esm-dir /path/to/esm2_t33_650M_UR50D \
  --cpz-dir /path/to/chembl27_pubchem_zinc_512 \
  --output-dir examples/output/my-complex-sequence
```

The output directory must be new. It receives `features.npy`, the supplied
`input.json`, and `RECEIPT.json` with recipe/model hashes, chain lengths,
ligand tokenization, feature hash and runtime versions. Add `--validate-only`
to check inputs and asset hashes without loading models; this does not produce
features. The script never fits a model or predicts affinity.

## Input and representation contract

- **Choose the target chains explicitly.** Supply uppercase full-chain amino-acid
  strings in a nonempty list. Repeated biological chains retain their multiplicity.
  Neither the encoder nor the example complexes automatically determines the
  intended target, chain set, biological assembly or missing sequence.
- **Prepare the ligand explicitly.** Supply the intended canonical isomeric
  SMILES with the correct stereochemistry, protonation and components. The encoder
  tokenizes a string; it does not validate chemistry or canonicalize it. PDB/MOL2/SDF
  to sequence/SMILES conversion is a separate preparation step.
- **Protein pooling:** final-layer residue mean, excluding BOS/EOS/padding.
  Nonoverlapping windows of at most 1,022 residues cover all residues. Means are
  weighted by residue count across windows and chains. This does not create
  attention across window boundaries or apply a spatial pocket crop.
- **Ligand pooling:** final hidden state at BOS, width 512, with a 256-token
  context including BOS and EOS. By default, unsupported tokens and overlength
  SMILES raise errors. Explicit `allow_unknown=True` / `--allow-unknown` maps
  whole unsupported tokens to UNK. Explicit `allow_truncation=True` /
  `--allow-truncation` keeps the first 254 content tokens and EOS. Both losses
  appear in the receipt; do not enable them silently to make a run pass.
- **Feature order:** ESM-2 protein coordinates `[0:1280]`, then CPZ ligand
  coordinates `[1280:1792]`; finite float32, with no scaling or labels.

For repeated inputs, instantiate one encoder and reuse it. The public
`protein_batch`, `ligand_batch`, `protein_windows`, `weighted_mean` and
`concatenate_embeddings` helpers support explicit caching/batching. Preserve
this pooling and profile identity when composing a larger application.

## Selected model versus historical helpers

[MODEL_SELECTION.json](MODEL_SELECTION.json) preserves the original FS-AU
selection receipt. Its paths describe external research assets and do not
resolve in a public clone. The feature recipe ID above identifies embeddings;
FS-AU's model recipe is `sequence-esm2-t33-cpz-bos-nmi-sqrt-gbdt-v1`.
Its selected GBDT configuration uses 10,000 trees, depth 7, learning rate 0.005,
minimum split 2, subsample 0.4, `max_features="sqrt"`, and
`random_state=None`, with a training-only StandardScaler in the saved pipeline.
These parameters are provenance, not an instruction to train.

The installed `make_gbdt`, `fit_gbdt`, `predict_gbdt` and module-level `RECIPE_ID`
retain their historical **FS-AQ / ChEMBL27** meanings. They do not implement
FS-AU bundle inference. In particular, `predict_gbdt` deliberately rejects a
CPZ/FS-AU bundle. Trained FS-AU pipelines, their compatible saved runtime,
benchmark data and study controllers are not bundled here. Do not send a
ChEMBL27-only vector to a CPZ model or standardize a vector twice.

The [earlier workflow log](history/README.before-public-guide-2026-09-24.md)
is retained for research provenance; its dataset-relative links refer to the
original research workspace. Current public usage is described on this page.
The [repository skill](../../skills/topokit-sequence/SKILL.md) helps an agent
follow the same feature recipe.
