# Protein–protein Topo features and affinity prediction

Use this workflow to turn a prepared protein–protein complex into **5,040 Topo
features**. Supply a PDB or mmCIF structure and the chain identifiers belonging
to each partner. The feature vector can be used for your own analysis or passed
to the optional affinity predictor. Extraction requires only NumPy and SciPy;
no labels, pretrained encoder or topology expertise are required.

The public recipe uses topology features only. Its installed Python API is
`topokit.workflows.protein_protein_prediction`; the commands below accompany a
source checkout. Run them from the directory containing `pyproject.toml` after
`python -m pip install .`.

## One prepared complex

```python
import numpy as np
from topokit.workflows.protein_protein_prediction import featurize

tensor = featurize("complex.pdb", ["A"], ["B"])
features = tensor.ravel(order="C")
np.save("ppi_topology_features.npy", features, allow_pickle=False)
```

The tensor is C-contiguous float32 with shape **`(10, 14, 36)`**. Flattening
in C order gives a float32 vector of **5,040 values**. A partner can contain
several chains: `featurize("complex.cif", ["A", "B"], ["C", "D"])` keeps the
first two chains together as one partner and the last two as the other.
The chain lists identify partners; the frozen recipe determines their internal
order from observed residue counts before the interface crop.

## A collection of complexes

Try the supplied [ten two-chain examples](../../examples/protein_protein/README.md):

```bash
python workflows/protein_protein_prediction/extract_features.py \
  --manifest examples/protein_protein/manifest.csv \
  --output examples/output/protein_protein
```

Each example assigns chain A to one partner and chain B to the other, with no
multi-chain groups or missing-chain exceptions. The completed matrix has shape
`(10, 5040)`. Structure hashes and source attribution accompany the examples.

For your own structures, use the same manifest format:

Create a UTF-8 CSV with these required columns:

```csv
sample_id,structure_file,partner_a_chains,partner_b_chains
pair_01,structures/pair_01.pdb,A,B
pair_02,structures/pair_02.cif,A;B,C;D
```

Use sample identifiers made from letters, digits, underscores or hyphens,
unique even when compared without case. Avoid Windows reserved names such as
`CON`, `AUX`, `NUL`, `COM1` and `LPT1` so files remain portable.
Structure paths are relative to the CSV's directory unless absolute. Separate
multiple chain identifiers with semicolons; do not combine them into a new chain.
Affinity labels are not required and are not used by the feature exporter.

```bash
python workflows/protein_protein_prediction/extract_features.py \
  --manifest my_pairs.csv \
  --output examples/output/ppi
```

Choose a new output directory. The exporter preserves sample order, records
input hashes and structure-selection details, and saves the exact schema used
by the prediction loader. A successful store contains:

| File | Meaning |
| --- | --- |
| `features.npy` | Float32 `(N, 5040)` matrix, one sample per row |
| `sample_ids.json` | Sample IDs in the matrix's row order |
| `manifest.csv` | Normalized sample inputs and explicit chain selections |
| `feature_schema.json` | Canonical feature identity and ordering |
| `run.json` | Completion status and artifact hashes |
| `samples/` | Per-sample float32 tensors |
| `records/` | Per-sample input-selection records, hashes and diagnostics |

Require exit code 0 and `run.json` status `complete` before using the matrix.
A failed sample has a recorded error; the exporter does not replace it with
zeros or silently drop its row. A complete matrix is written only when every
requested sample succeeds. For assistants, the
[PPI skill](../skills/topokit-protein-protein/SKILL.md) provides the corresponding
input and output checks.

## Structure preparation and chain identity

- Use the intended bound or docked complex with coordinates in **angstroms**.
  The workflow does not dock, align, repair structures, model missing residues
  or expand a biological assembly.
- Specify **author chain identifiers** for both PDB and mmCIF. Only the first
  coordinate model and protein ATOM rows are used. HETATM rows are excluded.
- Keep a multi-chain partner intact. Use disjoint, nonempty chain lists for the
  two partners; verify that the requested chains represent the intended binding
  partners in this structure.
- Alternate atom rows are selected by highest occupancy per residue/atom name.
  Equal occupancy retains the first input row; absent occupancy scores zero.
  This is atom-wise selection, not a global alternate-conformer optimization.
- Requested chains that are absent fail by default. An explicit optional CSV
  column `allowed_missing_chains` may list authorized missing chains separated
  by semicolons. Available chains in that partner are retained. The optional
  `empty_partner` column accepts `A` or `B` only when an intentionally empty
  partner is requested; it changes the representation to the available
  partner's Cα-anchored residues and single-partner channels. These choices
  remain visible in output records. No cohort-specific exception is automatic.

Normal affinity use requires a meaningful two-partner structure. Inspect the
selection diagnostics when chains or interface residues are missing; changing
the input changes what the descriptor represents.

## Fixed Topo recipe

The machine-readable [default recipe](DEFAULT_RECIPE.json) and saved schema
identify the exact feature definition as
`ppi-topo-cnos-ca-disjoint-crop20-alpha-r1p0-r7p5-step0p5-v1`.
Use **PPI Topo** or **topology features** as the display name.

| Setting | Definition |
| --- | --- |
| Partner order | Larger observed ATOM residue count first, before cropping; ties follow first-chain appearance in the structure |
| Interface | Residues with a Cα strictly less than **20 Å** from an opposite-partner Cα; keep their C/N/O/S atoms |
| Missing Cα | Exclude the residue and report its identity |
| Duplicate coordinates | Exact global keep-first, after partner ordering and cropping; no rounding |
| Categories per partner | `null, C, N, O, S, CA`; `CA` is carbon named CA and is excluded from `C` |
| Channels | 36 ordered category pairs, first-partner category outermost |
| Alpha radii | **1.0, 1.5, …, 7.5 Å**; 14 radii; 8.0 Å is excluded |
| Operator | Complete ordinary unweighted L0 at each radius |
| Output | Float32 `(10, 14, 36)`, flattened in C order |

Residue counts include insertion codes and use observed coordinates, not full
sequence lengths. Both interface selections use the original opposite-partner
Cα anchors. A qualifying residue contributes its supported atoms even when an
individual atom lies farther than 20 Å from the other partner. Deduplication
retains the first atom's partner and category and reports removed atoms.

`null` omits one partner; it does not mean all atoms. The channels comprise
25 mixed channels, ten single-partner channels and one intentional null/null
zero channel. Per channel, native alpha geometry computes full coface-dependent
births before selecting edges. Mixed channels keep only cross-partner edges;
single-partner channels keep internal edges. An edge enters when its squared
alpha birth is at most the squared radius. All selected vertices, including
isolated ones, remain present and edges have unit weight. This is not an
edge-length cutoff graph or a two-scale persistent Laplacian.

The ten statistics are the same as in the protein–ligand Topo recipe. Let
`λ+` contain eigenvalues greater than `1e-10` and `μ = mean(λ+)`:

| Index | Statistic |
| ---: | --- |
| 0 | Matrix size (number of selected vertices) |
| 1 | Zero-eigenvalue count, `abs(λ) ≤ 1e-10` |
| 2 | Positive mean `μ` |
| 3 | Positive minimum divided by `μ` |
| 4 | Positive mean absolute deviation from `μ`, divided by `μ` |
| 5 | Positive maximum divided by `μ` |
| 6–8 | Positive 25th percentile, median and 75th percentile, with linear interpolation |
| 9 | Positive spectral entropy `−Σ pᵢ ln(pᵢ)`, where `pᵢ = λᵢ / Σλ+` |

An empty positive spectrum gives zero for indices 2–9 while preserving counts.
An absent required category gives a zero channel; a geometry failure is an
error. Calculations use float64 and stored features use float32. Channels vary
fastest in the flattened vector:

```text
column = (statistic_index * 14 + radius_index) * 36 + channel_index
```

There is one ordered representation per complex and no partner-swap
augmentation. Changing radii, atom selection, channels, ordering or summaries
creates a different recipe and is incompatible with the selected model.
Resource limits fail explicitly; they do not shrink the crop or substitute a
partial spectrum. Use `--help` for the exporter's explicit resource controls.

## Optional binding-affinity prediction

The selected predictor averages three external GBDT pipelines trained on these
Topo features. Its target is **signed binding free energy, ΔG_bind, in kcal/mol**.
Negative values retain their sign; this is not mutation ΔΔG or a pKd conversion.

Follow the [GBDT guide](ml/README.md) for the pinned inference runtime and model
bundle identity. Once the three trusted bundles are available:

```bash
python workflows/protein_protein_prediction/ml/predict.py \
  --features examples/output/ppi \
  --models /path/to/trusted-ppi-models \
  --output examples/output/ppi_predictions.csv
```

The loader verifies feature identity and bundle hashes before loading the
training-fitted pipelines. It does not refit a scaler or train on the requested
samples. Model weights are external assets and are not downloaded by extraction.
The [model card](ml/MODEL_CARD.md) describes the selected topology-only model's
evaluation and applicability. The public workflow contains the reusable recipe,
extraction and inference interfaces; research run directories, training
controllers and intermediate experiment files remain outside the package.
