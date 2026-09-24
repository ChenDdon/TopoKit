# Protein–ligand topology features: default FS-AN recipe

This workflow converts prepared protein–ligand complexes into **27,500 topology
features per pair**, using the installed NumPy/SciPy implementation. It performs
feature extraction without a trained model or affinity labels. The scientific
recipe is fixed; the batch exporter only handles inputs, outputs and provenance.

## Install and run the ten examples

From the repository directory containing `pyproject.toml`:

```bash
python -m pip install .
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

Use a new output directory for every run. See the [ten-pair example collection](../../examples/protein_ligand/README.md)
for sample identities, source checksums and expected shapes. No model files,
PDBbind installation or surrounding research workspace are needed for this run.
The example structures retain their source-data terms; their local inclusion
does not establish unrestricted redistribution permission.

For one prepared pair in Python:

```python
import numpy as np
from topokit.workflows.protein_ligand_prediction import featurize, schema

tensor = featurize("protein.pdb", "ligand.mol2")
assert tensor.shape == (10, 50, 55)
vector = tensor.ravel(order="C")
np.save("topology_features.npy", vector, allow_pickle=False)
print(schema()["recipe_id"])
```

The API also accepts a single-record ligand SDF or MOL. The reference examples
use MOL2; preserve the intended atoms, coordinates and explicit hydrogen policy
when converting formats. `read_selected_atoms(...)` and `compute(...)` expose
atom counts, duplicate-coordinate diagnostics and channel presence.

## Use your own structures

Create a CSV with this header; relative paths are resolved against the CSV's
parent directory, independent of the current working directory:

```csv
sample_id,protein_file,ligand_file
complex_A,structures/A/protein.pdb,structures/A/ligand.mol2
complex_B,structures/B/protein.pdb,structures/B/ligand.mol2
```

IDs must be unique and contain only letters, digits, underscores or hyphens.
No four-character PDB ID, affinity label, train/test split or historical dataset
membership is required. Run the same command with your manifest path.

Inputs must describe an already prepared complex in a common coordinate frame
and angstrom units. Choose the intended protein model, biological assembly,
ligand and conformer upstream. TopoKit does not dock, align, repair, protonate,
choose alternate conformations, or infer absent hydrogens. Multi-model PDBs and
multi-record SDFs are rejected when a selection is ambiguous. The exporter
reports alternate-location and nonpositive-occupancy PDB rows in stderr and
sample records; the frozen FS-AN API retains raw eligible rows, so resolve these
warnings before treating the vectors as your intended single-conformer complex.

## Output and failure contract

| File | Meaning |
| --- | --- |
| `samples/<id>.npy` | C-contiguous float32 `(10, 50, 55)` tensor for a successful sample |
| `records/<id>.json` | Input hashes, result hash, counts, duplicate removals, channel presence, warnings, timing or failure |
| `feature_schema.json` | Exact canonical schema bytes for the installed FS-AN recipe |
| `manifest.csv` | All requested rows in order, with `pdb_id` as the prediction loader's identifier column |
| `features.npy` | Float32 `(N, 27500)` matrix, written only after every row succeeds |
| `sample_ids.json` | IDs in matrix-row order, written only for a complete batch |
| `run.json` | Recipe/implementation identity, runtime, limits, requested/successful/failed IDs and completion status |

The exporter returns exit code 0 only for a complete batch, 1 for a recorded
sample failure, and 2 for invalid invocation/preflight input. It never overwrites
an output directory, silently drops failed samples, substitutes zero tensors
for geometry errors, or changes the default recipe. Successful per-sample
outputs remain inspectable after a partial failure; consolidated features are
withheld. `run.json` must say `complete` before downstream use. Interrupted runs
may retain `running` status and must not be treated as completed stores.

The flattened column index is `(statistic_index * 50 + scale_index) * 55 +
channel_index`. Obtain the ordered statistics, radii and channels from `schema()`.
Numeric zeros in absent element channels are defined recipe values, not imputed
failed computations. No extra count or diagnostic columns are appended.

`--max-dense-entries` (default 25,000,000) and `--max-simplices` (default
1,000,000) are explicit resource caps, not scientific recipe switches. Full
spectra require dense matrices; increasing these caps can substantially increase
memory and runtime. This runner processes one pair at a time. Large datasets
may need an explicitly designed scheduling layer.

## Default recipe

[DEFAULT_RECIPE.json](DEFAULT_RECIPE.json) summarizes the recipe and links it to
the installed canonical schema. The following definition preserves the selected
FS-AN arithmetic; the exporter introduces no geometric or statistical change.

## Feature definition


1. **Inputs and protein field.** Read full-protein PDB `ATOM` records for
   C/N/O/S and the explicit ligand MOL2 atoms. Keep protein atoms with minimum
   distance **≤15 Å** to any supported explicit ligand atom. Supported ligand
   elements are C/N/O/S/P/F/Cl/Br/I/H. Explicit H is retained in both the
   feature categories and crop anchors; missing H is not inferred. Existing
   input bytes, including the provenance-pinned `6djc` normalized MOL2, are
   preserved unchanged on disk. After cropping, inspect the combined selected
   protein-then-ligand list and keep the **first atom at each exactly equal
   coordinate**. Keep its element and component labels; do not average or lift
   edges back to removed atoms. Deduplicate across element labels and both
   components **before creating channels**. Thus a protein/ligand coordinate
   collision keeps the protein row. Within a component, reader order wins.
   Distinct nearby coordinates are never rounded or merged. Crop anchors use
   the original ligand sites, so coordinate multiplicity cannot alter the
   distance test. Each JSON record lists the removed and retained atom rows,
   labels and coordinates, along with input/unique counts. Original coordinates
   of retained atoms determine alpha births.
2. **Categories.** Protein groups are `{null,C,N,O,S}`; ligand groups are
   `{null,C,N,O,S,P,F,Cl,Br,I,H}`. Their protein-major Cartesian product gives
   **55 channels**: 40 mixed channels, 4 protein-only, 10 ligand-only and one
   intentional null/null zero channel. Protein-only means the selected
   element in the cropped field, not the entire protein. `null` omits that
   side; it does not mean `all`.
3. **Geometry.** Construct alpha geometry separately on each selected cloud,
   using the public `native` backend, engine **1.1.1**. TopoKit computes
   circumspheres, empty-ball decisions and coface propagation; SciPy supplies
   candidate triangulation. Rational checks repair ill-conditioned cases;
   bounded small-cloud retriangulation is available without jitter or GUDHI.
   This is floating-point geometry with rational repair, not a universal
   exact-predicate guarantee. Process full coface births before taking the 1-skeleton. In mixed
   channels build on the selected protein/ligand union, then retain **only
   cross-component edges**. In null-side channels retain the selected
   component's internal edges. Delaunay support with an edge-length cutoff
   is not this alpha filtration. There is no r1 complete-connection override.
4. **Filtration.** Observe exactly **50 alpha radii `0.1,0.2,…,5.0 Å`**,
   including 5.0. These are physical radii, not squared values or diameters.
   The equivalent diameter grid is `0.2,0.4,…,10.0 Å`. Include an edge
   when its squared alpha birth is **≤ radius²**; do not divide by two again. These births are not
   Laplacian weights: every admitted edge has unit weight and is oriented
   once by row order. Keep isolated selected vertices.
5. **Spectrum.** Compute the complete ordinary unweighted Hyperdigraph L0
   spectrum at each scale. Build alpha once per channel and use the public
   incremental `L0Sweep`; insert each edge once and reuse summaries when the
   active edge set has not changed. These are successive ordinary L0
   spectra, not a two-scale persistent Laplacian. No L1/L2 features are used.
6. **Encoding.** Let `λ+` contain eigenvalues `>1e-10`, `μ=mean(λ+)`, and
   `MAD=mean(abs(λ+−μ))`. Zero eigenvalues satisfy `abs(λ)≤1e-10`.
   Store the following ten statistics in this exact order:

   | Index | Statistic | Scope |
   |---:|---|---|
   | 0 | Matrix size | Full spectrum, including isolated vertices |
   | 1 | Number of zero eigenvalues | Full spectrum |
   | 2 | Mean | Positive eigenvalues |
   | 3 | Minimum / mean | Positive eigenvalues; r5 is not applied |
   | 4 | MAD / mean | Positive mean absolute deviation, not standard deviation |
   | 5 | Maximum / mean | Positive eigenvalues; r5 is not applied |
   | 6 | 25th percentile | Positive eigenvalues; linear interpolation |
   | 7 | Median | Positive eigenvalues; linear interpolation |
   | 8 | 75th percentile | Positive eigenvalues; linear interpolation |
   | 9 | Spectral entropy | `−Σp ln(p)`, `p=λ+/Σλ+`; natural logs, no `log(k)` division |

   Empty positive spectra give zero positive-only summaries. A valid cloud
   with no edges still retains its matrix size and zero count. An absent
   required non-null element partner makes its entire channel zero.
   Null/null is always zero. Geometry failures are recorded and block any
   model requiring those samples; they are never encoded as missing-channel zeros.

The stored tensor is C-order **float32 `(10,50,55)`**, computed in float64.
Flattening yields **27,500 features**, with channels varying fastest. Presence
and atom-count diagnostics are stored in JSON, not appended to model inputs.
The null/null channel contributes 500 fixed zero coordinates. Ligand H and
normalized min/max remain included. The only changes from the previous
20 Å reference are the 15 Å crop and the alpha-radius grid shifted to 0.1–5.0 Å;
the explicit parameters in the selection receipt define this version.

## Optional affinity prediction

The completed batch directory is compatible with the installed guarded topology
prediction loader. A trusted external model bundle and its recorded runtime are
still required; they are not bundled with the examples or wheel. The selected
companion predictor averages seeds 0, 1 and 2 of StandardScaler + GBDT pipelines.
Each has 10,000 trees, learning rate 0.005, depth 7, min_samples_split 2,
subsample 0.4 and max_features `sqrt`. Scaling occurs once inside each pipeline.
See the [prediction guide](ml/README.md) and [installed model profile](../../src/topokit/workflows/protein_ligand_prediction/model_profile.json).

Older feature-selection experiments used different GBDT parameters. Their
records remain historical evidence and do not override the current model profile.
Changing crop, scales, channels, statistics or encoders defines different features
and makes them incompatible with an FS-AN model unless that model is retrained.

## Agent-assisted use and other workflows

[The topology skill](../skills/topokit-protein-ligand/SKILL.md) and
[workflow agent instructions](../AGENTS.md) guide an assistant from a manifest to
validated default features. These are repository-local instructions; they do not
install themselves into an assistant or authorize publication/training.

[ESM-2 + CPZ sequence features](sequence/README.md) are a distinct embedding
modality using protein sequences, ligand SMILES and pretrained encoders. The
[atom-deletion](../protein_ligand_prediction_version2/README.md) and
[H0 study](../protein_ligand_persistent_homology/README.md) remain historical
research controllers with fixed cohort assumptions, not alternate switches of
this default exporter. The optional supervised [TopoFormer](dl/README.md)
consumes existing topology features and is outside this extraction workflow.

The [previous workflow README](../../markdown/history/2026-09-23-readme-refresh/protein_ligand_README.original.md)
preserves complete experiment status and original references. Some historical
links require the original research workspace and are not public runtime dependencies.
