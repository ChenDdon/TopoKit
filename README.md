# TopoKit

[Source repository](https://github.com/ChenDdon/TopoKit) ·
[Releases](https://github.com/ChenDdon/TopoKit/releases) ·
[Report an issue](https://github.com/ChenDdon/TopoKit/issues)

TopoKit is a Python toolkit for constructing and analyzing **simplicial complexes,
sequence hyperdigraphs, and two-factor interaction complexes**. It turns explicit
objects or scientific coordinates into homology, persistence, Laplacian spectra,
and numerical features through composable APIs.

Version **0.3.0** includes a ready-to-run, non-DL protein–ligand feature workflow:
**FS-AN produces 27,500 topology features per prepared complex** using NumPy and
SciPy. A separate sequence recipe uses frozen **ESM-2 + CPZ** encoders to produce
1,792 embedding features; it requires optional dependencies and external model
assets.

[Install](#installation) · [Basic usage](#basic-usage) ·
[Tutorial notebooks](#tutorial-notebooks) ·
[Protein–ligand workflow](#proteinligand-topology-features) ·
[Sequence recipe](#sequence-based-features) · [Documentation](#documentation)

## Installation

Requires **Python 3.10 or newer**. Clone the repository and install it in a fresh
environment:

```bash
git clone https://github.com/ChenDdon/TopoKit.git
cd TopoKit
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install .
python -m topokit info
```

To reproduce this release, run `git checkout v0.3.0` before installation.
Alternatively, download the wheel or source archive from the
[v0.3.0 release](https://github.com/ChenDdon/TopoKit/releases/tag/v0.3.0).
The GitHub source archive includes the tutorials and example inputs; the wheel
contains the Python library and small runtime schemas. This release is hosted
on GitHub; the command above does not rely on a PyPI package of the same name.

The base installation depends only on NumPy and SciPy. Use an editable
installation when developing the package:

```bash
python -m pip install -e '.[plot,test]'
```

Optional extras are installed only when needed:

| Extra | Purpose |
| --- | --- |
| `.[plot]` | Matplotlib figures |
| `.[test]` | Pytest |
| `.[ml]` | Scikit-learn estimators and companion GBDT inference |
| `.[sequence]` | Frozen protein/ligand encoders: PyTorch, Transformers, RDKit and scikit-learn |
| `.[dl]` | Optional supervised TopoFormer application |
| `.[alpha_exact]` | Explicit GUDHI exact-alpha backend for reference or custom constructions |

Base topology extraction does not require pretrained weights, PyTorch, RDKit,
or GUDHI. Tutorials and sample structures are repository files under `examples/`;
they are not installed inside the Python wheel.

## Basic usage

This example reads a supplied point cloud, builds a simplicial alpha filtration,
and computes persistence and a complete ordinary degree-zero Laplacian spectrum:

```python
from topokit import readers, builders, core, postprocessing
from topokit.serialization import save_result

cloud = readers.read("examples/data/point_cloud_24.csv")
obj = builders.from_points(cloud, kind="simplicial", max_dimension=1)
bars = core.persistence(obj, max_dimension=1)
spectrum = core.laplacian(obj, dimension=0, scale=1.0)
summary = postprocessing.summarize_spectrum(
    spectrum, statistics=("min", "max", "mean", "zero_count")
)
save_result(
    {"persistence": bars, "spectrum": spectrum, "summary": summary},
    "examples/output/basic_usage.json",
)
```

The alpha scale in this example is **squared radius** in the coordinate units.
Hyperdigraph distance filtrations use distance scales. Select the construction
and units appropriate to your question; these scales are not interchangeable.

Use `builders.simplicial`, `builders.hyperdigraph`, or `builders.interaction`
for family-specific construction. `workflows.analyze_stationary` combines
fixed-scale homology, spectra, and named summaries; `workflows.laplacian_series`
computes spectra over an explicit scale schedule. With the plotting extra:

```python
from topokit import visualization as viz

axis = viz.plot_barcodes(bars)
viz.save_figure(axis.figure, "examples/output/barcodes.svg")
```

Readers support CSV, native point-cloud JSON, single-frame XYZ, PDB, MOL2,
single-record SDF/MOL, PDBQT, and atom-site CIF. Reader metadata preserves
available IDs, elements, charges, bonds and unit-cell information; construction
choices remain explicit. See [input contracts](markdown/CONTRACTS_0_3.md).

## Tutorial notebooks

The three notebooks in `examples/` introduce the current APIs with editable
inputs, numerical results and figures. Start with the point-cloud tutorial,
then explore explicit objects or your input format.

| Notebook | What it demonstrates |
| --- | --- |
| [Point-cloud representations and persistent analysis](examples/point_cloud_topology_workflow.ipynb) | A reproducible 34-point cloud; graph, simplicial, hyperdigraph and interaction representations; stationary homology/spectra; persistence, Betti curves, spectral curves and filtration GIFs. |
| [Fixed topological objects](examples/fixed_topological_objects_analysis.ipynb) | Editable dictionaries defining a graph, a closed simplicial shell, a hyperdigraph and an interaction complex; GF(2) homology and complete real Laplacian matrices/spectra through degree two. |
| [Seven input formats, one workflow](examples/different_input_formats_workflow.ipynb) | Independent PDB, MOL2, SDF, MOL, PDBQT, CIF and JSON examples; H0/H1 persistence, ordinary L0/L1 snapshots, spectral interpretation and custom descriptors. |

To run all tutorials with pip:

```bash
python -m pip install -e '.[plot]' jupyterlab ipykernel pandas pillow
python -m ipykernel install --user --name topokit --display-name 'Python (topokit)'
jupyter lab examples/point_cloud_topology_workflow.ipynb
```

Alternatively, the supplied [Conda environment](environment.yml) includes the
notebook dependencies:

```bash
conda env create -f environment.yml
conda activate topokit
jupyter lab
```

Select the Python environment in which TopoKit is installed. Run cells in order;
outputs go under `examples/output/`. The input-format tutorial uses explicit
small subsets for its larger structures. Ordinary Laplacian snapshot curves
are distinct from the two-scale persistent Laplacian API.

## Protein–ligand topology features

The default **FS-AN** workflow accepts a prepared protein PDB and ligand MOL2 in
the **same coordinate frame**, with coordinates in angstroms. It crops protein
atoms within 15 Å of supported ligand atoms, uses 50 alpha radii from 0.1 to
5.0 Å and 55 element/null channels, and computes ten summaries of ordinary
Hyperdigraph L0 spectra. Its output is a C-contiguous `float32` tensor of shape
`(10, 50, 55)`; C-order flattening gives **27,500 features**.

For one complex:

```python
from pathlib import Path
import numpy as np
from topokit.workflows.protein_ligand_prediction import featurize

# Replace these with a prepared protein–ligand pair.
tensor = featurize("protein.pdb", "ligand.mol2")
features = tensor.ravel(order="C")
Path("examples/output").mkdir(parents=True, exist_ok=True)
np.save("examples/output/topology_features.npy", features)
```

For the ten supplied test pairs, run the manifest-based exporter from the
repository root:

```bash
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

The exporter saves individual tensors and provenance records, the canonical
feature schema, and a complete `(N, 27500)` feature matrix with ordered sample
IDs when every row succeeds. Use the [sample guide](examples/protein_ligand/README.md)
and [manifest](examples/protein_ligand/manifest.csv) to prepare your own batch.
Sample provenance and redistribution status are recorded with the data.

Read the [full workflow guide](workflows/protein_ligand_prediction/README.md) for
input preparation, channel/statistic meanings, resource guards and output files.
An [agent skill](workflows/skills/topokit-protein-ligand/SKILL.md) guides an
assistant through the default recipe and checks its outputs.

The workflow expects an already prepared complex; docking and structure
preparation are separate steps. Resolve alternate conformations and choose the
intended receptor/ligand before extraction. Exact duplicate coordinates are
kept once, in cropped-protein-then-ligand order, with a removal receipt. Changes
to the crop, radii, channels or summaries define a different feature recipe and
must not be passed to a model trained for FS-AN.

Feature extraction produces descriptors. Affinity prediction additionally
requires a compatible trained model bundle and its fitted scaler. The selected
companion topology predictor is a three-seed GBDT ensemble; model assets are
separate from the package. See the [prediction guide](workflows/protein_ligand_prediction/ml/README.md).

## Sequence-based features

The selected **FS-AU** sequence recipe combines the **ESM-2
`esm2_t33_650M_UR50D`** protein encoder (1,280 features) with the **CPZ
`chembl27_pubchem_zinc_512`** ligand encoder (512 features), yielding a
**1,792-dimensional** vector. This is a separate pretrained embedding modality.
It takes protein sequence information and a ligand SMILES string; it does not
calculate topology from a bound three-dimensional complex.

Install `.[sequence]`, then follow the [sequence recipe and asset instructions](workflows/protein_ligand_prediction/sequence/README.md).
The [sequence agent skill](workflows/skills/topokit-sequence/SKILL.md) specifies
the selected encoders and verification steps. Encoder weights and downstream
GBDT models are external assets and are not downloaded at import time. Preserve
the documented model identities, pooling and token policies when reusing the
recipe; replacing CPZ with the ChEMBL27-only checkpoint changes the features.

## Architecture and scientific scope

TopoKit separates six responsibilities:

| Layer | Responsibility |
| --- | --- |
| `readers` | Parse inputs and preserve scientific metadata |
| `builders` | Construct explicit objects or filtrations |
| `core` | Analyze supplied objects: homology, persistence and Laplacians |
| `postprocessing` | Transform bars and spectra into statistics and vectors |
| `visualization` | Display supplied inputs, objects and numerical results |
| `workflows` | Compose public layers and optional downstream ML |

Core topology does not load datasets, train models, choose train/test splits, or
change supplied geometry. Homology over GF(2) and real Laplacian nullity need
not agree for every object. Alpha geometry uses floating-point calculations;
large or higher-dimensional constructions can be expensive. CIF parsing does
not perform symmetry or periodic-image expansion. See the [architecture](markdown/ARCHITECTURE.md),
[mathematical contracts](markdown/CONTRACTS.md) and [native alpha notes](markdown/NATIVE_ALPHA.md).

## Documentation

- [Documentation index](markdown/README.md), [notation](markdown/NOTATION.md) and [version 0.3 contracts](markdown/CONTRACTS_0_3.md)
- [Workflow index](workflows/README.md) and [visualization guide](markdown/VISUALIZATION.md)
- [Validation record](markdown/VALIDATION.md), [changelog](markdown/CHANGELOG.md) and [roadmap](markdown/ROADMAP.md)
- [Example-data provenance](examples/data/README.md) and [protein–ligand sample provenance](examples/protein_ligand/README.md)

For development checks:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Some tests require optional extras or reference assets; their availability and
scope are documented in the validation record. Historical research runs and
model-selection logs are retained in the [previous README snapshot](markdown/history/2026-09-23-readme-refresh/README.original.md).
Those records may refer to external research datasets absent from a clone.

## License

First-party code is licensed under the [MIT License](LICENSE). Example data,
pretrained weights and third-party dependencies retain their own terms;
see [NOTICE.md](NOTICE.md) and each data inventory for attribution and known
provenance gaps.
