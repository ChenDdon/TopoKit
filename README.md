# TopoKit

[Source repository](https://github.com/ChenDdon/TopoKit) ·
[Releases](https://github.com/ChenDdon/TopoKit/releases) ·
[Report an issue](https://github.com/ChenDdon/TopoKit/issues)

TopoKit is a Python toolkit for constructing and analyzing **simplicial complexes,
hyperdigraphs, and interaction complexes**. It turns explicit objects or scientific
coordinates into homology, persistence, Laplacian spectra and numerical features
through composable APIs.

Use the general APIs to build an analysis for your scientific question, or start
with a predefined workflow for protein–ligand or protein–protein complexes.
Tutorial notebooks and small example datasets demonstrate construction,
analysis, visualization and feature extraction.

[Install](#installation) · [Basic usage](#basic-usage) ·
[Tutorial notebooks](#tutorial-notebooks) · [Application workflows](#application-workflows) ·
[Citing the methods](#citing-the-methods) · [Documentation](#documentation)

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

The source checkout includes tutorials, example inputs and workflow scripts.
The Python wheel contains the library and its runtime schemas. Versioned
packages are available from [GitHub Releases](https://github.com/ChenDdon/TopoKit/releases).
The installation above uses this repository's source.

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
| `.[dl]` | Optional supervised affinity models |
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
choices remain explicit. See [input contracts](docs/CONTRACTS.md).

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

## Application workflows

Predefined recipes help turn prepared molecular inputs into feature arrays for
analysis or downstream modelling. Their guides specify input preparation,
feature definitions, optional dependencies and model requirements.

| Workflow | Inputs | Guide and examples |
| --- | --- | --- |
| Protein–ligand topology features | A prepared protein structure and ligand pose in the same coordinate frame | [Recipe and usage](workflows/protein_ligand_prediction/README.md) · [Ten example complexes](examples/protein_ligand/README.md) |
| Protein–protein topology features | A prepared complex structure and explicit partner-chain assignments | [Recipe and usage](workflows/protein_protein_prediction/README.md) · [Ten two-chain examples](examples/protein_protein/README.md) |
| Protein–ligand sequence features | Protein sequences and ligand SMILES | [Recipe and model assets](workflows/protein_ligand_prediction/sequence/README.md) |

Topology feature extraction uses the base installation. Optional pretrained
embeddings and affinity prediction require the assets and dependencies described
in their workflow guides. Inputs should already represent the intended molecular
system; docking and structure preparation are separate steps.

See the [workflow index](workflows/README.md) for commands and the accompanying
agent skills for guided use of each recipe.

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
not perform symmetry or periodic-image expansion. See the [architecture](docs/ARCHITECTURE.md),
[mathematical contracts](docs/CONTRACTS.md) and [native alpha notes](docs/NATIVE_ALPHA.md).

## Citing the methods

If TopoKit supports your research, cite the papers relevant to the methods you
use and report the package version and workflow recipe in your methods section.

1. Dong Chen, Jian Liu and Guo-Wei Wei. **Multiscale topology-enabled structure-to-sequence transformer for protein–ligand interaction predictions.** *Nature Machine Intelligence* **6**, 799–810 (2024). [doi:10.1038/s42256-024-00855-1](https://doi.org/10.1038/s42256-024-00855-1).
2. Dong Chen, Jian Liu, Chun-Long Chen and Guo-Wei Wei. **Interaction topology theory deciphers multiscale codes of MOF-like materials.** *Science Advances* **12**(34), eaee8016 (2026). [doi:10.1126/sciadv.aee8016](https://doi.org/10.1126/sciadv.aee8016).
3. Jian Liu, Dong Chen and Guo-Wei Wei. **Persistent interaction topology in data analysis.** *Foundations of Data Science* **9**, 34–60 (2026). [doi:10.3934/fods.2025011](https://doi.org/10.3934/fods.2025011).
4. Dong Chen, Jian Liu, Jie Wu and Guo-Wei Wei. **Persistent hyperdigraph homology and persistent hyperdigraph Laplacians.** *Foundations of Data Science* **5**(4), 558–588 (2023). [doi:10.3934/fods.2023010](https://doi.org/10.3934/fods.2023010).

## Documentation

- [Reference guide](docs/README.md): architecture, scientific contracts, notation and visualization
- [Workflow guides](workflows/README.md): Topo features, sequence embeddings and optional affinity prediction
- [Changelog](CHANGELOG.md), [v0.3.0 release notes](release-notes/v0.3.0.md) and [automated checks](https://github.com/ChenDdon/TopoKit/actions)
- [Example-data provenance](examples/data/README.md), [protein–ligand examples](examples/protein_ligand/README.md) and [protein–protein examples](examples/protein_protein/README.md)

For development checks:

```bash
python -m pip install -e '.[test,plot,ml]'
python -m pytest -q
```

Optional encoder, DL and reference-backend tests need their corresponding
extras or external assets.

## License

First-party code is licensed under the [MIT License](LICENSE). Example data,
pretrained weights and third-party dependencies retain their own terms;
see [NOTICE.md](NOTICE.md) and each data inventory for attribution and known
provenance gaps.
