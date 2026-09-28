# Examples

Start with the three notebooks, then use the small scripts or protein-complex
workflow with your own inputs. Run the commands below from the repository root.

## Notebooks

| Notebook | What it demonstrates |
| --- | --- |
| [Point-cloud topology](point_cloud_topology_workflow.ipynb) | Editable coordinates, graph/simplicial/hyperdigraph/interaction constructions, persistence, spectra, and filtration animations. |
| [Fixed topological objects](fixed_topological_objects_analysis.ipynb) | Explicit graph, simplicial, hyperdigraph, and interaction objects; homology and Laplacian matrices. |
| [Different input formats](different_input_formats_workflow.ipynb) | PDB, MOL2, SDF, MOL, PDBQT, CIF, and JSON readers followed by topological analysis. |

Install the notebook dependencies and open JupyterLab:

```bash
python -m pip install ".[plot]" jupyterlab pandas pillow
jupyter lab examples/point_cloud_topology_workflow.ipynb
```

The first two notebooks define their inputs in code. The input-format notebook
reads the bundled [data fixtures](data/README.md). That directory also contains
small reader and geometry regression fixtures used by the test suite.

## Small runnable scripts

| Script | Purpose |
| --- | --- |
| [point_cloud.py](point_cloud.py) | Run all three point-cloud routes and export results and feature arrays. |
| [configured_pipeline.py](configured_pipeline.py) | Configure constructions, fitted barcode features, and ordinary/persistent Laplacians. |
| [static_objects.py](static_objects.py) | Compute homology for three explicitly defined topological objects. |
| [features_to_ml.py](features_to_ml.py) | Create aligned training/test feature matrices for a later estimator. |
| [visualization_gallery.py](visualization_gallery.py) | Export figures illustrating simplicial and directed objects. |

For example:

```bash
python examples/point_cloud.py --output examples/output/point_cloud
python examples/static_objects.py
python examples/features_to_ml.py
```

`demo.py` and `data_fixture.py` support the point-cloud demo and tests. Generated
results are written to `examples/output/` when a tutorial or demo runs.

## Protein–ligand topology features

The [protein–ligand guide](protein_ligand/README.md) shows how to extract the
fixed **Topo** feature recipe for one complex or all ten included pairs, then
adapt the [manifest](protein_ligand/manifest.csv) to your own prepared complexes.
Extraction requires only the base package, NumPy, and SciPy.

## Protein–protein topology features

The [protein–protein guide](protein_protein/README.md) demonstrates the
5,040-feature PPI Topo recipe using ten compact complexes. Every structure
has exactly chains A and B, one chain per partner. Start with the supplied
[manifest](protein_protein/manifest.csv), then replace its rows with your own
prepared complexes. Extraction requires only the base package, NumPy and SciPy.

Structure provenance and applicable source-data terms are documented in the
[data inventory](data/README.md), [protein–ligand guide](protein_ligand/README.md)
and [protein–protein guide](protein_protein/README.md).
