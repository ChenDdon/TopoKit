# TopoKit v0.3.0

First public GitHub release of TopoKit, a Python toolkit for simplicial,
sequence-hyperdigraph and two-factor interaction topology.

## Included

- Six composable layers: readers, builders, core, postprocessing,
  visualization and workflows.
- Homology, persistence, ordinary and explicitly requested persistent
  Laplacians, spectral summaries and feature vectors.
- Native alpha geometry using NumPy/SciPy; GUDHI remains an optional explicit
  reference backend.
- Fixed FS-AN protein–ligand feature extraction: 15 Å crop, 50 alpha radii,
  55 element/null channels, ten summaries and 27,500 features per complex.
- A manifest-based exporter with canonical schema, sample identities, hashes,
  diagnostics and explicit failure records; ten small example complexes.
- Three executed tutorial notebooks covering point clouds, explicit objects
  and molecular/crystallographic input formats.
- Agent guides and skills for topology extraction and the optional frozen
  ESM-2 + CPZ sequence recipe. Sequence embeddings have 1,792 coordinates;
  encoder weights are supplied separately.

## Install

```bash
git clone https://github.com/ChenDdon/TopoKit.git
cd TopoKit
git checkout v0.3.0
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m topokit info
```

On Windows, activate the virtual environment with `.venv\Scripts\activate`.
Python 3.10+ is required. The base library depends only on NumPy and SciPy.
Install optional extras for plotting, ML, sequence encoders or supervised DL.

To run the supplied molecular examples from the repository root:

```bash
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

The completed matrix has shape `(10, 27500)`; individual tensors have shape
`(10, 50, 55)`. Output directories must be new.

## Assets and validation

The wheel contains the Python library and small runtime schemas. The source
archive includes scripts, notebooks, tests, examples and documentation.
`SHA256SUMS` gives the checksums of both distribution files. No pretrained
weights, fitted models, generated research outputs or transfer archives are
included. This is a GitHub release, not a PyPI publication.

The release workflow publishes only after the test matrix passes on Linux,
macOS and Windows with Python 3.10 and 3.12, including installed-wheel core and
ten-complex molecular smoke checks. Local validation also covered 1,279 relevant
tests and extraction from an unpacked source archive. The optional CPZ profile,
asset hashes and CLI were checked; full pretrained-model inference was not
rerun as part of release preparation.

## Scientific and data scope

Protein–ligand inputs must already be prepared in a common coordinate frame.
FS-AN computes complete ordinary L0 spectra over an alpha-radius schedule;
these descriptors are not themselves an affinity prediction. Geometry/resource
failures are explicit, and exact recipe identity matters for model reuse.

The MIT license covers first-party code. Example structures, optional weights
and dependencies retain their upstream terms. The data inventories and
`NOTICE.md` document provenance and unresolved redistribution metadata for
some local demonstration structures; this release does not assign them an MIT
data license. Historical H0/atom-deletion controllers preserve fixed-cohort
research protocols and are not generic alternatives to the default exporter.
