# Source and data provenance

`topokit` is the primary development home for three first-party research
implementations integrated with the copyright owner's authorization:

| Integrated kernel | Source package | Source version |
| --- | --- | --- |
| `topokit.core._simplicial` and simplicial builders | `simplicial-topo` | 0.1.0 |
| `topokit.core._hyperdigraph` and hyperdigraph builders | `hyperdigraph-topo` | 0.6.0 |
| `topokit.core._interaction` and interaction builders | `interaction-topology` | 0.2.0 |

The three original source directories are retained outside this package as
migration references. Installed `topokit` does not import those directories.
Private kernel namespaces are implementation details; new applications use
the public `topokit` facades. Migrated tests remain in `tests/legacy_*`.

The hyperdigraph source carried an MIT license, Copyright (c) 2026 Dong Chen.
The owner authorized MIT licensing of the integrated first-party code.
The combined MIT notice is in `LICENSE`. NumPy, SciPy, optional Matplotlib,
and other installed third-party dependencies retain their own licenses.

The 24-point numerical fixture preserves coordinates from
`examples/data/B12N12_cage.xyz` in the source workspace, whose provenance
references https://doi.org/10.1021/ja410088y. The public example deliberately
uses only coordinates, generic IDs, a selected ID subset, and assigned uniform
weights. It makes no chemical or predictive-performance claim. Coordinate
data and fixture-specific loaders now live only in `examples/`, outside the
installed library. Historical migration notes are in `markdown/`.
Coordinate provenance is not erased by renaming the fixture; the MIT code license does
not override source dataset notices. Check data redistribution metadata when
preparing a public release or adding further datasets.

The additional molecular-reader fixtures (`PDB`, `MOL2`, `SDF`, `MOL`,
`PDBQT`, and `CIF`) are inventoried in `examples/data/README.md`, including
embedded identifiers, exact renamed-copy relationships, checksums, and known
provenance gaps. No blanket license is asserted for that collection. Fixtures
whose original preparation or redistribution terms were not retained are
appropriate for local validation only until those terms are confirmed.

The ten complexes in `examples/protein_ligand/` were copied byte-for-byte from
the user's local protein–ligand dataset for requested smoke testing. Their
`SOURCE.json` records source paths, dataset-release metadata and SHA-256 values;
`EXPECTED.json` records feature-extraction checks. These files are not relicensed
under MIT. Upstream structure/preparation redistribution terms remain to be
confirmed before publishing the data. Their inclusion in a locally built source
archive does not resolve that provenance question. Wheels contain no structures.

`examples/data/AQUCOG_clean.cif` is byte-identical to
`structure_10143/AQUCOG_clean.cif` in the public CoRE MOF 2019 v1.1.4
all-solvent-removed dataset (DOI
[10.5281/zenodo.7691378](https://doi.org/10.5281/zenodo.7691378)), which is
redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please cite Chung et al., *J. Chem. Eng. Data* **64**, 5985–5998 (2019), DOI
[10.1021/acs.jced.9b00835](https://doi.org/10.1021/acs.jced.9b00835). This is a
computation-ready, symmetry-expanded P1 derivative of CSD refcode AQUCOG with
the neon guest and solvent removed; it is not a licensed WebCSD export. The
originating experimental structure is Wood et al., *Chem. Commun.* **52**,
10048 (2016), DOI
[10.1039/C6CC04808K](https://doi.org/10.1039/C6CC04808K).

The reader-layer Pauling electronegativity constants are an offline snapshot
of factual [PubChem periodic-table values](https://pubchem.ncbi.nlm.nih.gov/periodic-table/electronegativity).
Their source, date, checksum, missing entries, and override policy are documented
in [markdown/REFERENCE_DATA.md](markdown/REFERENCE_DATA.md). They are attributed
scientific reference constants, distinct from the example coordinate dataset
and the first-party algorithms. No runtime data download occurs.

Native topology algorithms use numerical/geometric primitives from SciPy.
GUDHI is an optional runtime dependency only for the explicitly selected
`gudhi_exact` alpha geometry backend (`topokit[alpha_exact]`). Default native
construction, the active protein–ligand workflow and mathematical cores do not
depend on it. TopoKit implements the native alpha filtration and rational
degeneracy checks itself, using SciPy for candidate triangulations. The adapter calls
GUDHI's public API; no GUDHI implementation is copied or vendored. GUDHI and
its dependencies retain their own upstream licenses. Independent external
reference comparisons also remain part of development/validation.
No path-complex, path-homology, or path-Laplacian module is exposed. The
hyperdigraph's directed-sequence construction and specialized internal
sequence algorithms preserve sequence-hyperdigraph embedded topology.

## Compact supervised TopoFormer

The optional `workflows.topoformer` module is an independent compact
implementation of a pre-norm transformer, following the width-first 2D
sine/cosine position and CLS/dense/tanh pooling conventions reviewed in
[WeilabMSU/TopoFormer](https://github.com/WeilabMSU/TopoFormer/tree/a63a8383be3cd9938a498d8a676840b531b3eb32).
The source commit and hashes are recorded in the protein–ligand DL application
receipt. This module does not import or bundle the upstream package, datasets
or pretrained weights, and does not claim its published pretrained scores.
PyTorch supplies neural-network primitives as an optional dependency.

## Optional ChEMBL sequence encoder

The workflow private `_chembl.py` is adapted from the user-supplied minimal PyTorch implementation in WeilabMSU/PretrainModels (`bt_fps/molecular_roberta.py`, local repository commit 454393fb57bfbf12982745449dc3b6752b00d04d). The upstream repository declares MIT in its README: https://github.com/WeilabMSU/PretrainModels. Source and asset-copy receipts are retained with the external dataset assets. ESM-2 is loaded through Transformers from the pinned official facebook/esm2_t33_650M_UR50D snapshot. Neither pretrained checkpoint is bundled into the TopoKit wheel.
