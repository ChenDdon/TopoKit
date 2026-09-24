# Protein–ligand feature examples

Ten small, existing protein–ligand structure pairs demonstrate the default
**FS-AN** topology feature recipe. The structures are byte-for-byte local copies
from the project's `datasets/protein_ligand_prediction/structures` collection;
[SOURCE.json](SOURCE.json) records original paths, upstream metadata, file sizes,
and SHA-256 hashes. The complete structure collection here is about 1.03 MB.

These examples require **Python 3.10+, NumPy, and SciPy**. No learned model,
GPU, docking program, or optional ML package is needed for topology extraction.
From the package root (the directory containing `pyproject.toml`), install:

```bash
python -m pip install .
```

## One complex

Run from the package root after installation:

```python
from pathlib import Path
import numpy as np
from topokit.workflows.protein_ligand_prediction import featurize

sample = Path("examples/protein_ligand/structures/2lk1")
tensor = featurize(sample / "protein.pdb", sample / "ligand.mol2")
assert tensor.shape == (10, 50, 55)
assert np.isfinite(tensor).all()
features = tensor.ravel(order="C")  # 27,500 values, float32
output = Path("examples/output/protein_ligand_single")
output.mkdir(parents=True, exist_ok=True)
np.save(output / "2lk1.npy", tensor)
```

The axes are **10 spectral statistics × 50 radii × 55 element channels**.
The fixed recipe selects protein C/N/O/S `ATOM` coordinates within 15 Å of
supported ligand atoms and computes element-specific alpha-complex L0 spectra.
Ligand atoms, including explicit H, are retained according to the recipe;
exact repeated coordinates are merged deterministically. The first channel
(`null`, `null`) and absent element channels contain zeros by design.

## All ten complexes

From the package root, export all ten pairs with the batch runner:

```bash
python workflows/protein_ligand_prediction/extract_features.py \
  --manifest examples/protein_ligand/manifest.csv \
  --output examples/output/protein_ligand
```

Choose a new output directory for each run; the runner refuses to overwrite
an existing directory. A successful run writes `features.npy` with shape
`(10, 27500)`, `sample_ids.json` in the same row order, individual tensors under
`samples/`, and schema and per-sample provenance records.

The manifest has columns `sample_id,protein_file,ligand_file`. Paths are
relative to the manifest's own directory. Copy its format to run prepared
complexes for your own question. Sample IDs identify feature rows; no binding
affinity labels or predicted affinities are included in these examples.

The installed Python API also supports the same batch in either package copy:

```python
import csv
from pathlib import Path
import numpy as np
from topokit.workflows.protein_ligand_prediction import featurize

manifest = Path("examples/protein_ligand/manifest.csv")
with manifest.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
X = np.stack([
    featurize(manifest.parent / row["protein_file"],
              manifest.parent / row["ligand_file"])
    for row in rows
])
assert X.shape == (10, 10, 50, 55)
assert np.isfinite(X).all()
X_flat = X.reshape(len(rows), -1, order="C")  # (10, 27500)
```

For a custom manifest, replace the literal sample count in the assertion with
`len(rows)`. Preserve row order and recipe metadata when using feature matrices
in downstream statistical or machine-learning analysis.

## Included pairs and observed checks

| Sample ID | Selected protein atoms | Selected ligand atoms | Ligand elements |
| --- | ---: | ---: | --- |
| 2lk1 | 226 | 27 | C, H, O, S |
| 6fhu | 342 | 42 | C, H, N, O |
| 6v1c | 298 | 51 | C, H, N, O |
| 2m3z | 376 | 37 | C, H, N, O, S |
| 2p0x | 445 | 43 | C, H, N, O, P |
| 2kaw | 552 | 41 | C, F, H, O, S |
| 2kfx | 593 | 44 | C, Cl, H, N, O, S |
| 4lwv | 596 | 71 | C, Cl, F, H, N, O, S |
| 4jv7 | 589 | 50 | Br, C, H, N, O |
| 4aq3 | 736 | 94 | C, Cl, H, I, N, O, S |

All ten pairs passed default FS-AN extraction on 2026-09-24: each yielded a
finite float32 `(10, 50, 55)` tensor and a 27,500-value C-order feature vector.
The ten-pair run took approximately **7.1 seconds** on the recorded macOS arm64
environment with numerical-library thread counts set to one. Runtime varies by
machine. [EXPECTED.json](EXPECTED.json) records per-pair counts, observed
nonzero counts, timings, recipe identity, and package/library versions.

The sample selection spans all ten supported ligand elements and favors small
files without explicit alternate-location labels or multiple `MODEL` records.
It is a functional demonstration set, not an accuracy benchmark or a
representative sample of binding targets. No candidates failed the extraction
check and no structure coordinates were edited.

## Input preparation and data provenance

Supply a prepared protein and ligand from the **same complex in the same
coordinate frame**, with coordinates measured in ångströms. The workflow does
not dock, align, protonate, choose alternate conformers, or repair structures.
Resolve alternate locations and multiple models before providing your own PDB.
A complex may produce a valid feature vector without being a scientifically
appropriate representation of your particular binding question.

The local source manifest identifies these pairs as **PDBbind v2020R1;
v2024 reprocessing workflow** and records `new_download/v2020_P-L.tar.gz` as the
source archive. That metadata and local file identity have been checked, but
the original acquisition/preparation chain and permission to redistribute the
processed data have not been independently verified. **The topokit MIT license
covers the software; it does not relicense these third-party structure files.**
Existing upstream terms apply. Confirm them before including this collection
in a public release; these copies currently provide local validation data.
