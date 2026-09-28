# Protein–protein topology feature examples

Ten compact prepared complexes demonstrate the **PPI Topo** recipe. Every file
contains exactly two coordinate chains, **A and B**, with **one chain per
partner**. Both chains contain at least 30 observed residues. The copied files
have no alternate-location labels or multiple-model records, and every protein
ATOM residue has a Cα atom. No missing-chain exceptions are needed.

The structures are unchanged copies from the local protein–protein dataset,
occupying about **1.57 MB** in total. They include protein domains and fragments;
they are not necessarily full-length proteins. This is a small functional
example set, not an affinity benchmark. No affinity labels, sequence features
or model weights are included.

## Install and try one complex

Run from the package root, which contains `pyproject.toml`:

```bash
python -m pip install .
```

Extraction uses only the base package, NumPy and SciPy:

```python
from pathlib import Path
import numpy as np
from topokit.workflows.protein_protein_prediction import featurize

tensor = featurize(
    "examples/protein_protein/structures/1q5w/complex.pdb", ["A"], ["B"]
)
assert tensor.shape == (10, 14, 36)
assert tensor.dtype == np.float32 and np.isfinite(tensor).all()
output = Path("examples/output/protein_protein_single")
output.mkdir(parents=True, exist_ok=True)
np.save(output / "1q5w.npy", tensor.ravel(order="C"), allow_pickle=False)
```

The result is **5,040 topology features**: ten statistics × fourteen radii ×
36 ordered atom-category channels, flattened in C order. The fixed recipe
orders partners by observed residue count before a strict 20 Å Cα interface
crop. Thus, naming A first in the manifest does not force A into the first
feature-channel axis. Use the API's ordering unchanged.

## Extract all ten with the installed API

The [manifest](manifest.csv) lists each structure and its partner chains. Paths
are relative to the manifest directory. This code works in either package copy:

```python
import csv
import json
from pathlib import Path
import numpy as np
from topokit.workflows.protein_protein_prediction import featurize, schema

manifest = Path("examples/protein_protein/manifest.csv")
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
X = np.stack([
    featurize(
        manifest.parent / row["structure_file"],
        row["partner_a_chains"].split(";"),
        row["partner_b_chains"].split(";"),
    ).ravel(order="C")
    for row in rows
])
assert X.shape == (10, 5040) and X.dtype == np.float32
assert np.isfinite(X).all()
output = Path("examples/output/protein_protein_api")
output.mkdir(parents=True, exist_ok=False)
np.save(output / "features.npy", X, allow_pickle=False)
(output / "sample_ids.json").write_text(
    json.dumps([row["sample_id"] for row in rows], indent=2) + "\n", encoding="utf-8"
)
(output / "feature_schema.json").write_text(
    json.dumps(schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
```

Choose a new output directory for each batch. For your own collection, change
the manifest and use `len(rows)` instead of the literal ten in the shape check.
This API example saves arrays for analysis. The public PPI workflow's batch
exporter additionally writes the complete receipts required by its optional
affinity predictor. No pretrained models are needed for these examples.

## Included complexes

All manifests assign chain **A** to partner A and chain **B** to partner B.
Residue counts below refer to observed ATOM residues before interface selection.

| Sample ID | Chain A residues | Chain B residues |
| --- | ---: | ---: |
| 1q5w | 31 | 76 |
| 2l14 | 59 | 49 |
| 2ktf | 76 | 32 |
| 2den | 46 | 76 |
| 2l0f | 76 | 45 |
| 2rms | 71 | 61 |
| 1otr | 49 | 76 |
| 2jy6 | 76 | 52 |
| 1wr1 | 76 | 58 |
| 1u5s | 71 | 66 |

All ten passed extraction on 2026-09-28 and yielded finite float32 `(10,14,36)`
tensors. [EXPECTED.json](EXPECTED.json) records the recipe, environment,
selected-atom counts and observed checks. No coordinates or chain names were
edited, and these are regular files rather than links to the source dataset.

## Provenance

[SOURCE.json](SOURCE.json) records local source paths, upstream membership,
file sizes and SHA-256 hashes. Partner annotations come from the saved
PLNet V2020 subset; coordinates are the local PDBbind v2020R1 references with
the recorded v2024 reprocessing workflow. This does not establish identity
with the exact structures used in the PLNet paper.

These third-party structures retain their upstream data terms; the TopoKit
MIT software license does not relicense them. Local file identities have been
verified, while original preparation and redistribution permission have not
been independently verified. See the package's [data notice](../../NOTICE.md).
