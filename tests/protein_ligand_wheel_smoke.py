"""Base-only installed-wheel smoke: python -I PATH/TO/THIS.py MANIFEST.csv.

Run outside the source checkout in a fresh environment containing the wheel,
NumPy and SciPy. This standalone script deliberately does not invoke pytest,
whose configured pythonpath would put the source tree ahead of the wheel.
"""
import csv
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

import numpy as np


class BlockOptionalAndLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "transformers", "rdkit", "sklearn", "gudhi",
            "simplicial_topo", "hyperdigraph_topo", "interaction_topology",
        }:
            raise AssertionError(f"Base feature extraction attempted an optional import: {fullname}")
        return None


sys.meta_path.insert(0, BlockOptionalAndLegacy())
import topokit
from topokit.workflows.protein_ligand_prediction import featurize, schema

assert sys.flags.isolated, "Use python -I to prevent source-tree/PYTHONPATH imports"
installed_path = Path(topokit.__file__).resolve()
assert installed_path.is_relative_to(Path(sys.prefix).resolve()), installed_path
assert "site-packages" in installed_path.parts, installed_path
assert not (installed_path.parent / "datasets").exists()

manifest = Path(sys.argv[1]).resolve()
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 10 and len({row["sample_id"] for row in rows}) == 10
provenance = json.loads((manifest.parent / "SOURCE.json").read_text(encoding="utf-8"))
expected_files = {
    entry["example_relative_path"]: entry["sha256"]
    for sample in provenance["samples"] for entry in sample["files"].values()
}
assert schema()["recipe_id"], "Packaged recipe schema is missing"

summary = []
for row in rows:
    inputs = {}
    for key in ("protein_file", "ligand_file"):
        path = (manifest.parent / row[key]).resolve()
        assert path.is_relative_to(manifest.parent), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_files[row[key]]
        inputs[key] = path
    tensor = featurize(**inputs)
    assert tensor.shape == (10, 50, 55), (row["sample_id"], tensor.shape)
    assert tensor.dtype == np.dtype("float32") and tensor.flags.c_contiguous
    assert np.isfinite(tensor).all() and np.count_nonzero(tensor) > 0
    assert tensor.ravel(order="C").shape == (27500,)
    # The reserved null/null channel is absent by definition for every input.
    np.testing.assert_array_equal(tensor[:, :, 0], np.zeros((10, 50)))
    summary.append({"sample_id": row["sample_id"], "shape": list(tensor.shape),
                    "finite": True, "feature_count": tensor.size})
print(json.dumps({"installed_package": str(installed_path), "python": sys.executable,
                  "base_dependency_isolation": "passed", "samples": summary}, indent=2))
