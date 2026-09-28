"""Exercise installed Topo PPI extraction using a tiny synthetic coordinate pair."""
import importlib.abc
import json
from pathlib import Path
import sys


class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "rdkit", "sklearn", "joblib", "gudhi"}:
            raise AssertionError(f"Base PPI extraction imported optional dependency: {fullname}")
        return None


sys.meta_path.insert(0, BlockOptional())
import numpy as np
import topokit
from topokit.workflows.protein_protein_prediction import featurize, schema

output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=False)
structure = output / "synthetic_pair.pdb"
structure.write_text(
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
    "ATOM      2  CA  ALA B   1       3.000   0.000   0.000  1.00 20.00           C\nEND\n",
    encoding="utf-8",
)
tensor = featurize(structure, ["A"], ["B"])
assert tensor.shape == (10, 14, 36) and tensor.dtype == np.float32
assert tensor.flags.c_contiguous and np.isfinite(tensor).all()
assert schema()["alpha_radii_angstrom"] == [1 + i / 2 for i in range(14)]
expected = np.zeros((10, 14, 36), dtype=np.float32)
# Single-partner CA channels each contain one isolated vertex at every radius.
expected[:2, :, 5] = 1
expected[:2, :, 30] = 1
# The two CA sites are 3 Å apart: their alpha edge first appears at radius1.5.
expected[0, :, 35] = 2
expected[1, :, 35] = 1
expected[1, 0, 35] = 2
expected[2, 1:, 35] = 2
expected[3, 1:, 35] = 1
expected[5, 1:, 35] = 1
expected[6:9, 1:, 35] = 2
np.testing.assert_array_equal(tensor, expected)
np.save(output / "topology_features.npy", tensor.ravel(order="C"), allow_pickle=False)
print(json.dumps({"status": "passed", "recipe_id": schema()["recipe_id"],
                  "shape": list(tensor.shape), "feature_count": tensor.size,
                  "installed_topokit": topokit.__file__}, indent=2))

# An optional manifest exercises the ten shipped real two-chain examples.
if len(sys.argv) > 2:
    import csv
    import hashlib
    from topokit.workflows.protein_protein_prediction import read_selected_atoms

    assert sys.flags.isolated, 'Use python -I for installed-wheel checks'
    installed = Path(topokit.__file__).resolve()
    assert installed.is_relative_to(Path(sys.prefix).resolve())
    assert 'site-packages' in installed.parts
    manifest = Path(sys.argv[2]).resolve()
    with manifest.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10 and len({row['sample_id'] for row in rows}) == 10
    sources = json.loads((manifest.parent / 'SOURCE.json').read_text(encoding='utf-8'))
    expected = json.loads((manifest.parent / 'EXPECTED.json').read_text(encoding='utf-8'))
    assert expected['recipe_id'] == schema()['recipe_id']
    sources = {row['sample_id']: row for row in sources['samples']}
    expected = {row['sample_id']: row for row in expected['samples']}
    assert set(sources) == set(expected) == {row['sample_id'] for row in rows}
    summary = []
    for row in rows:
        name = row['sample_id']
        assert row['partner_a_chains'] == 'A' and row['partner_b_chains'] == 'B'
        path = manifest.parent / row['structure_file']
        assert not path.is_symlink() and path.resolve().is_relative_to(manifest.parent)
        info = sources[name]['files']['structure']
        assert info['example_relative_path'] == row['structure_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        lines = path.read_text(encoding='utf-8').splitlines()
        atoms = [line for line in lines if line.startswith('ATOM  ')]
        assert {line[21] for line in lines if line.startswith(('ATOM  ', 'HETATM'))} == {'A', 'B'}
        assert not any(line.startswith('MODEL ') for line in lines)
        assert all(line[16] == ' ' for line in atoms)
        selected = read_selected_atoms(path, ['A'], ['B'])['diagnostics']
        assert selected['missing_requested_chains'] == [[], []]
        assert selected['missing_CA_residues_excluded'] == [[], []]
        assert selected['observed_requested_partner_residue_counts'] == [
            sources[name]['observed_residues'][chain] for chain in ['A', 'B']]
        assert min(selected['observed_requested_partner_residue_counts']) >= 30
        for key in ['ordered_partner_chains', 'cropped_residues', 'selected_atoms']:
            assert selected[key] == expected[name][key], (name, key)
        assert not selected['alternate_atom_rows_removed']
        assert not selected['duplicate_coordinate_rows_removed']
        tensor = featurize(path, ['A'], ['B'])
        assert tensor.shape == (10, 14, 36) and tensor.dtype == np.float32
        assert tensor.flags.c_contiguous and np.isfinite(tensor).all()
        assert np.count_nonzero(tensor[:, :, 35]) > 0
        np.testing.assert_array_equal(tensor[:, :, 0], np.zeros((10, 14)))
        summary.append({'sample_id': name, 'feature_count': tensor.size, 'finite': True})
    print(json.dumps({'installed_package': str(installed), 'samples': summary}, indent=2))
