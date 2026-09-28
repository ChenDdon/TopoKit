"""Scientific selection and spectral contracts using synthetic, nonbiological inputs."""
from __future__ import annotations


import numpy as np
import pytest


from topokit.workflows.protein_protein_prediction import features as ppi

def atom(index, chain, residue, name, point, *, element=None, occupancy=1.0,
         alt=None, record="ATOM", residue_name="ALA"):
    return dict(index=index, chain=chain, residue=(chain, str(residue), ""),
                residue_name=residue_name, name=name, element=element or name[0],
                point=np.asarray(point, dtype=float), occupancy=occupancy,
                alt=alt, record=record)


def test_disjoint_carbon_and_alpha_carbon_categories():
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "A", 1, "CB", (0, 1, 0)),
             atom(2, "A", 1, "N", (0, 2, 0)),
             atom(3, "A", 1, "O", (0, 3, 0)),
             atom(4, "A", 1, "SG", (0, 4, 0)),
             atom(5, "A", 1, "H", (0, 5, 0)),
             atom(6, "B", 1, "CA", (4, 0, 0)),
             atom(7, "A", 2, "CA", (0, 6, 0), element="Ca", record="HETATM")]
    selected = ppi.select_atoms(atoms, ["A"], ["B"])
    assert selected["partner1_categories"].tolist() == ["CA", "C", "N", "O", "S"]
    assert selected["partner2_categories"].tolist() == ["CA"]
    assert selected["diagnostics"]["source_partner_residue_counts"] == [1, 1]


def test_partner_length_uses_full_annotated_multichain_partner_before_crop():
    # A has more interface residues; B+C is longer before the interface crop.
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "A", 2, "CA", (0, 1, 0)),
             atom(2, "B", 1, "CA", (4, 0, 0)),
             atom(3, "C", 1, "CA", (100, 0, 0)),
             atom(4, "C", 2, "CA", (110, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A"], ["C", "B"])
    diagnostic = selected["diagnostics"]
    assert diagnostic["ordered_source_partners"] == ["B", "A"]
    assert diagnostic["ordered_partner_residue_counts"] == [3, 2]
    assert diagnostic["ordered_partner_chains"] == [["B", "C"], ["A"]]
    assert diagnostic["cropped_residues"] == [1, 2]


def test_equal_length_partners_follow_coordinate_chain_order():
    atoms = [atom(0, "Z", 1, "CA", (4, 0, 0)),
             atom(1, "A", 1, "CA", (0, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A"], ["Z"])
    assert selected["diagnostics"]["ordered_source_partners"] == ["B", "A"]
    np.testing.assert_array_equal(selected["partner1_points"], [[4, 0, 0]])


def test_interface_is_strict_ca_cutoff_and_retains_entire_qualified_residue():
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "A", 1, "CB", (-10, 0, 0)),
             atom(2, "A", 2, "CA", (-100, 0, 0)),
             atom(3, "A", 3, "CA", (-110, 0, 0)),
             atom(4, "B", 1, "CA", (19.9, 0, 0)),
             atom(5, "B", 1, "CB", (1000, 0, 0)),
             atom(6, "B", 2, "CA", (20, 0, 0)),
             atom(7, "B", 2, "CB", (1, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A"], ["B"])
    np.testing.assert_array_equal(selected["partner1_points"], [[0, 0, 0], [-10, 0, 0]])
    np.testing.assert_array_equal(selected["partner2_points"], [[19.9, 0, 0], [1000, 0, 0]])
    assert selected["diagnostics"]["cropped_residues"] == [1, 1]


def test_highest_occupancy_alternates_are_selected_before_interface_crop():
    atoms = [atom(0, "A", 1, "CA", (100, 0, 0), occupancy=0.2, alt="A"),
             atom(1, "A", 1, "CA", (0, 0, 0), occupancy=0.8, alt="B"),
             atom(2, "A", 1, "CB", (1, 0, 0), occupancy=0.6, alt="B"),
             atom(3, "A", 1, "CB", (2, 0, 0), occupancy=0.6, alt="A"),
             atom(4, "B", 1, "CA", (4, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A"], ["B"])
    np.testing.assert_array_equal(selected["partner1_points"], [[0, 0, 0], [1, 0, 0]])
    assert selected["diagnostics"]["alternate_atom_rows_removed"] == [0, 3]


def test_unanchored_residue_is_excluded_with_identity_diagnostic():
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "A", 2, "N", (1, 0, 0)),
             atom(2, "B", 1, "CA", (4, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A"], ["B"])
    assert selected["diagnostics"]["missing_CA_residues_excluded"] == [[("A", "2", "")], []]
    assert selected["diagnostics"]["source_partner_residue_counts"] == [2, 1]
    assert selected["partner1_categories"].tolist() == ["CA"]


def pdb_atom(serial, chain, x, *, name="CA", element="C"):
    return (f"ATOM  {serial:5d} {name:^4s} ALA {chain}   1    "
            f"{x:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{0:6.2f}          {element:>2s}\n")


def test_first_encountered_pdb_model_not_smallest_model_number(tmp_path):
    path = tmp_path / "synthetic.pdb"
    path.write_text("MODEL        7\n" + pdb_atom(1, "A", 0) + pdb_atom(2, "B", 4)
                    + "ENDMDL\nMODEL        1\n" + pdb_atom(1, "A", 100)
                    + pdb_atom(2, "B", 200) + "ENDMDL\nEND\n")
    selected = ppi.read_selected_atoms(path, ["A"], ["B"])
    assert selected["diagnostics"]["input"]["model"] == 7
    np.testing.assert_array_equal(selected["partner1_points"], [[0, 0, 0]])
    np.testing.assert_array_equal(selected["partner2_points"], [[4, 0, 0]])


def test_first_mmcif_model_and_author_chain_namespace(tmp_path):
    path = tmp_path / "synthetic.cif"
    columns = ("group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
               "label_asym_id", "label_seq_id", "auth_asym_id", "auth_seq_id",
               "pdbx_PDB_ins_code", "label_alt_id", "occupancy", "Cartn_x", "Cartn_y",
               "Cartn_z", "pdbx_PDB_model_num")
    path.write_text("data_synthetic\nloop_\n" + "".join("_atom_site." + c + "\n" for c in columns)
                    + "ATOM 1 C CA ALA X 1 A 101 ? . 1.0 0 0 0 7\n"
                    + "ATOM 2 C CA ALA Y 1 B 202 ? . 1.0 4 0 0 7\n"
                    + "ATOM 3 C CA ALA X 1 A 101 ? . 1.0 100 0 0 1\n"
                    + "ATOM 4 C CA ALA Y 1 B 202 ? . 1.0 200 0 0 1\n#\n")
    selected = ppi.read_selected_atoms(path, ["A"], ["B"])
    assert selected["diagnostics"]["input"]["model"] == "7"
    assert selected["diagnostics"]["input"]["available_models"] == ["7", "1"]
    np.testing.assert_array_equal(selected["partner1_points"], [[0, 0, 0]])
    np.testing.assert_array_equal(selected["partner2_points"], [[4, 0, 0]])
    with pytest.raises(ValueError, match="chains absent"):
        ppi.read_selected_atoms(path, ["X"], ["Y"])


def test_mixed_channels_exclude_internal_edges_but_null_channels_keep_them():
    first = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    second = np.asarray([[4, 0, 0]], dtype=float)
    empty = np.empty((0, 3))
    mixed = ppi.channel_summaries(first, second, cross_only=True)
    single = ppi.channel_summaries(first, empty, cross_only=False)
    # At radius 1 the only alpha edge is the within-first-partner 0--1 edge.
    assert mixed[0, 0] == 3
    assert mixed[1, 0] == 3
    assert single[0, 0] == 2
    assert single[1, 0] == 1
    assert single[2, 0] == pytest.approx(2)
    # At radius 1.5 the length-3 cross edge is born. The first vertex remains
    # isolated: graph spectrum is [0, 0, 2], not the three-node path spectrum.
    assert mixed[1, 1] == 2
    assert mixed[2, 1] == pytest.approx(2)
    assert np.all(mixed[1, 1:] == 2)


def test_schema_and_tensor_have_fourteen_radii_ending_before_8p0():
    assert len(ppi.RADII) == 14
    assert ppi.RADII[0] == 1.0
    assert ppi.RADII[-1] == 7.5
    np.testing.assert_allclose(np.diff(ppi.RADII), 0.5)
    np.testing.assert_allclose(ppi.THRESHOLDS, np.square(ppi.RADII))
    selected = ppi.select_atoms([atom(0, "A", 1, "CA", (0, 0, 0)),
                                 atom(1, "B", 1, "CA", (4, 0, 0))], ["A"], ["B"])
    tensor, present, counts = ppi.compute(selected)
    assert tensor.shape == (10, 14, 36)
    assert tensor.size == 5040
    assert np.isfinite(tensor).all()
    assert counts.shape == (36, 2)
    assert {ppi.CHANNELS[i] for i in np.flatnonzero(present)} == {
        ("CA", "null"), ("null", "CA"), ("CA", "CA")}
    assert not tensor[:, :, ppi.CHANNELS.index(("null", "null"))].any()
    assert not tensor[:, :, ~present].any()
    ca_pair = tensor[:, :, ppi.CHANNELS.index(("CA", "CA"))]
    assert ca_pair[1, 1] == 2  # r=1.5: disconnected
    assert ca_pair[1, 2] == 1  # r=2.0: the length-4 edge is present
    assert ppi.schema()["tensor"]["features"] == tensor.size


def test_extended_grid_computes_alpha_edges_beyond_previous_upper_radius():
    # The only edge is born at squared radius 56.25, beyond the old 34.81 cap.
    selected = ppi.select_atoms([atom(0, "A", 1, "CA", (0, 0, 0)),
                                 atom(1, "B", 1, "CA", (15, 0, 0))], ["A"], ["B"])
    tensor, _, _ = ppi.compute(selected)
    pair = tensor[:, :, ppi.CHANNELS.index(("CA", "CA"))]
    assert pair[1, ppi.RADII.index(7.0)] == 2
    assert np.all(pair[1, ppi.RADII.index(7.5):] == 1)
    np.testing.assert_allclose(pair[2, ppi.RADII.index(7.5):], 2)


def test_approved_missing_chain_keeps_present_chains_in_same_partner():
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "B", 1, "CA", (100, 0, 0)),
             atom(2, "J", 1, "CA", (4, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A", "B"], ["H", "J"],
                                allowed_missing_chains=["H"])
    diagnostic = selected["diagnostics"]
    assert diagnostic["source_partner_chains"] == [["A", "B"], ["H", "J"]]
    assert diagnostic["source_partner_residue_counts"] == [2, 1]
    assert diagnostic["ordered_partner_chains"] == [["A", "B"], ["J"]]
    assert diagnostic["missing_requested_chains"] == [[], ["H"]]
    assert diagnostic["explicit_empty_partner"] is None
    assert diagnostic["interface_crop_mode"] == "cross_partner_CA_distance"
    np.testing.assert_array_equal(selected["partner1_points"], [[0, 0, 0]])
    np.testing.assert_array_equal(selected["partner2_points"], [[4, 0, 0]])


def test_wholly_missing_approved_partner_preserves_available_internal_channels():
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "A", 1, "CB", (1, 0, 0)),
             atom(2, "A", 2, "CA", (100, 0, 0)),
             atom(3, "A", 3, "N", (200, 0, 0))]
    selected = ppi.select_atoms(atoms, ["H"], ["A"], allowed_missing_chains=["H"])
    diagnostic = selected["diagnostics"]
    assert diagnostic["ordered_source_partners"] == ["B", "A"]
    assert diagnostic["missing_requested_chains"] == [["H"], []]
    assert diagnostic["interface_crop_mode"] == "available_partner_CA_residues_no_opponent"
    assert diagnostic["missing_CA_residues_excluded"] == [[("A", "3", "")], []]
    np.testing.assert_array_equal(selected["partner1_points"], [[0, 0, 0], [1, 0, 0], [100, 0, 0]])
    assert selected["partner2_points"].shape == (0, 3)
    tensor, present, counts = ppi.compute(selected)
    assert {ppi.CHANNELS[i] for i in np.flatnonzero(present)} == {("C", "null"), ("CA", "null")}
    assert tensor[:, :, ppi.CHANNELS.index(("CA", "null"))].any()
    assert not tensor[:, :, ~present].any()
    assert not counts[:, 1].any()


@pytest.mark.parametrize("empty_partner,expected_source_order,expected_lengths,expected_points", [
    ("B", ["A", "B"], [1, 0], [[0, 0, 0]]),
    ("A", ["B", "A"], [0, 2], [[100, 0, 0], [200, 0, 0]]),
])
def test_explicit_whole_empty_partner_is_distinct_from_missing_chain(
        empty_partner, expected_source_order, expected_lengths, expected_points):
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "J", 1, "CA", (100, 0, 0)),
             atom(2, "J", 2, "CA", (200, 0, 0))]
    selected = ppi.select_atoms(atoms, ["A"], ["H", "J"],
                                allowed_missing_chains=["H"], empty_partner=empty_partner)
    diagnostic = selected["diagnostics"]
    assert diagnostic["explicit_empty_partner"] == empty_partner
    assert diagnostic["source_partner_chains"] == [["A"], ["H", "J"]]
    assert diagnostic["source_partner_residue_counts"] == expected_lengths
    assert diagnostic["ordered_source_partners"] == expected_source_order
    assert diagnostic["interface_crop_mode"] == "available_partner_CA_residues_no_opponent"
    np.testing.assert_array_equal(selected["partner1_points"], expected_points)
    assert selected["partner2_points"].shape == (0, 3)


def test_reader_forwards_explicit_missing_chain_policy(tmp_path):
    path = tmp_path / "synthetic.pdb"
    path.write_text(pdb_atom(1, "A", 0) + pdb_atom(2, "J", 4) + "END\n")
    selected = ppi.read_selected_atoms(path, ["A"], ["H", "J"],
                                       allowed_missing_chains=["H"])
    assert selected["diagnostics"]["missing_requested_chains"] == [[], ["H"]]
    np.testing.assert_array_equal(selected["partner2_points"], [[4, 0, 0]])


def test_missing_partner_and_dense_resource_failures_are_explicit():
    atoms = [atom(0, "A", 1, "CA", (0, 0, 0)),
             atom(1, "B", 1, "CA", (4, 0, 0))]
    with pytest.raises(ValueError, match="chains absent"):
        ppi.select_atoms(atoms, ["A"], ["C"])
    with pytest.raises(ValueError, match="chains absent"):
        ppi.select_atoms(atoms, ["A"], ["C", "D"], allowed_missing_chains=["D"])
    with pytest.raises(ValueError, match="disjoint"):
        ppi.select_atoms(atoms, ["A"], ["A", "B"])
    with pytest.raises(ppi.ResourceLimitError, match="dense entries"):
        ppi.channel_summaries(np.asarray([[0, 0, 0]]), np.asarray([[4, 0, 0]]),
                              cross_only=True, max_dense_entries=3)


def test_public_schema_records_dense_parent_projection_and_angstrom_units():
    contract = ppi.schema()
    assert contract['recipe_id'] == ppi.RECIPE_ID
    assert contract['coordinate_units'] == 'angstrom'
    assert contract['parent_recipe']['recipe_id'] == ppi.PARENT_RECIPE_ID
    assert contract['parent_recipe']['retained_radius_indices'] == list(range(0, 70, 5))
    assert contract['parent_recipe']['tensor_shape'] == [10, 70, 36]
    assert contract['augmentation'] == 'none'
    assert contract['tensor'] == {'shape': [10, 14, 36], 'dtype': 'float32',
                                  'order': 'C', 'features': 5040}
    assert contract['channels'] == [[a, b] for a in ppi.GROUPS for b in ppi.GROUPS]
    contract['parent_recipe']['retained_radius_indices'].clear()
    assert len(ppi.schema()['parent_recipe']['retained_radius_indices']) == 14


def test_global_exact_deduplication_preserves_order_and_category():
    atoms = [atom(0, 'A', 1, 'CA', (0, 0, 0)),
             atom(1, 'A', 1, 'CB', (1, 0, 0)),
             atom(2, 'B', 1, 'CA', (0, 0, 0)),
             atom(3, 'B', 1, 'CB', (2, 0, 0))]
    selected = ppi.select_atoms(atoms, ['A'], ['B'])
    assert selected['partner1_categories'].tolist() == ['CA', 'C']
    assert selected['partner2_categories'].tolist() == ['C']
    removed = selected['diagnostics']['duplicate_coordinate_rows_removed']
    assert removed == [{'input_index': 2, 'kept_input_index': 0,
                        'removed_partner': 2, 'category': 'CA',
                        'point': [0.0, 0.0, 0.0]}]
    _, present, _ = ppi.compute(selected)
    assert not present[ppi.CHANNELS.index(('CA', 'CA'))]
    assert present[ppi.CHANNELS.index(('CA', 'C'))]


def test_twenty_angstrom_interface_boundary_is_rejected():
    atoms = [atom(0, 'A', 1, 'CA', (0, 0, 0)),
             atom(1, 'B', 1, 'CA', (20, 0, 0))]
    with pytest.raises(ValueError, match='no interacting residues'):
        ppi.select_atoms(atoms, ['A'], ['B'])


def test_eight_angstrom_radius_is_excluded():
    # Both residues are in the interface, but their length-16 edge is born at
    # radius 8, beyond every retained alpha radius in the public recipe.
    selected = ppi.select_atoms([atom(0, 'A', 1, 'CA', (0, 0, 0)),
                                 atom(1, 'B', 1, 'CA', (16, 0, 0))], ['A'], ['B'])
    tensor, _, _ = ppi.compute(selected)
    channel = tensor[:, :, ppi.CHANNELS.index(('CA', 'CA'))]
    np.testing.assert_array_equal(channel[0], np.full(14, 2.0))
    np.testing.assert_array_equal(channel[1], np.full(14, 2.0))
    assert not channel[2:].any()


@pytest.mark.parametrize('seed', [17, 123])
def test_direct_sparse_radii_match_dense_grid_projection(seed, monkeypatch):
    # Sampling fewer thresholds must not change native-alpha births or the
    # cached L0 sweep result. Exercise all 35 nonempty/null-side channels with
    # noncoplanar, disjoint category clouds and both partner orientations.
    rng = np.random.default_rng(seed)
    atoms = []
    for chain, offset in [('B', 0.0), ('A', 3.0)]:
        for residue in range(1, 4):
            for name in ['CA', 'CB', 'N', 'O', 'SG']:
                point = rng.normal(size=3) * 2 + np.array([offset, residue, 0.0])
                atoms.append(atom(len(atoms), chain, residue, name, point))
    selected = ppi.select_atoms(atoms, ['A'], ['B'])
    sparse, sparse_present, sparse_counts = ppi.compute(selected)
    dense_radii = tuple(i / 10 for i in range(10, 80))
    with monkeypatch.context() as patch:
        patch.setattr(ppi, 'RADII', dense_radii)
        patch.setattr(ppi, 'THRESHOLDS', np.square(dense_radii))
        patch.setattr(ppi, 'TENSOR_SHAPE', (10, 70, 36))
        dense, dense_present, dense_counts = ppi.compute(selected)
    np.testing.assert_array_equal(sparse.astype('<f4'), dense[:, ::5, :].astype('<f4'))
    np.testing.assert_array_equal(sparse_present, dense_present)
    np.testing.assert_array_equal(sparse_counts, dense_counts)
    assert sparse_present.sum() == 35


def test_featurize_returns_canonical_float32_c_order_tensor(tmp_path):
    path = tmp_path / 'synthetic.pdb'
    path.write_text(pdb_atom(1, 'A', 0) + pdb_atom(2, 'B', 4) + 'END\n')
    tensor = ppi.featurize(path, ['A'], ['B'])
    assert tensor.shape == (10, 14, 36)
    assert tensor.dtype == np.dtype('<f4')
    assert tensor.flags.c_contiguous
    selected = ppi.read_selected_atoms(path, ['A'], ['B'])
    np.testing.assert_array_equal(tensor, ppi.compute(selected)[0].astype('<f4'))
    flat = tensor.reshape(-1)
    for stat, radius, channel in [(0, 0, 35), (1, 2, 35), (2, 13, 35)]:
        assert flat[(stat * 14 + radius) * 36 + channel] == tensor[stat, radius, channel]


def test_feature_import_keeps_optional_model_dependencies_lazy():
    import os
    from pathlib import Path
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path(ppi.__file__).resolve().parents[3]))
    code = ('import sys; from topokit.workflows.protein_protein_prediction import featurize; '
            'assert not ({"torch", "transformers", "sklearn", "joblib", "pandas", "Bio"} '
            '& set(sys.modules))')
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
