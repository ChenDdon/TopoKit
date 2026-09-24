"""Focused checks for dependency-free molecular coordinate readers."""

from pathlib import Path

import numpy as np
import pytest

from topokit import readers


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.parametrize(("filename", "count", "first_id", "first_element"), [
    ("data_6PT3_ligand.pdb", 36, 1, "C"),
    ("data_6PT3_ligand.mol2", 36, 1, "C"),
    ("data_molecule.sdf", 36, 1, "C"),
    ("data_molecule.mol", 24, 1, "O"),
    ("data_protein.pdbqt", 3646, 1, "N"),
    ("data_material.cif", 29, "Ni1", "Ni"),
])
def test_registered_molecular_readers_return_xyz_like_point_clouds(
        filename, count, first_id, first_element):
    cloud = readers.read(DATA / filename)
    assert len(cloud) == count
    assert cloud.points.shape == (count, 3)
    assert cloud.ids[0] == first_id
    assert cloud.metadata["labels"][0] == first_element
    assert cloud.metadata["columns"]["element"][0] == first_element
    assert cloud.metadata["source_file"] == filename
    assert cloud.metadata["format"] == Path(filename).suffix[1:]
    assert cloud.metadata["coordinate_units"] == "angstrom"
    np.testing.assert_array_equal(cloud.weights, np.ones(count))
    assert np.isfinite(cloud.points).all()


def test_ligand_pdb_and_mol2_preserve_ids_and_matching_coordinates():
    pdb = readers.read_pdb(DATA / "data_6PT3_ligand.pdb")
    mol2 = readers.read_mol2(DATA / "data_6PT3_ligand.mol2")
    assert pdb.ids == mol2.ids == tuple(range(1, 37))
    np.testing.assert_allclose(pdb.points, mol2.points)
    assert pdb.metadata["columns"]["atom_name"][:3] == ("CBI", "CBG", "NBF")
    assert pdb.metadata["columns"]["residue_name"][0] == "OWY"
    assert mol2.metadata["molecule"]["charge_type"] == "GASTEIGER"
    assert mol2.metadata["columns"]["atom_type"][:3] == ("C.3", "C.3", "N.am")
    np.testing.assert_allclose(
        mol2.metadata["columns"]["partial_charge"][:3], (0.0233, 0.0938, -0.2761)
    )
    assert len(mol2.metadata["bonds"]) == 39
    assert mol2.metadata["bonds"][0]["atoms"] == (1, 2)


def test_pdb_preserves_explicit_neutral_formal_charge_as_zero(tmp_path):
    neutral = "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O0 "
    path = tmp_path / "explicit-neutral-charge.pdb"
    path.write_text(f"{neutral}\nEND\n")

    cloud = readers.read_pdb(path)

    assert cloud.metadata["columns"]["formal_charge"] == (0,)
    assert cloud.metadata["columns"]["source_record"] == (neutral,)


def test_mdl_v2000_and_v3000_metadata_and_sdf_fields_are_retained():
    mol = readers.read_mol(DATA / "data_molecule.mol")
    sdf = readers.read_sdf(DATA / "data_molecule.sdf")
    assert mol.metadata["ctfile_version"] == "V2000"
    assert mol.metadata["header"]["name"] == "5793"
    assert len(mol.metadata["bonds"]) == 24
    assert mol.metadata["columns"]["formal_charge"] == (0,) * 24
    assert sdf.metadata["ctfile_version"] == "V3000"
    assert sdf.metadata["header"]["name"] == "train__NCGC00013037-01"
    assert len(sdf.metadata["bonds"]) == 37
    assert sdf.metadata["properties"]["s_lp_Force_Field"] == "S-OPLS"
    assert sdf.metadata["properties"]["r_lp_Energy"] == "41.525813999999997"
    assert all(line.startswith("M  ") for line in sdf.metadata["mdl_property_records"])
    assert "41.525813999999997" not in sdf.metadata["mdl_property_records"]


def test_pdbqt_retains_partial_charge_and_docking_atom_type():
    cloud = readers.read_pdbqt(DATA / "data_protein.pdbqt")
    columns = cloud.metadata["columns"]
    assert columns["atom_type"][:4] == ("NA", "C", "C", "OA")
    assert columns["element"][:4] == ("N", "C", "C", "O")
    np.testing.assert_array_equal(columns["partial_charge"][:4], np.zeros(4))
    assert cloud.metadata["charge_type"] == "partial"
    assert cloud.metadata["non_atom_records"][0].startswith("REMARK")


def test_cif_fractional_sites_are_cartesianized_without_symmetry_expansion():
    cloud = readers.read_cif(DATA / "data_material.cif")
    assert cloud.metadata["data_block"] == "4066832"
    assert cloud.metadata["coordinate_source"] == "fractional_converted_to_cartesian"
    assert cloud.metadata["symmetry_expansion"] == "not_applied"
    assert cloud.metadata["unit_cell"]["length_a"] == pytest.approx(15.253)
    assert cloud.metadata["unit_cell"]["length_b"] == pytest.approx(11.484)
    assert cloud.metadata["unit_cell"]["length_c"] == pytest.approx(11.8579)
    np.testing.assert_allclose(
        cloud.points[0], (0.75 * 15.253, 0.70660 * 11.484, 0.01922 * 11.8579),
        atol=1e-12,
    )
    assert cloud.metadata["columns"]["_atom_site_fract_y"][0] == "0.70660(8)"


def test_molecular_columns_and_labels_stay_aligned_on_subset():
    cloud = readers.read_mol2(DATA / "data_6PT3_ligand.mol2")
    selected = cloud.subset([3, 1])
    assert selected.ids == (3, 1)
    assert selected.metadata["atom_count"] == selected.metadata["selected_atom_count"] == 2
    assert selected.metadata["source_atom_count"] == 36
    assert selected.metadata["selection_source_ids"] == tuple(range(1, 37))
    assert selected.metadata["labels"] == ("N", "C")
    assert selected.metadata["columns"]["atom_name"] == ("NBF", "CBI")
    assert selected.metadata["columns"]["partial_charge"] == (-0.2761, 0.0233)
    assert selected.metadata["declared_counts"] == {"atoms": 36, "bonds": 39}
    assert len(selected.metadata["source_bonds"]) == 39
    assert selected.metadata["bonds"] == ()
    assert selected.metadata["selected_bond_count"] == 0
    assert selected.metadata["bond_selection_policy"] == "induced_by_selected_atom_ids"
    repeated = selected.subset([3])
    assert repeated.metadata["source_atom_count"] == 36
    assert repeated.metadata["selected_atom_count"] == 1
    assert len(repeated.metadata["source_bonds"]) == 39


def test_pdb_multiple_models_require_explicit_selection(tmp_path):
    atom_one = "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C"
    atom_two = "ATOM      1  O   UNK A   1       1.000   2.000   3.000  1.00  0.00           O"
    path = tmp_path / "models.pdb"
    path.write_text(
        f"MODEL        1\n{atom_one}\nENDMDL\nMODEL        2\n{atom_two}\nENDMDL\n"
    )
    with pytest.raises(ValueError, match="explicit model"):
        readers.read_pdb(path)
    second = readers.read_pdb(path, model=2)
    assert second.metadata["model"] == 2
    assert second.metadata["available_models"] == (1, 2)
    assert second.metadata["labels"] == ("O",)
    np.testing.assert_array_equal(second.points, [[1, 2, 3]])


def test_pdb_selected_model_exposes_only_its_induced_connectivity(tmp_path):
    model_one = (
        "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2  O   UNK A   1       1.000   0.000   0.000  1.00  0.00           O",
    )
    model_two = (
        "ATOM     10  N   UNK A   1       2.000   0.000   0.000  1.00  0.00           N",
        "ATOM     11  H   UNK A   1       3.000   0.000   0.000  1.00  0.00           H",
    )
    path = tmp_path / "model-connectivity.pdb"
    path.write_text(
        "MODEL        1\n" + "\n".join(model_one) + "\nENDMDL\n"
        "MODEL        2\n" + "\n".join(model_two) + "\nENDMDL\n"
        "CONECT    1    2\nCONECT   10   11\nEND\n"
    )

    cloud = readers.read_pdb(path, model=1)

    assert cloud.ids == (1, 2)
    assert cloud.metadata["connectivity_records"] == ((1, 2),)
    assert cloud.metadata["source_connectivity_records"] == ((1, 2), (10, 11))
    assert cloud.metadata["selected_connectivity_record_count"] == 1
    assert cloud.metadata["source_connectivity_record_count"] == 2
    assert cloud.metadata["connectivity_selection_policy"] == "induced_by_selected_atom_ids"
    assert cloud.subset(cloud.ids).metadata["connectivity_records"] == ((1, 2),)


def test_cartesian_cif_and_uncertainty_numbers(tmp_path):
    path = tmp_path / "cartesian.cif"
    path.write_text(
        "data_example\n"
        "_cell_length_a 10\n"
        "_cell_length_b 10\n"
        "_cell_length_c 10\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n"
        "_atom_site_Cartn_y\n"
        "_atom_site_Cartn_z\n"
        "C1 C 1.25(2) -2 3.5\n"
        "O2 O 0 0 0\n"
    )
    cloud = readers.read_cif(path, coordinate_units="nm")
    assert cloud.ids == ("C1", "O2")
    assert cloud.metadata["coordinate_source"] == "cartesian"
    assert cloud.metadata["coordinate_units"] == "nm"
    assert cloud.metadata["unit_cell"]["length_a"] == 10
    np.testing.assert_array_equal(cloud.points, [[1.25, -2, 3.5], [0, 0, 0]])


def test_mmcif_dotted_names_are_normalized_for_atom_and_cell_fields(tmp_path):
    path = tmp_path / "fractional.mmcif"
    path.write_text(
        "data_example\n"
        "_cell.length_a 10\n"
        "_cell.length_b 20\n"
        "_cell.length_c 30\n"
        "_cell.angle_alpha 90\n"
        "_cell.angle_beta 90\n"
        "_cell.angle_gamma 90\n"
        "loop_\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.fract_x\n"
        "_atom_site.fract_y\n"
        "_atom_site.fract_z\n"
        "site-1 C 0.1 0.2 0.3\n"
    )
    cloud = readers.read(path)
    assert cloud.ids == ("site-1",)
    assert cloud.metadata["labels"] == ("C",)
    np.testing.assert_allclose(cloud.points, [[1, 4, 9]], atol=1e-12)
    assert cloud.metadata["columns"]["_atom_site.fract_x"] == ("0.1",)
    assert cloud.metadata["cif_normalized_tags"]["_cell_length_a"] == "10"


@pytest.mark.parametrize("text, message", [
    (
        "molecule\nprogram\ncomment\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n",
        "M  END",
    ),
    (
        "molecule\nprogram\ncomment\n"
        "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        "M  V30 BEGIN CTAB\n"
        "M  V30 COUNTS 1 0 0 0 0\n"
        "M  V30 BEGIN ATOM\n"
        "M  V30 1 C 0 0 0 0\n"
        "M  V30 END ATOM\n"
        "M  END\n",
        "CTAB",
    ),
])
def test_mdl_requires_explicit_record_and_v3000_section_closure(tmp_path, text, message):
    path = tmp_path / "truncated.mol"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        readers.read_mol(path)


def test_v2000_rejects_negative_bond_count_and_accepts_blank_charge(tmp_path):
    path = tmp_path / "minimal.mol"
    atom = f"{0:10.4f}{0:10.4f}{0:10.4f} {'C':<3}{'':2}{'':3}"
    path.write_text(
        "molecule\nprogram\ncomment\n"
        f"  1  0  0  0  0  0  0  0  0  0999 V2000\n{atom}\nM  END\n"
    )
    cloud = readers.read_mol(path)
    assert cloud.metadata["columns"]["formal_charge"] == (0,)
    path.write_text(
        "molecule\nprogram\ncomment\n"
        f"  1 -1  0  0  0  0  0  0  0  0999 V2000\n{atom}\nM  END\n"
    )
    with pytest.raises(ValueError, match="nonnegative bond"):
        readers.read_mol(path)


def test_v3000_formal_charge_must_be_integral(tmp_path):
    path = tmp_path / "fractional-charge.mol"
    path.write_text(
        "molecule\nprogram\ncomment\n"
        "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        "M  V30 BEGIN CTAB\n"
        "M  V30 COUNTS 1 0 0 0 0\n"
        "M  V30 BEGIN ATOM\n"
        "M  V30 1 C 0 0 0 0 CHG=1.5\n"
        "M  V30 END ATOM\n"
        "M  V30 END CTAB\n"
        "M  END\n"
    )
    with pytest.raises(ValueError, match="formal charge must be an integer"):
        readers.read_mol(path)


@pytest.mark.parametrize("angles", [(-90, 90, 90), (120, 120, 120)])
def test_cif_rejects_invalid_or_degenerate_unit_cells(tmp_path, angles):
    alpha, beta, gamma = angles
    path = tmp_path / "bad-cell.cif"
    path.write_text(
        "data_bad\n"
        "_cell_length_a 1\n_cell_length_b 1\n_cell_length_c 1\n"
        f"_cell_angle_alpha {alpha}\n_cell_angle_beta {beta}\n_cell_angle_gamma {gamma}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C1 C 0 0 0\n"
    )
    with pytest.raises(ValueError, match="angles|volume"):
        readers.read_cif(path)


@pytest.mark.parametrize("body, message", [
    (
        "_cell_length_a\n_cell_length_b 2\n",
        "missing its value",
    ),
    (
        "_cell_length_a 1\n_CELL_LENGTH_A 2\n",
        "unique",
    ),
])
def test_cif_rejects_missing_scalar_values_and_duplicate_data_names(tmp_path, body, message):
    path = tmp_path / "bad-tags.cif"
    path.write_text(
        "data_bad\n" + body
        + "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "C1 C 0 0 0\n"
    )
    with pytest.raises(ValueError, match=message):
        readers.read_cif(path)


def test_cif_rejects_duplicate_case_normalized_loop_headers(tmp_path):
    path = tmp_path / "duplicate-loop.cif"
    path.write_text(
        "data_bad\nloop_\n"
        "_atom_site_label\n_ATOM_SITE_LABEL\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "C1 C1 0 0 0\n"
    )
    with pytest.raises(ValueError, match="duplicate case-normalized"):
        readers.read_cif(path)


def test_cif_without_type_symbol_preserves_site_label_without_element_guessing(tmp_path):
    path = tmp_path / "unknown-element.cif"
    path.write_text(
        "data_unknown\n"
        "loop_\n_atom_site_label\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "CA 0 0 0\n"
    )
    cloud = readers.read_cif(path)
    assert cloud.metadata["labels"] == ("CA",)
    assert cloud.metadata["columns"]["element"] == (None,)


def test_cif_quoted_control_like_values_retain_value_semantics(tmp_path):
    path = tmp_path / "quoted-values.cif"
    path.write_text(
        "data_quoted\n"
        "_description 'data_not_a_block'\n"
        "_annotation \"_not_a_data_name\"\n"
        "_author 'O'Brien'\n"
        "_multiline\n;\ndata_semicolon_value\n;\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "'_C1' C 0 0 0\n"
        "'data_site_2' O 1 2 3\n"
    )
    cloud = readers.read_cif(path)
    assert cloud.ids == ("_C1", "data_site_2")
    assert cloud.metadata["cif_tags"]["_description"] == "data_not_a_block"
    assert cloud.metadata["cif_tags"]["_annotation"] == "_not_a_data_name"
    assert cloud.metadata["cif_tags"]["_author"] == "O'Brien"
    assert cloud.metadata["cif_tags"]["_multiline"] == "data_semicolon_value"
    np.testing.assert_array_equal(cloud.points, [[0, 0, 0], [1, 2, 3]])


def test_cif_quoted_numeric_atom_ids_remain_strings(tmp_path):
    path = tmp_path / "quoted-numeric-ids.cif"
    path.write_text(
        "data_quoted_ids\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "'12' C 0 0 0\n"
        "12 O 1 2 3\n"
    )

    cloud = readers.read_cif(path)

    assert cloud.ids == ("12", 12)
    assert getattr(cloud.metadata["columns"]["_atom_site_label"][0], "quoted")
    assert not getattr(cloud.metadata["columns"]["_atom_site_label"][1], "quoted")


@pytest.mark.parametrize("text", [
    (
        "data_quoted_coordinate\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "C1 C '0' 0 0\n"
    ),
    (
        "data_quoted_cell\n"
        "_cell_length_a '10'\n_cell_length_b 10\n_cell_length_c 10\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C1 C 0 0 0\n"
    ),
    (
        "data_quoted_charge\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        "_atom_site_charge\n"
        "C1 C 0 0 0 '1'\n"
    ),
])
def test_cif_quoted_numeric_fields_are_not_coerced(tmp_path, text):
    path = tmp_path / "quoted-number.cif"
    path.write_text(text)

    with pytest.raises(ValueError, match="unquoted CIF number"):
        readers.read_cif(path)


def test_reader_subset_preserves_source_declarations_and_induces_mdl_bonds():
    cloud = readers.read_mol(DATA / "data_molecule.mol")
    selected = cloud.subset([1, 9, 11])
    assert selected.metadata["source_atom_count"] == 24
    assert selected.metadata["atom_count"] == selected.metadata["selected_atom_count"] == 3
    assert selected.metadata["declared_counts"] == {"atoms": 24, "bonds": 24}
    assert selected.metadata["source_declared_counts"] == {"atoms": 24, "bonds": 24}
    assert len(selected.metadata["source_bonds"]) == 24
    assert {bond["atoms"] for bond in selected.metadata["bonds"]} == {(1, 9), (1, 11)}
    assert selected.metadata["selected_bond_count"] == 2


def test_pdb_subset_induces_connectivity_and_preserves_source_records(tmp_path):
    atoms = [
        "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2  N   UNK A   1       1.000   0.000   0.000  1.00  0.00           N",
        "ATOM      3  O   UNK A   1       2.000   0.000   0.000  1.00  0.00           O",
    ]
    path = tmp_path / "connected.pdb"
    path.write_text("\n".join(atoms + ["CONECT    1    2    3", "CONECT    2    1", "END"]) + "\n")
    cloud = readers.read_pdb(path)
    assert cloud.metadata["source_connectivity_records"] == ((1, 2, 3), (2, 1))
    selected = cloud.subset([1, 3])
    assert selected.metadata["source_atom_count"] == 3
    assert selected.metadata["selected_atom_count"] == 2
    assert selected.metadata["connectivity_records"] == ((1, 3),)
    assert selected.metadata["source_connectivity_records"] == ((1, 2, 3), (2, 1))
    assert selected.metadata["source_connectivity_record_count"] == 2
    assert selected.metadata["selected_connectivity_record_count"] == 1
    assert selected.metadata["connectivity_selection_policy"] == "induced_by_selected_atom_ids"


@pytest.mark.parametrize(("filename", "source_count"), [
    ("data_protein.pdbqt", 3646),
    ("data_material.cif", 29),
])
def test_bondless_reader_subsets_report_source_and_selected_counts(filename, source_count):
    cloud = readers.read(DATA / filename)
    selected = cloud.subset(cloud.ids[:4])
    assert selected.metadata["source_atom_count"] == source_count
    assert selected.metadata["selected_atom_count"] == selected.metadata["atom_count"] == 4


def test_multi_record_sdf_and_empty_molecular_files_fail_explicitly(tmp_path):
    sdf = tmp_path / "many.sdf"
    sdf.write_text("first\n\n\n$$$$\nsecond\n\n\n$$$$\n")
    with pytest.raises(ValueError, match="exactly one"):
        readers.read_sdf(sdf)
    for suffix, function in {
        ".pdb": readers.read_pdb,
        ".mol2": readers.read_mol2,
        ".mol": readers.read_mol,
        ".pdbqt": readers.read_pdbqt,
        ".cif": readers.read_cif,
    }.items():
        path = tmp_path / f"empty{suffix}"
        path.write_text("")
        with pytest.raises(ValueError):
            function(path)


def test_default_reader_registry_advertises_molecular_formats():
    assert readers.available_readers() == (
        "cif", "csv", "json", "mol", "mol2", "pdb", "pdbqt", "sdf", "xyz"
    )
