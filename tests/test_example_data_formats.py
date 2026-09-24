"""Integration checks for the public AQUCOG and native-JSON examples."""

from collections import Counter
from pathlib import Path

import numpy as np

from topokit import readers


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_aqucog_core_mof_fixture_dispatches_as_cartesian_cif():
    cloud = readers.read(DATA / "AQUCOG_clean.cif")

    assert cloud.points.shape == (162, 3)
    assert cloud.ids[:4] == ("Ni1", "O1", "O2", "O3")
    assert len(set(cloud.ids)) == len(cloud)
    assert Counter(cloud.metadata["labels"]) == {
        "Ni": 18, "O": 54, "C": 72, "H": 18,
    }
    assert cloud.metadata["coordinate_units"] == "angstrom"
    assert cloud.metadata["coordinate_source"] == "fractional_converted_to_cartesian"
    assert cloud.metadata["symmetry_expansion"] == "not_applied"
    np.testing.assert_allclose(
        [cloud.metadata["unit_cell"][name] for name in (
            "length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma",
        )],
        [25.7542, 25.7542, 6.755, 90.0, 90.0, 120.0],
    )


def test_native_json_fixture_matches_source_cif_and_explicit_weights():
    source = readers.read(DATA / "data_material.cif")
    expected = readers.assign_element_weights(source)
    cloud = readers.read(DATA / "data_material_from_cif.json")

    np.testing.assert_allclose(cloud.points, expected.points, atol=1e-10, rtol=0)
    assert cloud.ids == expected.ids
    assert cloud.metadata["labels"] == expected.metadata["labels"]
    np.testing.assert_array_equal(cloud.weights, expected.weights)
    assert cloud.metadata["coordinate_units"] == "angstrom"
    assert cloud.metadata["json_coordinate_key"] == "coordinates"
    assert cloud.metadata["source_structure"]["file"] == "data_material.cif"
    assert cloud.metadata["weight_semantics"] == "Pauling electronegativity (dimensionless)"
