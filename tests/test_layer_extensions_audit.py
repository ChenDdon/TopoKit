"""Independent adversarial checks for reader and numerical extension contracts."""
import math

import numpy as np
import pytest

from topokit.data import PointCloud
from topokit.readers import ReaderRegistry, read_csv, read_xyz
from topokit.postprocessing import histogram_features, spectral_entropy, summarize_spectrum
from topokit.results import PersistenceInterval, PersistenceResult, SpectrumResult


@pytest.mark.parametrize("text", [
    "id,x,x,z,weight\na,0,1,0,1\n",
    "id,x,y,z,weight\na,0,0,0,1,extra\n",
    "id,x,y,z,weight\na,0,0,0\n",
    "id,x,y,z,weight,\na,0,0,0,1,note\n",
    "id,x,y,weight\na,0,0,1\n",
])
def test_csv_rejects_structural_information_loss(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_csv(path)


def test_csv_preserves_raw_attributes_and_explicit_units(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text('\ufeffatom,px,py,w,annotation\nC_2,1.0,2.00,2.5,"a,b"\nN_1,3,4,-1,"second"\n')
    cloud = read_csv(path, coordinate_columns=("px", "py"), id_column="atom", weight_column="w",
                     coordinate_units=" angstrom ")
    assert cloud.ids == ("C_2", "N_1")
    np.testing.assert_array_equal(cloud.points, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(cloud.weights, [2.5, -1])
    assert cloud.metadata["columns"]["py"] == ("2.00", "4")
    assert cloud.metadata["columns"]["annotation"] == ("a,b", "second")
    assert cloud.metadata["coordinate_columns"] == ("px", "py")
    assert cloud.metadata["coordinate_units"] == "angstrom"
    selected = cloud.subset(["N_1", "C_2"])
    assert selected.metadata["columns"]["annotation"] == ("second", "a,b")
    assert cloud.metadata["columns"]["annotation"] == ("a,b", "second")


def test_csv_explicitly_optional_ids_and_weights(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("x,y,z\n0,1,2\n3,4,5\n")
    cloud = read_csv(path, id_column=None, weight_column=None)
    assert cloud.ids == (0, 1)
    np.testing.assert_array_equal(cloud.weights, [1, 1])


@pytest.mark.parametrize("options", [
    {"coordinate_columns": "xyz"}, {"coordinate_columns": ()},
    {"coordinate_columns": ("x", "x")}, {"coordinate_units": ""},
])
def test_csv_invalid_options_fail_before_interpreting_rows(tmp_path, options):
    path = tmp_path / "points.csv"
    path.write_text("id,x,y,z,weight\na,0,0,0,1\n")
    with pytest.raises(ValueError):
        read_csv(path, **options)


def test_xyz_preserves_tokens_and_comment_without_inferring_chemistry(tmp_path):
    path = tmp_path / "sample.xyz"
    path.write_text("2\nenergy=-1.0 any other metadata\nB 0 1 2 custom 7\nN 3 4 5 tag\n")
    cloud = read_xyz(path, coordinate_units="nm")
    assert cloud.ids == (0, 1)
    assert cloud.metadata["labels"] == ("B", "N")
    assert cloud.metadata["extra_fields"] == (("custom", "7"), ("tag",))
    assert cloud.metadata["comment"] == "energy=-1.0 any other metadata"
    assert cloud.metadata["coordinate_units"] == "nm"
    np.testing.assert_array_equal(cloud.weights, [1, 1])
    assert cloud.subset([1]).metadata["labels"] == ("N",)
    assert cloud.subset([1]).metadata["extra_fields"] == (("tag",),)


@pytest.mark.parametrize("text", [
    "1\nframe1\nC 0 0 0\n1\nframe2\nC 1 1 1\n",
    "0\nempty\n", "1\nnonfinite\nC nan 0 0\n", "2\nmissing\nC 0 0 0\n",
])
def test_xyz_rejects_ambiguous_or_invalid_records(tmp_path, text):
    path = tmp_path / "bad.xyz"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_xyz(path)


def test_reader_registry_normalizes_names_and_does_not_rebuild_cloud():
    registry = ReaderRegistry()
    cloud = PointCloud([[0, 0]], metadata={"custom": {"setting": 3}})
    calls = []

    def custom(path, **options):
        calls.append((path, options))
        return cloud

    registry.register(" Custom ", custom, extensions=".XYZ")
    assert registry.read("unused.XYZ", mode="scientific") is cloud
    assert calls == [("unused.XYZ", {"mode": "scientific"})]
    assert registry.read("unused.other", format=" CUSTOM ") is cloud
    with pytest.raises(ValueError):
        registry.read("unused.x")  # string extensions never register individual letters


def test_registry_collision_is_atomic_and_compound_suffix_is_specific():
    registry = ReaderRegistry()
    first = PointCloud([[0]])
    second = PointCloud([[1]])
    registry.register("plain", lambda path: first, extensions=("gz", "xyz"))
    with pytest.raises(ValueError, match="registered"):
        registry.register("failed", lambda path: second, extensions=("new", "xyz"))
    assert registry.available() == ("plain",)
    with pytest.raises(ValueError):
        registry.read("unused.new")
    registry.register("compound", lambda path: second, extensions=("xyz.gz",))
    assert registry.read("unused.XYZ.GZ") is second
    assert registry.read("unused.gz") is first
    registry.register("plain", lambda path: second, extensions=("new",), replace=True)
    with pytest.raises(ValueError):
        registry.read("unused.xyz")
    assert registry.read("unused.new") is second


@pytest.mark.parametrize("values", [[], [0, 0], [1, 1], [1e308, 1e308], [1e-300, 1e-300]])
def test_entropy_handles_empty_zero_and_extreme_scales(values):
    expected = 0 if not values or not any(values) else math.log(2)
    assert spectral_entropy(values) == pytest.approx(expected)


def test_entropy_subnormal_normalization_and_tolerance_do_not_mutate_input():
    assert spectral_entropy([1, 1, 1, np.nextafter(0., 1.)]) == pytest.approx(math.log(3))
    values = np.array([-1e-12, 0, 1, 1])
    original = values.copy()
    assert spectral_entropy(values, base=2, tol=1e-10) == pytest.approx(1)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("values, options", [
    ([1, -1e-3], {}), ([math.inf], {}), ([math.nan], {}),
    (np.array([1+2j]), {}), ([[1, 2]], {}), ([1, 2], {"tol": True}),
    ([1, 2], {"base": 1}), ([1, 2], {"base": math.inf}),
])
def test_entropy_rejects_uninterpretable_inputs(values, options):
    with pytest.raises(ValueError):
        spectral_entropy(values, **options)


def test_summary_callbacks_are_isolated_from_each_other_and_source():
    values = np.array([0., 1., 2.])
    result = SpectrumResult(0, values)

    def mutate(copy):
        copy[:] = 9
        return copy.sum()

    vector = summarize_spectrum(result, statistics={"mutating": mutate, "sum": np.sum}, positive_only=False)
    np.testing.assert_array_equal(vector.values, [27, 3])
    np.testing.assert_array_equal(values, [0, 1, 2])
    assert vector.metadata["entropy_policy"] is None
    for callback in (lambda values: np.array([1]), lambda values: np.complex128(1+2j)):
        with pytest.raises(ValueError, match="scalar"):
            summarize_spectrum(result, statistics={"invalid": callback})


def test_spectral_summary_empty_and_partial_policies_remain_explicit():
    result = SpectrumResult(2, np.array([]), complete=False)
    with pytest.raises(ValueError, match="partial"):
        summarize_spectrum(result)
    with pytest.raises(ValueError, match="boolean"):
        summarize_spectrum(result, allow_partial="yes")
    vector = summarize_spectrum(result, allow_partial=True,
                               statistics=("min", "max", "mean", "std", "median", "sum", "count", "spectral_entropy"))
    assert np.isnan(vector.values[:5]).all()
    np.testing.assert_array_equal(vector.values[5:], [0, 0, 0])
    assert vector.metadata["partial_scope"] == "supplied_eigenvalues_only"
    assert vector.metadata["supplied_eigenvalues"] == 0
    assert not vector.metadata["complete_spectrum"]
    assert vector.metadata["entropy_policy"]["zero_operator"] == "zero_entropy"


def test_summary_avoids_intermediate_overflow_without_hiding_true_trace_overflow():
    vector = summarize_spectrum(SpectrumResult(0, np.array([1e308, 1e308])),
                               statistics=("mean", "std", "median", "sum"))
    np.testing.assert_array_equal(vector.values[:3], [1e308, 0, 1e308])
    assert np.isposinf(vector.values[3])
    assert vector.metadata["nonfinite_statistics"] == ("sum",)
    vector = summarize_spectrum(SpectrumResult(0, np.array([0., 1e308])), statistics=("std",), positive_only=False)
    assert vector.values[0] == pytest.approx(5e307)


def test_postprocessing_preserves_scientific_schema_with_independent_provenance():
    metadata = {"route": "interaction", "definition_id": "overlap-v1", "scale_units": "length",
                "coordinate_units": "angstrom", "filtration_start": 0., "filtration_end": 2.,
                "settings": {"overlap": [1, 2]}}
    bars = PersistenceResult((PersistenceInterval(0, 0., 1.),), field="GF(3)", max_dimension=0,
                             metadata=metadata)
    edges = np.array([0., 1., 2.])
    features = histogram_features(bars, birth_edges=edges, death_edges=edges)
    spectrum = SpectrumResult(2, np.array([0., 2.]), start=0., end=1., kind="persistent", metadata=metadata)
    summary = summarize_spectrum(spectrum)
    for derived in (features, summary):
        for key in ("route", "definition_id", "scale_units", "coordinate_units", "filtration_start", "filtration_end"):
            assert derived.metadata[key] == metadata[key]
        assert derived.metadata["source_metadata"] == metadata
    assert features.metadata["coefficient_field"] == features.metadata["field"] == "GF(3)"
    assert summary.metadata["coefficient_field"] == summary.metadata["field"] == "R"
    assert summary.metadata["operator_kind"] == "persistent"
    assert summary.metadata["partial_scope"] == "full_operator"
    metadata["settings"]["overlap"].append(3)
    edges[1] = .25
    for derived in (features, summary):
        assert derived.metadata["source_metadata"]["settings"]["overlap"] == [1, 2]
    np.testing.assert_array_equal(features.metadata["birth_edges"], [0, 1, 2])
    assert not features.metadata["birth_edges"].flags.writeable
