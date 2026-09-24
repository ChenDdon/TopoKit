"""Independent cross-cutting regression checks, including adversarial inputs."""

import json
import math
import warnings

import numpy as np
import pytest
from scipy import sparse

from topokit import PointCloud
from topokit import api
from topokit.core import simplicial
from topokit.builders import simplicial as simplicial_builder
from topokit.core._spectral import solve_symmetric
from topokit.postprocessing import PersistenceVectorizer, histogram_features
from topokit.results import PersistenceInterval, PersistenceResult, SpectrumResult
from topokit.serialization import load_result, save_result
from topokit.workflows import analyze


def bars(intervals=(), *, maximum=1, units="squared_radius"):
    return PersistenceResult(tuple(PersistenceInterval(*value) for value in intervals),
                             max_dimension=maximum,
                             metadata={"scale_units": units, "filtration_start": 0.,
                                       "filtration_end": math.inf})


def test_strict_json_roundtrip_preserves_nonfinite_values_and_tuple_keys(tmp_path):
    value = {("factor", 7): np.array([[math.inf, -math.inf, math.nan]]),
             7: "integer", "7": "string", "__float__": "ordinary user metadata",
             "empty": np.empty((2, 0)), "scalar": np.array(1.5),
             "ids": (("left", 1), ("right", 2))}
    path = save_result(value, tmp_path / "roundtrip.json")
    # Reject JavaScript-style NaN/Infinity constants; the format must be strict JSON.
    json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    loaded = load_result(path)
    assert loaded[7] == "integer" and loaded["7"] == "string"
    assert loaded["__float__"] == "ordinary user metadata"
    assert loaded["ids"] == (("left", 1), ("right", 2))
    assert loaded["empty"].shape == (2, 0) and loaded["scalar"].shape == ()
    assert np.isposinf(loaded[("factor", 7)][0, 0])
    assert np.isneginf(loaded[("factor", 7)][0, 1])
    assert np.isnan(loaded[("factor", 7)][0, 2])


def test_sparse_spectrum_roundtrip_preserves_basis_and_numerical_arrays(tmp_path):
    matrix = sparse.csr_matrix(np.array([[2., -1.], [-1., 2.]]))
    values, vectors = np.linalg.eigh(matrix.toarray())
    result = SpectrumResult(1, values, vectors, matrix,
                            basis=((("a", 1), ("a", 2)), (("b", 1), ("b", 2))),
                            start=.5, end=math.inf, kind="persistent",
                            metadata={"mapping": {(0, 1): ("a", "b")}})
    restored = load_result(save_result(result, tmp_path / "spectrum.json"))
    assert isinstance(restored, SpectrumResult)
    assert sparse.isspmatrix_csr(restored.matrix)
    assert restored.basis == result.basis
    assert restored.metadata == result.metadata
    assert restored.end == math.inf
    np.testing.assert_array_equal(restored.matrix.indptr, matrix.indptr)
    np.testing.assert_array_equal(restored.matrix.indices, matrix.indices)
    np.testing.assert_array_equal(restored.matrix.data, matrix.data)
    np.testing.assert_array_equal(restored.eigenvectors, vectors)


def test_point_cloud_serialization_preserves_attributes_and_immutability(tmp_path):
    original = PointCloud([[0., 1.], [2., 3.]], ids=[8, "n"],
                          weights={8: -2., "n": 4.}, metadata={"source": ("unit", 1)})
    loaded = load_result(save_result(original, tmp_path / "cloud.json"))
    assert loaded.ids == original.ids and loaded.metadata == original.metadata
    np.testing.assert_array_equal(loaded.weights, original.weights)
    assert not loaded.points.flags.writeable and not loaded.weights.flags.writeable
    selected = loaded.subset(["n", 8])
    assert selected.ids == ("n", 8)
    np.testing.assert_array_equal(selected.weights, [4., -2.])


def test_topology_export_describes_but_does_not_rehydrate_live_engine(tmp_path):
    topology = simplicial_builder.from_points([[0.], [1.]], filtration_start=.1)
    restored = load_result(save_result(topology, tmp_path / "description.json"))
    assert restored["type"] == "TopologyDescription"
    assert restored["kind"] == "simplicial"
    assert "native" not in restored
    assert restored["metadata"]["raw_filtration"] == topology.metadata["raw_filtration"]


def test_feature_channels_conserve_finite_essential_and_overflow_intervals():
    result = bars([(0, 0., 1.), (0, .5, 1.5), (0, 1.5, 2.),
                   (0, 2., 2.5), (0, 3., 4.), (0, -1., 1.),
                   (0, .5, math.inf), (0, 3., math.inf),
                   (0, .4, .4), (0, .2, .25)])
    options = dict(birth_edges=[0., 1., 2.], death_edges=[0., 1., 2.],
                   dimensions=(0, 1), min_persistence=.1)
    raw = histogram_features(result, **options)
    assert raw.finite_histograms[0].sum() == 3
    assert raw.essential_histograms[0].sum() == 1
    np.testing.assert_array_equal(raw.out_of_range[0], [3, 1])
    assert raw.values.sum() == 8
    assert len(raw.values) == len(raw.names) == 16
    assert np.count_nonzero(raw.finite_histograms[1]) == 0
    normalized = histogram_features(result, normalize=True, **options)
    assert normalized.values.sum() == pytest.approx(1.)
    assert len(result.intervals) == 10  # feature filtering never edits raw bars


def test_histogram_includes_final_right_edge_without_clipping_essential_bars():
    result = bars([(0, 1., 2.), (0, 2., math.inf)])
    features = histogram_features(result, birth_edges=[0., 1., 2.],
                                  death_edges=[0., 1., 2.], dimensions=[0])
    assert features.finite_histograms[0, 1, 1] == 1
    assert features.essential_histograms[0, 1] == 1
    assert features.out_of_range.sum() == 0


@pytest.mark.parametrize("option", [
    {"birth_edges": [0., 0., 1.]}, {"death_edges": [0., math.inf]},
    {"dimensions": [0, 0]}, {"dimensions": [True]},
    {"min_persistence": -.1}, {"min_persistence": math.nan},
])
def test_feature_invalid_parameters_are_rejected(option):
    options = dict(birth_edges=[0., 1., 2.], death_edges=[0., 1., 2.], dimensions=[0])
    options.update(option)
    with pytest.raises((ValueError, TypeError)):
        histogram_features(bars([(0, 0., 1.)]), **options)


def test_features_do_not_silently_encode_uncomputed_dimensions_as_zero():
    result = bars([(0, 0., 1.)], maximum=0)
    with pytest.raises(ValueError, match="dimension|computed"):
        histogram_features(result, birth_edges=[0., 1.], death_edges=[0., 1.],
                           dimensions=[0, 1])


def test_fixed_vectorizer_is_repeatable_and_checks_units():
    data = [bars([(0, 0., 1.), (1, .5, math.inf)]), bars([(0, .2, 1.5)])]
    vectorizer = PersistenceVectorizer(birth_edges=[0., 1., 2.],
                                      death_edges=[0., 1., 2.], dimensions=(0, 1))
    first = vectorizer.fit_transform(iter(data))
    np.testing.assert_array_equal(first, vectorizer.transform(iter(data)))
    assert vectorizer.transform([]).shape == (0, first.shape[1])
    assert len(vectorizer.get_feature_names_out()) == first.shape[1]
    with pytest.raises(ValueError, match="units"):
        vectorizer.transform([bars([(0, 0., 1.)], units="edge_length")])
    with pytest.raises(ValueError, match="units"):
        vectorizer.fit([data[0], bars(units="edge_length")])
    vectorizer.set_params(normalize=True)
    with pytest.raises(RuntimeError, match="fit"):
        vectorizer.transform(data)


def test_fitted_schema_cannot_change_silently_via_mutable_bin_edges():
    birth_edges = [0., 1., 2.]
    data = [bars([(0, .5, 1.5)])]
    vectorizer = PersistenceVectorizer(birth_edges=birth_edges,
                                      death_edges=[0., 1., 2.], dimensions=(0,))
    original = vectorizer.fit_transform(data)
    birth_edges[1] = .25
    # Either freeze the fitted schema or explicitly invalidate it; silently
    # reassigning columns while retaining feature names is not acceptable.
    try:
        transformed = vectorizer.transform(data)
    except (RuntimeError, ValueError):
        return
    np.testing.assert_array_equal(transformed, original)


def test_root_dispatch_accepts_native_and_wrapped_simplicial_objects():
    explicit = simplicial.SimplicialComplex([(0, 1), (1, 2), (0, 2)])
    assert api.homology(explicit).betti_numbers == (1, 1, 0)
    assert api.laplacian(explicit, 1).nullity == 1
    wrapped = api.from_points([[0.], [1.]], kind="simplicial")
    assert api.homology(wrapped).betti_numbers == (1, 0, 0)
    with pytest.raises(TypeError):
        api.homology([(0, 1)])
    with pytest.raises(ValueError):
        api.from_points([[0.], [1.]], kind="unknown")


def test_coface_truncation_warns_at_construction_and_later_higher_requests():
    points = [[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]
    with pytest.warns(UserWarning, match="truncated"):
        truncated = simplicial_builder.from_points(points, max_dimension=2, max_simplex_dimension=2)
    assert truncated.metadata["dimension_warnings"]
    low_degree = simplicial_builder.from_points(points, max_dimension=1)
    with pytest.warns(UserWarning, match="truncated"):
        result = simplicial.persistence(low_degree, max_dimension=2)
    assert result.metadata["dimension_warnings"]
    with pytest.warns(UserWarning, match="truncated"):
        simplicial.laplacian(low_degree, dimension=2)


def test_affine_rank_limit_is_not_falsely_reported_as_missing_cofaces():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        line = simplicial_builder.from_points([[0.], [1.], [2.]], max_dimension=2,
                                       max_simplex_dimension=1)
    assert not recorded
    assert line.metadata["requested_death_dimension_available"]
    assert simplicial.homology(line).betti_numbers == (1, 0, 0)


def test_workflow_uses_snapshot_laplacians_without_mutating_barcodes(monkeypatch):
    topology = simplicial_builder.from_points([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    original = api.persistence(topology).as_array().copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("persistent Laplacians must be opt-in")

    monkeypatch.setattr(api, "persistent_laplacian", forbidden)
    result = analyze(topology, scales=[.3, .6])
    np.testing.assert_array_equal(result.persistence.as_array(), original)
    assert result.snapshots[.3]["homology"].betti_numbers == (1, 1, 0)
    assert result.snapshots[.6]["homology"].betti_numbers == (1, 0, 0)
    assert set(result.snapshots[.3]["laplacians"]) == {0, 1, 2}
    assert all(value.kind == "ordinary" for snapshot in result.snapshots.values()
               for value in snapshot["laplacians"].values())


@pytest.mark.parametrize("scales", [[.3, .3], [math.inf], [math.nan], [True]])
def test_workflow_rejects_invalid_or_boolean_scales(scales):
    topology = simplicial_builder.from_points([[0.], [1.]])
    with pytest.raises((ValueError, TypeError)):
        analyze(topology, scales=scales)


def test_partial_spectrum_validates_symmetry_before_arpack():
    matrix = sparse.csr_matrix([[1., 5., 0., 0.], [0., 2., 0., 0.],
                                [0., 0., 3., 0.], [0., 0., 0., 4.]])
    with pytest.raises(ValueError, match="symmetric"):
        solve_symmetric(matrix, k=1)


def test_integer_sparse_operators_work_consistently_for_partial_spectra():
    result = solve_symmetric(sparse.eye(5, format="csr", dtype=np.int64), k=2,
                             return_eigenvectors=True)
    np.testing.assert_allclose(result.values, [1., 1.])
    assert result.nullity is None and not result.complete
    assert result.residual_max < 1e-10


def test_repeated_eigenspaces_are_checked_by_projector_not_vector_sign():
    matrix = np.array([[2., -1., 0., -1.], [-1., 2., -1., 0.],
                       [0., -1., 2., -1.], [-1., 0., -1., 2.]])
    permutation = np.array([2, 0, 3, 1])
    first = solve_symmetric(matrix, return_eigenvectors=True)
    second = solve_symmetric(matrix[np.ix_(permutation, permutation)],
                              return_eigenvectors=True)
    inverse = np.argsort(permutation)
    left = first.vectors[:, np.isclose(first.values, 2.)]
    right = second.vectors[inverse][:, np.isclose(second.values, 2.)]
    np.testing.assert_allclose(left @ left.T, right @ right.T, atol=1e-12)
    np.testing.assert_allclose(first.vectors.T @ first.vectors, np.eye(4), atol=1e-12)
    assert first.nullity == 1 and first.residual_max < 1e-12


@pytest.mark.parametrize("options", [{"k": 0}, {"k": True}, {"tol": 0.},
                                      {"tol": math.nan}, {"max_dense_entries": True}])
def test_spectral_invalid_options_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        solve_symmetric(np.eye(3), **options)
