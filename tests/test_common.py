import json
import numpy as np
import pytest
from scipy import sparse
from topokit import PointCloud, PersistenceInterval, PersistenceResult, SpectrumResult, ResourceLimitError
from topokit.core._spectral import solve_symmetric
from topokit.postprocessing import histogram_features, PersistenceVectorizer
from examples.data_fixture import load_demo_cloud as demo_point_cloud
from topokit.serialization import save_result, load_result


def sample():
    return PersistenceResult((PersistenceInterval(0, 0, 1), PersistenceInterval(0, 0, float("inf"), True),
                              PersistenceInterval(1, 1, 3), PersistenceInterval(1, 5, 7)),
                             max_dimension=2, metadata={"scale_units": "squared_radius", "filtration_start": 0})


def test_point_identity_and_weights():
    cloud = PointCloud([[0, 1], [2, 3]], ids=("a", "b"), weights={"b": 7, "a": 4})
    assert tuple(cloud.weights) == (4, 7)
    assert cloud.subset(("b",)).ids == ("b",)
    with pytest.raises(ValueError):
        cloud.points[0, 0] = 9
    with pytest.raises(ValueError):
        PointCloud([[0, 1], [2, 3]], ids=("a", "a"))
    with pytest.raises(ValueError):
        PointCloud([[0, 1]], weights={"wrong": 1})
    assert demo_point_cloud().subset(demo_point_cloud().metadata["subset_ids"]).points.shape == (12, 3)


def test_feature_channels_and_fixed_ml_schema():
    result = sample()
    args = dict(birth_edges=[0, 1, 2, 4], death_edges=[0, 1, 2, 4])
    feature = histogram_features(result, **args)
    assert feature.finite_histograms.sum() == 2
    assert feature.essential_histograms.sum() == 1
    assert feature.out_of_range.sum() == 1
    transformer = PersistenceVectorizer(**args)
    matrix = transformer.fit_transform([result, result])
    assert matrix.shape == (2, len(feature.values))
    np.testing.assert_array_equal(matrix[0], feature.values)
    other_units = PersistenceResult(result.intervals, metadata={"scale_units": "distance"})
    with pytest.raises(ValueError):
        transformer.transform([other_units])


def test_standard_json_roundtrip(tmp_path):
    result = sample()
    path = save_result({"bars": result, "ids": {(1, "a"): (3, "b")}}, tmp_path / "result.json")
    document = json.loads(path.read_text())
    assert document["schema_version"] == 1
    assert "Infinity" not in path.read_text()
    restored = load_result(path)
    assert restored["bars"].intervals == result.intervals
    assert restored["ids"] == {(1, "a"): (3, "b")}
    matrix = sparse.csr_matrix([[1., -1.], [-1., 1.]])
    spectrum = SpectrumResult(0, np.array([0., 2.]), matrix=matrix, basis=("a", "b"))
    restored = load_result(save_result(spectrum, tmp_path / "spectrum.json"))
    np.testing.assert_allclose(restored.matrix.toarray(), matrix.toarray())
    assert restored.basis == ("a", "b")


def test_spectral_contract_and_budget():
    matrix = sparse.csr_matrix([[1., -1.], [-1., 1.]])
    default = solve_symmetric(matrix)
    assert default.vectors is None and default.nullity == 1
    solved = solve_symmetric(matrix, return_eigenvectors=True)
    assert solved.residual_max < 1e-10
    assert solved.vectors.shape == (2, 2)
    with pytest.raises(ResourceLimitError):
        solve_symmetric(matrix, max_dense_entries=1)


def test_domain_and_initial_stage():
    result = PersistenceResult((PersistenceInterval(0, 5, float("inf"), True),), max_dimension=0,
                               metadata={"filtration_start": 5, "filtration_end": 9})
    assert result.betti_at(5) == (1,)
    with pytest.raises(ValueError):
        result.betti_at(4)
