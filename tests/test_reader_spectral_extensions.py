"""Reader reference data, explicit filtration bounds, and spectral composition."""
import math
import numpy as np
import pytest

from topokit import PointCloud, builders, core, readers
from topokit.exceptions import ResourceLimitError
from topokit.filtrations import resolve_range
from topokit.results import SpectrumResult
from topokit.postprocessing import summarize_spectrum
from topokit.workflows import laplacian_series, critical_scales


def test_xyz_weight_assignment_is_explicit_sourced_and_nonmutating(tmp_path):
    path = tmp_path / "points.xyz"
    path.write_text("3\nsynthetic coordinates\nB 0 0 0 extra\nN 1 0 0\nB 0 1 0\n")
    raw = readers.read_xyz(path, coordinate_units="angstrom")
    weighted = readers.assign_element_weights(raw)
    np.testing.assert_array_equal(raw.weights, [1, 1, 1])
    np.testing.assert_array_equal(weighted.weights, [2.04, 3.04, 2.04])
    np.testing.assert_array_equal(weighted.points, raw.points)
    assert weighted.ids == raw.ids
    assert weighted.metadata["extra_fields"] == (("extra",), (), ())
    assert weighted.metadata["weight_assignment"]["scale"] == "Pauling"
    assert "weight_assignment" not in raw.metadata
    assert len(readers.PAULING_ELECTRONEGATIVITY) == 118
    assert readers.PAULING_ELECTRONEGATIVITY["Ne"] is None
    with pytest.raises(TypeError):
        readers.PAULING_ELECTRONEGATIVITY["B"] = 0
    a = builders.from_points(raw, max_dimension=1)
    b = builders.from_points(weighted, max_dimension=1)
    assert list(a.native.get_filtration()) == list(b.native.get_filtration())


def test_element_assignment_never_imputes_or_guesses_symbols():
    cloud = PointCloud([[0.], [1.]], metadata={"labels": ("He", "C")})
    with pytest.raises(ValueError, match="no PubChem"):
        readers.assign_element_weights(cloud)
    enriched = readers.assign_element_weights(cloud, overrides={"He": 1.})
    assert enriched.metadata["weight_assignment"]["contains_user_overrides"]
    np.testing.assert_array_equal(enriched.weights, [1, 2.55])
    for symbols in (("C1", "C"), "CC", ("C",)):
        with pytest.raises(ValueError):
            readers.assign_element_weights(cloud, symbols=symbols)
    for override in ({"He": math.nan}, {"He": True}, {"Xx": 1}, [("He", 1)]):
        with pytest.raises(ValueError):
            readers.assign_element_weights(cloud, overrides=override)
    relabeled = readers.assign_element_weights(cloud, symbols=("B", "N"))
    assert relabeled.metadata["labels"] == ("He", "C")
    assert relabeled.metadata["weight_assignment"]["symbols_by_point_id"] == {0: "B", 1: "N"}
    with pytest.raises(ValueError):
        readers.assign_element_weights(cloud, symbols={"C", "B"})


def test_resolve_range_is_not_an_observation_schedule():
    assert resolve_range() == (0., math.inf)
    assert resolve_range([1, 3]) == (1., 3.)
    assert resolve_range([1, 3], filtration_start=1, max_scale=3) == (1., 3.)
    for bad in ([2, 1], [0], [0, 1, 2], [math.inf, math.inf], [0, math.nan],
                [-1, 2], [False, 1], "01", [0, 1+0j], {0, 2}, {0: "x", 2: "y"}):
        with pytest.raises(ValueError):
            resolve_range(bad)
    with pytest.raises(ValueError, match="conflicts"):
        resolve_range([0, 2], max_scale=3)


def test_default_spectrum_summary_uses_positive_values_and_full_zero_count():
    values = np.array([-1e-12, 0., 2., 4.])
    answer = summarize_spectrum(SpectrumResult(2, values))
    assert answer.names == ("L2:mean", "L2:max", "L2:min", "L2:std", "L2:zero_count")
    np.testing.assert_allclose(answer.values, [3., 4., 2., 1., 2.])
    assert answer.metadata["positive_only"]
    np.testing.assert_array_equal(values, [-1e-12, 0., 2., 4.])
    empty = summarize_spectrum(SpectrumResult(2, np.zeros(3)))
    assert np.isnan(empty.values[:4]).all()
    assert empty.values[-1] == 3
    with pytest.raises(ValueError, match="positive-semidefinite"):
        summarize_spectrum(SpectrumResult(0, np.array([-1., 2.])), statistics=("mean",))


def test_spectrum_relative_tolerance_partial_scope_and_custom_callbacks():
    result = SpectrumResult(0, np.array([-.01, .01, 1e6]), kind="persistent")
    summary = summarize_spectrum(result, atol=1e-12, rtol=1e-7,
                                 statistics=("positive_count", "zero_count"))
    np.testing.assert_array_equal(summary.values, [1, 2])
    assert summary.metadata["zero_count_semantics"] == "real_persistent_image_rank"
    partial = summarize_spectrum(SpectrumResult(0, np.array([0., 1.]), complete=False), allow_partial=True)
    assert math.isnan(partial.values[-1])
    assert partial.metadata["observed_zero_count"] == 1
    assert not partial.metadata["zero_count_available"]
    custom = summarize_spectrum(SpectrumResult(0, np.array([0., 1., 4.])),
                                statistics={"range": lambda values: np.ptp(values)})
    assert custom.values[0] == 3


def test_relative_tolerance_schema_compares_configuration_not_sample_threshold():
    from topokit.workflows.ml import feature_matrix
    summaries = [summarize_spectrum(SpectrumResult(0, np.array(values)), rtol=.1)
                 for values in ([0., 1., 2.], [0., 2., 4.])]
    assert summaries[0].metadata["tolerance"] != summaries[1].metadata["tolerance"]
    assert feature_matrix(summaries).shape == (2, 5)
    with pytest.raises(ValueError, match="relative_tolerance"):
        feature_matrix([summaries[0], summarize_spectrum(SpectrumResult(0, np.array([0., 1., 2.])), rtol=.2)])


def test_series_observation_guard_stops_consuming_unbounded_input():
    obj = builders.from_points([[0.], [1.]])
    def scales():
        yield from (0., 1., 2.)
        raise AssertionError("must stop at budget+1")
    with pytest.raises(ResourceLimitError):
        laplacian_series(obj, scales=scales(), max_snapshots=2)


def test_spectral_schema_rejects_different_paired_thresholds():
    from topokit.builders.interaction import from_points, regrade_pair_filtration
    from topokit.workflows.ml import feature_matrix
    obj = from_points([[0.], [1.]], max_dimension=0)
    records = []
    for thresholds in ([0., 1.], [0., 2.]):
        path = regrade_pair_filtration(obj, thresholds, max_dimension=0)
        spectrum = core.laplacian(path, dimension=0, scale=1.)
        records.append(summarize_spectrum(spectrum, statistics=("zero_count",)))
    with pytest.raises(ValueError, match="filtration_a|schema"):
        feature_matrix(records)


@pytest.mark.parametrize("kind", ["simplicial", "hyperdigraph", "interaction"])
def test_laplacian_collections_are_ordinary_by_default_and_ragged(kind):
    obj = builders.from_points([[0.], [1.], [2.]], kind=kind, max_dimension=1)
    times = critical_scales(obj)
    assert times[0] == 0
    collection = laplacian_series(obj, max_dimension=1)
    assert tuple(collection) == times
    for by_degree in collection.values():
        assert tuple(by_degree) == (0, 1)
        for spectrum in by_degree.values():
            assert spectrum.kind == "ordinary"
            assert spectrum.eigenvectors is None and spectrum.matrix is None
    paired = laplacian_series(obj, max_dimension=1, mode="persistent", scale_pairs=[(times[0], times[-1])])
    assert paired[times[0], times[-1]][1].kind == "persistent"
    static = core.laplacians(obj, max_dimension=1)
    assert tuple(static) == (0, 1)
    with pytest.raises(ValueError, match="scale_pairs"):
        laplacian_series(obj, mode="persistent")
    with pytest.raises(ValueError, match="increasing"):
        laplacian_series(obj, scales=[1, 0])
    with pytest.raises(ResourceLimitError):
        laplacian_series(obj, max_snapshots=1)
