"""Independent interval/bin predicates and real persistence integration."""
from copy import deepcopy
import math
import numpy as np
import pytest

from topokit import PersistenceInterval, PersistenceResult, ResourceLimitError
from topokit.core import hyperdigraph
from topokit.postprocessing import barcode_bin_counts
from topokit.serialization import load_result, save_result
from topokit.workflows.ml import feature_matrix


def result(records=(), *, maximum=0, **metadata):
    return PersistenceResult(tuple(PersistenceInterval(*bar) for bar in records),
                             max_dimension=maximum, metadata=metadata)


def test_exact_boundaries_partial_bins_zero_length_and_essential_intervals():
    bars = result(((0, 0, .05), (0, .02, .08), (0, .05, .1),
                   (0, 0, math.inf), (0, .05, .05), (0, .1, math.inf)))
    axis = [0, .05, .1]
    np.testing.assert_array_equal(barcode_bin_counts(bars, bin_edges=axis).values, [3, 3])
    np.testing.assert_array_equal(barcode_bin_counts(bars, bin_edges=axis, mode="cover").values, [2, 2])
    np.testing.assert_array_equal(barcode_bin_counts(bars, bin_edges=axis, mode="death").values, [0, 3])


@pytest.mark.parametrize("mode", ["overlap", "cover", "death"])
def test_random_intervals_match_independent_bin_by_bin_predicates(mode):
    rng = np.random.default_rng(741)
    axis = np.asarray([-1., -.2, .1, .5, 2., 5.])
    records = []
    for q in range(3):
        for index in range(100):
            birth = float(rng.uniform(-3, 7))
            death = birth + float(rng.uniform(0, 3))
            if index % 7 == 0:
                death = math.inf
            elif index % 9 == 0:
                death = birth
            records.append((q, birth, death))
    records += [(q, float(b), float(d)) for q in range(3)
                for b in axis for d in axis if b <= d]
    source = result(records, maximum=2)
    degrees = (2, 0, 1)
    expected = []
    for q in degrees:
        for i, (left, right) in enumerate(zip(axis[:-1], axis[1:])):
            count = 0
            for degree, birth, death in records:
                if degree != q or death <= birth:
                    continue
                if mode == "overlap":
                    count += min(death, right) > max(birth, left)
                elif mode == "cover":
                    count += birth <= left and death >= right
                else:
                    count += math.isfinite(death) and (left <= death < right or
                                                      (i == len(axis)-2 and death == right))
            expected.append(count)
    actual = barcode_bin_counts(source, bin_edges=axis, dimensions=degrees, mode=mode)
    np.testing.assert_array_equal(actual.values, expected)
    assert actual.names[0] == f"H2:barcode_{mode}:bin[0]"


def test_101_boundaries_give_100_bins_and_essential_is_not_a_death_at_cutoff():
    source = result(((0, 0, math.inf),), filtration_start=0, filtration_end=5)
    axis = np.arange(101, dtype=float) / 20
    feature = barcode_bin_counts(source, bin_edges=axis)
    np.testing.assert_array_equal(feature.values, np.ones(100))
    assert barcode_bin_counts(source, bin_edges=axis, mode="death").values.sum() == 0
    assert source.intervals[0].death == math.inf
    assert feature.metadata["learned_from_data"] is False


def test_empty_barcodes_and_explicitly_empty_dimension_selection():
    np.testing.assert_array_equal(barcode_bin_counts(result(), bin_edges=[0, 1, 2]).values, [0, 0])
    empty = barcode_bin_counts(result(), bin_edges=[0, 1], dimensions=())
    assert empty.values.shape == (0,)
    assert empty.names == ()


def test_real_hyperdigraph_barcodes_then_bins_including_higher_degree():
    graph = hyperdigraph.FilteredHyperdigraph((0, 1, 2, 3),
        [((0, 1), .5), ((1, 2), 1.), ((0, 2), 1.5), ((0, 1, 2), 2.)],
        include_all_vertices=True, vertex_birth=0.)
    bars = hyperdigraph.persistence(graph, max_dimension=1)
    feature = barcode_bin_counts(bars, bin_edges=[0, .5, 1, 1.5, 2, 2.5], dimensions=(0, 1))
    # H0 merges at .5 and 1; vertex 3 remains isolated. Triangle fills at 2.
    np.testing.assert_array_equal(feature.values, [4, 3, 2, 2, 2, 0, 0, 0, 1, 0])
    for scale, index in ((.25, 0), (.75, 1), (1.25, 2), (1.75, 3), (2.25, 4)):
        fixed = hyperdigraph.homology(graph, max_dimension=1, scale=scale)
        assert fixed.betti_numbers == (int(feature.values[index]), int(feature.values[index+5]))


def test_roundtrip_provenance_independence_and_ml_schema(tmp_path):
    source = result(((0, 0, 1), (0, 0, math.inf)), source_id="original", filtration_end=2)
    original = deepcopy(source)
    axis = np.array([0, 1, 2.])
    feature = barcode_bin_counts(source, bin_edges=axis)
    restored = load_result(save_result(feature, tmp_path / "counts.json"))
    np.testing.assert_array_equal(feature_matrix([feature, restored]), [[2, 1], [2, 1]])
    axis[-1] = 3
    assert feature.metadata["bin_edges"][-1] == 2
    feature.metadata["source_metadata"]["source_id"] = "edited"
    assert source == original
    different_bins = barcode_bin_counts(source, bin_edges=[0, .5, 2])
    with pytest.raises(ValueError, match="schema"):
        feature_matrix([feature, different_bins])


@pytest.mark.parametrize("kwargs,match", [
    ({"bin_edges": [0]}, "bin_edges"),
    ({"bin_edges": [0, 0, 1]}, "bin_edges"),
    ({"bin_edges": [0, math.inf]}, "bin_edges"),
    ({"bin_edges": [0, 1j]}, "real"),
    ({"mode": "betti"}, "mode"),
    ({"dimensions": (1,)}, "not computed"),
    ({"dimensions": (0, 0)}, "unique"),
    ({"max_features": True}, "max_features"),
])
def test_invalid_options_fail_explicitly(kwargs, match):
    options = {"bin_edges": [0, 1], **kwargs}
    with pytest.raises(ValueError, match=match):
        barcode_bin_counts(result(), **options)


def test_window_budget_and_invalid_source_guards():
    with pytest.raises(ResourceLimitError):
        barcode_bin_counts(result(), bin_edges=[0, 1, 2], max_features=1)
    for metadata, match in (({"filtration_start": .5}, "before"),
                            ({"filtration_end": .5}, "beyond"),
                            ({"filtration_end": math.nan}, "finite")):
        with pytest.raises(ValueError, match=match):
            barcode_bin_counts(result(**metadata), bin_edges=[0, 1])
    with pytest.raises(ValueError, match="death"):
        barcode_bin_counts(result(((0, 2, 1),)), bin_edges=[0, 1])
    with pytest.raises(TypeError, match="PersistenceResult"):
        barcode_bin_counts([], bin_edges=[0, 1])
