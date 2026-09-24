"""Connection support, scientific scale, and object-expansion contracts."""
import math

import numpy as np
import pytest
from scipy import sparse

from topokit import PointCloud, ResourceLimitError
from topokit.builders import simplicial as sb, hyperdigraph as hb
from topokit.builders import connections
from topokit.core import simplicial as sc, hyperdigraph as hc


def line():
    return PointCloud([[0.], [1.], [3.]], ids=("left", "middle", "right"), weights=[1, 2, 3])


@pytest.mark.parametrize("mode", ["graph", "flag"])
def test_none_infers_delaunay_but_empty_bonds_leave_all_vertices(mode):
    cloud = line()
    inferred = sb.from_points(cloud, complex_type=mode)
    empty = sb.from_points(cloud, complex_type=mode, bonds=[])
    assert inferred.native.simplices(1) == ((0, 1), (1, 2))
    assert inferred.metadata["connection_support"] == "delaunay"
    assert empty.native.simplices(0) == ((0,), (1,), (2,))
    assert empty.native.simplices(1) == ()
    assert empty.metadata["connection_support"] == "supplied"
    assert sc.homology(empty).betti_numbers == (3, 0, 0)


@pytest.mark.parametrize("route", ["graph", "flag", "digraph", "hyperdigraph"])
def test_supplied_stable_id_bonds_replace_support_without_geometry(monkeypatch, route):
    def forbidden(*args, **kwargs):
        raise AssertionError("supplied bonds must not invoke Delaunay")
    monkeypatch.setattr(connections, "build_point_cloud_support", forbidden)
    cloud = line()
    options = {"bonds": [("right", "left")], "max_dimension": 2}
    if route in {"graph", "flag"}:
        obj = sb.from_points(cloud, complex_type=route, **options)
        assert obj.native.simplices(1) == ((0, 2),)
        assert sc.laplacian(obj, 1).basis == (("left", "right"),)
    else:
        obj = hb.from_points(cloud, object_type=route, **options)
        assert obj.native.snapshot(4).directed_hyperedges(1) == (("left", "right"),)
    assert obj.cloud.ids == cloud.ids
    assert obj.metadata["connection_pairs"] == (("left", "right"),)


def test_graph_caps_dimension_while_flag_fills_bond_cliques():
    cloud = PointCloud([[0, 0], [1, 0], [.5, math.sqrt(3)/2], [4, 4]], ids=("a", "b", "c", "other"))
    pairs = [("a", "b"), ("b", "c"), ("a", "c")]
    graph = sb.from_points(cloud, complex_type="graph", bonds=pairs, max_simplex_dimension=7)
    flag = sb.from_points(cloud, complex_type="flag", bonds=pairs)
    assert graph.metadata["max_simplex_dimension"] == graph.native.dimension == 1
    assert flag.native.dimension == 2
    assert sc.homology(graph).betti_numbers == (2, 1, 0)
    assert sc.homology(flag).betti_numbers == (2, 0, 0)
    assert sc.laplacian(graph, 1).nullity == 1
    assert sc.laplacian(flag, 1).nullity == 0


@pytest.mark.parametrize("route", ["graph", "flag", "digraph", "hyperdigraph"])
def test_cutoff_prunes_supplied_bonds_inclusively_and_records_ids(route):
    cloud = line()
    pairs = [("left", "middle"), ("middle", "right"), ("left", "right")]
    options = {"bonds": pairs, "cutoff": 2., "filtration_range": (.5, 1.5)}
    obj = (sb.from_points(cloud, complex_type=route, **options) if route in {"graph", "flag"}
           else hb.from_points(cloud, object_type=route, **options))
    assert obj.metadata["cutoff_pruned_bonds"] == (("left", "right"),)
    assert obj.metadata["cutoff_pruned_count"] == 1
    assert obj.metadata["retained_connection_count"] == 2  # length 2 is included by cutoff
    assert obj.metadata["filtration_excluded_connection_count"] == 1
    assert obj.metadata["filtration_range"] == (.5, 1.5)
    if route in {"graph", "flag"}:
        assert obj.native.simplices(1) == ((0, 1),)
        assert dict(obj.metadata["raw_filtration"])[(0,)] == 0
        assert obj.native.filtration((0,)) == .5
    else:
        assert obj.native.snapshot(1.5).directed_hyperedges(1) == (("left", "middle"),)
        assert dict(obj.metadata["raw_births"])[("left",)] == 0
        assert obj.native.birth_of(("left",)) == .5


def test_none_cutoff_preserves_long_supplied_bond_and_cutoff_zero_preserves_coincident_ids():
    long = sb.from_points(line(), complex_type="graph", bonds=[("left", "right")])
    assert long.native.filtration((0, 2)) == 3.
    cloud = PointCloud([[0.], [0.]], ids=(10, 20))
    coincident = hb.from_points(cloud, bonds=[(10, 20)], cutoff=0)
    assert set(coincident.native.snapshot(0).directed_hyperedges(1)) == {(10, 20), (20, 10)}
    assert coincident.metadata["cutoff_pruned_count"] == 0


def test_alpha_remains_squared_radius_not_bond_or_diameter_filtration():
    points = [[0, 0], [2, 0], [1, math.sqrt(3)]]
    alpha = sb.from_points(points, filtration_range=(0, 1.1))
    assert alpha.metadata["scale_units"] == "squared_radius"
    assert len(alpha.native.simplices(1)) == 3 and alpha.native.simplices(2) == ()
    for options in ({"bonds": []}, {"bonds": [(0, 1)]}, {"cutoff": 2.}):
        with pytest.raises(ValueError, match="alpha"):
            sb.from_points(points, **options)


def test_rips_uses_full_distance_graph_and_retains_separate_distance_cap():
    cloud = line()
    rips = sb.from_points(cloud, complex_type="rips")
    flag = sb.from_points(cloud, complex_type="flag")
    assert len(rips.native.simplices(1)) == 3
    assert len(flag.native.simplices(1)) == 2
    assert rips.native.simplices(2) == ((0, 1, 2),) and flag.native.simplices(2) == ()
    restricted = sb.from_points(cloud, complex_type="rips", cutoff=2., filtration_range=(0, 10))
    assert restricted.native.simplices(1) == ((0, 1), (1, 2))
    assert restricted.metadata["cutoff_distance"] == 2.
    assert restricted.metadata["filtration_end"] == 10.
    assert restricted.metadata["scale_units"] == "edge_length"
    with pytest.raises(ValueError, match="rips"):
        sb.from_points(cloud, complex_type="rips", bonds=[])


def test_higher_hyperedges_remain_consecutive_sequences_not_cliques():
    cloud = PointCloud([[0.], [1.], [2.]], weights=[1, 2, 3])
    hyper = hb.from_points(cloud, bonds=[(0, 1), (1, 2)])
    graph = hb.from_points(cloud, bonds=[(0, 1), (1, 2)], object_type="digraph")
    assert hyper.native.snapshot(2).directed_hyperedges(2) == ((0, 1, 2),)
    assert (0, 2) not in hyper.native.snapshot(2).directed_hyperedges(1)
    assert graph.native.snapshot(2).directed_hyperedges(2) == ()
    assert graph.metadata["construction_max_dimension"] == 1
    assert hc.homology(graph).betti_numbers == (1, 0, 0)


def test_weight_overrides_are_explicit_and_do_not_mutate_cloud_or_alpha():
    cloud = line()
    unchanged = hb.from_points(cloud, weights=None)
    overridden = hb.from_points(cloud, weights={"left": 3, "middle": 2, "right": 1})
    assert unchanged.cloud is cloud
    assert overridden.native.snapshot(3).directed_hyperedges(1) == (("middle", "left"), ("right", "middle"))
    np.testing.assert_array_equal(cloud.weights, [1, 2, 3])
    np.testing.assert_array_equal(overridden.cloud.weights, [3, 2, 1])
    first = sb.from_points(cloud)
    second = sb.from_points(cloud, weights=[9, 2, 4])
    assert first.metadata["raw_filtration"] == second.metadata["raw_filtration"]
    uniform = hb.from_points([[0.], [1.]])
    assert set(uniform.native.snapshot(1).directed_hyperedges(1)) == {(0, 1), (1, 0)}


@pytest.mark.parametrize("as_sparse", [False, True])
def test_adjacency_converter_is_binary_explicit_and_preserves_id_order(as_sparse):
    array = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    matrix = sparse.csr_matrix(array) if as_sparse else array
    pairs = connections.bonds_from_adjacency(matrix, ids=("z", 4, "a"))
    assert pairs == (("z", 4), (4, "a"))
    directed = connections.bonds_from_adjacency(matrix, ids=("z", 4, "a"), directed=True)
    assert directed == (("z", 4), (4, "z"), (4, "a"), ("a", 4))
    np.testing.assert_array_equal(array, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@pytest.mark.parametrize("matrix, ids", [
    ([[0, 2], [2, 0]], ("a", "b")), ([[0, 1], [0, 0]], ("a", "b")),
    ([[1, 0], [0, 0]], ("a", "b")), ([[0, 1], [1, 0]], ("a", "a")),
    ([[0, 1], [1, 0]], ("a",)), ([[0, math.nan], [math.nan, 0]], ("a", "b")),
    ([[0, 1], [1, 0]], "ab"),
])
def test_adjacency_rejects_ambiguous_values_shape_labels_or_direction(matrix, ids):
    with pytest.raises(ValueError):
        connections.bonds_from_adjacency(matrix, ids=ids)


@pytest.mark.parametrize("bonds", [[("left", "unknown")], [(0, 1)], [("left", "left")],
                                  [("left", "middle", 2)], [(True, "middle")]])
def test_bonds_require_known_distinct_stable_id_pairs(bonds):
    with pytest.raises((ValueError, TypeError)):
        sb.from_points(line(), complex_type="flag", bonds=bonds)


@pytest.mark.parametrize("cutoff", [-1, True, math.inf, math.nan, "2"])
def test_invalid_distance_cutoffs_are_rejected(cutoff):
    for builder in (lambda: sb.from_points(line(), complex_type="graph", cutoff=cutoff),
                    lambda: hb.from_points(line(), cutoff=cutoff)):
        with pytest.raises(ValueError):
            builder()


def test_reverse_duplicates_collapse_once_with_explicit_metadata():
    obj = sb.from_points(line(), complex_type="graph",
                         bonds=[("left", "right"), ("right", "left")])
    assert obj.native.simplices(1) == ((0, 2),)
    assert obj.metadata["duplicate_bonds_collapsed"] == 1


def test_construction_budgets_refuse_without_implicit_edge_pruning():
    pairs = [("left", "middle"), ("middle", "right"), ("left", "right")]
    with pytest.raises(ResourceLimitError):
        sb.from_points(line(), complex_type="flag", bonds=pairs, max_simplices=6)
    with pytest.raises(ResourceLimitError):
        hb.from_points(line(), bonds=pairs, max_hyperedges=6)
    assert hb.from_points(line(), bonds=[], max_hyperedges=3).metadata["hyperedge_counts"] == (3, 0, 0, 0)
