from itertools import combinations
import ast
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from topokit import PointCloud, ResourceLimitError, Topology
from topokit.core import simplicial as st
from topokit.builders import simplicial as sb


SQUARE = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
TETRAHEDRON = np.array([[1., 1., 1.], [1., -1., -1.],
                        [-1., 1., -1.], [-1., -1., 1.]])


def diagram(result, dimension):
    return np.array([(i.birth, i.death) for i in result.intervals
                     if i.dimension == dimension]).reshape(-1, 2)


def test_weighted_cloud_alpha_retains_ids_and_unweighted_geometry():
    cloud = PointCloud(SQUARE, ids=["a", "b", "c", "d"], weights=[1., 2., 4., 8.])
    obj = sb.from_points(cloud)
    assert obj.kind == "simplicial"
    assert obj.cloud is cloud
    assert obj.metadata["max_simplex_dimension"] == 3
    assert obj.metadata["scale_units"] == "squared_radius"
    assert obj.metadata["geometry_weighted"] is False
    assert obj.metadata["point_weights_role"] == "attributes_only"
    assert obj.metadata["raw_filtration"] == sb.from_points(SQUARE).metadata["raw_filtration"]
    assert obj.metadata["vertex_id_map"] == {0: "a", 1: "b", 2: "c", 3: "d"}
    assert obj.metadata["vertex_to_id"] == obj.metadata["vertex_id_map"]
    result = st.laplacian(obj, 0, scale=.3, return_matrix=True)
    assert result.basis == (("a",), ("b",), ("c",), ("d",))
    assert result.eigenvectors is None
    assert result.matrix.shape == (4, 4)
    np.testing.assert_allclose(diagram(st.persistence(obj), 1), [[.25, .5]])


def test_initial_stage_clamping_preserves_raw_births_and_query_domain():
    obj = sb.from_points(SQUARE, filtration_start=.3, max_scale=.6)
    assert min(value for _, value in obj.metadata["raw_filtration"]) == 0
    assert min(value for _, value in obj.native.get_filtration()) == .3
    result = st.persistence(obj)
    np.testing.assert_allclose(diagram(result, 1), [[.3, .5]])
    assert all(i.at_initial_stage for i in result.intervals)
    assert result.betti_at(.3) == (1, 1, 0)
    assert st.homology(obj, scale=.3).betti_numbers == (1, 1, 0)
    for scale in [.29, .61, np.inf, np.nan]:
        with pytest.raises(ValueError):
            st.homology(obj, scale=scale)
        with pytest.raises(ValueError):
            st.laplacian(obj, scale=scale)


def test_cutoff_essential_means_survival_in_constructed_domain():
    obj = sb.from_points(SQUARE, filtration_start=.3, max_scale=.4)
    result = st.persistence(obj)
    np.testing.assert_allclose(diagram(result, 1), [[.3, np.inf]])
    assert result.metadata["filtration_end"] == .4
    with pytest.raises(ValueError):
        result.betti_at(.5)


def test_geometry_route_has_h2_and_l2_snapshots_by_explicit_scale():
    obj = sb.from_points(PointCloud(TETRAHEDRON, ids=["p", "q", "r", "s"],
                                   weights=[1, 3, 2, 4]))
    assert obj.native.dimension == 3
    np.testing.assert_allclose(diagram(st.persistence(obj), 2), [[8 / 3, 3]])
    h = st.homology(obj, scale=2.8)
    assert h.betti_numbers == (1, 0, 1)
    for q, expected in enumerate(h.betti_numbers):
        result = st.laplacian(obj, q, scale=2.8, return_eigenvectors=True, return_matrix=True)
        assert result.complete and result.nullity == expected
        assert result.metadata["residual_max"] < 1e-10
        np.testing.assert_allclose(result.matrix @ result.eigenvectors,
                                   result.eigenvectors * result.eigenvalues, atol=1e-10)
    assert st.homology(obj, scale=3.1).betti_numbers == (1, 0, 0)


def test_static_explicit_objects_and_higher_degree():
    sphere = st.SimplicialComplex(combinations(range(5), 4))
    assert st.homology(sphere, max_dimension=3).betti_numbers == (1, 0, 0, 1)
    result = st.laplacian(sphere, dimension=3)
    assert result.basis == tuple(combinations(range(5), 4))
    assert result.nullity == 1
    assert result.matrix is None and result.eigenvectors is None
    assert st.homology([(10, 20), (20, 30)], max_dimension=1).betti_numbers == (1, 0)


def test_explicit_graph_is_unfilled_unless_flag_is_requested():
    edges = [(10, 20), (20, 30), (10, 30)]
    graph = sb.from_graph(edges)
    filled = sb.from_graph(edges, flag=True)
    assert graph.native.dimension == 1 and filled.native.dimension == 2
    assert st.homology(graph).betti_numbers == (1, 1, 0)
    assert st.homology(filled).betti_numbers == (1, 0, 0)
    assert st.laplacian(graph, 1).nullity == 1
    assert st.laplacian(filled, 1).nullity == 0
    assert graph.metadata["graph_expansion"] == "none"


def test_graph_weights_are_births_not_laplacian_inner_products():
    weighted = sb.from_graph([(0, 1, 3.), (1, 2, 5.)], vertices=[0, 1, 2])
    plain = sb.from_graph([(0, 1), (1, 2)], vertices=[0, 1, 2])
    assert st.homology(weighted, scale=2.).betti_numbers == (3, 0, 0)
    assert st.homology(weighted, scale=4.).betti_numbers == (2, 0, 0)
    np.testing.assert_allclose(st.laplacian(weighted, return_matrix=True).matrix.toarray(),
                               st.laplacian(plain, return_matrix=True).matrix.toarray())
    assert weighted.metadata["edge_weights_role"] == "filtration_births"


def test_explicit_real_homology_is_marked_separately():
    obj = st.SimplicialComplex([(0, 1, 2)])
    assert st.homology(obj, field="real").field == "R"
    assert st.homology(obj, field=3).field == "GF(3)"
    with pytest.raises((TypeError, ValueError)):
        st.persistence(obj, field="real")


def test_partial_spectrum_cannot_report_total_nullity_and_respects_budget():
    obj = st.SimplicialComplex([(0, 1, 2, 3, 4)])
    with pytest.raises(ResourceLimitError):
        st.laplacian(obj, 0, max_dense_entries=4)
    result = st.laplacian(obj, 0, k=2, return_eigenvectors=True,
                          max_dense_entries=4)
    assert not result.complete
    assert result.nullity is None
    assert result.eigenvectors.shape == (5, 2)
    assert result.metadata["residual_max"] < 1e-8


def test_persistent_laplacian_is_opt_in_and_equal_scale_matches_ordinary():
    obj = sb.from_points(SQUARE)
    ordinary = st.laplacian(obj, 1, scale=.3, return_matrix=True)
    same = st.persistent_laplacian(obj, 1, start=.3, end=.3, return_matrix=True)
    later = st.persistent_laplacian(obj, 1, start=.3, end=.6)
    np.testing.assert_allclose(same.matrix.toarray(), ordinary.matrix.toarray())
    assert ordinary.kind == "ordinary" and ordinary.nullity == 1
    assert later.kind == "persistent" and later.nullity == 0
    assert later.start == .3 and later.end == .6
    for start, end in [(.6, .3), (None, .3), (.3, np.inf)]:
        with pytest.raises(ValueError):
            st.persistent_laplacian(obj, 1, start=start, end=end)


def test_duplicate_merge_maps_to_first_stable_id():
    cloud = PointCloud([[0., 0.], [1., 0.], [0., 0.]], ids=["first", "next", "alias"])
    with pytest.raises(ValueError, match="Duplicate"):
        sb.from_points(cloud, duplicates="error")
    obj = sb.from_points(cloud)
    assert obj.metadata["original_id_to_vertex_id"] == {
        "first": "first", "next": "next", "alias": "first"}
    assert st.laplacian(obj, 0).basis == (("first",), ("next",))


def test_explicit_skeleton_still_uses_full_coface_alpha_births():
    obj = sb.from_points([[0., 0.], [2., 0.], [.2, .2]],
                         max_dimension=1, max_simplex_dimension=1)
    assert obj.native.dimension == 1
    assert obj.native.filtration((0, 1)) == pytest.approx(1.64)
    assert not obj.metadata["requested_death_dimension_available"]


def test_rips_scale_units_and_default_death_dimension():
    obj = sb.from_points(SQUARE, complex_type="rips")
    assert obj.native.dimension == 3
    assert obj.metadata["scale_units"] == "edge_length"
    np.testing.assert_allclose(diagram(st.persistence(obj), 1), [[1, np.sqrt(2)]])


def test_empty_cloud_and_empty_optional_eigenvectors():
    obj = sb.from_points(np.empty((0, 3)))
    assert st.homology(obj).betti_numbers == (0, 0, 0)
    assert st.persistence(obj).intervals == ()
    result = st.laplacian(obj, 2, return_eigenvectors=True, return_matrix=True)
    assert result.nullity == 0 and result.complete
    assert result.basis == ()
    assert result.eigenvectors.shape == result.matrix.shape == (0, 0)


@pytest.mark.parametrize("options", [
    {"filtration_start": -.1}, {"filtration_start": np.nan},
    {"filtration_start": .5, "max_scale": .4}, {"max_scale": True},
    {"max_scale": -np.inf}, {"max_dimension": True},
    {"max_simplex_dimension": -1}, {"weights": [1, 1, 1]},
])
def test_invalid_construction_contract(options):
    with pytest.raises((TypeError, ValueError)):
        sb.from_points(SQUARE, **options)


def test_distinct_cores_are_not_silently_reinterpreted():
    wrong = Topology("interaction", st.SimplicialComplex([(0, 1)]))
    for operation in [st.homology, st.persistence, st.laplacian]:
        with pytest.raises(ValueError, match="not interchangeable"):
            operation(wrong)


def test_alpha_vertex_budget_precedes_qhull(monkeypatch):
    from topokit.builders import _alpha_native as builders

    def forbidden(*args, **kwargs):
        raise AssertionError("Qhull should not run after a known vertex-budget failure")

    monkeypatch.setattr(builders, "Delaunay", forbidden)
    with pytest.raises(ResourceLimitError, match="before Delaunay"):
        sb.from_points(SQUARE, max_simplices=3)


def test_alpha_edges_use_midpoint_without_least_squares(monkeypatch):
    from topokit.builders import _simplicial as builders

    def forbidden(*args, **kwargs):
        raise AssertionError("Edge circumcenters do not need least squares")

    monkeypatch.setattr(builders.np.linalg, "lstsq", forbidden)
    obj = sb.from_points([[0.], [2.], [5.]])
    assert obj.native.filtration((0, 1)) == pytest.approx(1)
    assert obj.native.filtration((1, 2)) == pytest.approx(2.25)


def test_canonical_math_core_contains_no_constructor_or_outer_layer_imports():
    from topokit.core import _simplicial as native

    assert not hasattr(st, "from_points") and not hasattr(st, "from_graph")
    for name in ("alpha_complex", "rips_complex", "graph_complex", "build_complex", "analyze_points"):
        assert not hasattr(native, name)
    native_directory = Path(native.__file__).parent
    assert not (native_directory / "builders.py").exists()
    assert not (native_directory / "api.py").exists()
    forbidden = {"builders", "readers", "plots", "workflows", "features", "featurization"}
    for source in [Path(st.__file__), *native_directory.glob("*.py")]:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not forbidden.intersection((node.module or "").split(".")), source
            elif isinstance(node, ast.Import):
                assert not any(forbidden.intersection(alias.name.split("."))
                               for alias in node.names), source


def test_simplicial_core_runs_with_outer_layers_blocked():
    code = '''
import sys
import topokit
# Root convenience aliases are permitted; they are not dependencies of the
# mathematical core. Remove their imported builder dispatcher before checking.
blocked = ("topokit.builders", "topokit.readers", "topokit.plots",
           "topokit.workflows", "topokit.features", "topokit.featurization")
for name in tuple(sys.modules):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        del sys.modules[name]
class BlockOuterLayers:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise AssertionError("Core imported an outer layer: " + fullname)
sys.meta_path.insert(0, BlockOuterLayers())
from topokit.core import simplicial
obj = simplicial.SimplicialComplex([(0, 1), (1, 2), (0, 2)])
assert simplicial.homology(obj).betti_numbers == (1, 1, 0)
assert simplicial.laplacian(obj, 1).nullity == 1
assert simplicial.persistence(obj).max_dimension == 2
assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules for prefix in blocked)
'''
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-B", "-c", code], env=environment,
                   check=True, capture_output=True, text=True)
