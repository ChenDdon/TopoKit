"""Object views preserve cell membership, filtration, and stable identity."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from topokit import PointCloud
from topokit.builders import simplicial as simplex_builders, hyperdigraph as hyper_builders
from topokit.core.simplicial import SimplicialComplex
from topokit.core.hyperdigraph import Hyperdigraph, FilteredHyperdigraph
from topokit.exceptions import ResourceLimitError
from topokit.results import Topology
from topokit.visualization import topology as views


@pytest.fixture
def captured(monkeypatch):
    from topokit.visualization import objects

    calls = []

    def capture(cloud, cells, **options):
        axis = SimpleNamespace(cloud=cloud, cells=tuple(cells), options=options)
        calls.append(axis)
        return axis

    monkeypatch.setattr(objects, "plot_simplices", capture)
    monkeypatch.setattr(objects, "plot_hyperedges", capture)
    return calls


def test_built_simplex_view_maps_ids_and_keeps_required_cofaces(captured):
    cloud = PointCloud([[0, 0], [1, 0], [0, 1]], ids=["alpha", "beta", "gamma"])
    obj = simplex_builders.from_points(cloud, complex_type="flag", max_dimension=1,
                                      bonds=[("alpha", "beta"), ("beta", "gamma"), ("gamma", "alpha")])
    before = obj.native.get_filtration()
    metadata = deepcopy(obj.metadata)
    axis = views.plot_simplicial_complex(obj, dimensions=2, max_cells=1, max_scan_cells=1)
    assert axis.cells == (("alpha", "beta", "gamma"),)
    assert axis.cloud.ids == cloud.ids
    assert obj.native.get_filtration() == before
    assert obj.native.dimension == 2  # q+1 construction stays available.
    assert obj.metadata == metadata
    np.testing.assert_array_equal(obj.cloud.points, cloud.points)
    assert not obj.cloud.points.flags.writeable


def test_explicit_simplex_labels_are_not_coordinate_indices(captured):
    native = SimplicialComplex([(10, 42)])
    cloud = PointCloud([[3, 0], [0, 0]], ids=[42, 10])
    axis = views.plot_simplicial_complex(native, cloud=cloud, dimensions=1)
    assert axis.cells == ((10, 42),)
    assert axis.cloud.ids == (42, 10)
    np.testing.assert_array_equal(axis.cloud.points, [[3, 0], [0, 0]])


def test_simplicial_envelope_without_builder_mapping_preserves_explicit_ids(captured):
    native = SimplicialComplex([(4, 9)])
    cloud = PointCloud([[0, 0], [1, 0]], ids=[4, 9])
    assert views.plot_simplicial_complex(Topology("simplicial", native, cloud), dimensions=1).cells == ((4, 9),)


def test_coordinate_override_uses_stable_ids_after_reordering(captured):
    original = PointCloud([[0, 0], [1, 0]], ids=["a", "b"])
    obj = simplex_builders.from_points(original, complex_type="graph", bonds=[("a", "b")])
    override = PointCloud([[0, 8, 0], [0, 0, 0]], ids=["b", "a"])
    axis = views.plot_simplicial_complex(obj, cloud=override, dimensions=1)
    assert axis.cells == (("a", "b"),)
    assert axis.cloud.ids == ("b", "a")
    np.testing.assert_array_equal(obj.cloud.points, original.points)


@pytest.mark.parametrize("kind", ["simplicial", "hyperdigraph"])
def test_filtration_boundaries_are_inclusive_and_unborn_points_are_hidden(captured, kind):
    cloud = PointCloud([[0, 0], [1, 0], [2, 0]], ids=[0, 1, 2])
    if kind == "simplicial":
        native = SimplicialComplex({(0,): 1, (1,): 2, (0, 1): 2, (2,): 3})
        plot = views.plot_simplicial_complex
    else:
        native = FilteredHyperdigraph([0, 1, 2], [((0,), 1), ((1,), 2), ((0, 1), 2), ((2,), 3)])
        plot = views.plot_hyperdigraph
    obj = Topology(kind, native, cloud, {"filtration_start": 1, "filtration_end": 3})
    axis = plot(obj, scale=2)
    assert set(axis.cells) == {(0,), (1,), (0, 1)}
    assert axis.cloud.ids == (0, 1)
    assert plot(obj, scale=1).cells == ((0,),)
    assert plot(obj, scale=3).cloud.ids == (0, 1, 2)
    for scale in [0.99, 3.01]:
        with pytest.raises(ValueError, match="domain"):
            plot(obj, scale=scale)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf"), True, "1", 1j])
def test_invalid_scales_fail_before_native_read(captured, scale):
    obj = Topology("simplicial", object(), PointCloud([[0, 0]]))
    with pytest.raises(ValueError, match="scale"):
        views.plot_simplicial_complex(obj, scale=scale)
    assert captured == []


@pytest.mark.parametrize("dimensions", [True, -1, 0.5, "2", [0, -1], [1, True], {"dimension": 1}])
def test_invalid_dimension_filters_fail_before_native_read(captured, dimensions):
    obj = Topology("simplicial", object(), PointCloud([[0, 0]]))
    with pytest.raises(ValueError, match="dimensions"):
        views.plot_simplicial_complex(obj, dimensions=dimensions)
    assert captured == []


def test_empty_dimension_filter_is_an_empty_view(captured):
    native = SimplicialComplex([(0, 1)])
    axis = views.plot_simplicial_complex(native, cloud=PointCloud([[0, 0], [1, 0]]), dimensions=[])
    assert axis.cells == ()
    assert axis.cloud.points.shape == (0, 2)


def test_built_hyperdigraph_uses_string_ids_and_preserves_both_orders(captured):
    cloud = PointCloud([[0, 0], [1, 0], [2, 0]], ids=["left", "middle", "right"])
    obj = hyper_builders.from_points(cloud, max_dimension=1,
                                    bonds=[("left", "middle"), ("middle", "right")])
    before = obj.native.weighted_hyperedges()
    axis = views.plot_hyperdigraph(obj, dimensions=2)
    assert set(axis.cells) == {("left", "middle", "right"), ("right", "middle", "left")}
    assert obj.native.weighted_hyperedges() == before
    assert obj.native.max_dimension == 2


def test_static_hyperdigraph_never_synthesizes_subedges(captured):
    cloud = PointCloud([[0, 0], [1, 0], [1, 1]], ids=["a", "b", "c"])
    native = Hyperdigraph(cloud.ids, [("c", "a", "b")])
    axis = views.plot_hyperdigraph(native, cloud=cloud)
    assert axis.cells == (("c", "a", "b"),)
    assert native.directed_hyperedges(0) == ()
    assert native.directed_hyperedges(1) == ()
    with pytest.raises(ValueError, match="filtered"):
        views.plot_hyperdigraph(native, cloud=cloud, scale=0)


def test_max_cells_limits_selected_cells_separately_from_scanning(captured):
    native = FilteredHyperdigraph(["a", "b"], [(("a",), 0), (("b",), 3), (("a", "b"), 3)])
    cloud = PointCloud([[0, 0], [1, 0]], ids=["a", "b"])
    assert views.plot_hyperdigraph(native, cloud=cloud, scale=0, max_cells=1, max_scan_cells=3).cells == (("a",),)
    assert views.plot_hyperdigraph(native, cloud=cloud, scale=0, max_primitives=0).cells == (("a",),)
    with pytest.raises(ResourceLimitError, match="max_cells"):
        views.plot_hyperdigraph(native, cloud=cloud, max_cells=1)
    with pytest.raises(ResourceLimitError, match="max_scan_cells"):
        views.plot_hyperdigraph(native, cloud=cloud, scale=0, max_cells=1, max_scan_cells=2)


def test_discarded_infinite_record_stream_has_a_scan_bound(captured):
    reads = []

    class Stream:
        def weighted_hyperedges(self):
            while True:
                reads.append(True)
                yield SimpleNamespace(vertices=("a",), birth=3)

    with pytest.raises(ResourceLimitError, match="max_scan_cells"):
        views.plot_hyperdigraph(Stream(), cloud=PointCloud([[0, 0]], ids=["a"]),
                               scale=0, max_scan_cells=4)
    assert len(reads) == 5
    assert captured == []


@pytest.mark.parametrize("name", ["max_cells", "max_primitives", "max_scan_cells"])
@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_invalid_budgets_fail_before_native_access(captured, name, value):
    with pytest.raises(ValueError, match=name):
        views.plot_hyperdigraph(object(), cloud=PointCloud([[0, 0]]), **{name: value})
    assert captured == []


def test_missing_coordinates_and_mismatched_ids_are_actionable(captured):
    native = SimplicialComplex([(10, 20)])
    with pytest.raises(ValueError, match="cloud=PointCloud"):
        views.plot_simplicial_complex(native)
    with pytest.raises(ValueError, match="unknown point ID"):
        views.plot_simplicial_complex(native, cloud=PointCloud([[0, 0], [1, 0]]))
    with pytest.raises(ValueError, match="project explicitly"):
        views.plot_simplicial_complex(native, cloud=PointCloud(np.zeros((2, 4)), ids=[10, 20]))


def test_simplicial_public_filtration_fallback_supports_stable_string_ids(captured):
    native = SimpleNamespace(get_filtration=lambda: iter([(("a",), 0), (("b",), 0), (("a", "b"), 1)]))
    axis = views.plot_simplicial_complex(native, cloud=PointCloud([[0, 0], [1, 0]], ids=["a", "b"]),
                                         scale=0)
    assert axis.cells == (("a",), ("b",))


def test_dispatch_rejects_interaction_and_cross_family_use(captured):
    with pytest.raises(ValueError, match="plot_interaction_factors"):
        views.plot_topology(Topology("interaction", object()))
    with pytest.raises(ValueError, match="Expected a simplicial"):
        views.plot_simplicial_complex(Topology("hyperdigraph", object()))
    cloud = PointCloud([[0, 0], [1, 0]])
    assert views.plot_topology(Topology("simplicial", SimplicialComplex([(0, 1)]), cloud)).cells
    assert views.plot_topology(Hyperdigraph([0, 1], [(0, 1)]), cloud=cloud).cells == ((0, 1),)


@pytest.mark.parametrize("ambient_dimension", [2, 3])
def test_object_wrappers_render_on_real_axes(ambient_dimension):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5]])[:, :ambient_dimension]
    cloud = PointCloud(points)
    objects = [SimplicialComplex([(0, 1, 2)]), Hyperdigraph(cloud.ids, [(0, 1, 2)])]
    for obj in objects:
        axis = views.plot_topology(obj, cloud=cloud, labels=True)
        axis.figure.canvas.draw()
        assert axis.name == ("3d" if ambient_dimension == 3 else "rectilinear")
        plt.close(axis.figure)
