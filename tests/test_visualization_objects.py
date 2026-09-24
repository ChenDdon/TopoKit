"""Rendering contracts: display explicit cells without changing their topology."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from topokit.data import PointCloud


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot
    before = set(pyplot.get_fignums())
    yield pyplot
    for number in set(pyplot.get_fignums()) - before:
        pyplot.close(number)


@pytest.fixture
def cloud():
    # Mixed stable IDs catch algorithms that sort IDs or treat them as rows.
    return PointCloud(
        [[0.0, 0.0], [1.5, 0.0], [0.4, 1.2], [1.2, 1.1]],
        ids=("left", 9, "top", 2), weights=(0.3, 0.1, 0.9, 0.7),
        metadata={"units": "nm", "labels": ("A", "B", "C", "D")},
    )


def _renderer(name):
    from topokit import visualization
    return getattr(visualization, name)


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
@pytest.mark.parametrize("cells", (
    [()], [("left", "left")], [("left", "missing")],
    {2: [("left", 9)]}, {-1: [("left",)]}, {None: [("left", 9)]},
))
def test_invalid_cells_fail_before_drawing(plt, cloud, name, cells):
    _, ax = plt.subplots()
    ax.plot([0, 1], [1, 0], color="red")
    old_children = tuple(ax.get_children())
    old_title = ax.get_title()
    old_limits = ax.get_xlim(), ax.get_ylim()
    with pytest.raises((ValueError, TypeError)):
        _renderer(name)(cloud, cells, ax=ax)
    assert tuple(ax.get_children()) == old_children
    assert ax.get_title() == old_title
    assert (ax.get_xlim(), ax.get_ylim()) == old_limits


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
def test_invalid_input_does_not_create_a_figure(plt, cloud, name):
    figures = plt.get_fignums()
    with pytest.raises((ValueError, TypeError)):
        _renderer(name)(cloud, [("left", "missing")])
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
def test_max_cells_stops_consuming_a_stream(plt, cloud, name):
    consumed = []

    def cells():
        for index in range(4):
            consumed.append(index)
            yield ("left", 9)
        raise AssertionError("max_cells must stop an unbounded input stream")

    before = plt.get_fignums()
    with pytest.raises(ValueError, match="max_cells|[Tt]oo many|subset"):
        _renderer(name)(cloud, cells(), max_cells=3)
    assert len(consumed) == 4
    assert plt.get_fignums() == before


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
def test_primitive_budget_fails_before_drawing(plt, name):
    points = np.column_stack((np.arange(20), np.arange(20) % 3))
    _, ax = plt.subplots()
    before = tuple(ax.get_children())
    with pytest.raises(ValueError, match="max_primitives|[Pp]rimitive|subset"):
        _renderer(name)(points, [tuple(range(20))], ax=ax, max_primitives=10)
    assert tuple(ax.get_children()) == before


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
def test_dimension_mapping_and_inputs_are_preserved(plt, cloud, name):
    cells = {0: [("left",)], 1: [["left", 9]], 2: [("left", 9, "top")]}
    original = deepcopy(cells)
    points, weights = cloud.points.copy(), cloud.weights.copy()
    metadata = deepcopy(cloud.metadata)
    ax = _renderer(name)(cloud, cells, labels=True)
    assert cells == original
    np.testing.assert_array_equal(cloud.points, points)
    np.testing.assert_array_equal(cloud.weights, weights)
    assert cloud.metadata == metadata
    assert not cloud.points.flags.writeable
    assert not cloud.weights.flags.writeable
    assert set(ax._topokit_view["cells"]) == {
        ("left",), ("left", 9), ("left", 9, "top")
    }
    assert {"left", "9", "top", "2"} <= {text.get_text() for text in ax.texts}


def test_a_graph_triangle_does_not_acquire_a_filled_simplex(plt, cloud):
    ax = _renderer("plot_simplices")(
        cloud, [("left", 9), (9, "top"), ("top", "left")]
    )
    view = ax._topokit_view
    assert view["kind"] == "simplicial"
    assert view["primitive_counts"]["edges"] == 3
    assert view["primitive_counts"]["faces"] == 0
    assert view["faces"] == ()
    assert not any(artist.get_gid() == "topokit-faces" for artist in ax.collections)


def test_explicit_tetrahedron_displays_its_six_edges_and_four_faces(plt, cloud):
    supplied = ("left", 9, "top", 2)
    ax = _renderer("plot_simplices")(cloud, [supplied, ("left", 9, "top")])
    view = ax._topokit_view
    assert view["cells"] == (supplied, ("left", 9, "top"))
    assert view["primitive_counts"]["edges"] == 6
    assert view["primitive_counts"]["faces"] == 4
    assert {frozenset(face) for face in view["faces"]} == {
        frozenset(("left", 9, "top")), frozenset(("left", 9, 2)),
        frozenset(("left", "top", 2)), frozenset((9, "top", 2)),
    }
    faces = [artist for artist in ax.collections if artist.get_gid() == "topokit-faces"]
    edges = [artist for artist in ax.collections if artist.get_gid() == "topokit-edges"]
    assert len(faces) == len(edges) == 1
    assert len(faces[0].get_paths()) == 4
    assert len(edges[0].get_segments()) == 6


@pytest.mark.parametrize("degree", (1, 2, 3))
def test_hyperedge_uses_only_consecutive_arrows(plt, cloud, degree):
    cell = cloud.ids[:degree + 1]
    ax = _renderer("plot_hyperedges")(cloud, [cell])
    view = ax._topokit_view
    assert view["kind"] == "hyperdigraph"
    assert view["cells"] == (cell,)
    assert view["arrows"] == tuple(zip(cell[:-1], cell[1:]))
    assert view["primitive_counts"]["arrows"] == degree
    assert view["primitive_counts"]["faces"] == 0
    arrows = [patch for patch in ax.patches
              if (patch.get_gid() or "").startswith("topokit-arrow-")]
    assert len(arrows) == degree
    assert {patch.get_gid() for patch in arrows} == {
        f"topokit-arrow-0-{segment}" for segment in range(degree)
    }


def test_reciprocal_hyperedges_remain_distinct_in_data_and_geometry(plt, cloud):
    cells = [("left", 9), (9, "left")]
    ax = _renderer("plot_hyperedges")(cloud, cells)
    assert ax._topokit_view["cells"] == tuple(cells)
    assert ax._topokit_view["arrows"] == tuple(cells)
    ax.figure.canvas.draw()
    arrows = [patch for patch in ax.patches
              if (patch.get_gid() or "").startswith("topokit-arrow-")]
    assert len(arrows) == 2
    # Compare shaft midpoints rather than arrow heads: reversing an arrow head
    # alone must not count as separating two otherwise overlapping supports.
    shaft_heights = [np.mean(patch.get_path().vertices[:3, 1]) for patch in arrows]
    assert abs(shaft_heights[0] - shaft_heights[1]) > 1e-5


def test_duplicate_hyperedges_keep_separate_rendered_lanes(plt, cloud):
    cells = [("left", 9), ("left", 9)]
    ax = _renderer("plot_hyperedges")(cloud, cells)
    assert ax._topokit_view["cells"] == tuple(cells)
    assert ax._topokit_view["primitive_counts"]["arrows"] == 2
    ax.figure.canvas.draw()
    arrows = [patch for patch in ax.patches
              if (patch.get_gid() or "").startswith("topokit-arrow-")]
    assert len(arrows) == 2
    assert not np.allclose(arrows[0].get_path().vertices, arrows[1].get_path().vertices)


def test_style_can_be_reused_without_global_matplotlib_changes(plt, cloud):
    from topokit.visualization import PlotStyle
    style = PlotStyle(node_size=71, node_color="#a53142", linewidth=2.2)
    old_rcparams = dict(plt.rcParams)
    ax = _renderer("plot_simplices")(cloud, [("left", 9, "top")], style=style)
    assert dict(plt.rcParams) == old_rcparams
    nodes = next(artist for artist in ax.collections if artist.get_gid() == "topokit-nodes")
    np.testing.assert_array_equal(nodes.get_sizes(), [71])
    edges = next(artist for artist in ax.collections if artist.get_gid() == "topokit-edges")
    np.testing.assert_allclose(edges.get_linewidths(), [2.2])
    with pytest.raises(FrozenInstanceError):
        style.node_size = 12


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
@pytest.mark.parametrize("ambient,view", (
    (2, "auto"), (2, "3d"), (3, "3d"), (3, "projected"),
))
def test_rendering_exports_png_and_svg(plt, name, ambient, view):
    points = np.array([[0, 0, 0], [1.5, 0, 0], [0.6, 1.2, 0], [0.8, 0.4, 1.1]])
    ax = _renderer(name)(
        points[:, :ambient], [(0, 1, 2, 3)], view=view,
        labels=True, title="Explicit cell", elev=24, azim=-40,
    )
    assert hasattr(ax, "get_zlim") == (view == "3d")
    assert ax._topokit_view["projection"] == ("2d" if view == "auto" else view)
    assert ax.get_title() == "Explicit cell"
    ax.figure.canvas.draw()
    for format_, signature in (("png", b"\x89PNG\r\n\x1a\n"), ("svg", b"<?xml")):
        buffer = BytesIO()
        ax.figure.savefig(buffer, format=format_, dpi=80)
        assert buffer.getvalue().startswith(signature)
        assert len(buffer.getvalue()) > 1000


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
def test_view_does_not_infer_higher_dimensional_embedding(plt, name):
    with pytest.raises(ValueError, match="2D|3D|dimension|project"):
        _renderer(name)(np.zeros((3, 4)), [(0, 1, 2)])


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
@pytest.mark.parametrize("ambient,view", ((2, "auto"), (3, "3d"), (3, "projected")))
def test_empty_objects_render_without_invented_cells(plt, name, ambient, view):
    ax = _renderer(name)(np.empty((0, ambient)), [], view=view)
    assert ax._topokit_view["cells"] == ()
    assert sum(ax._topokit_view["primitive_counts"].values()) == 0
    ax.figure.canvas.draw()


@pytest.mark.parametrize("name", ("plot_simplices", "plot_hyperedges"))
@pytest.mark.parametrize("axis_projection,view", ((None, "3d"), ("3d", "projected")))
def test_mismatched_axes_are_rejected_before_drawing(plt, name, axis_projection, view):
    ax = plt.figure().add_subplot(projection=axis_projection)
    before = tuple(ax.get_children())
    with pytest.raises(ValueError, match="axis|axes|3D|2D"):
        _renderer(name)([[0, 0, 0], [1, 1, 1]], [(0, 1)], ax=ax, view=view)
    assert tuple(ax.get_children()) == before


def test_plotting_api_imports_without_matplotlib():
    """The optional dependency is required only when a render is requested."""
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class NoPlotDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
                    raise ImportError('matplotlib deliberately unavailable')
        sys.meta_path.insert(0, NoPlotDependency())
        from topokit.visualization import PlotStyle, plot_simplices, plot_hyperedges
        from topokit.visualization import blocks
        assert not any(name.startswith('matplotlib') for name in sys.modules)
        for renderer in (plot_simplices, plot_hyperedges):
            try:
                renderer([[0, 0], [1, 0]], [(0, 1)])
            except ImportError as error:
                assert 'topokit[plot]' in str(error)
            else:
                raise AssertionError('plotting should request its optional dependency')
    """)
    environment = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
