"""Semantic checks for reusable diagrams; no topology construction is needed."""

import itertools
import math

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topokit.visualization.blocks import (
    plot_building_blocks, plot_hyperdigraph_block, plot_simplex_block,
)


@pytest.fixture(autouse=True)
def close_test_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("dimension", range(5))
def test_simplex_block_contains_only_its_displayed_faces(dimension):
    axis = plot_simplex_block(dimension)
    view = axis._topokit_view
    assert view["cells"] == (tuple(range(dimension + 1)),)
    assert view["primitive_counts"]["edges"] == math.comb(dimension + 1, 2)
    assert view["primitive_counts"]["faces"] == math.comb(dimension + 1, 3)
    assert view["primitive_counts"]["arrows"] == 0
    nodes = next(artist for artist in axis.collections
                 if artist.get_gid() == "topokit-nodes")
    assert len(nodes.get_offsets()) == dimension + 1
    assert {str(label) for label in range(dimension + 1)} <= {
        text.get_text() for text in axis.texts}
    axis.figure.canvas.draw()


@pytest.mark.parametrize("dimension", range(5))
def test_hyperedge_block_has_consecutive_arrows_without_clique_completion(dimension):
    axis = plot_hyperdigraph_block(dimension)
    view = axis._topokit_view
    assert view["cells"] == (tuple(range(dimension + 1)),)
    assert view["arrows"] == tuple(zip(range(dimension), range(1, dimension + 1)))
    assert view["primitive_counts"]["arrows"] == dimension
    assert view["primitive_counts"]["ribbons"] == dimension
    assert view["primitive_counts"]["faces"] == 0
    arrows = [artist for artist in axis.get_children()
              if str(artist.get_gid()).startswith("topokit-arrow-")]
    assert len(arrows) == dimension
    axis.figure.canvas.draw()


def test_custom_hyperedge_order_preserves_ids_positions_and_input():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    original = points.copy()
    fig, axis = plt.subplots()
    returned = plot_hyperdigraph_block(2, ax=axis, points=points,
                                      ids=("left", "right", "top"),
                                      order=("top", "left", "right"))
    assert returned is axis
    assert axis._topokit_view["cells"] == (("top", "left", "right"),)
    assert axis._topokit_view["arrows"] == (("top", "left"), ("left", "right"))
    nodes = next(artist for artist in axis.collections
                 if artist.get_gid() == "topokit-nodes")
    np.testing.assert_array_equal(nodes.get_offsets(), points)
    np.testing.assert_array_equal(points, original)
    assert len(fig.axes) == 1


def test_projected_tetrahedron_and_opt_in_three_dimensional_axes():
    fig, axis = plt.subplots()
    assert plot_simplex_block(3, ax=axis) is axis
    assert axis._topokit_view["projection"] == "projected"
    axis3d = plot_simplex_block(3, view="3d")
    assert hasattr(axis3d, "get_zlim")
    assert axis3d._topokit_view["projection"] == "3d"
    fig.canvas.draw()
    axis3d.figure.canvas.draw()


@pytest.mark.parametrize("kind", ("simplicial", "hyperdigraph"))
def test_gallery_keeps_supplied_axes_and_requested_dimension_order(kind):
    fig, grid = plt.subplots(2, 2)
    before = tuple(plt.get_fignums())
    dimensions = (3, 1, 0, 2)
    returned_fig, axes = plot_building_blocks(kind, dimensions, axes=grid)
    assert returned_fig is fig
    assert axes.shape == (4,)
    assert all(actual is expected for actual, expected in zip(axes, grid.flat))
    assert tuple(plt.get_fignums()) == before
    assert [len(axis._topokit_view["cells"][0]) - 1 for axis in axes] == list(dimensions)
    fig.canvas.draw()


def test_gallery_supports_single_axis_repeated_dimensions_and_3d():
    fig, axis = plt.subplots()
    returned_fig, axes = plot_building_blocks(dimensions=(1,), axes=axis)
    assert returned_fig is fig and axes[0] is axis
    fig3d, axes3d = plot_building_blocks("hyperdigraph", (2, 2, 3), view="3d")
    assert all(hasattr(axis, "get_zlim") for axis in axes3d)
    assert [axis._topokit_view["cells"] for axis in axes3d] == [
        ((0, 1, 2),), ((0, 1, 2),), ((0, 1, 2, 3),)]
    fig3d.canvas.draw()


def test_gallery_frames_remain_aligned_for_point_segment_triangle_and_tetrahedron():
    fig, axes = plot_building_blocks()
    fig.canvas.draw()
    for axis in axes:
        assert np.diff(axis.get_xlim()) == pytest.approx(np.diff(axis.get_ylim()))
        assert axis.get_position().y0 == pytest.approx(axes[0].get_position().y0)
        assert axis.get_position().y1 == pytest.approx(axes[0].get_position().y1)


def test_schematic_high_dimension_captions_and_custom_title():
    assert "schematic 2-skeleton" in plot_simplex_block(4).get_title()
    assert "schematic ordered sequence" in plot_hyperdigraph_block(4).get_title()
    assert plot_simplex_block(2, title="Custom cell", labels=False).get_title() == "Custom cell"


def test_building_blocks_do_not_change_global_plot_style():
    keys = ("font.family", "font.size", "axes.prop_cycle", "svg.fonttype", "figure.dpi")
    previous = {key: matplotlib.rcParams[key] for key in keys}
    plot_building_blocks()
    assert {key: matplotlib.rcParams[key] for key in keys} == previous


@pytest.mark.parametrize("plot", (plot_simplex_block, plot_hyperdigraph_block))
@pytest.mark.parametrize("dimension,error", ((-1, ValueError), (1.5, TypeError),
                                             (True, TypeError), (32, ValueError)))
def test_invalid_dimension_is_rejected_before_creating_figure(plot, dimension, error):
    before = tuple(plt.get_fignums())
    with pytest.raises(error):
        plot(dimension)
    assert tuple(plt.get_fignums()) == before


def test_large_dimension_and_infinite_gallery_stop_at_resource_guards(monkeypatch):
    import topokit.visualization.blocks as blocks

    def forbidden_layout(dimension):
        raise AssertionError("layout allocated before resource check")

    monkeypatch.setattr(blocks, "_layout", forbidden_layout)
    with pytest.raises(ValueError, match="max_vertices"):
        plot_simplex_block(10**9)
    with pytest.raises(ValueError, match="max_vertices"):
        plot_simplex_block(np.int64(np.iinfo(np.int64).max))
    with pytest.raises(ValueError, match="max_blocks"):
        plot_building_blocks(dimensions=itertools.repeat(0))


@pytest.mark.parametrize("options", (
    {"points": [[0, 0], [1, 0]]},
    {"points": [[0], [1], [2]]},
    {"ids": ("a", "a", "b")},
    {"ids": itertools.repeat("a")},
    {"order": (0, 1, 1)},
    {"order": (0, 1, 4)},
    {"order": (0, 1)},
))
def test_inconsistent_geometry_ids_or_order_rejected(options):
    with pytest.raises(ValueError):
        plot_hyperdigraph_block(2, **options)


def test_invalid_gallery_axes_or_kind_rejected():
    fig, axes = plt.subplots(1, 2)
    another_fig, another_axis = plt.subplots()
    with pytest.raises(ValueError, match="same figure"):
        plot_building_blocks(dimensions=(1, 2), axes=[axes[0], another_axis])
    with pytest.raises(ValueError, match="exactly one"):
        plot_building_blocks(dimensions=(1,), axes=axes)
    with pytest.raises(ValueError, match="kind"):
        plot_building_blocks(kind="directed_clique")
    with pytest.raises(ValueError, match="at least one"):
        plot_building_blocks(dimensions=())
    assert len(fig.axes) == 2 and len(another_fig.axes) == 1
