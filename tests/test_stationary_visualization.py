"""Reusable fixed-scale analysis and presentation APIs."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from topokit import PointCloud, SpectrumResult
from topokit.builders import interaction, simplicial
from topokit.core.hyperdigraph import Hyperdigraph
from topokit.postprocessing import spectral_energy, summarize_spectrum
from topokit.results import Topology
from topokit.workflows import analyze_stationary


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
    return PointCloud(
        [[0, 0], [1, 0], [0.4, 0.8], [1.3, 0.9]],
        ids=("a", "b", "c", "d"), weights=(0.2, 0.2, 0.8, 0.6),
        metadata={"coordinate_units": "units"},
    )


def test_energy_is_a_builtin_spectral_summary():
    assert spectral_energy([-2, 0, 3]) == 5
    result = summarize_spectrum(
        SpectrumResult(0, np.array([0., 1., 3.])),
        statistics=("min", "energy"), positive_only=False,
    )
    assert result.names == ("L0:min", "L0:energy")
    np.testing.assert_allclose(result.values, [0, 4])


def test_stationary_workflow_returns_explicit_homology_spectra_and_rows(cloud):
    graph = simplicial.from_points(
        cloud, max_dimension=0, complex_type="graph",
        bonds=[("a", "b"), ("b", "c")], cutoff=2, max_scale=2,
    )
    result = analyze_stationary(graph, scale=2, dimensions=(0,))
    assert result.topology is graph
    assert result.scale == 2
    assert result.dimensions == (0,)
    assert result.homology.betti_numbers == (2,)
    assert tuple(result.spectra) == (0,)
    assert result.spectra[0].complete
    assert result.summaries[0]["zero_count"] == 2
    assert result.summaries[0]["energy"] == pytest.approx(
        np.abs(result.spectra[0].eigenvalues).sum()
    )
    rows = result.summary_rows((0, 1))
    assert rows[0]["dimension"] == 0
    assert rows[1]["dimension"] == 1
    assert all(np.isnan(rows[1][name]) for name in result.metadata["summary_statistics"])


@pytest.mark.parametrize("dimensions", [(), (0, 0), (-1,), (True,), "01"])
def test_stationary_workflow_rejects_invalid_dimensions(cloud, dimensions):
    graph = simplicial.from_points(cloud, max_dimension=0, complex_type="graph", bonds=[])
    with pytest.raises(ValueError, match="dimensions"):
        analyze_stationary(graph, dimensions=dimensions)


def test_stationary_graph_and_simplicial_views_are_distinct(plt, cloud):
    from topokit import visualization as viz
    bonds = [("a", "b"), ("b", "c"), ("c", "a")]
    graph = simplicial.from_points(
        cloud, max_dimension=0, complex_type="graph", bonds=bonds,
        cutoff=2, max_scale=2,
    )
    complex_ = simplicial.from_points(
        cloud, max_dimension=1, complex_type="flag", bonds=bonds,
        cutoff=2, max_scale=2,
    )
    graph_axis = viz.plot_stationary_graph(graph, scale=2)
    simplex_axis = viz.plot_stationary_simplicial(
        complex_, scale=2, proximity_radius=0.5,
    )
    assert graph_axis._topokit_view["kind"] == "graph"
    assert graph_axis._topokit_view["proximity_circles"] is False
    assert graph_axis._topokit_view["primitive_counts"]["faces"] == 0
    assert not [patch for patch in graph_axis.patches
                if (patch.get_gid() or "").startswith("topokit-proximity")]
    assert simplex_axis._topokit_view["kind"] == "simplicial"
    assert len(simplex_axis._topokit_view["cells_by_dimension"][2]) == 1
    assert simplex_axis._topokit_view["proximity_circle_count"] == len(cloud)
    assert len([patch for patch in simplex_axis.patches
                if (patch.get_gid() or "").startswith("topokit-proximity")]) == 2 * len(cloud)
    for axis in (graph_axis, simplex_axis):
        axis.figure.canvas.draw()
        assert not any(line.get_visible() for line in axis.get_xgridlines() + axis.get_ygridlines())


def test_layered_hyperdigraph_deduplicates_segments_and_preserves_cells(plt, cloud):
    from topokit import visualization as viz
    cells = [
        *(tuple([point]) for point in cloud.ids),
        ("a", "b"), ("b", "a"),
        ("a", "b", "c"), ("a", "b", "d"),
    ]
    native = Hyperdigraph(cloud.ids, cells)
    before = deepcopy(native.directed_hyperedges())
    topology = Topology("hyperdigraph", native, cloud)
    axis = viz.plot_stationary_hyperdigraph(
        topology, proximity_radius=0.4, colorbar=False,
        annotate_weights=("a", "b"),
    )
    view = axis._topokit_view
    assert native.directed_hyperedges() == before
    assert view["arrows_by_dimension"][1] == (("a", "b"), ("b", "a"))
    assert view["arrows_by_dimension"][2] == (
        ("a", "b"), ("b", "c"), ("b", "d")
    )
    assert view["max_connections_per_support"] == {1: 2, 2: 1, "total": 3}
    blue = [patch for patch in axis.patches
            if (patch.get_gid() or "").startswith("topokit-stationary-directed-1")]
    orange = [patch for patch in axis.patches
              if (patch.get_gid() or "").startswith("topokit-stationary-directed-2")]
    assert len(blue) == 2 and len(orange) == 3
    assert {patch.get_connectionstyle().rad for patch in blue + orange} == {0.09}
    assert min(patch.get_linewidth() for patch in orange) > max(
        patch.get_linewidth() for patch in blue
    )
    assert max(patch.get_zorder() for patch in orange) < min(
        patch.get_zorder() for patch in blue
    )


def test_hyperdigraph_display_count_is_explicit_and_deterministic(plt, cloud):
    from topokit import visualization as viz
    native = Hyperdigraph(cloud.ids, [
        *(tuple([point]) for point in cloud.ids),
        ("a", "b", "c"), ("a", "b", "d"), ("c", "b", "d"),
    ])
    axis = viz.plot_stationary_hyperdigraph(
        Topology("hyperdigraph", native, cloud), display_counts={2: 2},
    )
    view = axis._topokit_view
    assert len(view["source_cells_by_dimension"][2]) == 3
    assert view["cells_by_dimension"][2] == (
        ("a", "b", "c"), ("c", "b", "d")
    )


def test_interaction_view_uses_retained_factors_and_explicit_overlap(plt, cloud):
    from topokit import visualization as viz
    bonds = [("a", "b"), ("b", "c"), ("c", "a")]
    topology = interaction.from_points(
        cloud, max_dimension=1, max_scale=2,
        factor_max_dimensions=(2, 2),
        factor_options={"complex_type": "flag", "bonds": bonds, "cutoff": 2},
    )
    before = deepcopy(topology.metadata["source_factor_filtrations"])
    figure, axes = viz.plot_stationary_interaction(
        topology, scale=2, proximity_radius=0.5, limits_margin=0.7,
    )
    assert topology.metadata["source_factor_filtrations"] == before
    assert figure._topokit_view["kind"] == "interaction"
    assert figure._topokit_view["overlap_pairs"] == tuple((item, item) for item in cloud.ids)
    assert all(axis._topokit_view["proximity_circle_count"] == len(cloud)
               for axis in axes)
    assert all(axis._topokit_view["overlap_ids"] == cloud.ids for axis in axes)
    figure.canvas.draw()


def test_stationary_dispatch_and_diagnostics(plt, cloud):
    from topokit import visualization as viz
    graph = simplicial.from_points(
        cloud, max_dimension=0, complex_type="graph",
        bonds=[("a", "b")], cutoff=2, max_scale=2,
    )
    axis = viz.plot_stationary(graph, scale=2)
    assert axis._topokit_view["kind"] == "graph"
    result = analyze_stationary(graph, scale=2, dimensions=(0,))
    figure, axes = viz.plot_stationary_diagnostics(
        result, title="Graph Representation", display_dimensions=(0, 1),
    )
    assert figure._topokit_view["kind"] == "stationary_diagnostics"
    assert "not requested" in axes[1, 0].texts[0].get_text()
    figure.canvas.draw()


def test_stationary_visualization_import_remains_lazy():
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class NoPlotDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
                    raise ImportError('matplotlib deliberately unavailable')
        sys.meta_path.insert(0, NoPlotDependency())
        from topokit.visualization import StationaryStyle, plot_stationary
        assert StationaryStyle().weight_cmap == 'cividis'
        assert not any(name.startswith('matplotlib') for name in sys.modules)
    """)
    environment = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
