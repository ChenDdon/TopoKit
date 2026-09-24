"""Layer 5: independent displays. Numerical results remain authoritative."""
from .plots import (plot_points, plot_simplices, plot_hyperedges, plot_interaction_factors,
                    plot_barcodes, plot_betti_curves, plot_spectral_summary_series,
                    plot_spectrum, plot_features, plot_eigenvector)
from .style import PlotStyle
from .blocks import plot_simplex_block, plot_hyperdigraph_block, plot_building_blocks
from .topology import plot_simplicial_complex, plot_hyperdigraph, plot_topology
from .export import save_figure
from .stationary import (
    StationaryStyle, clean_axis, empty_axis, add_panel_label,
    set_point_limits, add_proximity_circles, unique_directed_segments,
    maximum_connections_per_support, plot_stationary_graph,
    plot_stationary_simplicial, plot_stationary_hyperdigraph,
    plot_stationary_interaction, plot_stationary,
    plot_stationary_diagnostics,
)


def plot_graph(cloud, edges, **options):
    """Display an explicit undirected graph without implicit clique expansion."""
    return plot_simplices(cloud, edges, **options)


__all__ = ["plot_points", "plot_simplices", "plot_hyperedges", "plot_interaction_factors",
           "plot_graph", "plot_barcodes", "plot_betti_curves",
           "plot_spectral_summary_series", "plot_spectrum", "plot_features", "plot_eigenvector",
           "PlotStyle", "plot_simplicial_complex", "plot_hyperdigraph", "plot_topology",
           "plot_simplex_block", "plot_hyperdigraph_block", "plot_building_blocks", "save_figure"]
__all__ += [
    "StationaryStyle", "clean_axis", "empty_axis", "add_panel_label",
    "set_point_limits", "add_proximity_circles", "unique_directed_segments",
    "maximum_connections_per_support", "plot_stationary_graph",
    "plot_stationary_simplicial", "plot_stationary_hyperdigraph",
    "plot_stationary_interaction", "plot_stationary",
    "plot_stationary_diagnostics",
]
