"""Simple fixed-scale views for TopoKit objects and numerical results.

These functions compose existing explicit-cell renderers. They select only
supplied cells at an explicit scale, never construct topology, and never run
homology or Laplacian algorithms. Animation remains an example-level concern.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from numbers import Integral, Real

import numpy as np

from .._validation import dimension as _dimension, validate_query
from ..data import PointCloud, as_point_cloud
from ..exceptions import ResourceLimitError
from ..results import StationaryResult, Topology
from .plots import _plt
from .style import PlotStyle
from .topology import _select_hyperdigraph_view, _select_simplicial_view


@dataclass(frozen=True)
class StationaryStyle:
    """Reusable colors and sizes for compact fixed-scale demonstrations.

    Importing this record does not load Matplotlib or change global ``rcParams``.
    Per-call ``PlotStyle`` overrides remain available on the object renderers.
    """

    point_color: str = "#272727"
    edge_color: str = "#0F4D92"
    face_color: str = "#75B878"
    higher_edge_color: str = "#EC8B45"
    mean_color: str = "#75B878"
    annotation_color: str = "#767676"
    highlight_color: str = "#B64342"
    mid_gray: str = "#A7A7A7"
    light_color: str = "#F4F6F8"
    circle_color: str = "#9EC3E6"
    weight_cmap: str = "cividis"
    node_size: float = 28.0
    node_linewidth: float = 0.45
    edge_width: float = 0.75
    face_alpha: float = 0.34
    arrow_width: float = 0.95
    higher_edge_width: float = 3
    higher_edge_alpha: float = 1
    arrow_size: float = 7.3
    curvature: float = 0.09
    circle_face_alpha: float = 0.018
    circle_edge_alpha: float = 0.12
    circle_linewidth: float = 0.40
    overlap_size: float = 52.0
    overlap_linewidth: float = 1.0
    padding: float = 0.10

    def __post_init__(self):
        alpha_names = {"face_alpha", "higher_edge_alpha", "circle_face_alpha",
                       "circle_edge_alpha"}
        color_names = {name for name in self.__dataclass_fields__
                       if name.endswith("_color") or name in {"mid_gray", "weight_cmap"}}
        for item in fields(self):
            if item.name in color_names:
                value = getattr(self, item.name)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{item.name} must be a nonempty color or colormap name")
                continue
            value = getattr(self, item.name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"{item.name} must be finite and nonnegative")
            if item.name in alpha_names and value > 1:
                raise ValueError(f"{item.name} must lie in [0, 1]")
        for name in ("node_size", "arrow_size", "overlap_size"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")

    @property
    def palette(self):
        """Return a fresh shorthand color mapping for custom companion plots."""
        return {
            "blue": self.edge_color,
            "green": self.face_color,
            "teal": self.mean_color,
            "orange": self.higher_edge_color,
            "red": self.highlight_color,
            "dark": self.point_color,
            "gray": self.annotation_color,
            "mid_gray": self.mid_gray,
            "light": self.light_color,
            "blue_soft": self.circle_color,
        }


def _resolve_style(style):
    value = StationaryStyle() if style is None else style
    if not isinstance(value, StationaryStyle):
        raise TypeError("style must be a StationaryStyle")
    return value


def _optional_nonnegative(value, name):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _degrees(value, *, maximum=2):
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        result = (int(value),)
    elif isinstance(value, (str, bytes, Mapping, bool, np.bool_)):
        raise ValueError("dimensions must contain unique nonnegative integers")
    else:
        try:
            result = tuple(value)
        except TypeError as error:
            raise ValueError("dimensions must contain unique nonnegative integers") from error
    if (any(isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral)
            or q < 0 or q > maximum for q in result) or len(set(result)) != len(result)):
        raise ValueError(f"dimensions must contain unique integers from 0 through {maximum}")
    return tuple(sorted(map(int, result)))


def clean_axis(ax, *, color="#272727"):
    """Apply local clean-axis styling without changing global Matplotlib state."""
    ax.tick_params(direction="out", length=3, width=0.6, colors=color)
    ax.grid(False)
    return ax


def empty_axis(ax, message, *, color="#767676", title=None):
    """Mark an intentionally empty panel while retaining a stable layout."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, fontsize=7.5,
            ha="center", va="center", color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def add_panel_label(ax, label, *, color="#272727"):
    """Add a compact bold panel label and return its text artist."""
    return ax.text(-0.11, 1.035, str(label), transform=ax.transAxes,
                   fontsize=10, fontweight="bold", color=color,
                   ha="left", va="bottom")


def _cloud_sequence(clouds):
    if isinstance(clouds, PointCloud):
        return (clouds,)
    if (isinstance(clouds, (tuple, list)) and clouds
            and all(isinstance(cloud, PointCloud) for cloud in clouds)):
        return tuple(clouds)
    return (as_point_cloud(clouds),)


def set_point_limits(ax, clouds, *, margin=0.12):
    """Set equal 2D limits from one cloud or a sequence of reference clouds."""
    margin = _optional_nonnegative(margin, "margin")
    if margin is None:
        raise ValueError("margin must be finite and nonnegative")
    records = _cloud_sequence(clouds)
    if any(cloud.points.shape[1] != 2 for cloud in records):
        raise ValueError("set_point_limits requires explicit 2D coordinates")
    arrays = [cloud.points for cloud in records if len(cloud)]
    if arrays:
        points = np.vstack(arrays)
        low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    return ax


def add_proximity_circles(ax, cloud, radius, *, style=None,
                          face_alpha=None, edge_alpha=None, linewidth=None):
    """Draw one exact-radius background circle around each supplied 2D point."""
    cloud = as_point_cloud(cloud)
    if cloud.points.shape[1] != 2:
        raise ValueError("proximity circles require explicit 2D coordinates")
    radius = _optional_nonnegative(radius, "radius")
    if radius is None:
        raise ValueError("radius must be finite and nonnegative")
    style = _resolve_style(style)
    face_alpha = style.circle_face_alpha if face_alpha is None else _optional_nonnegative(
        face_alpha, "face_alpha")
    edge_alpha = style.circle_edge_alpha if edge_alpha is None else _optional_nonnegative(
        edge_alpha, "edge_alpha")
    linewidth = style.circle_linewidth if linewidth is None else _optional_nonnegative(
        linewidth, "linewidth")
    if face_alpha > 1 or edge_alpha > 1:
        raise ValueError("circle alpha values must lie in [0, 1]")
    from matplotlib.patches import Circle
    artists = []
    for index, point in enumerate(cloud.points):
        fill = Circle(point, radius, facecolor=style.circle_color,
                      edgecolor="none", alpha=face_alpha, zorder=0)
        fill.set_gid(f"topokit-proximity-fill-{index}")
        ax.add_patch(fill)
        artists.append(fill)
    for index, point in enumerate(cloud.points):
        outline = Circle(point, radius, fill=False, edgecolor=style.edge_color,
                         linewidth=linewidth, alpha=edge_alpha, zorder=0)
        outline.set_gid(f"topokit-proximity-outline-{index}")
        ax.add_patch(outline)
        artists.append(outline)
    return tuple(artists)


def unique_directed_segments(hyperedges):
    """Return each consecutive oriented segment once, preserving first use."""
    seen, segments = set(), []
    for hyperedge in hyperedges:
        for start, end in zip(hyperedge[:-1], hyperedge[1:]):
            segment = (start, end)
            if segment not in seen:
                seen.add(segment)
                segments.append(segment)
    return tuple(segments)


def maximum_connections_per_support(*segment_layers):
    """Count the most displayed arrows over any unordered point pair."""
    counts = {}
    for segments in segment_layers:
        for start, end in segments:
            support = frozenset((start, end))
            counts[support] = counts.get(support, 0) + 1
    return max(counts.values(), default=0)


def _plot_style(style, *, node_size=None, edge_width=None, face_alpha=None):
    return PlotStyle(
        node_size=style.node_size if node_size is None else node_size,
        node_color=style.point_color,
        node_edge_color="white",
        node_linewidth=style.node_linewidth,
        linewidth=style.edge_width if edge_width is None else edge_width,
        edge_color=style.edge_color,
        face_alpha=style.face_alpha if face_alpha is None else face_alpha,
        arrow_size=style.arrow_size,
        arrow_color=style.edge_color,
        curvature=style.curvature,
        padding=style.padding,
    )


def _cells_by_dimension(cells):
    groups = {}
    for cell in cells:
        groups.setdefault(len(cell) - 1, []).append(tuple(cell))
    return {q: tuple(group) for q, group in groups.items()}


def _basic_legend(ax, style, entries, *, fontsize=7.0, loc="upper left"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = []
    for kind, label in entries:
        if kind == "point":
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markerfacecolor=style.point_color,
                                  markeredgecolor="white", markersize=6, label=label))
        elif kind == "edge":
            handles.append(Line2D([], [], color=style.edge_color,
                                  linewidth=1.35, label=label))
        elif kind == "face":
            handles.append(Patch(facecolor=style.face_color,
                                 alpha=style.face_alpha, label=label))
        elif kind == "higher":
            handles.append(Line2D([], [], color=style.higher_edge_color,
                                  linewidth=style.higher_edge_width,
                                  alpha=style.higher_edge_alpha, label=label))
        elif kind == "overlap":
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markerfacecolor="none",
                                  markeredgecolor=style.higher_edge_color,
                                  markeredgewidth=style.overlap_linewidth,
                                  label=label))
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=fontsize, frameon=False)


def plot_stationary_graph(topology, *, scale=None, ax=None, title=None,
                          style=None, legend=True, show_axes=True,
                          limits_margin=None, max_cells=1000,
                          max_primitives=20_000, max_scan_cells=1_000_000,
                          node_size=None, edge_width=None):
    """Plot the explicit 0/1-dimensional graph view of a simplicial topology.

    Higher-dimensional simplices are not displayed or filled. Proximity circles
    are intentionally absent from this graph representation.
    """
    style = _resolve_style(style)
    options = {"max_cells": max_cells, "max_primitives": max_primitives}
    cloud, cells, value, source_metadata = _select_simplicial_view(
        topology, scale=scale, dimensions=(0, 1), max_scan_cells=max_scan_cells,
        options=options)
    from .objects import plot_simplices
    label = title if title is not None else (
        "Graph Representation" if value is None
        else f"Graph Representation (cutoff = {value:g})"
    )
    axis = plot_simplices(
        cloud, cells, ax=ax, labels=False, show_axes=show_axes, legend=False,
        title=label, style=_plot_style(style, node_size=node_size,
                                      edge_width=edge_width),
        colors={1: style.edge_color}, max_cells=max_cells,
        max_primitives=max_primitives,
    )
    clean_axis(axis, color=style.point_color)
    if limits_margin is not None:
        set_point_limits(axis, cloud, margin=limits_margin)
    if legend:
        _basic_legend(axis, style, (("point", "point"), ("edge", "edge")))
    metadata = dict(axis._topokit_view)
    metadata.update({
        "kind": "graph", "scale": value,
        "cells_by_dimension": _cells_by_dimension(cells),
        "source_topology_kind": "simplicial",
        "source_construction": source_metadata.get("graph_expansion"),
        "proximity_circles": False,
    })
    axis._topokit_view = metadata
    return axis


def plot_stationary_simplicial(topology, *, scale=None, dimensions=(0, 1, 2),
                               ax=None, title="Simplicial Complex Representation",
                               style=None, legend=True, show_axes=True,
                               proximity_radius=None, limits_margin=None,
                               circle_face_alpha=None, circle_edge_alpha=None,
                               max_cells=1000, max_primitives=20_000,
                               max_scan_cells=1_000_000, node_size=None,
                               edge_width=None, face_alpha=None):
    """Plot selected simplices at one scale with optional exact-radius circles."""
    style = _resolve_style(style)
    options = {"max_cells": max_cells, "max_primitives": max_primitives}
    cloud, cells, value, _ = _select_simplicial_view(
        topology, scale=scale, dimensions=dimensions,
        max_scan_cells=max_scan_cells, options=options)
    from .objects import plot_simplices
    axis = plot_simplices(
        cloud, cells, ax=ax, labels=False, show_axes=show_axes, legend=False,
        title=title, style=_plot_style(style, node_size=node_size,
                                      edge_width=edge_width,
                                      face_alpha=face_alpha),
        colors={1: style.edge_color, 2: style.face_color},
        max_cells=max_cells, max_primitives=max_primitives,
    )
    circles = ()
    if proximity_radius is not None:
        circles = add_proximity_circles(
            axis, cloud, proximity_radius, style=style,
            face_alpha=circle_face_alpha, edge_alpha=circle_edge_alpha,
        )
    clean_axis(axis, color=style.point_color)
    if limits_margin is not None:
        set_point_limits(axis, cloud, margin=limits_margin)
    if legend:
        present = set(len(cell) - 1 for cell in cells)
        entries = []
        if 0 in present:
            entries.append(("point", "0-simplex"))
        if 1 in present:
            entries.append(("edge", "1-simplex"))
        if 2 in present:
            entries.append(("face", "2-simplex"))
        _basic_legend(axis, style, entries)
    metadata = dict(axis._topokit_view)
    metadata.update({
        "kind": "simplicial", "scale": value,
        "cells_by_dimension": _cells_by_dimension(cells),
        "proximity_radius": proximity_radius,
        "proximity_circle_count": len(circles) // 2,
    })
    axis._topokit_view = metadata
    return axis


def _display_counts(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("display_counts must map dimensions to nonnegative counts")
    result = {}
    for degree, count in value.items():
        degree = _dimension(degree, "display dimension")
        if degree > 2:
            raise ValueError("display_counts supports dimensions 0, 1, and 2")
        count = _dimension(count, "display count")
        result[degree] = count
    return result


def _even_selection(cells, count):
    if count >= len(cells):
        return tuple(cells)
    if count == 0:
        return ()
    indices = np.linspace(0, len(cells) - 1, count, dtype=int)
    return tuple(cells[index] for index in indices)


def plot_stationary_hyperdigraph(
        topology, *, scale=None, dimensions=(0, 1, 2), ax=None,
        title="Hyperdigraph Representation", style=None,
        proximity_radius=None, display_counts=None, display_hyperedges=None,
        circle_face_alpha=None, circle_edge_alpha=None,
        legend=True,
        color_by_weights=True, colorbar=False, annotate_weights=(),
        annotation_offsets=None, weight_precision=3, show_selection_summary=False,
        show_axes=True, limits_margin=None, max_cells=500,
        max_primitives=20_000, max_scan_cells=1_000_000,
        node_size=None, arrow_size=None):
    """Plot a clear 0/1/2-dimensional directed-hyperedge view at one scale.

    Consecutive oriented segments are deduplicated inside each dimension.
    Directed 2-hyperedge segments are wider translucent arrows below directed
    1-hyperedges on the same arc. ``display_counts={2: 14}`` requests an evenly
    spaced deterministic display subset while retaining every cell in the
    supplied numerical topology.
    """
    style = _resolve_style(style)
    degrees = _degrees(dimensions, maximum=2)
    options = {"max_cells": max_cells, "max_primitives": max_primitives}
    cloud, cells, value, _ = _select_hyperdigraph_view(
        topology, scale=scale, dimensions=degrees,
        max_scan_cells=max_scan_cells, options=options)
    if cloud.points.shape[1] != 2:
        raise ValueError("the layered stationary Hyperdigraph view requires 2D coordinates")
    source_groups = _cells_by_dimension(cells)
    selected_groups = dict(source_groups)
    requested_counts = _display_counts(display_counts)
    explicit_groups = {}
    if display_hyperedges is not None:
        if not isinstance(display_hyperedges, Mapping):
            raise TypeError("display_hyperedges must map dimensions to explicit cells")
        for degree, records in display_hyperedges.items():
            degree = _dimension(degree, "display dimension")
            if degree > 2:
                raise ValueError("display_hyperedges supports dimensions 0, 1, and 2")
            if degree in requested_counts:
                raise ValueError("choose display_counts or display_hyperedges for each dimension, not both")
            if isinstance(records, (str, bytes, Mapping)):
                raise ValueError("display_hyperedges values must be iterables of explicit cells")
            chosen = tuple(tuple(cell) for cell in records)
            if any(len(cell) != degree + 1 for cell in chosen):
                raise ValueError("each displayed hyperedge length must equal its dimension plus one")
            available = set(source_groups.get(degree, ()))
            if any(cell not in available for cell in chosen):
                raise ValueError("display_hyperedges must be selected from cells present at the requested scale")
            explicit_groups[degree] = chosen
            selected_groups[degree] = chosen
    for degree, count in requested_counts.items():
        selected_groups[degree] = _even_selection(source_groups.get(degree, ()), count)

    from .objects import plot_hyperedges
    resolved_node_size = style.node_size if node_size is None else node_size
    resolved_arrow_size = style.arrow_size if arrow_size is None else arrow_size
    resolved_arrow_size = _optional_nonnegative(resolved_arrow_size, "arrow_size")
    if resolved_arrow_size == 0:
        raise ValueError("arrow_size must be positive")
    base_style = _plot_style(style, node_size=resolved_node_size)
    axis = plot_hyperedges(
        cloud, selected_groups.get(0, ()), ax=ax, labels=False,
        show_axes=show_axes, legend=False, color_by_weights=color_by_weights,
        title=title, style=base_style, max_cells=max_cells,
        max_primitives=max_primitives,
    )
    if color_by_weights and axis.collections:
        axis.collections[-1].set_cmap(style.weight_cmap)
    circles = ()
    if proximity_radius is not None:
        circles = add_proximity_circles(
            axis, cloud, proximity_radius, style=style,
            face_alpha=circle_face_alpha, edge_alpha=circle_edge_alpha,
        )

    one_segments = unique_directed_segments(selected_groups.get(1, ()))
    two_segments = unique_directed_segments(selected_groups.get(2, ()))
    point_shrink = max(4.5, math.sqrt(float(resolved_node_size)) * 0.72)
    layer_specs = (
        (2, two_segments, style.higher_edge_color, style.higher_edge_width,
         style.higher_edge_alpha, resolved_arrow_size, 4),
        (1, one_segments, style.edge_color, style.arrow_width,
         0.98, resolved_arrow_size, 6),
    )
    from matplotlib.patches import FancyArrowPatch
    for degree, segments, color, width, alpha, mutation_scale, zorder in layer_specs:
        for index, (start, end) in enumerate(segments):
            arrow = FancyArrowPatch(
                cloud.points[cloud.index[start]], cloud.points[cloud.index[end]],
                arrowstyle="-|>", connectionstyle=f"arc3,rad={style.curvature}",
                mutation_scale=mutation_scale, shrinkA=point_shrink,
                shrinkB=point_shrink, linewidth=width, color=color,
                alpha=alpha, capstyle="round", joinstyle="round", zorder=zorder,
            )
            arrow.set_gid(f"topokit-stationary-directed-{degree}-segment-{index}")
            axis.add_patch(arrow)

    annotation_offsets = {} if annotation_offsets is None else dict(annotation_offsets)
    if isinstance(annotate_weights, (str, bytes, Mapping)):
        raise ValueError("annotate_weights must be an iterable of point IDs")
    annotated = tuple(annotate_weights)
    if isinstance(weight_precision, bool) or not isinstance(weight_precision, Integral) or weight_precision < 0:
        raise ValueError("weight_precision must be a nonnegative integer")
    for point_id in annotated:
        if point_id not in cloud.index:
            raise ValueError(f"unknown annotated point ID: {point_id!r}")
        offset = annotation_offsets.get(point_id, (7, 7))
        if (not isinstance(offset, (tuple, list)) or len(offset) != 2
                or any(isinstance(v, bool) or not isinstance(v, Real)
                       or not math.isfinite(float(v)) for v in offset)):
            raise ValueError("annotation offsets must be finite (x, y) pairs")
        point = cloud.points[cloud.index[point_id]]
        axis.annotate(
            f"{point_id}\nw={cloud.weights[cloud.index[point_id]]:.{weight_precision}f}",
            point, xytext=offset, textcoords="offset points", ha="left",
            va="top" if offset[1] < 0 else "bottom", fontsize=6.3,
            color=style.highlight_color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6),
            arrowprops=dict(arrowstyle="-", color=style.highlight_color,
                            linewidth=0.45, alpha=0.65, shrinkA=0, shrinkB=2),
        )

    if legend:
        entries = []
        if selected_groups.get(0):
            entries.append(("point", "directed 0-hyperedge"))
        if selected_groups.get(1):
            entries.append(("edge", "directed 1-hyperedge"))
        if selected_groups.get(2):
            entries.append(("higher", "directed 2-hyperedge"))
        _basic_legend(axis, style, entries, fontsize=6.6)
    if colorbar and color_by_weights and len(cloud):
        import matplotlib as mpl
        low, high = float(cloud.weights.min()), float(cloud.weights.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        source = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=low, vmax=high), cmap=style.weight_cmap,
        )
        source.set_array([])
        axis.figure.colorbar(source, ax=axis, fraction=0.035, pad=0.025,
                             label="direction weight")
    if show_selection_summary and 2 in source_groups:
        axis.text(
            0.02, 0.015,
            f"{len(selected_groups.get(2, ()))} of {len(source_groups.get(2, ()))} "
            f"directed 2-hyperedges; {len(two_segments)} unique segments",
            transform=axis.transAxes, color=style.annotation_color, fontsize=6.5,
        )
    clean_axis(axis, color=style.point_color)
    if limits_margin is not None:
        set_point_limits(axis, cloud, margin=limits_margin)
    metadata = dict(axis._topokit_view)
    metadata.update({
        "kind": "hyperdigraph", "scale": value,
        "source_cells_by_dimension": source_groups,
        "cells_by_dimension": selected_groups,
        "display_counts": requested_counts,
        "explicit_display_dimensions": tuple(sorted(explicit_groups)),
        "arrows_by_dimension": {1: one_segments, 2: two_segments},
        "unique_segment_counts": {1: len(one_segments), 2: len(two_segments)},
        "max_connections_per_support": {
            1: maximum_connections_per_support(one_segments),
            2: maximum_connections_per_support(two_segments),
            "total": maximum_connections_per_support(one_segments, two_segments),
        },
        "curve_semantics": "same-curvature directed-2 underlay below directed-1 arrow",
        "proximity_radius": proximity_radius,
        "proximity_circle_count": len(circles) // 2,
        "annotated_weight_ids": annotated,
    })
    axis._topokit_view = metadata
    return axis


def _interaction_factor_data(topology, scale, dimensions, max_cells, max_scan_cells):
    if not isinstance(topology, Topology) or topology.kind != "interaction":
        raise TypeError("topology must be an interaction Topology")
    if (not isinstance(topology.cloud, (tuple, list)) or len(topology.cloud) != 2
            or any(not isinstance(cloud, PointCloud) for cloud in topology.cloud)):
        raise ValueError("interaction factor coordinates are unavailable; build from two point clouds")
    value = validate_query(topology.metadata, scale)
    degrees = _degrees(dimensions, maximum=2)
    factors = topology.metadata.get("source_factor_filtrations")
    maps = topology.metadata.get("factor_vertex_id_maps")
    if (not isinstance(factors, (tuple, list)) or len(factors) != 2
            or not isinstance(maps, (tuple, list)) or len(maps) != 2):
        raise ValueError("interaction topology lacks retained factor filtrations or ID maps")
    cell_budget = _dimension(max_cells, "max_cells")
    scan_budget = _dimension(max_scan_cells, "max_scan_cells")
    selected = []
    for items, mapping in zip(factors, maps):
        cells, scanned = [], 0
        for simplex, birth in items:
            scanned += 1
            if scanned > scan_budget:
                raise ResourceLimitError("Factor inspection exceeds max_scan_cells")
            if value is not None and birth > value:
                continue
            if len(simplex) - 1 not in degrees:
                continue
            if len(cells) >= cell_budget:
                raise ResourceLimitError("Selected factor cells exceed max_cells")
            try:
                cells.append(tuple(mapping[vertex] for vertex in simplex))
            except KeyError as error:
                raise ValueError("interaction factor vertex ID map is incomplete") from error
        selected.append(tuple(cells))
    return tuple(topology.cloud), tuple(selected), value


def plot_stationary_interaction(
        topology, *, scale=None, dimensions=(0, 1, 2), axes=None,
        title="Interaction Complex Representation", factor_titles=None,
        style=None, proximity_radius=None, show_axes=True,
        circle_face_alpha=None, circle_edge_alpha=None,
        shared_limits=True, limits_margin=None, legend=True,
        max_cells=1000, max_primitives=20_000,
        max_scan_cells=1_000_000, node_size=None,
        edge_width=None, face_alpha=None, figsize=(7.2, 3.7)):
    """Plot two retained interaction factors and their explicit overlap.

    The panels are factor-complex views. No embedding of the quotient interaction
    chain is inferred.
    """
    style = _resolve_style(style)
    clouds, factor_cells, value = _interaction_factor_data(
        topology, scale, dimensions, max_cells, max_scan_cells)
    if axes is None:
        figure, axes = _plt().subplots(1, 2, figsize=figsize, constrained_layout=True)
    else:
        axes = np.asarray(axes, dtype=object).reshape(-1)
        if len(axes) != 2 or axes[0].figure is not axes[1].figure:
            raise ValueError("axes must contain two axes from the same figure")
        figure = axes[0].figure
    axes = tuple(np.asarray(axes, dtype=object).reshape(-1))
    if factor_titles is None:
        factor_titles = tuple(
            f"Factor {label}: {len(cloud)} points"
            for label, cloud in zip(("A", "B"), clouds)
        )
    else:
        factor_titles = tuple(factor_titles)
        if len(factor_titles) != 2:
            raise ValueError("factor_titles must contain exactly two strings")
    from .objects import plot_simplices
    overlap_pairs = tuple(topology.metadata.get("overlap_pairs", ()))
    circle_counts = []
    for side, (axis, cloud, cells, factor_title) in enumerate(
            zip(axes, clouds, factor_cells, factor_titles)):
        plot_simplices(
            cloud, cells, ax=axis, labels=False, show_axes=show_axes,
            legend=False, title=factor_title,
            style=_plot_style(style, node_size=node_size,
                              edge_width=edge_width, face_alpha=face_alpha),
            colors={1: style.edge_color, 2: style.face_color},
            max_cells=max_cells, max_primitives=max_primitives,
        )
        circles = ()
        if proximity_radius is not None:
            circles = add_proximity_circles(
                axis, cloud, proximity_radius, style=style,
                face_alpha=circle_face_alpha, edge_alpha=circle_edge_alpha,
            )
        overlap_ids = tuple(pair[side] for pair in overlap_pairs)
        if overlap_ids:
            try:
                coordinates = cloud.points[[cloud.index[item] for item in overlap_ids]]
            except KeyError as error:
                raise ValueError("overlap pair references a point absent from its factor") from error
            rings = axis.scatter(
                *coordinates.T, s=style.overlap_size, facecolors="none",
                edgecolors=style.higher_edge_color,
                linewidths=style.overlap_linewidth, zorder=12,
            )
            rings.set_gid(f"topokit-interaction-overlap-{side}")
        clean_axis(axis, color=style.point_color)
        reference = clouds if shared_limits else cloud
        if limits_margin is not None:
            set_point_limits(axis, reference, margin=limits_margin)
        add_panel_label(axis, "a" if side == 0 else "b", color=style.point_color)
        metadata = dict(axis._topokit_view)
        metadata.update({
            "kind": "interaction_factor", "factor": side, "scale": value,
            "cells_by_dimension": _cells_by_dimension(cells),
            "overlap_ids": overlap_ids,
            "proximity_radius": proximity_radius,
            "proximity_circle_count": len(circles) // 2,
        })
        axis._topokit_view = metadata
        circle_counts.append(len(circles) // 2)
    if legend and overlap_pairs:
        _basic_legend(axes[1], style, (("overlap", "explicit overlap vertex"),),
                      fontsize=6.6, loc="lower right")
    figure.suptitle(title)
    figure._topokit_view = {
        "kind": "interaction", "scale": value,
        "factor_cells": factor_cells, "overlap_pairs": overlap_pairs,
        "proximity_circle_counts": tuple(circle_counts),
        "view_semantics": "two factor complexes with explicit overlap",
    }
    return figure, axes


def plot_stationary(topology, *, representation=None, **options):
    """Dispatch a simple fixed-scale graph, simplicial, Hyperdigraph, or interaction view."""
    if not isinstance(topology, Topology):
        raise TypeError("topology must be a TopoKit Topology")
    allowed = {None, "graph", "simplicial", "hyperdigraph", "interaction"}
    if representation not in allowed:
        raise ValueError("representation must be graph, simplicial, hyperdigraph, interaction, or None")
    selected = representation
    if selected is None:
        if topology.kind == "simplicial":
            selected = ("graph" if topology.metadata.get("graph_expansion") == "none"
                        and topology.metadata.get("max_simplex_dimension", 2) <= 1
                        else "simplicial")
        else:
            selected = topology.kind
    if selected in {"graph", "simplicial"} and topology.kind != "simplicial":
        raise ValueError(f"{selected} presentation requires a simplicial Topology")
    if selected == "hyperdigraph" and topology.kind != "hyperdigraph":
        raise ValueError("Hyperdigraph presentation requires a hyperdigraph Topology")
    if selected == "interaction" and topology.kind != "interaction":
        raise ValueError("interaction presentation requires an interaction Topology")
    return {
        "graph": plot_stationary_graph,
        "simplicial": plot_stationary_simplicial,
        "hyperdigraph": plot_stationary_hyperdigraph,
        "interaction": plot_stationary_interaction,
    }[selected](topology, **options)


def plot_stationary_diagnostics(result, *, axes=None, title=None,
                                display_dimensions=(0, 1), style=None,
                                figsize=(7.2, 5.2)):
    """Plot homology, complete spectra, and summary values from one result."""
    if not isinstance(result, StationaryResult):
        raise TypeError("result must be a StationaryResult from workflows.analyze_stationary")
    style = _resolve_style(style)
    display_degrees = _degrees(display_dimensions, maximum=2)
    if len(display_degrees) != 2:
        raise ValueError("display_dimensions must contain exactly two dimensions")
    if axes is None:
        figure, axes = _plt().subplots(2, 2, figsize=figsize, constrained_layout=True)
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != (2, 2) or len({id(axis.figure) for axis in axes.flat}) != 1:
            raise ValueError("axes must be a 2-by-2 array from one figure")
        figure = axes.flat[0].figure
    axes = np.asarray(axes, dtype=object)

    homology_degrees = tuple(range(len(result.homology.betti_numbers)))
    betti = np.asarray(result.homology.betti_numbers, dtype=float)
    colors = [style.edge_color if q == 0 else style.higher_edge_color
              for q in homology_degrees]
    bars = axes[0, 0].bar(
        [f"H{q}" for q in homology_degrees], betti, color=colors,
        width=0.52, edgecolor="white", linewidth=0.45,
    )
    axes[0, 0].bar_label(
        bars, labels=[str(int(value)) for value in betti], padding=2, fontsize=7.5,
    )
    clean_axis(axes[0, 0], color=style.point_color)
    axes[0, 0].set_ylabel(f"Betti number over {result.homology.field}")
    axes[0, 0].set_ylim(0, max(1.0, betti.max(initial=0.0)) * 1.18)
    axes[0, 0].set_title("Stationary homology")

    spectrum_axes = (axes[0, 1], axes[1, 0])
    for degree, axis in zip(display_degrees, spectrum_axes):
        spectrum = result.spectra.get(degree)
        if spectrum is None:
            empty_axis(axis, f"Intentionally left empty\nL{degree} was not requested",
                       color=style.annotation_color, title=f"L{degree} spectrum")
            continue
        values = np.sort(np.asarray(spectrum.eigenvalues, dtype=float))
        if values.size:
            axis.plot(
                np.arange(values.size), values,
                color=style.edge_color if degree == 0 else style.higher_edge_color,
                marker="o", markersize=2.0, markeredgewidth=0, linewidth=0.9,
            )
            axis.axhline(0, color=style.mid_gray, linewidth=0.55)
            clean_axis(axis, color=style.point_color)
        else:
            empty_axis(axis, "Empty chain group", color=style.annotation_color)
        axis.set_xlabel("Sorted eigenvalue index")
        axis.set_ylabel("Eigenvalue")
        axis.set_title(f"Complete L{degree} spectrum")

    metric_names = ("min", "max", "mean", "zero_count", "energy")
    matrix = np.asarray([
        [result.summaries.get(q, {}).get(name, math.nan) for name in metric_names]
        for q in display_degrees
    ], dtype=float)
    column_scale = np.nanmax(np.abs(matrix), axis=0)
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 0),
                            column_scale, 1.0)
    shading = np.abs(matrix) / column_scale
    import matplotlib as mpl
    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "topokit_stationary_blues",
        [style.light_color, style.circle_color, style.edge_color],
    )
    color_map.set_bad(style.light_color)
    axes[1, 1].imshow(np.ma.masked_invalid(shading), cmap=color_map,
                      vmin=0, vmax=1, aspect="auto")
    axes[1, 1].set_xticks(
        range(len(metric_names)), ("min", "max", "mean", "zeros", "energy"),
        rotation=28, ha="right",
    )
    axes[1, 1].set_yticks(range(len(display_degrees)),
                          [f"L{q}" for q in display_degrees])
    for row in range(len(display_degrees)):
        for column in range(len(metric_names)):
            value = matrix[row, column]
            label = "—" if not np.isfinite(value) else (
                f"{int(value)}" if metric_names[column] == "zero_count" else f"{value:.3g}"
            )
            text_color = ("white" if np.isfinite(shading[row, column])
                          and shading[row, column] > 0.62 else style.point_color)
            axes[1, 1].text(column, row, label, ha="center", va="center",
                            fontsize=7, color=text_color)
    axes[1, 1].set_title("Stationary Laplacian summaries")
    axes[1, 1].tick_params(length=0)
    axes[1, 1].grid(False)
    for spine in axes[1, 1].spines.values():
        spine.set_visible(False)
    for label, axis in zip("abcd", axes.flat):
        add_panel_label(axis, label, color=style.point_color)
    figure.suptitle(title or "Stationary homology and Laplacian summaries")
    figure._topokit_view = {
        "kind": "stationary_diagnostics", "scale": result.scale,
        "dimensions": display_degrees, "metrics": metric_names,
    }
    return figure, axes


__all__ = [
    "StationaryStyle", "clean_axis", "empty_axis", "add_panel_label",
    "set_point_limits", "add_proximity_circles", "unique_directed_segments",
    "maximum_connections_per_support", "plot_stationary_graph",
    "plot_stationary_simplicial", "plot_stationary_hyperdigraph",
    "plot_stationary_interaction", "plot_stationary",
    "plot_stationary_diagnostics",
]
