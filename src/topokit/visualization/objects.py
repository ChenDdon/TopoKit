"""Explicit-cell views with filled faces and ordered, separated arrow ribbons.

These functions draw supplied objects. They never infer cliques, triangulate
scientific data, or compute homology. Three-dimensional coordinates can be
shown on a rotatable 3D axis or as an orthographic 2D projection for vector art.
"""
from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations
import math
from numbers import Integral, Real

import numpy as np
from ..data import as_point_cloud
from .style import PlotStyle


_SIMPLEX_COLORS = {1: "#36485B", 2: "#6699CC", 3: "#234F7D"}
_HYPEREDGE_COLORS = {1: "#4A9778", 2: "#4D85B6", 3: "#DBAD42"}
_EXTRA_COLORS = ("#8C79AC", "#CB8A66", "#5E9298")
_Arrow3D = None


def _limit(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _angle(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _cells(records, lookup, maximum, primitive_budget, kind):
    """Consume input incrementally; no unbounded tuple(generator) allocation."""
    if isinstance(records, (str, bytes)):
        raise ValueError("cells must be an iterable of vertex sequences")
    grouped = isinstance(records, Mapping)
    if grouped:
        groups = records.items()
    else:
        groups = ((None, records),)
    cells, indexed = [], []
    work = 0
    for degree, group in groups:
        if grouped:
            degree = _limit(degree, "cell dimension")
        if isinstance(group, (str, bytes, Mapping)):
            raise ValueError("each dimension must contain an iterable of cells")
        for raw in group:
            if len(cells) >= maximum:
                raise ValueError("Too many cells for this view; select an explicit subset or raise max_cells")
            if isinstance(raw, (str, bytes, Mapping, Set)):
                raise ValueError("each cell must be a nonempty vertex sequence")
            cell, indices = [], []
            seen = set()
            for label in raw:
                if isinstance(label, (bool, np.bool_)) or not isinstance(label, (str, Integral)):
                    raise ValueError("cell point IDs must be strings or integers")
                if label not in lookup:
                    raise ValueError(f"{kind} references an unknown point ID: {label!r}")
                if label in seen:
                    raise ValueError("a cell cannot contain repeated point IDs")
                seen.add(label)
                cell.append(label)
                indices.append(lookup[label])
                n = len(cell)
                needed = (math.comb(n, 2) + math.comb(n, 3) if kind == "simplicial"
                          else 2 * max(0, n - 1))
                if work + needed > primitive_budget:
                    raise ValueError("Cell expansion exceeds max_primitives; select explicit cells or raise the budget")
            if not cell:
                raise ValueError("empty cells are not supported")
            if degree is not None and len(cell) != degree + 1:
                raise ValueError("cell length must equal its dimension plus one")
            work += needed
            cells.append(tuple(cell))
            indexed.append(tuple(indices))
    return tuple(cells), tuple(indexed)


def _coordinates(cloud, view, elev, azim):
    if view not in ("auto", "projected", "3d"):
        raise ValueError("view must be 'auto', 'projected', or '3d'")
    points = cloud.points
    if points.shape[1] not in (2, 3):
        raise ValueError("Object views need 2D/3D coordinates; project explicitly before plotting")
    if view == "3d" and points.shape[1] == 2:
        return np.column_stack((points, np.zeros(len(points)))), np.zeros(len(points)), "3d"
    if points.shape[1] == 2:
        return points, np.zeros(len(points)), "2d"
    if view != "projected":
        return points, np.zeros(len(points)), "3d"
    # Unit camera vectors give a true orthographic view without figure creation
    # or data-dependent anisotropic scaling. Coordinates are never perturbed.
    theta, phi = math.radians(azim), math.radians(elev)
    right = np.array([-math.sin(theta), math.cos(theta), 0.])
    up = np.array([-math.sin(phi)*math.cos(theta), -math.sin(phi)*math.sin(theta), math.cos(phi)])
    towards = np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)])
    origin = points[0] if len(points) else np.zeros(3)
    centered = points - origin
    projected = np.column_stack((centered @ right, centered @ up))
    depth = centered @ towards
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(depth)):
        raise ValueError("Projection overflow; rescale coordinates before plotting")
    return projected, depth, "projected"


def _prepare(cloud, records, kind, *, max_cells, max_primitives, style, colors,
             view, elev, azim, labels, show_axes, legend, color_by_weights, ax):
    maximum = _limit(max_cells, "max_cells")
    primitive_budget = _limit(max_primitives, "max_primitives")
    cloud = as_point_cloud(cloud)
    style = PlotStyle() if style is None else style
    if not isinstance(style, PlotStyle):
        raise TypeError("style must be a PlotStyle")
    for value in (labels, show_axes, legend, color_by_weights):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("labels, show_axes, legend and color_by_weights must be booleans")
    elev, azim = _angle(elev, "elev"), _angle(azim, "azim")
    points, depth, projection = _coordinates(cloud, view, elev, azim)
    if len(points):
        with np.errstate(over="ignore", invalid="ignore"):
            spans = np.max(points, axis=0) - np.min(points, axis=0)
            pad = max(float(np.max(spans)), 1.) * max(style.padding, style.label_offset)
            bounds = np.max(np.abs(points), axis=0) + pad
        if not np.all(np.isfinite(spans)) or not np.all(np.isfinite(bounds)):
            raise ValueError("Coordinate span overflow; rescale before plotting")
    cells, indexed = _cells(records, cloud.index, maximum, primitive_budget, kind)
    palette = dict(_SIMPLEX_COLORS if kind == "simplicial" else _HYPEREDGE_COLORS)
    if colors is not None:
        if not isinstance(colors, Mapping):
            raise TypeError("colors must map dimensions to Matplotlib colors")
        palette.update({_limit(q, "color dimension"): color for q, color in colors.items()})
    # Invalid inputs fail before a figure is created or caller axes are changed.
    from .plots import _plt
    plt = _plt()
    from matplotlib.colors import is_color_like
    for color in (*palette.values(), style.node_color, style.node_edge_color, style.edge_color, style.arrow_color):
        if not is_color_like(color):
            raise ValueError(f"Invalid Matplotlib color: {color!r}")
    if ax is not None and (getattr(ax, "name", None) == "3d") != (projection == "3d"):
        raise ValueError("The supplied axis must match the requested 2D/3D view")
    if ax is None:
        figure = plt.figure(figsize=(6.2, 5.0))
        ax = figure.add_subplot(projection="3d" if projection == "3d" else None)
    return cloud, cells, indexed, points, depth, projection, style, palette, ax


def _color(palette, degree):
    return palette.get(degree, _EXTRA_COLORS[(degree - 4) % len(_EXTRA_COLORS)])


def _finish(ax, cloud, points, projection, *, style, labels, show_axes,
            color_by_weights, title, elev, azim, metadata):
    kwargs = dict(s=style.node_size, c=cloud.weights if color_by_weights else style.node_color,
                  edgecolors=style.node_edge_color, linewidths=style.node_linewidth, zorder=10)
    if color_by_weights:
        kwargs["cmap"] = "viridis"
    if projection == "3d":
        kwargs["depthshade"] = False
        ax.computed_zorder = False
    scatter = ax.scatter(*points.T, **kwargs)
    scatter.set_gid("topokit-nodes")
    if len(points):
        minimum, maximum = np.min(points, axis=0), np.max(points, axis=0)
        spans = maximum - minimum
        if not np.all(np.isfinite(spans)):
            raise ValueError("Coordinate span overflow; rescale before plotting")
        span = max(float(np.max(spans)), 1e-8)
        # An isolated point should use a useful, finite display window.
        span = 1.0 if not np.any(spans) else span
        centers = minimum * .5 + maximum * .5
    else:
        span, centers, spans = 1.0, np.zeros(points.shape[1]), np.zeros(points.shape[1])
    if labels:
        offset = style.label_offset * span
        for label, point in zip(cloud.ids, points):
            position = point.copy()
            position[0] += offset
            position[1] += offset
            text = ax.text(*position, str(label), fontsize=style.label_size,
                           color=style.node_color, zorder=12, ha="left", va="bottom")
            text.set_gid("topokit-label")
    half = np.maximum(spans / 2, span * .08) + style.padding * span
    for index, setter in enumerate((ax.set_xlim, ax.set_ylim) + ((ax.set_zlim,) if projection == "3d" else ())):
        setter(centers[index] - half[index], centers[index] + half[index])
    if projection == "3d":
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set_box_aspect(2 * half)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
    else:
        ax.set_aspect("equal", adjustable="box")
    units = cloud.metadata.get("coordinate_units", "input units")
    if show_axes:
        ax.set_axis_on()
        names = ("u", "v") if projection == "projected" else ("x", "y", "z")
        ax.set_xlabel(f"{names[0]} [{units}]", fontsize=9)
        ax.set_ylabel(f"{names[1]} [{units}]", fontsize=9)
        if projection == "3d":
            ax.set_zlabel(f"z [{units}]", fontsize=9)
        ax.tick_params(labelsize=8)
    else:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11, pad=12)
    metadata.update(projection=projection, coordinate_units=units,
                    camera={"elev": float(elev), "azim": float(azim)},
                    displayed_point_ids=cloud.ids,
                    node_semantics="coordinate markers for supplied points")
    ax._topokit_view = metadata
    return ax


def plot_simplices(cloud, simplices, *, ax=None, max_cells=1000,
                   max_primitives=20000, title="Simplicial complex", style=None,
                   colors=None, view="auto", elev=18, azim=-65, labels=False,
                   show_axes=False, legend=False, color_by_weights=False):
    """Draw explicit simplices, or ``{dimension: cells}``, without clique inference.

    Triangular faces and edges of supplied higher cells are drawn once, with
    dimension colors. Above degree 3 this is only a 2-skeleton view. ``projected``
    uses an orthographic camera for 3D points; ``auto`` retains true 3D axes.
    ``max_primitives`` bounds pre-deduplication edge/face incidence expansion.
    All input-cloud points are shown, including isolated coordinates. The
    returned axes can be composed, restyled or saved by the caller.
    """
    cloud, cells, indexed, points, depth, projection, style, palette, ax = _prepare(
        cloud, simplices, "simplicial", max_cells=max_cells, max_primitives=max_primitives,
        style=style, colors=colors, view=view, elev=elev, azim=azim,
        labels=labels, show_axes=show_axes, legend=legend, color_by_weights=color_by_weights, ax=ax)
    edges, faces = set(), {}
    for cell in indexed:
        degree = len(cell) - 1
        edges.update(tuple(sorted(edge)) for edge in combinations(cell, 2))
        for face in combinations(cell, 3):
            key = tuple(sorted(face))
            faces[key] = max(degree, faces.get(key, 0))
    ordered_faces = sorted(faces, key=lambda face: (float(np.mean(depth[list(face)])), face))
    if faces:
        polygons = [points[list(face)] for face in ordered_faces]
        facecolors = [_color(palette, faces[face]) for face in ordered_faces]
        if projection == "3d":
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            artist = Poly3DCollection(polygons, facecolors=facecolors,
                edgecolors="none", alpha=style.face_alpha, zorder=1)
            ax.add_collection3d(artist)
        else:
            from matplotlib.collections import PolyCollection
            artist = PolyCollection(polygons, facecolors=facecolors,
                edgecolors="none", alpha=style.face_alpha, zorder=1)
            ax.add_collection(artist)
        artist.set_gid("topokit-faces")
    ordered_edges = sorted(edges)
    if edges:
        segments = [points[list(edge)] for edge in ordered_edges]
        if projection == "3d":
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            artist = Line3DCollection(segments, colors=palette[1] if colors is not None and 1 in colors else style.edge_color,
                                      linewidths=style.linewidth, zorder=4)
            ax.add_collection3d(artist)
        else:
            from matplotlib.collections import LineCollection
            artist = LineCollection(segments, colors=palette[1] if colors is not None and 1 in colors else style.edge_color,
                                    linewidths=style.linewidth, zorder=4)
            ax.add_collection(artist)
        artist.set_gid("topokit-edges")
    if legend and cells:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        dimensions = sorted({len(cell)-1 for cell in cells})
        handles = [Patch(facecolor=_color(palette, q), alpha=style.face_alpha,
                         label=f"{q}-simplex") if q >= 2 else
                   Line2D([], [], color=style.edge_color, marker="o" if q == 0 else None,
                          linestyle="none" if q == 0 else "-", label=f"{q}-simplex") for q in dimensions]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    highest = max((len(cell) - 1 for cell in cells), default=0)
    if highest > 3 and title:
        title += " (2-skeleton view)"
    labels_for = lambda records: tuple(tuple(cloud.ids[i] for i in cell) for cell in records)
    metadata = {"kind": "simplicial", "cells": cells,
        "edges": labels_for(ordered_edges), "faces": labels_for(ordered_faces), "arrows": (),
        "primitive_counts": {"edges": len(edges), "faces": len(faces), "arrows": 0, "ribbons": 0},
        "higher_cell_view": "2-skeleton" if highest > 3 else "edges_and_triangular_faces"}
    return _finish(ax, cloud, points, projection, style=style, labels=labels,
        show_axes=show_axes, color_by_weights=color_by_weights, title=title,
        elev=elev, azim=azim, metadata=metadata)


def _arrow_type():
    global _Arrow3D
    if _Arrow3D is None:
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        class Arrow3D(FancyArrowPatch):
            def __init__(self, a, b, **kwargs):
                super().__init__((0, 0), (0, 0), **kwargs)
                self._vertices3d = np.array([a, b], copy=True)

            def do_3d_projection(self, renderer=None):
                xs, ys, zs = proj3d.proj_transform(*self._vertices3d.T, self.axes.get_proj())
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return float(np.min(zs))

            def draw(self, renderer):
                self.do_3d_projection()
                super().draw(renderer)
        _Arrow3D = Arrow3D
    return _Arrow3D


def plot_hyperedges(cloud, hyperedges, *, ax=None, max_cells=500,
                    max_primitives=20000, title="Ordered hyperdigraph", style=None,
                    colors=None, view="auto", elev=18, azim=-65, labels=False,
                    show_axes=False, legend=False, color_by_weights=False):
    """Draw ordered hyperedges as colored ribbons with consecutive arrows.

    A q-hyperedge has q arrows, never all pairwise clique edges or a filled
    simplex. Repeated/reciprocal support receives separate curved display lanes;
    these curves are schematic offsets, not new geometry. Colors encode degree.
    Tuple order and every supplied hyperedge occurrence are preserved. Supply
    selected hyperedges for a crowded view; nothing is silently sampled.
    """
    cloud, cells, indexed, points, depth, projection, style, palette, ax = _prepare(
        cloud, hyperedges, "hyperdigraph", max_cells=max_cells, max_primitives=max_primitives,
        style=style, colors=colors, view=view, elev=elev, azim=azim,
        labels=labels, show_axes=show_axes, legend=legend, color_by_weights=color_by_weights, ax=ax)
    from matplotlib.patches import FancyArrowPatch
    segments, groups = [], defaultdict(list)
    for cell_index, cell in enumerate(indexed):
        for segment_index, (a, b) in enumerate(zip(cell[:-1], cell[1:])):
            groups[tuple(sorted((a, b)))].append(len(segments))
            segments.append((cell_index, segment_index, a, b, len(cell)-1))
    curves = [0.] * len(segments)
    for positions in groups.values():
        for lane, index in enumerate(positions):
            a, b = segments[index][2:4]
            # The sign change makes lanes refer to a common undirected axis.
            center = (len(positions)-1)/2
            curves[index] = (lane - center) / max(1., center) * style.curvature * (1 if a < b else -1)
    for index, (cell_index, segment_index, a, b, degree) in enumerate(segments):
        color = _color(palette, degree)
        common = {"connectionstyle": f"arc3,rad={curves[index]}",
                  "capstyle": "round", "joinstyle": "round"}
        radius = math.sqrt(style.node_size) * .5 + style.node_linewidth
        for ribbon in (True, False):
            kwargs = {**common, "arrowstyle": "-" if ribbon else "-|>",
                "mutation_scale": style.arrow_size,
                "linewidth": style.ribbon_width if ribbon else style.linewidth,
                "color": color if ribbon else style.arrow_color,
                "alpha": style.ribbon_alpha if ribbon else 1.,
                "shrinkA": 0. if ribbon else radius,
                "shrinkB": 0. if ribbon else radius,
                "zorder": 2 if ribbon else 6}
            if projection == "3d":
                artist = _arrow_type()(points[a], points[b], **kwargs)
            else:
                artist = FancyArrowPatch(points[a], points[b], **kwargs)
            artist.set_gid(f"topokit-{'ribbon' if ribbon else 'arrow'}-{cell_index}-{segment_index}")
            ax.add_patch(artist)
    if legend and cells:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=_color(palette, q) if q else style.node_color,
                          linewidth=4 if q else 0, marker="o" if q == 0 else None,
                          label=f"{q}-hyperedge") for q in sorted({len(cell)-1 for cell in cells})]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    arrows = tuple((cloud.ids[a], cloud.ids[b]) for _, _, a, b, _ in segments)
    metadata = {"kind": "hyperdigraph", "cells": cells, "edges": (), "faces": (),
        "arrows": arrows, "arrow_curvatures": tuple(curves),
        "primitive_counts": {"edges": 0, "faces": 0, "arrows": len(segments), "ribbons": len(segments)},
        "direction_semantics": "consecutive vertices in each supplied ordered hyperedge",
        "curve_semantics": "display separation of repeated/reciprocal support"}
    return _finish(ax, cloud, points, projection, style=style, labels=labels,
        show_axes=show_axes, color_by_weights=color_by_weights, title=title,
        elev=elev, azim=azim, metadata=metadata)


__all__ = ["plot_simplices", "plot_hyperedges"]
