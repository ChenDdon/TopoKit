"""Reusable schematic building blocks, independent of topology construction.

The coordinates here are diagram layouts, not inferred scientific geometry.
An ordered hyperedge is rendered as its consecutive arrows, never as a
directed clique. Matplotlib remains optional until a plot is requested.
"""

from itertools import islice
from numbers import Integral

import numpy as np

from ..data import PointCloud
from .objects import plot_hyperedges, plot_simplices
from .style import PlotStyle

__all__ = ["plot_simplex_block", "plot_hyperdigraph_block", "plot_building_blocks"]


def _dimension(value, max_vertices):
    if isinstance(max_vertices, bool) or not isinstance(max_vertices, Integral):
        raise TypeError("max_vertices must be a positive integer")
    if max_vertices < 1:
        raise ValueError("max_vertices must be a positive integer")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("dimension must be a nonnegative integer")
    value = int(value)  # Avoid fixed-width NumPy integer overflow in value + 1.
    if value < 0:
        raise ValueError("dimension must be a nonnegative integer")
    if value + 1 > max_vertices:
        raise ValueError("building block exceeds max_vertices; raise the limit explicitly")
    return value


def _layout(dimension):
    if dimension == 0:
        return np.array([[0.0, 0.0]])
    if dimension == 1:
        return np.array([[-0.5, 0.0], [0.5, 0.0]])
    base = np.array([[-0.5, -np.sqrt(3) / 6],
                     [0.5, -np.sqrt(3) / 6],
                     [0.0, np.sqrt(3) / 3]])
    if dimension == 2:
        return base
    if dimension == 3:
        return np.vstack((np.column_stack((base, np.zeros(3))),
                          [0.0, 0.0, np.sqrt(2 / 3)]))
    angles = np.pi / 2 - np.linspace(0, 2 * np.pi, dimension + 1, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _cloud(dimension, points, ids):
    count = dimension + 1
    coordinates = _layout(dimension) if points is None else points
    if hasattr(coordinates, "__len__") and len(coordinates) != count:
        raise ValueError("points must contain dimension + 1 coordinates")
    # Bound iterable consumption too: accidental infinite ID iterators must
    # fail rather than allocating indefinitely.
    labels = None if ids is None else tuple(islice(iter(ids), count + 1))
    cloud = PointCloud(coordinates, ids=labels,
                       metadata={"visualization_layout": "building_block",
                                 "block_dimension": dimension})
    if len(cloud) != count:
        raise ValueError("points must contain dimension + 1 coordinates")
    if cloud.points.shape[1] not in (2, 3):
        raise ValueError("building block points need 2D or 3D coordinates")
    return cloud


def _options(options, dimension, kind):
    options = dict(options)
    options.setdefault("view", "projected")
    options.setdefault("style", PlotStyle(node_size=65, face_alpha=0.4, ribbon_width=10))
    # The regular tetrahedron has three clear silhouette vertices and one
    # interior vertex from this camera, with no nearly overlapping edges.
    options.setdefault("azim", -90)
    if "title" not in options:
        caption = f"{dimension}-{'simplex' if kind == 'simplicial' else 'hyperedge'}"
        if dimension > 3:
            caption += ("\nschematic 2-skeleton" if kind == "simplicial"
                        else "\nschematic ordered sequence")
        options["title"] = caption
    return options


def _square_frame(axis):
    if not hasattr(axis, "get_zlim"):
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        mid_x, mid_y = sum(xlim) / 2, sum(ylim) / 2
        axis.set_xlim(mid_x - span / 2, mid_x + span / 2)
        axis.set_ylim(mid_y - span / 2, mid_y + span / 2)
    return axis


def plot_simplex_block(dimension, *, ax=None, labels=True, points=None,
                       ids=None, max_vertices=32, **options):
    """Draw one simplex building block and return its Matplotlib axis.

    Dimensions 0--3 use a point, segment, equilateral triangle, and regular
    tetrahedron. The default orthographic projected view works on ordinary 2D
    axes. Above dimension 3, the vertices use a regular polygon and the plot
    shows a *schematic 2-skeleton*, not a faithful geometric embedding or the
    simplex interior. No topological object or closure is constructed.

    ``points`` may replace the layout with exactly ``dimension + 1`` finite
    2D or 3D coordinates; ``ids`` supplies unique string/integer labels.
    ``max_vertices`` bounds the requested block before layout allocation.
    Other options (including ``style``, ``view``, ``elev``, ``azim``, and
    ``title``) are forwarded to :func:`plot_simplices`. The function never
    saves, shows, closes, or changes global Matplotlib settings.
    """
    dimension = _dimension(dimension, max_vertices)
    cloud = _cloud(dimension, points, ids)
    axis = plot_simplices(cloud, (cloud.ids,), ax=ax, labels=labels,
                          **_options(options, dimension, "simplicial"))
    return _square_frame(axis)


def plot_hyperdigraph_block(dimension, *, ax=None, labels=True, points=None,
                           ids=None, order=None, max_vertices=32, **options):
    """Draw one ordered hyperedge with exactly ``dimension`` directed steps.

    The block contains ``dimension + 1`` distinct vertices. Arrows connect
    consecutive entries only; deleting a vertex or filling a directed clique
    is not implied. ``order`` is an optional permutation of the supplied
    ``ids`` (the default IDs are 0 through ``dimension``), and changes the
    directed sequence while leaving vertex positions and labels unchanged.

    Layouts and forwarding options match :func:`plot_simplex_block`.
    Dimensions above 3 use a schematic polygon for the ordered sequence.
    Returns the supplied or newly created axis, without saving or showing it.
    """
    dimension = _dimension(dimension, max_vertices)
    cloud = _cloud(dimension, points, ids)
    sequence = cloud.ids if order is None else tuple(islice(iter(order), dimension + 2))
    if any(isinstance(label, bool) or not isinstance(label, (str, Integral))
           for label in sequence):
        raise TypeError("order IDs must be strings or integers (not booleans)")
    sequence = tuple(int(label) if isinstance(label, Integral) else label for label in sequence)
    if len(sequence) != len(cloud) or set(sequence) != set(cloud.ids):
        raise ValueError("order must be a permutation of the building block point IDs")
    axis = plot_hyperedges(cloud, (sequence,), ax=ax, labels=labels,
                           **_options(options, dimension, "hyperdigraph"))
    return _square_frame(axis)


def plot_building_blocks(kind="simplicial", dimensions=(0, 1, 2, 3), *,
                         axes=None, labels=True, max_vertices=32,
                         max_blocks=16, **options):
    """Draw a row of blocks and return ``(figure, flat_axes_array)``.

    ``kind`` is ``'simplicial'`` or ``'hyperdigraph'``. ``dimensions`` preserves
    caller order, including repeated dimensions. An existing axis or array of
    axes must have exactly one entry per dimension, all on the same figure.
    Otherwise a new row is created. No existing artists are cleared.

    ``max_blocks`` bounds gallery size, and ``max_vertices`` bounds each block.
    Additional options are passed to every block. For different custom IDs,
    coordinates, or ordering in each panel, call the individual block
    functions on your own axes instead. ``view='3d'`` creates 3D gallery axes;
    the default ``view='projected'`` creates ordinary 2D axes.
    """
    if kind not in ("simplicial", "hyperdigraph"):
        raise ValueError("kind must be 'simplicial' or 'hyperdigraph'")
    if isinstance(max_blocks, bool) or not isinstance(max_blocks, Integral):
        raise TypeError("max_blocks must be a positive integer")
    if max_blocks < 1:
        raise ValueError("max_blocks must be a positive integer")
    dimensions = tuple(islice(iter(dimensions), int(max_blocks) + 1))
    if not dimensions:
        raise ValueError("dimensions must contain at least one dimension")
    if len(dimensions) > max_blocks:
        raise ValueError("gallery exceeds max_blocks; raise the limit explicitly")
    dimensions = tuple(_dimension(value, max_vertices) for value in dimensions)
    view = options.get("view", "projected")
    if view not in ("auto", "projected", "3d"):
        raise ValueError("view must be 'auto', 'projected', or '3d'")
    if axes is None:
        from .plots import _plt
        figure = _plt().figure(figsize=(2.35 * len(dimensions), 2.8))
        supplied_points = options.get("points")
        ambient_dimension = (None if supplied_points is None
                             else np.asarray(supplied_points).shape[-1])
        axes = []
        for index, dimension in enumerate(dimensions):
            use_3d = view == "3d" or (view == "auto" and
                                     (ambient_dimension == 3 or
                                      (ambient_dimension is None and dimension == 3)))
            axes.append(figure.add_subplot(1, len(dimensions), index + 1,
                                           projection="3d" if use_3d else None))
        flat_axes = np.asarray(axes, dtype=object)
        figure.subplots_adjust(left=0.025, right=0.975, bottom=0.06,
                               top=0.80, wspace=0.18)
    else:
        flat_axes = np.asarray(axes, dtype=object).reshape(-1)
        if flat_axes.size != len(dimensions):
            raise ValueError("axes must contain exactly one axis per dimension")
        if any(not hasattr(axis, "figure") for axis in flat_axes):
            raise TypeError("axes must contain Matplotlib axes")
        figure = flat_axes[0].figure
        if any(axis.figure is not figure for axis in flat_axes):
            raise ValueError("all axes must belong to the same figure")
    plot = plot_simplex_block if kind == "simplicial" else plot_hyperdigraph_block
    for axis, dimension in zip(flat_axes, dimensions):
        plot(dimension, ax=axis, labels=labels, max_vertices=max_vertices, **options)
    return figure, flat_axes
