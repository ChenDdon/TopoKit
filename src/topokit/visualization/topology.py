"""Read-only object views using public cell accessors and explicit coordinates.

These adapters select supplied cells. They do not construct missing faces,
expand digraphs, or compute topology. Their rendering cloud contains just the
vertices referenced by the selected cells; include dimension zero to retain
isolated vertices. Endpoint markers do not imply singleton hyperedge membership.
"""

from collections.abc import Mapping
from itertools import islice
from numbers import Integral, Real
import math

from .._validation import dimension as _nonnegative_integer, validate_query
from ..data import as_point_cloud
from ..exceptions import ResourceLimitError
from ..results import Topology


__all__ = ["plot_simplicial_complex", "plot_hyperdigraph", "plot_topology"]


def _dimensions(value, scan_budget):
    if value is None:
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return (_nonnegative_integer(value, "dimensions"),)
    if isinstance(value, (str, bytes, Mapping, bool)):
        raise ValueError("dimensions must be a nonnegative integer or an iterable of them")
    try:
        iterator = iter(value)
    except TypeError as error:
        raise ValueError("dimensions must be a nonnegative integer or an iterable of them") from error
    result = set()
    for count, degree in enumerate(iterator, 1):
        if count > max(scan_budget, 1):
            raise ResourceLimitError("The dimensions iterable exceeds max_scan_cells")
        result.add(_nonnegative_integer(degree, "dimensions"))
    return tuple(sorted(result))


def _prepare(topology, kind, cloud, scale, dimensions, options, max_scan_cells):
    # Validate resource controls before accessing a potentially large object.
    cell_budget = _nonnegative_integer(options.get("max_cells", 1000 if kind == "simplicial" else 500),
                                       "max_cells")
    primitive_budget = _nonnegative_integer(options.get("max_primitives", 20_000), "max_primitives")
    scan_budget = _nonnegative_integer(max_scan_cells, "max_scan_cells")
    degrees = _dimensions(dimensions, scan_budget)
    if isinstance(topology, Topology):
        if topology.kind != kind:
            raise ValueError(f"Expected a {kind} Topology, received {topology.kind!r}")
        native = topology.native
        metadata = dict(getattr(native, "metadata", {}))
        metadata.update(topology.metadata)
        cloud = topology.cloud if cloud is None else cloud
    else:
        native = topology
        metadata = dict(getattr(native, "metadata", {}))
    if cloud is None:
        raise ValueError("This object has no coordinates; supply cloud=PointCloud(points, ids=vertex_ids) "
                         "with explicit 2D or 3D coordinates")
    cloud = as_point_cloud(cloud)
    if cloud.points.shape[1] not in (2, 3):
        raise ValueError("Object views need 2D/3D coordinates; project explicitly and pass cloud=PointCloud(...)")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, Real)
                              or not math.isfinite(float(scale))):
        raise ValueError("scale must be a finite real number")
    value = validate_query(metadata, scale)
    return native, metadata, cloud, value, degrees, cell_budget, primitive_budget, scan_budget


def _groups(accessor, degrees):
    if degrees is None:
        yield accessor()
    else:
        for degree in degrees:
            yield accessor(degree)


def _cell(raw, primitive_budget):
    if isinstance(raw, (str, bytes, Mapping, set, frozenset)):
        raise ValueError("Each cell must be an ordered, nonempty sequence of vertex IDs")
    # A long or nonterminating cell is refused without first materializing it.
    try:
        result = tuple(islice(iter(raw), primitive_budget + 2))
    except TypeError as error:
        raise ValueError("Each cell must be an ordered, nonempty sequence of vertex IDs") from error
    if len(result) > primitive_budget + 1:
        raise ResourceLimitError("A cell exceeds max_primitives; select a smaller view or raise the budget")
    if not result:
        raise ValueError("Cells must be nonempty")
    return result


def _select(groups, unpack, scale, degrees, cell_budget, primitive_budget, scan_budget):
    selected = []
    scanned = 0
    for records in groups:
        for record in records:
            scanned += 1
            if scanned > scan_budget:
                raise ResourceLimitError("Cell inspection exceeds max_scan_cells; select dimensions or raise the budget")
            raw, birth = unpack(record)
            if scale is not None and birth > scale:
                continue
            cell = _cell(raw, primitive_budget)
            if degrees is not None and len(cell) - 1 not in degrees:
                continue
            if len(selected) >= cell_budget:
                raise ResourceLimitError("Selected cells exceed max_cells; select a scale/dimensions or raise the budget")
            selected.append(cell)
    return tuple(selected)


def _view_cloud(cloud, cells):
    lookup = cloud.index
    used = set()
    for cell in cells:
        for label in cell:
            if label not in lookup:
                raise ValueError(f"Cell references unknown point ID {label!r}; supply coordinates with matching stable IDs")
            used.add(label)
    return cloud.subset(label for label in cloud.ids if label in used)


def _select_simplicial_view(topology, *, scale=None, dimensions=None, cloud=None,
                            max_scan_cells=1_000_000, options=None):
    """Return validated display data shared by simplicial presentation APIs."""
    options = {} if options is None else options
    native, metadata, cloud, value, degrees, cell_budget, primitive_budget, scan_budget = _prepare(
        topology, "simplicial", cloud, scale, dimensions, options, max_scan_cells)
    if degrees is None and hasattr(native, "__len__") and len(native) > scan_budget:
        raise ResourceLimitError("Cell inspection exceeds max_scan_cells; select dimensions or raise the budget")
    if degrees is None and callable(getattr(native, "get_filtration", None)):
        groups = (native.get_filtration(),)
        unpack = lambda record: record
    elif callable(getattr(native, "simplices", None)) and callable(getattr(native, "filtration", None)):
        groups = _groups(native.simplices, degrees)
        unpack = lambda cell: (cell, native.filtration(cell))
    elif callable(getattr(native, "get_filtration", None)):
        groups = (native.get_filtration(),) if degrees != () else ()
        unpack = lambda record: record
    else:
        raise TypeError("Supply a simplicial Topology or a native object with public simplex/filtration accessors")
    cells = _select(groups, unpack, value, degrees, cell_budget, primitive_budget, scan_budget)
    mapping = metadata.get("vertex_to_id", metadata.get("vertex_id_map"))
    if mapping is not None:
        try:
            cells = tuple(tuple(mapping[label] for label in cell) for cell in cells)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("The simplicial object's vertex ID map is incomplete or invalid") from error
    return _view_cloud(cloud, cells), cells, value, metadata


def _select_hyperdigraph_view(topology, *, scale=None, dimensions=None, cloud=None,
                              max_scan_cells=1_000_000, options=None):
    """Return validated display data shared by hyperdigraph presentation APIs."""
    options = {} if options is None else options
    native, metadata, cloud, value, degrees, cell_budget, primitive_budget, scan_budget = _prepare(
        topology, "hyperdigraph", cloud, scale, dimensions, options, max_scan_cells)
    if callable(getattr(native, "weighted_hyperedges", None)):
        groups = _groups(native.weighted_hyperedges, degrees)
        unpack = lambda record: (record.vertices, record.birth)
    elif callable(getattr(native, "directed_hyperedges", None)):
        if value is not None:
            raise ValueError("scale applies only to a filtered hyperdigraph")
        groups = _groups(native.directed_hyperedges, degrees)
        unpack = lambda cell: (cell, 0.0)
    else:
        raise TypeError("Supply a hyperdigraph Topology or a native object with public hyperedge accessors")
    cells = _select(groups, unpack, value, degrees, cell_budget, primitive_budget, scan_budget)
    return _view_cloud(cloud, cells), cells, value, metadata


def plot_simplicial_complex(topology, *, scale=None, dimensions=None, cloud=None,
                            max_scan_cells=1_000_000, **options):
    """Display the selected cells of a simplicial object or ``Topology``.

    ``scale`` selects inclusive births in the recorded filtration domain;
    ``None`` displays all supplied cells. ``dimensions`` is an integer or an
    iterable of exact degrees, not a construction cap. Higher-dimensional
    boundaries remain present in the original object. Coordinates come from
    the envelope or an explicit ``cloud``. Builder ID maps are respected;
    static native vertex labels are matched exactly to cloud IDs.

    ``max_cells`` (default 1000) limits selected cells. ``max_scan_cells`` caps
    examined records independently, including cells discarded by the scale
    filter. Native degree accessors restrict inspection where available.
    ``max_primitives`` (default 20000) and other options are forwarded to
    :func:`plot_simplices`. Only vertices in selected cells are displayed.
    """
    from .objects import plot_simplices

    visible_cloud, cells, value, _ = _select_simplicial_view(
        topology, scale=scale, dimensions=dimensions, cloud=cloud,
        max_scan_cells=max_scan_cells, options=options)
    options.setdefault("title", "Simplicial complex" if value is None else f"Simplicial complex at {value:g}")
    return plot_simplices(visible_cloud, cells, **options)


def plot_hyperdigraph(topology, *, scale=None, dimensions=None, cloud=None,
                      max_scan_cells=1_000_000, **options):
    """Display supplied ordered hyperedges, preserving their sequence order.

    Accepts a hyperdigraph ``Topology`` or native object exposing
    ``directed_hyperedges``/``weighted_hyperedges``. ``scale`` selects inclusive
    births for filtered objects only. ``dimensions`` selects exact degrees;
    it never expands paths into ordered cliques or inserts boundary hyperedges.
    Vertex IDs match the explicit or attached cloud exactly, including strings.

    ``max_cells`` (default 500) caps selected hyperedges, and ``max_scan_cells``
    caps inspected records even when the scale filter discards them. Rendering
    options, including ``max_primitives`` (default 20000), are forwarded to
    :func:`plot_hyperedges`. The view contains only selected-cell endpoints;
    their coordinate markers do not assert singleton hyperedge membership.
    """
    from .objects import plot_hyperedges

    visible_cloud, cells, value, _ = _select_hyperdigraph_view(
        topology, scale=scale, dimensions=dimensions, cloud=cloud,
        max_scan_cells=max_scan_cells, options=options)
    options.setdefault("title", "Sequence hyperdigraph" if value is None else f"Sequence hyperdigraph at {value:g}")
    return plot_hyperedges(visible_cloud, cells, **options)


def plot_topology(topology, **options):
    """Dispatch simplicial and sequence-hyperdigraph views only.

    Interaction quotient chains require the separate ``plot_interaction_factors``
    view; no geometric embedding is inferred for them.
    """
    kind = topology.kind if isinstance(topology, Topology) else None
    if kind == "simplicial" or (kind is None and (callable(getattr(topology, "simplices", None))
                                                  or callable(getattr(topology, "get_filtration", None)))):
        return plot_simplicial_complex(topology, **options)
    if kind == "hyperdigraph" or (kind is None and callable(getattr(topology, "directed_hyperedges", None))):
        return plot_hyperdigraph(topology, **options)
    raise ValueError("plot_topology supports simplicial and hyperdigraph objects only; "
                     "use plot_interaction_factors for interaction factor views")
