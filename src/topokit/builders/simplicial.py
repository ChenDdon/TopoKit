"""Point-cloud and graph constructors for defined simplicial objects.

Scientific point weights remain attributes, not alpha radii or Hodge metrics.
Alpha scales are squared radii; Rips scales are edge lengths. No homology or
Laplacian calculation runs implicitly during construction.
"""

import math

from ..results import Topology
from ..exceptions import ResourceLimitError
from .._validation import dimension as validate_dimension
from ..filtrations import resolve_range
from ..core.simplicial import SimplicialComplex, _unpack, _warn_coface_truncation
from . import _simplicial as _geometry
from .connections import prepare_cloud, resolve_connections, validate_cutoff


GeometryError = _geometry.GeometryError
__all__ = ["from_points", "from_graph", "GeometryError"]


def from_points(cloud, *, max_dimension=2, filtration_start=0,
                max_scale=math.inf, complex_type="alpha",
                max_simplex_dimension=None, bonds=None, cutoff=None, weights=None,
                filtration_range=None, **options):
    """Construct geometry once and retain raw births and stable point IDs.

    ``max_dimension`` is the planned analysis degree. Construction defaults
    to degree q+1 so Hq deaths and the full Lq up-term remain available. A lower
    explicit simplex dimension describes a deliberately truncated object.
    Births below ``filtration_start`` are clamped, not shifted; the raw values
    remain in metadata. ``max_scale`` is the inclusive native-filtration bound.

    Alpha (default) uses squared-radius births and does not accept bonds or a
    distance cutoff; use ``filtration_range``/``max_scale`` in alpha units.
    Alpha accepts ``backend='native'`` (default) or explicit optional
    ``backend='gudhi_exact'``. Both check coordinates before geometry and default
    to ``duplicates='merge'``: keep the first exact coordinate once, preserving
    its stable ID. ``duplicates='error'`` opts into strict rejection. Near points
    are never merged. Original-to-retained mappings and counts are in metadata.
    Native alpha uses batched circumspheres and
    coface propagation, with rational checks for uncertain geometry and bounded
    small-cloud repair; it never imports GUDHI. The latter uses exact geometry on unmodified
    coordinates and records its dependency version; it is not an automatic
    fallback. Its full-complex budget is checked after external construction.
    Rips uses the full distance graph and does not accept bonds. ``graph`` is
    a 1-complex; ``flag`` fills cliques. These latter two infer Delaunay support
    only when ``bonds is None``; supplied stable-ID pairs replace it, and ``[]``
    keeps vertices only. ``cutoff`` is an inclusive Euclidean-distance cap,
    also applied to supplied bonds. It is independent of the filtration range.
    ``weights=None`` preserves PointCloud weights; an explicit array/ID mapping
    overrides attributes only, never alpha radii or Laplacian inner products.
    """
    cloud = prepare_cloud(cloud, weights)
    maximum_dimension = validate_dimension(max_dimension)
    simplex_dimension = maximum_dimension + 1 if max_simplex_dimension is None else validate_dimension(
        max_simplex_dimension, "max_simplex_dimension")
    range_start, range_end = resolve_range(filtration_range, filtration_start=filtration_start, max_scale=max_scale)
    cutoff = validate_cutoff(cutoff)
    if "vertex_weights" in options:
        raise ValueError("Use weights for point attributes; weighted alpha geometry is not implemented")
    if complex_type not in {"alpha", "rips", "graph", "flag"}:
        raise ValueError("complex_type must be 'alpha', 'rips', 'graph', or 'flag'")
    if complex_type in {"alpha", "rips"} and bonds is not None:
        raise ValueError(f"{complex_type} does not accept bonds; choose complex_type='graph' or 'flag'")
    if complex_type == "alpha" and cutoff is not None:
        raise ValueError("alpha does not accept a distance cutoff; use max_scale/filtration_range in squared-radius units, or choose graph/flag")
    connection_metadata = {}
    if complex_type in {"graph", "flag"}:
        simplex_dimension = min(simplex_dimension, 1) if complex_type == "graph" else simplex_dimension
        if set(options) - {"max_simplices"}:
            raise ValueError("point graph/flag builders support only max_simplices as an additional option")
        simplex_budget = validate_dimension(options.get("max_simplices", 1_000_000), "max_simplices")
        if len(cloud) > simplex_budget:
            raise ResourceLimitError("Vertex count exceeds max_simplices before connection construction")
        connections = resolve_connections(cloud, bonds=bonds, cutoff=cutoff,
                                          max_connections=None)
        edges = tuple((i, j, distance) for (i, j), distance in zip(connections.pairs, connections.distances)
                      if distance <= range_end)
        if simplex_dimension >= 1 and len(cloud) + len(edges) > simplex_budget:
            raise ResourceLimitError("Graph vertices and edges exceed max_simplices before flag expansion")
        raw_complex = _geometry.graph_complex(
            edges, vertices=range(len(cloud)), flag=complex_type == "flag",
            max_dimension=simplex_dimension, **options)
        raw_complex.metadata["scale_units"] = "edge_length"
        connection_metadata = {**connections.metadata, "graph_expansion": "flag" if complex_type == "flag" else "none",
                               "filtration_excluded_connection_count": len(connections.pairs) - len(edges),
                               "edge_weights_role": "euclidean_distance_births"}
    else:
        raw_complex = _geometry.build_complex(
            cloud.points, complex_type=complex_type, max_dimension=simplex_dimension,
            max_scale=min(range_end, cutoff) if cutoff is not None else range_end, **options)
        connection_metadata = {"connection_support": "delaunay" if complex_type == "alpha" else "complete_distance",
                               "cutoff_distance": cutoff, "cutoff_policy": "inclusive_euclidean_distance"}
    native = SimplicialComplex(max_simplices=raw_complex.max_simplices)
    raw_filtration = raw_complex.get_filtration()
    for simplex, birth in raw_filtration:
        native._store(simplex, max(range_start, birth))
    original_to_vertex = raw_complex.metadata.get("original_to_vertex", tuple(range(len(cloud))))
    vertex_ids = {simplex[0]: cloud.ids[simplex[0]] for simplex in native.simplices(0)}
    geometry_dimension_bound = (
        min(1, max(len(cloud) - 1, 0)) if complex_type == "graph"
        else raw_complex.metadata.get("affine_dimension", max(len(cloud) - 1, 0)))
    metadata = dict(raw_complex.metadata)
    metadata.update({
        **connection_metadata,
        "route": "simplicial", "source": "topokit.builders.simplicial.from_points",
        "construction_kind": "point_cloud",
        "filtration_start": range_start, "filtration_end": range_end,
        "filtration_range": (range_start, range_end),
        "max_homology_dimension": maximum_dimension, "max_simplex_dimension": simplex_dimension,
        "geometry_dimension_bound": geometry_dimension_bound,
        "requested_death_dimension_available": simplex_dimension >= min(maximum_dimension + 1, geometry_dimension_bound),
        "raw_filtration": raw_filtration, "point_ids": cloud.ids,
        "vertex_id_map": vertex_ids, "vertex_to_id": vertex_ids,
        "original_id_to_vertex_id": {cloud.ids[i]: cloud.ids[j] for i, j in enumerate(original_to_vertex)},
        "geometry_weighted": False, "point_weights_role": "attributes_only",
        "point_metadata": dict(cloud.metadata),
        "coordinate_units": cloud.metadata.get("coordinate_units", cloud.metadata.get("coordinate_unit", "unspecified")),
    })
    _warn_coface_truncation(metadata, maximum_dimension)
    native.metadata = metadata.copy()
    native.validate()
    return Topology("simplicial", native, cloud, metadata)


def from_graph(edges=(), *, vertices=(), flag=False, max_dimension=2, **options):
    """Construct an undirected graph; clique filling requires ``flag=True``.

    Vertices use integer labels. Edges are ``(u, v)`` or ``(u, v, birth)``;
    ``vertices`` can also map labels to births. Supplied weights are filtration
    births, not Hodge inner-product weights. ``max_dimension`` controls optional
    flag expansion; without ``flag=True`` the graph stays a 1-complex.
    """
    maximum_dimension = validate_dimension(max_dimension)
    native = _geometry.graph_complex(edges, vertices=vertices, flag=flag,
                                      max_dimension=maximum_dimension, **options)
    _, _, metadata = _unpack(native)
    metadata.update({"source": "topokit.builders.simplicial.from_graph",
                     "construction_kind": "explicit_graph",
                     "graph_expansion": "flag" if flag else "none",
                     "max_simplex_dimension": maximum_dimension if flag else min(maximum_dimension, 1),
                     "edge_weights_role": "filtration_births",
                     "laplacian_inner_product": "unweighted_euclidean"})
    native.metadata = metadata.copy()
    return Topology("simplicial", native, None, metadata)
