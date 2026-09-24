"""Generic point-cloud and explicit-digraph hyperdigraph constructors."""
import math
import numpy as np
from ..results import Topology
from ..exceptions import ResourceLimitError
from .._validation import dimension as _dimension
from ..filtrations import resolve_range
from ..core._hyperdigraph.model import DEFINITION_ID, Hyperdigraph, FilteredHyperdigraph
from .connections import prepare_cloud, resolve_connections

def from_digraph(edges, vertices=None):
    """Use explicit directed edges as 1-hyperedges, without higher expansion."""
    edges = tuple(tuple(edge) for edge in edges)
    if any(len(edge) != 2 for edge in edges):
        raise ValueError("digraph edges must be ordered pairs")
    if vertices is None:
        vertices = tuple(dict.fromkeys(vertex for edge in edges for vertex in edge))
    return Hyperdigraph(vertices, edges, include_all_vertices=True)

def from_points(cloud, max_dimension=2, filtration_start=0, max_scale=math.inf,
                *, max_hyperedges=500_000, bonds=None, cutoff=None, weights=None,
                object_type="hyperdigraph", filtration_range=None):
    """Build a filtered sequence hyperdigraph from a generic weighted cloud.

    A retained pair points from lower to higher weight; exact equal
    weights retain BOTH directions. Directed sequences contain distinct IDs
    and extend along consecutive allowed edges through max_dimension + 1.
    Effective birth is max(filtration_start, raw distance birth). The start
    truncates the filtration; it does not translate geometric scale values.

    ``bonds=None`` infers Delaunay support; explicit stable-ID pairs replace it,
    and ``bonds=[]`` retains vertices only. ``cutoff`` is an inclusive physical
    distance cap, including for supplied bonds, independently of the filtration
    domain. ``weights=None`` preserves a PointCloud's weights; raw coordinates
    default to uniform weights, and explicit arrays/ID mappings override them.
    ``object_type='digraph'`` caps construction at degree 1, without expanding
    higher sequences. Explicitly directed static edges still use from_digraph.
    """
    cloud = prepare_cloud(cloud, weights)
    q = _dimension(max_dimension)
    start, end_scale = resolve_range(filtration_range, filtration_start=filtration_start, max_scale=max_scale)
    if object_type not in {"hyperdigraph", "digraph"}:
        raise ValueError("object_type must be 'hyperdigraph' or 'digraph'")
    build_dimension = 1 if object_type == "digraph" else q + 1
    budget = _dimension(max_hyperedges, "max_hyperedges")
    if len(cloud) == 0:
        raise ValueError("a native hyperdigraph requires at least one point")
    if len(cloud) > budget:
        raise ResourceLimitError("Singleton count exceeds max_hyperedges")
    connections = resolve_connections(cloud, bonds=bonds, cutoff=cutoff)
    adjacency = [[] for _ in cloud.ids]
    edge_count = 0
    filtration_excluded = 0
    for (i, j), distance in zip(connections.pairs, connections.distances):
        i, j = int(i), int(j)
        if distance > end_scale:
            filtration_excluded += 1
            continue
        if cloud.weights[i] <= cloud.weights[j]:
            adjacency[i].append((j, distance)); edge_count += 1
        if cloud.weights[j] <= cloud.weights[i]:
            adjacency[j].append((i, distance)); edge_count += 1
    for neighbors in adjacency:
        neighbors.sort()
    # Extend one sequence degree at a time, keeping raw and clamped births separate.
    previous_sequences = [((i,), 0.0) for i in range(len(cloud))]
    hyperedge_total = len(previous_sequences)
    raw_birth_records = [((label,), 0.0) for label in cloud.ids]
    effective_birth_records = []
    hyperedge_counts = [len(cloud)]
    for _q in range(1, build_dimension + 1):
        current_sequences = []
        for sequence, birth in previous_sequences:
            for target, distance in adjacency[sequence[-1]]:
                if target in sequence:
                    continue
                hyperedge_total += 1
                if hyperedge_total > budget:
                    raise ResourceLimitError(
                        f"Directed hyperedge construction exceeds max_hyperedges={budget:,}; "
                        "no topology was silently pruned"
                    )
                extended = sequence + (target,)
                raw_birth = max(birth, distance)
                edge = tuple(cloud.ids[i] for i in extended)
                current_sequences.append((extended, raw_birth))
                raw_birth_records.append((edge, raw_birth))
                effective_birth_records.append((edge, max(start, raw_birth)))
        hyperedge_counts.append(len(current_sequences))
        previous_sequences = current_sequences
    native = FilteredHyperdigraph(cloud.ids, effective_birth_records, include_all_vertices=True, vertex_birth=start)
    coordinate_units = cloud.metadata.get("coordinate_units", cloud.metadata.get("coordinate_unit", "input_coordinate_unit"))
    metadata = {
        "route": "hyperdigraph",
        "definition_id": DEFINITION_ID, "core_version": "0.6.0",
        "construction": (f"{connections.metadata['connection_support']}_supported_distinct_vertex_directed_sequences"
                         if object_type == "hyperdigraph" else f"{connections.metadata['connection_support']}_supported_digraph"),
        "object_type": object_type, "orientation": "lower_to_higher_equal_both",
        "weight_role": "direction_only", "filtration_coordinate": "euclidean_distance",
        "scale_units": "coordinate_distance", "coordinate_units": coordinate_units,
        "coordinate_unit": coordinate_units,
        "filtration_start": start, "filtration_end": end_scale,
        "filtration_range": (start, end_scale),
        "filtration_start_policy": "clamp_not_shift", "raw_births": tuple(raw_birth_records),
        "construction_max_dimension": build_dimension, "max_analysis_dimension": q,
        "hyperedge_counts": tuple(hyperedge_counts), "directed_edge_count": edge_count,
        "vertex_ids": cloud.ids, "point_weights": tuple(map(float, cloud.weights)),
        "point_metadata": dict(cloud.metadata),
        "filtration_excluded_connection_count": filtration_excluded,
        "left_truncated": start > 0, "right_censored": math.isfinite(end_scale),
        **connections.metadata,
    }
    import scipy
    metadata["geometry_versions"] = {"numpy": np.__version__, "scipy": scipy.__version__}
    return Topology("hyperdigraph", native, cloud, metadata)

__all__ = ["from_points", "from_digraph"]
