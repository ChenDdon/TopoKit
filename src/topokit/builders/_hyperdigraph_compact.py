"""Construction of compact score-ordered hyperdigraph inputs; no analysis."""
from __future__ import annotations
from array import array
from collections.abc import Hashable, Iterable, Sequence
from numbers import Real
from typing import Any
import math
from ..core._hyperdigraph.model import SequenceHyperdigraph
from ..core._hyperdigraph.path_compatible import ScoreDirectedEdgeFiltration, _MAX_PACKED_VERTEX_COUNT
from ._hyperdigraph_geometry import (ConnectionSupport, PointCloudSupport, build_point_cloud_support,
                                    distance_cutoff, materialize_points, score_order)

def score_directed_filtration_from_points(
    points: Iterable[Iterable[Any]],
    point_weights: Iterable[Any],
    max_distance: Any | None = None,
    *,
    connection_support: ConnectionSupport = "delaunay",
    storage_backend: str = "auto",
) -> ScoreDirectedEdgeFiltration:
    """Construct a score-directed Euclidean edge filtration.

    ``connection_support='delaunay'`` (the default) uses the undirected
    Delaunay 1-skeleton as a geometric maximum constraint before orienting
    each retained edge from lower to higher direction score.  Use
    ``'complete'`` to recover the legacy all-pairs construction.
    """

    coordinates = materialize_points(points)
    order = score_order(point_weights, len(coordinates))
    cutoff = distance_cutoff(max_distance)
    support = build_point_cloud_support(coordinates, connection_support)
    if storage_backend not in {"auto", "numpy", "python"}:
        raise ValueError("storage_backend must be 'auto', 'numpy', or 'python'")
    if storage_backend != "python":
        try:
            return _build_numpy_filtration(coordinates, order, cutoff, support)
        except ImportError:
            if storage_backend == "numpy":
                raise
    return _build_python_filtration(coordinates, order, cutoff, support)


def _critical_count(sorted_weights: Sequence[float]) -> int:
    if not sorted_weights:
        return 1
    distinct_count = 1
    previous_weight = float(sorted_weights[0])
    for index in range(1, len(sorted_weights)):
        current_weight = float(sorted_weights[index])
        if current_weight != previous_weight:
            distinct_count += 1
            previous_weight = current_weight
    return distinct_count + int(float(sorted_weights[0]) != 0.0)

def _build_python_filtration(
    coordinates: tuple[tuple[float, ...], ...],
    order: tuple[int, ...],
    cutoff: float | None,
    support: PointCloudSupport,
) -> ScoreDirectedEdgeFiltration:
    count = len(coordinates)
    if count > _MAX_PACKED_VERTEX_COUNT:
        raise ValueError("packed point-cloud filtrations support at most 65,535 points")
    edge_records: list[tuple[float, int]] = []
    inverse_order = [0] * count
    for score_rank, original_index in enumerate(order):
        inverse_order[original_index] = score_rank
    candidate_pairs = support.pairs
    if candidate_pairs is None:
        for source in range(count - 1):
            original_left = order[source]
            for target in range(source + 1, count):
                original_right = order[target]
                distance = math.dist(
                    coordinates[original_left], coordinates[original_right]
                )
                if cutoff is None or distance <= cutoff:
                    edge_records.append((distance, (source << 16) | target))
    else:
        for original_left, original_right in candidate_pairs:
            source = inverse_order[original_left]
            target = inverse_order[original_right]
            if source > target:
                source, target = target, source
            distance = math.dist(
                coordinates[original_left], coordinates[original_right]
            )
            if cutoff is None or distance <= cutoff:
                edge_records.append((distance, (source << 16) | target))
    edge_records.sort()
    weights = array("d", (record[0] for record in edge_records))
    pairs = array("I", (record[1] for record in edge_records))
    return ScoreDirectedEdgeFiltration(
        count,
        pairs,
        weights,
        order,
        "python_packed_arrays",
        _critical_count(weights),
        support.name,
        support.ambient_dimension,
        support.intrinsic_dimension,
        support.edge_count,
        support.maximal_simplex_count,
        support.projected_to_affine_hull,
    )

def score_directed_distance_matrix(
    filtration: ScoreDirectedEdgeFiltration,
) -> tuple[tuple[float, ...], ...]:
    """Materialize directed distances from a compact score-oriented support."""

    if not isinstance(filtration, ScoreDirectedEdgeFiltration):
        raise TypeError("filtration must be a ScoreDirectedEdgeFiltration")
    distances = [
        [
            0.0 if row == column else math.inf
            for column in range(filtration.vertex_count)
        ]
        for row in range(filtration.vertex_count)
    ]
    for packed, weight in zip(filtration.packed_pairs, filtration.edge_weights):
        ordered_source = int(packed) >> 16
        ordered_target = int(packed) & 0xFFFF
        source = filtration.score_order[ordered_source]
        target = filtration.score_order[ordered_target]
        distances[source][target] = float(weight)
    return tuple(tuple(row) for row in distances)

def _build_numpy_filtration(
    coordinates: tuple[tuple[float, ...], ...],
    order: tuple[int, ...],
    cutoff: float | None,
    support: PointCloudSupport,
) -> ScoreDirectedEdgeFiltration:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "storage_backend='numpy' requires NumPy; use 'python' for small inputs"
        ) from exc

    count = len(coordinates)
    if count > _MAX_PACKED_VERTEX_COUNT:
        raise ValueError("packed point-cloud filtrations support at most 65,535 points")
    cloud = np.asarray(coordinates, dtype=np.float64)
    if support.pairs is None:
        pair_capacity = count * (count - 1) // 2
        ordered = cloud[np.asarray(order)]
        weights = np.empty(pair_capacity, dtype=np.float64)
        pairs = np.empty(pair_capacity, dtype=np.uint32)
        edge_cursor = 0
        for source in range(count - 1):
            differences = ordered[source + 1 :] - ordered[source]
            distances = np.sqrt(np.einsum("ij,ij->i", differences, differences))
            targets = np.arange(source + 1, count, dtype=np.uint32)
            if cutoff is not None:
                retained = distances <= cutoff
                distances = distances[retained]
                targets = targets[retained]
            retained_count = len(distances)
            weights[edge_cursor : edge_cursor + retained_count] = distances
            pairs[edge_cursor : edge_cursor + retained_count] = (
                np.uint32(source) << np.uint32(16)
            ) | targets
            edge_cursor += retained_count
        weights = weights[:edge_cursor]
        pairs = pairs[:edge_cursor]
    else:
        original_pairs = np.asarray(support.pairs, dtype=np.int64)
        differences = cloud[original_pairs[:, 1]] - cloud[original_pairs[:, 0]]
        weights = np.sqrt(np.einsum("ij,ij->i", differences, differences))
        inverse_order = np.empty(count, dtype=np.uint32)
        inverse_order[np.asarray(order, dtype=np.int64)] = np.arange(
            count, dtype=np.uint32
        )
        left_ranks = inverse_order[original_pairs[:, 0]]
        right_ranks = inverse_order[original_pairs[:, 1]]
        sources = np.minimum(left_ranks, right_ranks)
        targets = np.maximum(left_ranks, right_ranks)
        pairs = (sources << np.uint32(16)) | targets
        if cutoff is not None:
            retained = weights <= cutoff
            weights = weights[retained]
            pairs = pairs[retained]
    permutation = np.lexsort((pairs, weights))
    sorted_weights = np.ascontiguousarray(weights[permutation])
    sorted_pairs = np.ascontiguousarray(pairs[permutation])
    sorted_weights.setflags(write=False)
    sorted_pairs.setflags(write=False)
    if len(sorted_weights):
        distinct = int(np.count_nonzero(sorted_weights[1:] != sorted_weights[:-1])) + 1
        critical_value_count = distinct + int(sorted_weights[0] != 0.0)
    else:
        critical_value_count = 1
    return ScoreDirectedEdgeFiltration(
        count,
        sorted_pairs,
        sorted_weights,
        order,
        "numpy_packed_arrays",
        critical_value_count,
        support.name,
        support.ambient_dimension,
        support.intrinsic_dimension,
        support.edge_count,
        support.maximal_simplex_count,
        support.projected_to_affine_hull,
    )


def _score_directed_distance_matrix(
    points: Iterable[Iterable[Real]],
    point_weights: Iterable[Real],
    max_distance: Real | None,
    connection_support: ConnectionSupport,
    storage_backend: str,
) -> tuple[
    tuple[tuple[float, ...], ...],
    ScoreDirectedEdgeFiltration,
]:
    """Reuse the certified point-cloud orientation used by homology."""

    compact = score_directed_filtration_from_points(
        points,
        point_weights,
        max_distance=max_distance,
        connection_support=connection_support,
        storage_backend=storage_backend,
    )
    return score_directed_distance_matrix(compact), compact

def _point_cloud_maximum_adjacency(
    compact: ScoreDirectedEdgeFiltration,
    minimum_adjacency: object | None,
    maximum_adjacency: object | None,
) -> tuple[
    tuple[tuple[bool, ...], ...] | None,
    tuple[tuple[bool, ...], ...],
]:
    """Intersect public adjacency bounds with the oriented geometric support."""

    from ._hyperdigraph_converters import _binary_matrix

    size = compact.vertex_count
    support = [[False for _ in range(size)] for _ in range(size)]
    for packed in compact.packed_pairs:
        ordered_source = int(packed) >> 16
        ordered_target = int(packed) & 0xFFFF
        source = compact.score_order[ordered_source]
        target = compact.score_order[ordered_target]
        support[source][target] = True

    if maximum_adjacency is not None:
        requested_maximum = _binary_matrix(maximum_adjacency, "maximum_adjacency")
        if len(requested_maximum) != size:
            raise ValueError("maximum_adjacency shape must match the point cloud")
        for source in range(size):
            for target in range(size):
                support[source][target] = (
                    support[source][target] and requested_maximum[source][target]
                )

    requested_minimum: tuple[tuple[bool, ...], ...] | None = None
    if minimum_adjacency is not None:
        requested_minimum = _binary_matrix(minimum_adjacency, "minimum_adjacency")
        if len(requested_minimum) != size:
            raise ValueError("minimum_adjacency shape must match the point cloud")
        for source in range(size):
            for target in range(size):
                if requested_minimum[source][target] and not support[source][target]:
                    raise ValueError(
                        "minimum_adjacency must be contained in the oriented "
                        "point-cloud connection support, max_distance cutoff, "
                        "and maximum_adjacency"
                    )
    return requested_minimum, tuple(tuple(row) for row in support)

def _validate_minimum_hyperdigraph_point_support(
    minimum_hyperdigraph: SequenceHyperdigraph | None,
    labels: tuple[Hashable, ...],
    maximum_adjacency: tuple[tuple[bool, ...], ...],
) -> None:
    if minimum_hyperdigraph is None:
        return
    label_index = {label: index for index, label in enumerate(labels)}
    for dimension in minimum_hyperdigraph.dimensions:
        if dimension == 0:
            continue
        for hyperedge in minimum_hyperdigraph.directed_hyperedges(dimension):
            try:
                indexed = tuple(label_index[label] for label in hyperedge)
            except KeyError as exc:
                raise ValueError(
                    "minimum_hyperdigraph contains a vertex outside the point cloud"
                ) from exc
            if any(
                not maximum_adjacency[source][target]
                for source, target in zip(indexed[:-1], indexed[1:])
            ):
                raise ValueError(
                    "minimum_hyperdigraph must use consecutive directed edges "
                    "contained in the point-cloud connection support, "
                    "max_distance cutoff, and maximum bounds"
                )
