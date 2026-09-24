"""Input adapters and bounded path-hyperdigraph filtrations.

The matrix adapters intentionally generate only *simple directed paths*:
vertices cannot repeat inside a sequence.  This agrees with the native
sequence-hyperdigraph definition used by the package and with the legacy
``connected_map_to_dihyperedge`` example.

Two independent kinds of bounds are supported:

``minimum``
    Connections or explicit hyperedges that are always present.  In a
    filtration they enter at ``filtration_start`` (zero by default).

``maximum``
    An allow-list.  It prevents unnecessary directed connections or explicit
    higher hyperedges from being generated.

All bounds default to ``None`` and therefore do not modify the standard
construction.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from math import inf, isfinite, isnan
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

from ..core._hyperdigraph.model import (
    FilteredSequenceHyperdigraph,
    Hyperedge,
    SequenceHyperdigraph,
    WeightedHyperedge,
)

if TYPE_CHECKING:
    from ..core._hyperdigraph.results import HomologyResult, PersistenceResult


_RealValue = Real | float


def _validate_nonnegative_dimension(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _square_rows(matrix: object, name: str) -> tuple[tuple[object, ...], ...]:
    if isinstance(matrix, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a nonempty square matrix")
    try:
        iterable = cast(Iterable[Iterable[object]], matrix)
        rows = tuple(tuple(row) for row in iterable)
    except TypeError as exc:
        raise TypeError(f"{name} must be a nonempty square matrix") from exc
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError(f"{name} must be a nonempty square matrix")
    return rows


def _binary_matrix(matrix: object, name: str) -> tuple[tuple[bool, ...], ...]:
    rows = _square_rows(matrix, name)
    converted: list[tuple[bool, ...]] = []
    for row_index, row in enumerate(rows):
        converted_row: list[bool] = []
        for column_index, value in enumerate(row):
            try:
                number = float(cast(Any, value))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} entries must be binary") from exc
            if number not in {0.0, 1.0}:
                raise ValueError(f"{name} entries must be 0 or 1")
            active = bool(number)
            if row_index == column_index and active:
                raise ValueError(
                    f"{name} must have a zero diagonal; repeated vertices "
                    "are not valid hyperedges"
                )
            converted_row.append(active)
        converted.append(tuple(converted_row))
    return tuple(converted)


def _distance_matrix(matrix: object) -> tuple[tuple[float, ...], ...]:
    rows = _square_rows(matrix, "distance_matrix")
    converted: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        converted_row: list[float] = []
        for column_index, value in enumerate(row):
            if value is None:
                number = inf
            else:
                try:
                    number = float(cast(Any, value))
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "distance_matrix entries must be non-negative real "
                        "numbers, infinity, or None"
                    ) from exc
            if isnan(number) or number < 0:
                raise ValueError(
                    "distance_matrix entries must be non-negative and not NaN"
                )
            if row_index == column_index:
                number = 0.0
            converted_row.append(number)
        converted.append(tuple(converted_row))
    return tuple(converted)


def _vertex_labels(
    size: int, vertices: Iterable[Hashable] | None
) -> tuple[Hashable, ...]:
    if vertices is None:
        labels: tuple[Hashable, ...] = tuple(range(size))
    else:
        labels = tuple(vertices)
    if len(labels) != size:
        raise ValueError("vertices must have one label per matrix row")
    # Reuse the model's complete validation of hashability and uniqueness.
    return SequenceHyperdigraph(labels).vertices


def _mask_or_default(
    matrix: object | None,
    size: int,
    name: str,
    default: bool,
) -> tuple[tuple[bool, ...], ...]:
    if matrix is None:
        return tuple(
            tuple(default and row != column for column in range(size))
            for row in range(size)
        )
    converted = _binary_matrix(matrix, name)
    if len(converted) != size:
        raise ValueError(f"{name} shape must match the input matrix")
    return converted


def _effective_adjacency(
    adjacency: tuple[tuple[bool, ...], ...],
    minimum_adjacency: object | None,
    maximum_adjacency: object | None,
) -> tuple[tuple[bool, ...], ...]:
    size = len(adjacency)
    minimum_mask = _mask_or_default(
        minimum_adjacency, size, "minimum_adjacency", False
    )
    maximum_mask = _mask_or_default(
        maximum_adjacency, size, "maximum_adjacency", True
    )
    for row in range(size):
        for column in range(size):
            if minimum_mask[row][column] and not maximum_mask[row][column]:
                raise ValueError(
                    "minimum_adjacency must be contained in maximum_adjacency"
                )
    return tuple(
        tuple(
            minimum_mask[row][column]
            or (adjacency[row][column] and maximum_mask[row][column])
            for column in range(size)
        )
        for row in range(size)
    )


def _index_paths(
    adjacency: tuple[tuple[bool, ...], ...], max_path_dimension: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    neighbors = tuple(
        tuple(column for column, active in enumerate(row) if active)
        for row in adjacency
    )
    paths_by_dimension: list[tuple[tuple[int, ...], ...]] = [
        tuple((vertex,) for vertex in range(len(adjacency)))
    ]
    for _dimension in range(1, max_path_dimension + 1):
        paths: list[tuple[int, ...]] = []
        for path in paths_by_dimension[-1]:
            for target in neighbors[path[-1]]:
                if target not in path:
                    paths.append(path + (target,))
        paths_by_dimension.append(tuple(paths))
    return tuple(paths_by_dimension)


def _index_path_births(
    edge_births: Mapping[tuple[int, int], float],
    size: int,
    max_path_dimension: int,
    filtration_start: float,
) -> tuple[tuple[tuple[tuple[int, ...], float], ...], ...]:
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(size)]
    for (source, target), birth in edge_births.items():
        neighbors[source].append((target, birth))
    weighted_paths_by_dimension: list[tuple[tuple[tuple[int, ...], float], ...]] = [
        tuple(((vertex,), filtration_start) for vertex in range(size))
    ]
    for _dimension in range(1, max_path_dimension + 1):
        weighted_paths: list[tuple[tuple[int, ...], float]] = []
        for path, path_birth in weighted_paths_by_dimension[-1]:
            for target, edge_birth in neighbors[path[-1]]:
                if target not in path:
                    weighted_paths.append(
                        (path + (target,), max(path_birth, edge_birth))
                    )
        weighted_paths_by_dimension.append(tuple(weighted_paths))
    return tuple(weighted_paths_by_dimension)


def _bound_hyperedges(
    bound: SequenceHyperdigraph | None,
    vertices: tuple[Hashable, ...],
    name: str,
) -> tuple[Hyperedge, ...] | None:
    if bound is None:
        return None
    if not isinstance(bound, SequenceHyperdigraph):
        raise TypeError(f"{name} must be a Hyperdigraph or None")
    if set(bound.vertices) != set(vertices):
        raise ValueError(f"{name} must use the same ambient vertices")
    return bound.directed_hyperedges()


def _apply_hyperdigraph_bounds(
    generated: Iterable[Hyperedge],
    vertices: tuple[Hashable, ...],
    max_path_dimension: int,
    minimum_hyperdigraph: SequenceHyperdigraph | None,
    maximum_hyperdigraph: SequenceHyperdigraph | None,
) -> tuple[Hyperedge, ...]:
    minimum_order = _bound_hyperedges(
        minimum_hyperdigraph, vertices, "minimum_hyperdigraph"
    )
    maximum_order = _bound_hyperedges(
        maximum_hyperdigraph, vertices, "maximum_hyperdigraph"
    )
    required_hyperedges = set(minimum_order or ())
    allowed_hyperedges = set(maximum_order) if maximum_order is not None else None
    if allowed_hyperedges is not None and not required_hyperedges <= allowed_hyperedges:
        raise ValueError(
            "minimum_hyperdigraph must be contained in maximum_hyperdigraph"
        )
    over_dimension_hyperedges = [
        edge for edge in required_hyperedges if len(edge) - 1 > max_path_dimension
    ]
    if over_dimension_hyperedges:
        raise ValueError(
            "minimum_hyperdigraph contains a hyperedge above "
            "max_path_dimension"
        )

    selected_hyperedges: list[Hyperedge] = []
    seen: set[Hyperedge] = set()
    for edge in generated:
        # F_0 is controlled by include_all_vertices or explicit singleton data,
        # rather than by a maximum positive-dimensional path allow-list.
        allowed = len(edge) == 1 or allowed_hyperedges is None or edge in allowed_hyperedges
        if allowed and edge not in seen:
            selected_hyperedges.append(edge)
            seen.add(edge)
    for edge in minimum_order or ():
        if edge not in seen:
            selected_hyperedges.append(edge)
            seen.add(edge)
    return tuple(selected_hyperedges)


def hyperdigraph_from_adjacency_matrix(
    adjacency_matrix: object,
    *,
    vertices: Iterable[Hashable] | None = None,
    max_path_dimension: int = 2,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    include_all_vertices: bool = True,
) -> SequenceHyperdigraph:
    """Generate a native hyperdigraph from a binary directed adjacency matrix.

    Positive-dimensional hyperedges are all simple directed paths through
    ``max_path_dimension`` after applying the optional minimum/maximum bounds.
    """

    _validate_nonnegative_dimension(max_path_dimension, "max_path_dimension")
    adjacency = _binary_matrix(adjacency_matrix, "adjacency_matrix")
    labels = _vertex_labels(len(adjacency), vertices)
    effective = _effective_adjacency(
        adjacency, minimum_adjacency, maximum_adjacency
    )
    indexed_paths = _index_paths(effective, max_path_dimension)
    generated = tuple(
        tuple(labels[index] for index in path)
        for paths in indexed_paths
        for path in paths
        if len(path) > 1
    )
    selected_hyperedges = _apply_hyperdigraph_bounds(
        generated,
        labels,
        max_path_dimension,
        minimum_hyperdigraph,
        maximum_hyperdigraph,
    )
    return SequenceHyperdigraph(
        labels, selected_hyperedges, include_all_vertices=include_all_vertices
    )


def filtered_hyperdigraph_from_distance_matrix(
    distance_matrix: object,
    *,
    vertices: Iterable[Hashable] | None = None,
    max_path_dimension: int = 2,
    filtration_start: _RealValue = 0.0,
    minimum_adjacency: object | None = None,
    maximum_adjacency: object | None = None,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    include_all_vertices: bool = True,
) -> FilteredSequenceHyperdigraph:
    """Create a path-hyperdigraph filtration from directed distances.

    An allowed edge ``(i,j)`` is born at ``max(filtration_start, d[i,j])``.
    Minimum connections/hyperedges are born at ``filtration_start``.  A path's
    birth is the maximum birth of its consecutive edges.
    """

    _validate_nonnegative_dimension(max_path_dimension, "max_path_dimension")
    if isinstance(filtration_start, bool) or not isinstance(
        filtration_start, Real
    ):
        raise TypeError("filtration_start must be a finite real number")
    start = float(filtration_start)
    if not isfinite(start):
        raise ValueError("filtration_start must be a finite real number")

    distances = _distance_matrix(distance_matrix)
    size = len(distances)
    labels = _vertex_labels(size, vertices)
    minimum_mask = _mask_or_default(
        minimum_adjacency, size, "minimum_adjacency", False
    )
    maximum_mask = _mask_or_default(
        maximum_adjacency, size, "maximum_adjacency", True
    )
    edge_births: dict[tuple[int, int], float] = {}
    for source in range(size):
        for target in range(size):
            if minimum_mask[source][target] and not maximum_mask[source][target]:
                raise ValueError(
                    "minimum_adjacency must be contained in maximum_adjacency"
                )
            if source == target or not maximum_mask[source][target]:
                continue
            if minimum_mask[source][target]:
                edge_births[(source, target)] = start
            elif isfinite(distances[source][target]):
                edge_births[(source, target)] = max(
                    start, distances[source][target]
                )

    indexed_path_births = _index_path_births(
        edge_births, size, max_path_dimension, start
    )
    generated_order: list[Hyperedge] = []
    generated_birth: dict[Hyperedge, float] = {}
    for paths in indexed_path_births[1:]:
        for path, birth in paths:
            edge = tuple(labels[index] for index in path)
            generated_order.append(edge)
            generated_birth[edge] = birth

    selected_hyperedges = _apply_hyperdigraph_bounds(
        generated_order,
        labels,
        max_path_dimension,
        minimum_hyperdigraph,
        maximum_hyperdigraph,
    )
    required_hyperedges = set(
        _bound_hyperedges(
            minimum_hyperdigraph, labels, "minimum_hyperdigraph"
        )
        or ()
    )
    weighted_hyperedges = tuple(
        WeightedHyperedge(
            edge,
            start if edge in required_hyperedges else generated_birth[edge],
        )
        for edge in selected_hyperedges
        if len(edge) > 1
    )
    explicit_minimum_vertices = tuple(
        WeightedHyperedge(edge, start)
        for edge in selected_hyperedges
        if len(edge) == 1
    )
    return FilteredSequenceHyperdigraph(
        labels,
        (*explicit_minimum_vertices, *weighted_hyperedges),
        include_all_vertices=include_all_vertices,
        vertex_birth=start,
    )


def bounded_filtration(
    filtration: FilteredSequenceHyperdigraph,
    *,
    minimum_hyperdigraph: SequenceHyperdigraph | None = None,
    maximum_hyperdigraph: SequenceHyperdigraph | None = None,
    filtration_start: _RealValue = 0.0,
) -> FilteredSequenceHyperdigraph:
    """Apply explicit minimum/maximum hyperdigraphs to an existing filtration."""

    if not isinstance(filtration, FilteredSequenceHyperdigraph):
        raise TypeError("filtration must be a FilteredHyperdigraph")
    if isinstance(filtration_start, bool) or not isinstance(
        filtration_start, Real
    ):
        raise TypeError("filtration_start must be a finite real number")
    start = float(filtration_start)
    if not isfinite(start):
        raise ValueError("filtration_start must be a finite real number")

    vertices = filtration.vertices
    minimum_order = _bound_hyperedges(
        minimum_hyperdigraph, vertices, "minimum_hyperdigraph"
    )
    maximum_order = _bound_hyperedges(
        maximum_hyperdigraph, vertices, "maximum_hyperdigraph"
    )
    required_hyperedges = set(minimum_order or ())
    allowed_hyperedges = set(maximum_order) if maximum_order is not None else None
    if allowed_hyperedges is not None and not required_hyperedges <= allowed_hyperedges:
        raise ValueError(
            "minimum_hyperdigraph must be contained in maximum_hyperdigraph"
        )

    selected_records: list[WeightedHyperedge] = []
    seen: set[Hyperedge] = set()
    for record in filtration.weighted_hyperedges():
        # A maximum bounds connections and higher hyperedges.  It must not
        # silently delete F_0 singleton faces when its allow-list contains
        # only positive-dimensional hyperedges.
        if (
            len(record.vertices) > 1
            and allowed_hyperedges is not None
            and record.vertices not in allowed_hyperedges
        ):
            continue
        birth = (
            min(record.birth, start)
            if record.vertices in required_hyperedges
            else record.birth
        )
        selected_records.append(WeightedHyperedge(record.vertices, birth))
        seen.add(record.vertices)
    for edge in minimum_order or ():
        if edge not in seen:
            selected_records.append(WeightedHyperedge(edge, start))
    return FilteredSequenceHyperdigraph(vertices, selected_records)


def hyperdigraph_from_dimension_dict(
    hyperedges_by_dimension: Mapping[int, Iterable[Iterable[Hashable]]],
    *,
    vertices: Iterable[Hashable] | None = None,
    include_all_vertices: bool = False,
) -> SequenceHyperdigraph:
    """Convert the legacy ``{dimension: [sequences...]}`` representation."""

    if not isinstance(hyperedges_by_dimension, Mapping):
        raise TypeError("hyperedges_by_dimension must be a mapping")
    flattened: list[Hyperedge] = []
    inferred_vertices: list[Hashable] = []
    seen_vertices: set[Hashable] = set()
    for dimension in sorted(hyperedges_by_dimension):
        _validate_nonnegative_dimension(dimension, "dimension key")
        for raw_edge in hyperedges_by_dimension[dimension]:
            edge = tuple(raw_edge)
            if len(edge) != dimension + 1:
                raise ValueError(
                    f"dimension {dimension} contains a sequence of length "
                    f"{len(edge)}"
                )
            flattened.append(edge)
            for vertex in edge:
                if vertex not in seen_vertices:
                    seen_vertices.add(vertex)
                    inferred_vertices.append(vertex)
    labels = tuple(vertices) if vertices is not None else tuple(inferred_vertices)
    if not labels:
        raise ValueError("vertices cannot be inferred from an empty dictionary")
    return SequenceHyperdigraph(
        labels, flattened, include_all_vertices=include_all_vertices
    )


__all__ = ['bounded_filtration', 'filtered_hyperdigraph_from_distance_matrix', 'hyperdigraph_from_adjacency_matrix', 'hyperdigraph_from_dimension_dict']
