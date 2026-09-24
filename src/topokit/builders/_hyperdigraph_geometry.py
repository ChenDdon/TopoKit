"""Shared point-cloud validation and connection-support construction.

``delaunay`` builds an undirected Delaunay 1-skeleton and leaves orientation
to the caller.  It is a geometric maximum-connection constraint, not an alpha
complex: filtration values are still Euclidean edge lengths and higher
hyperedges are still directed paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Literal, cast


ConnectionSupport = Literal["delaunay", "complete"]


@dataclass(frozen=True, slots=True)
class PointCloudSupport:
    """Undirected candidate edges and geometry metadata for a point cloud.

    ``pairs`` is ``None`` for complete support so the dense builder can retain
    its blockwise implementation without materializing a Python tuple for
    every possible pair.
    """

    name: ConnectionSupport
    pairs: tuple[tuple[int, int], ...] | None
    ambient_dimension: int
    intrinsic_dimension: int | None
    edge_count: int
    maximal_simplex_count: int | None
    projected_to_affine_hull: bool


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def materialize_points(
    points: Iterable[Iterable[Any]],
) -> tuple[tuple[float, ...], ...]:
    if isinstance(points, (str, bytes, bytearray)):
        raise TypeError("points must be an iterable of coordinate rows")
    try:
        rows = tuple(tuple(row) for row in points)
    except TypeError as exc:
        raise TypeError("points must be an iterable of coordinate rows") from exc
    if len(rows) < 2:
        raise ValueError("at least two points are required")
    dimension = len(rows[0])
    if dimension == 0 or any(len(row) != dimension for row in rows):
        raise ValueError("points must form a nonempty rectangular array")
    return tuple(
        tuple(finite_float(value, f"points[{i}][{j}]") for j, value in enumerate(row))
        for i, row in enumerate(rows)
    )


def score_order(point_weights: Iterable[Any], count: int) -> tuple[int, ...]:
    """Return the strict direction-score order.

    ``point_weights`` is retained as the public compatibility name.  These
    values orient edges; they are not weighted-alpha radii.
    """

    if isinstance(point_weights, (str, bytes, bytearray)):
        raise TypeError("point_weights must be an iterable of real numbers")
    raw = tuple(point_weights)
    if len(raw) != count:
        raise ValueError("point_weights must contain exactly one value per point")
    weights = tuple(
        finite_float(value, f"point_weights[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(weights)) != count:
        raise ValueError(
            "the implicit two-family engine requires distinct point weights"
        )
    return tuple(sorted(range(count), key=weights.__getitem__))


def distance_cutoff(value: Any | None) -> float | None:
    if value is None:
        return None
    result = finite_float(value, "max_distance")
    if result < 0.0:
        raise ValueError("max_distance must be nonnegative")
    return result


def validate_connection_support(value: str) -> ConnectionSupport:
    if value not in {"delaunay", "complete"}:
        raise ValueError("connection_support must be 'delaunay' or 'complete'")
    return cast(ConnectionSupport, value)


def build_point_cloud_support(
    points: Iterable[Iterable[Any]],
    connection_support: str = "delaunay",
) -> PointCloudSupport:
    """Construct the requested undirected candidate-edge support.

    Rank-deficient clouds are projected onto their intrinsic affine hull before
    triangulation.  Exact duplicate points and Qhull omissions are rejected;
    the routine never silently changes a requested Delaunay support into a
    complete graph.
    """

    coordinates = materialize_points(points)
    support_mode = validate_connection_support(connection_support)
    point_count = len(coordinates)
    ambient_dimension = len(coordinates[0])
    if support_mode == "complete":
        return PointCloudSupport(
            name=support_mode,
            pairs=None,
            ambient_dimension=ambient_dimension,
            intrinsic_dimension=None,
            edge_count=point_count * (point_count - 1) // 2,
            maximal_simplex_count=None,
            projected_to_affine_hull=False,
        )

    first_occurrence: dict[tuple[float, ...], int] = {}
    for index, point in enumerate(coordinates):
        previous = first_occurrence.setdefault(point, index)
        if previous != index:
            raise ValueError(
                "connection_support='delaunay' requires distinct point "
                f"coordinates; points {previous} and {index} are duplicates. "
                "Deduplicate the cloud or use connection_support='complete'."
            )

    try:
        import numpy as np
        from scipy.spatial import Delaunay, QhullError
    except ImportError as exc:
        raise ImportError(
            "connection_support='delaunay' requires NumPy and SciPy; install "
            "the package dependencies or use connection_support='complete'"
        ) from exc

    cloud = np.asarray(coordinates, dtype=np.float64)
    centered = cloud - cloud[0]
    _left, singular_values, right_transpose = np.linalg.svd(
        centered, full_matrices=False
    )
    if singular_values.size == 0 or float(singular_values[0]) == 0.0:
        raise ValueError(
            "connection_support='delaunay' requires positive affine dimension"
        )
    rank_tolerance = (
        np.finfo(np.float64).eps * max(centered.shape) * float(singular_values[0])
    )
    intrinsic_dimension = int(np.count_nonzero(singular_values > rank_tolerance))
    if intrinsic_dimension == 0:
        raise ValueError(
            "connection_support='delaunay' requires positive affine dimension"
        )
    projected_coordinates = centered @ right_transpose[:intrinsic_dimension, :].T
    projected_to_affine_hull = intrinsic_dimension < ambient_dimension

    if intrinsic_dimension == 1:
        order = np.argsort(projected_coordinates[:, 0], kind="stable")
        pairs = tuple(
            sorted(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in zip(order[:-1], order[1:])
            )
        )
        return PointCloudSupport(
            name=support_mode,
            pairs=pairs,
            ambient_dimension=ambient_dimension,
            intrinsic_dimension=1,
            edge_count=len(pairs),
            maximal_simplex_count=point_count - 1,
            projected_to_affine_hull=projected_to_affine_hull,
        )

    try:
        triangulation = Delaunay(projected_coordinates)
    except QhullError as exc:
        raise ValueError(
            "Delaunay construction failed for this point cloud. Resolve the "
            "geometric degeneracy or explicitly use "
            "connection_support='complete'; no silent fallback was applied."
        ) from exc

    simplices = np.asarray(triangulation.simplices, dtype=np.int64)
    used_vertices = {int(vertex) for vertex in simplices.ravel()}
    missing_vertices = sorted(set(range(point_count)) - used_vertices)
    if missing_vertices:
        raise ValueError(
            "Delaunay/Qhull omitted input vertices "
            f"{missing_vertices}; perturb or deduplicate the cloud, or "
            "explicitly use connection_support='complete'. No silent fallback "
            "was applied."
        )

    support_pairs: set[tuple[int, int]] = set()
    for simplex in simplices:
        for left_offset in range(len(simplex) - 1):
            left = int(simplex[left_offset])
            for right_offset in range(left_offset + 1, len(simplex)):
                right = int(simplex[right_offset])
                support_pairs.add((min(left, right), max(left, right)))
    pairs = tuple(sorted(support_pairs))
    return PointCloudSupport(
        name=support_mode,
        pairs=pairs,
        ambient_dimension=ambient_dimension,
        intrinsic_dimension=intrinsic_dimension,
        edge_count=len(pairs),
        maximal_simplex_count=len(simplices),
        projected_to_affine_hull=projected_to_affine_hull,
    )


__all__ = [
    "ConnectionSupport",
    "PointCloudSupport",
    "build_point_cloud_support",
]
