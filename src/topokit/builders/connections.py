"""Explicit connection contracts shared by geometric construction recipes.

Pairs refer to stable point IDs, not implicit row numbers. No chemistry, bond
order, orientation, or topological expansion is inferred by this module.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from ..data import PointCloud, as_point_cloud
from ..exceptions import ResourceLimitError
from ._hyperdigraph_geometry import build_point_cloud_support


@dataclass(frozen=True)
class ConnectionResult:
    pairs: tuple[tuple[int, int], ...]
    distances: tuple[float, ...]
    metadata: dict


def prepare_cloud(data, weights=None):
    """Preserve existing weights unless an explicit array/ID mapping overrides."""
    cloud = as_point_cloud(data)
    return cloud if weights is None else PointCloud(cloud.points, cloud.ids, weights, cloud.metadata)


def validate_cutoff(cutoff):
    """A physical distance cap, never an alpha squared-radius parameter."""
    if cutoff is None:
        return None
    if (isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, Real)
            or not math.isfinite(cutoff) or cutoff < 0):
        raise ValueError("cutoff must be None or a finite nonnegative Euclidean distance")
    return float(cutoff)


def _ids(ids):
    if isinstance(ids, (str, bytes)):
        raise ValueError("IDs must be an explicit sequence, not a string")
    labels = tuple(ids)
    if any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (str, Integral)) for label in labels):
        raise ValueError("IDs must be unique strings or integers, not booleans")
    if len(set(labels)) != len(labels):
        raise ValueError("IDs must be unique strings or integers, not booleans")
    return labels


def bonds_from_adjacency(adjacency, *, ids, directed=False):
    """Convert an explicit binary adjacency matrix using its supplied ID order.

    Undirected input must be symmetric; diagonal entries must be zero. Numeric
    entries other than 0/1 are rejected, since they could mean bond orders,
    distances, or filtration births. ``directed=True`` returns ordered edges
    suitable for ``from_digraph``; point-cloud ``bonds`` remain undirected.
    Dense and SciPy sparse adjacency matrices are supported.
    """
    labels = _ids(ids)
    if not isinstance(directed, (bool, np.bool_)):
        raise ValueError("directed must be a boolean")
    if sparse.issparse(adjacency):
        matrix = adjacency.tocsr(copy=True)
        if matrix.shape != (len(labels), len(labels)):
            raise ValueError("adjacency shape must match the explicit ID order")
        matrix.sum_duplicates()
        if np.iscomplexobj(matrix.data) or not np.isin(matrix.data, (0, 1)).all():
            raise ValueError("adjacency must contain only binary 0/1 entries")
        if np.any(matrix.diagonal() != 0):
            raise ValueError("self loops are not supported")
        if not directed and (matrix != matrix.T).nnz:
            raise ValueError("undirected adjacency must be symmetric")
        rows, columns = matrix.nonzero()
    else:
        matrix = np.asarray(adjacency)
        if matrix.shape != (len(labels), len(labels)):
            raise ValueError("adjacency shape must match the explicit ID order")
        if np.iscomplexobj(matrix) or not np.isin(matrix, (0, 1)).all():
            raise ValueError("adjacency must contain only binary 0/1 entries")
        if np.any(np.diag(matrix) != 0):
            raise ValueError("self loops are not supported")
        if not directed and not np.array_equal(matrix, matrix.T):
            raise ValueError("undirected adjacency must be symmetric")
        rows, columns = np.nonzero(matrix)
    indices = sorted((int(i), int(j)) for i, j in zip(rows, columns) if directed or i < j)
    return tuple((labels[i], labels[j]) for i, j in indices)


def resolve_connections(cloud, *, bonds=None, cutoff=None, max_connections=None):
    """Resolve Delaunay or supplied undirected pairs, then apply distance pruning.

    ``None`` infers Delaunay; an empty iterable means no edges. Explicit pairs
    replace Delaunay and are normalized/deduplicated without adding connections.
    Result pairs use internal row indices; metadata records stable external IDs.
    The cutoff is inclusive and applies equally to supplied and inferred pairs.
    All points remain available to the caller, including isolated vertices.
    """
    cloud = as_point_cloud(cloud)
    cutoff = validate_cutoff(cutoff)
    if max_connections is not None and (isinstance(max_connections, (bool, np.bool_))
            or not isinstance(max_connections, Integral) or max_connections < 0):
        raise ValueError("max_connections must be a nonnegative integer or None")
    support_metadata = {"ambient_dimension": cloud.points.shape[1]}
    duplicate_count = 0
    if bonds is None:
        connection_source = "delaunay"
        if len(cloud) < 2:
            pairs = ()
            support_metadata.update({"intrinsic_dimension": 0, "support_maximal_simplex_count": len(cloud),
                                     "projected_to_affine_hull": cloud.points.shape[1] > 0})
        else:
            support = build_point_cloud_support(cloud.points, "delaunay")
            pairs = support.pairs
            support_metadata.update({"intrinsic_dimension": support.intrinsic_dimension,
                                     "support_maximal_simplex_count": support.maximal_simplex_count,
                                     "projected_to_affine_hull": support.projected_to_affine_hull})
    else:
        if isinstance(bonds, (str, bytes, dict)):
            raise ValueError("bonds must be a sequence of stable-ID pairs")
        connection_source = "supplied"
        id_to_index = cloud.index
        unique_pairs = set()
        for raw_bond in bonds:
            if isinstance(raw_bond, (str, bytes, dict, set)):
                raise ValueError("each bond must be a pair of stable point IDs")
            pair = tuple(raw_bond)
            if len(pair) != 2:
                raise ValueError("each bond must have exactly two stable point IDs; attributes are separate")
            _ids(pair)
            try:
                i, j = (id_to_index[label] for label in pair)
            except KeyError as exc:
                raise ValueError(f"bond references unknown point ID {exc.args[0]!r}") from exc
            if i == j:
                raise ValueError("self loops are not supported")
            edge = (min(i, j), max(i, j))
            duplicate_count += edge in unique_pairs
            unique_pairs.add(edge)
        pairs = tuple(sorted(unique_pairs))
    retained_pairs, distances, removed_bonds = [], [], []
    for i, j in pairs:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = cloud.points[i] - cloud.points[j]
        distance = math.hypot(*difference)
        if not math.isfinite(distance):
            raise ValueError("Euclidean distances overflow; rescale the point cloud")
        if cutoff is not None and distance > cutoff:
            removed_bonds.append((cloud.ids[i], cloud.ids[j]))
            continue
        if max_connections is not None and len(retained_pairs) >= max_connections:
            raise ResourceLimitError("Retained connection count exceeds the construction budget; no edges pruned")
        retained_pairs.append((i, j))
        distances.append(distance)
    metadata = {
        "connection_support": connection_source, "candidate_connection_count": len(pairs),
        "support_edge_count": len(pairs), "retained_connection_count": len(retained_pairs),
        "connection_pairs": tuple((cloud.ids[i], cloud.ids[j]) for i, j in retained_pairs),
        "duplicate_bonds_collapsed": int(duplicate_count), "cutoff_distance": cutoff,
        "cutoff_policy": "inclusive_euclidean_distance",
        "cutoff_pruned_bonds": tuple(removed_bonds), "cutoff_pruned_count": len(removed_bonds),
        **support_metadata,
    }
    return ConnectionResult(tuple(retained_pairs), tuple(distances), metadata)


__all__ = ["ConnectionResult", "bonds_from_adjacency", "prepare_cloud", "resolve_connections"]
