"""Native alpha filtrations, neighbor-intersection Rips, and graph complexes.

The default backend uses SciPy geometric primitives. Exact GUDHI geometry is
available only by explicit opt-in. Alpha values are squared radii; Rips values
are diameters.
"""

from itertools import combinations
import math

import numpy as np
from scipy.spatial import cKDTree

from ..core._simplicial._validation import ComplexityLimitError, nonnegative_int, real, tolerance
from ..core._simplicial.complex import SimplexTree, normalize_simplex


class GeometryError(ValueError):
    """Geometry is too degenerate for reliable floating-point construction."""


def _points(points):
    result = np.asarray(points, dtype=float)
    if result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError("points must be a finite (n_points, ambient_dimension) array")
    return result


def _cutoff(value):
    value = real(value, "max_scale", finite=False)
    if value < 0:
        raise ValueError("max_scale must be nonnegative")
    return value


def _expand_flag(tree, max_dimension):
    """Enumerate cliques through intersections of forward neighbor sets."""
    forward_neighbors = {s[0]: set() for s in tree.simplices(0)}
    for u, v in tree.simplices(1):
        forward_neighbors[u].add(v)

    def visit(prefix, candidates, value):
        if len(prefix) >= max_dimension + 1:
            return
        for vertex in sorted(candidates):
            simplex = prefix + (vertex,)
            birth = max(value, *(tree.filtration((u, vertex)) for u in prefix))
            if len(simplex) > 2:
                tree._store(simplex, birth)
            visit(simplex, candidates.intersection(forward_neighbors[vertex]), birth)

    if max_dimension > 1:
        for vertex in forward_neighbors:
            visit((vertex,), forward_neighbors[vertex], tree.filtration((vertex,)))
    return tree


def graph_complex(edges=(), *, vertices=(), flag=False, max_dimension=2,
                  vertex_filtration=0.0, max_simplices=1_000_000):
    """Undirected graph as a 1-complex, or its optional clique/flag complex.

    Edges are (u,v) or (u,v,birth); vertices are labels or {label: birth}.
    An edge enters at max(its supplied birth, both endpoint births).
    Duplicate edges use the earliest birth. Self loops are rejected.
    """
    max_dimension = nonnegative_int(max_dimension, "max_dimension")
    default_birth = real(vertex_filtration, "vertex_filtration")
    tree = SimplexTree(max_simplices=max_simplices)
    items = vertices.items() if hasattr(vertices, "items") else ((v, default_birth) for v in vertices)
    for vertex, birth in items:
        tree.insert((vertex,), birth)
    for edge in edges:
        edge = tuple(edge)
        if len(edge) not in (2, 3):
            raise ValueError("Edges must be (u,v) or (u,v,birth)")
        simplex = normalize_simplex(edge[:2])
        birth = default_birth if len(edge) == 2 else real(edge[2], "edge birth")
        for vertex in simplex:
            if (vertex,) not in tree:
                tree.insert((vertex,), default_birth)
        birth = max(birth, *(tree.filtration((v,)) for v in simplex))
        if max_dimension >= 1:
            tree._store(simplex, birth)
    if flag:
        _expand_flag(tree, max_dimension)
    tree.metadata = {"complex_type": "flag" if flag else "graph", "scale_units": "edge_birth"}
    return tree


def rips_complex(points=None, *, distance_matrix=None, max_dimension=2,
                 max_scale=math.inf, max_simplices=1_000_000):
    """Vietoris--Rips filtration, truncated by dimension and edge length.

    Finite point-cloud cutoffs use a KD-tree radius graph instead of an n*n
    distance matrix. A symmetric nonnegative dissimilarity matrix is also
    supported (triangle inequality not required); +inf means missing edges.
    """
    max_dimension = nonnegative_int(max_dimension, "max_dimension")
    max_scale = _cutoff(max_scale)
    if (points is None) == (distance_matrix is None):
        raise ValueError("Supply exactly one of points or distance_matrix")
    tree = SimplexTree(max_simplices=max_simplices)
    if distance_matrix is not None:
        distances = np.asarray(distance_matrix, dtype=float)
        if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
                or np.isnan(distances).any() or (distances < 0).any()
                or not np.array_equal(distances, distances.T)
                or np.any(np.diag(distances) != 0)):
            raise ValueError("distance_matrix must be square, symmetric, nonnegative, with zero diagonal")
        n = len(distances)
        pairs = ((i, j) for i in range(n) for j in range(i + 1, n)
                 if np.isfinite(distances[i, j]) and distances[i, j] <= max_scale)
        edge_length = lambda i, j: float(distances[i, j])
    else:
        points = _points(points)
        n = len(points)
        if n > tree.max_simplices:
            raise ComplexityLimitError("Vertex count exceeds max_simplices")
        if max_dimension == 0:
            pairs = ()
        elif math.isfinite(max_scale):
            spatial_index = cKDTree(points)
            edge_count = (int(spatial_index.count_neighbors(spatial_index, max_scale)) - n) // 2
            if n + edge_count > tree.max_simplices:
                raise ComplexityLimitError("Radius graph alone exceeds max_simplices")
            pairs = spatial_index.query_pairs(max_scale, output_type="ndarray")
        else:
            if n + n * (n - 1) // 2 > tree.max_simplices:
                raise ComplexityLimitError("Complete radius graph alone exceeds max_simplices")
            pairs = combinations(range(n), 2)
        edge_length = lambda i, j: float(np.linalg.norm(points[i] - points[j]))
    for i in range(n):
        tree._store((i,), 0.0)
    if max_dimension >= 1:
        for i, j in pairs:
            value = edge_length(i, j)
            if not np.isfinite(value):
                raise GeometryError("Distances overflow; rescale the point cloud")
            tree._store((int(i), int(j)), value)
        _expand_flag(tree, max_dimension)
    tree.metadata = {"complex_type": "rips", "scale_units": "edge_length", "max_scale": max_scale}
    return tree


def alpha_complex(points, *, max_dimension=None, max_scale=math.inf,
                  duplicates="merge", geometry_tolerance=1e-12,
                  max_simplices=1_000_000, backend="native"):
    """Unweighted alpha filtration with native empty-ball/coface propagation.

    Triangulates with SciPy/Qhull, computes batched Gabriel circumradii on
    original coordinates, and propagates coface births downward. Ill-conditioned
    spheres use rational arithmetic; bounded small-cloud repair is available.
    Values and max_scale are SQUARED RADII. All Delaunay dimensions must be
    processed before returning a skeleton. No random jitter is applied.

    Default duplicates='merge' retains the first original index as each vertex label;
    metadata['original_to_vertex'] records the correspondence.
    Explicit duplicates='error' retains strict rejection for historical replay.
    backend='gudhi_exact' explicitly delegates geometry to optional GUDHI;
    it is never an automatic fallback from a native geometry error.
    """
    if backend not in {"native", "gudhi_exact"}:
        raise ValueError("alpha backend must be 'native' or 'gudhi_exact'")
    if backend == "gudhi_exact":
        from ._alpha_gudhi import alpha_complex as exact_alpha
        return exact_alpha(points, max_dimension=max_dimension, max_scale=max_scale,
                           duplicates=duplicates, geometry_tolerance=geometry_tolerance,
                           max_simplices=max_simplices)
    from ._alpha_native import alpha_complex as native_alpha
    return native_alpha(points, max_dimension=max_dimension, max_scale=max_scale,
                        duplicates=duplicates, geometry_tolerance=geometry_tolerance,
                        max_simplices=max_simplices)


def build_complex(points=None, *, complex_type="alpha", **kwargs):
    """Unified point-cloud entry point; alpha is the default."""
    if complex_type == "alpha":
        return alpha_complex(points, **kwargs)
    if complex_type == "rips":
        return rips_complex(points, **kwargs)
    raise ValueError("complex_type must be 'alpha' or 'rips'")
