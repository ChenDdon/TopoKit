"""Explicit optional exact-alpha geometry adapter; no topological analysis."""
import math

import numpy as np

from ..core._simplicial._validation import ComplexityLimitError, nonnegative_int, tolerance
from ..core._simplicial.complex import SimplexTree
from ..data import _coordinate_representatives


def alpha_complex(points, *, max_dimension=None, max_scale=math.inf,
                  duplicates="merge", geometry_tolerance=1e-12,
                  max_simplices=1_000_000):
    """Convert GUDHI exact alpha geometry to the native explicit object.

    Full cofaces are built before radius/dimension truncation. The simplex
    budget covers that full complex and is checked after external construction,
    before conversion; it is not a bound on GUDHI's peak allocation. Exact
    arithmetic uses the supplied floating-point coordinates without projection
    or perturbation. Exact births are converted to float by GUDHI.
    """
    from ._simplicial import GeometryError, _cutoff, _points

    points = _points(points)
    maximum = _cutoff(max_scale)
    tolerance(geometry_tolerance)  # validated for API compatibility; unused by exact geometry
    if max_dimension is not None:
        max_dimension = nonnegative_int(max_dimension, "max_dimension")
    if duplicates not in {"error", "merge"}:
        raise ValueError("duplicates must be 'error' or 'merge'")
    tree = SimplexTree(max_simplices=max_simplices)
    labels, representatives = _coordinate_representatives(points)
    if len(labels) != len(points) and duplicates == "error":
        raise ValueError("Duplicate coordinates; pass duplicates='merge' to merge explicitly")
    original_to_vertex = tuple(int(i) for i in representatives)
    selected = points[labels]
    if len(selected) > tree.max_simplices:
        raise ComplexityLimitError("Unique vertex count exceeds max_simplices before exact alpha construction")
    try:
        import gudhi
    except ImportError as error:
        raise ImportError("backend='gudhi_exact' requires the optional topokit[alpha_exact] dependency") from error

    metadata = {
        "complex_type": "alpha", "scale_units": "squared_radius", "max_scale": maximum,
        "original_to_vertex": original_to_vertex, "geometry_backend": "gudhi_exact",
        "duplicate_policy": duplicates, "duplicate_representative": "first_input_row",
        "input_point_count": len(points), "unique_point_count": len(selected),
        "duplicate_point_count": len(points)-len(selected),
        "geometry_backend_version": gudhi.__version__, "geometry_precision": "exact",
        "geometry_tolerance_applied": False, "coordinate_perturbation": False,
        "cofaces_processed_before_truncation": True,
        "simplex_budget_scope": "full Delaunay complex; checked after external construction",
    }
    if not len(selected):
        tree.metadata = dict(metadata, affine_dimension=0, full_simplex_count=0)
        return tree
    external = gudhi.AlphaComplex(points=selected, precision="exact").create_simplex_tree(
        max_alpha_square=math.inf, output_squared_values=True)
    if external.num_simplices() > tree.max_simplices:
        raise ComplexityLimitError("Full exact alpha complex exceeds max_simplices before skeleton extraction")
    vertices = {int(s[0]) for s, _ in external.get_skeleton(0)}
    if vertices != set(range(len(selected))):
        raise GeometryError("Exact alpha construction omitted input vertices")
    for simplex, birth in external.get_filtration():
        value = float(birth)
        if not np.isfinite(value) or value < 0:
            raise GeometryError("Invalid or overflowing exact squared alpha radius")
        if (max_dimension is None or len(simplex) <= max_dimension + 1) and value <= maximum:
            tree._store(tuple(int(labels[v]) for v in simplex), value)
    tree.metadata = dict(metadata, affine_dimension=max(0, external.dimension()),
                         full_simplex_count=external.num_simplices())
    tree.validate()
    return tree
