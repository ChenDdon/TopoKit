"""Simplicial convenience workflows composing constructors and math cores.

These helpers preserve the native research workflow results. The core itself
has no dependency on this layer or on geometric point-cloud construction.
"""

from dataclasses import dataclass

from ..core._simplicial._validation import nonnegative_int
from ..builders._simplicial import build_complex
from ..core._simplicial.complex import SimplexTree, as_complex
from ..core._simplicial.homology import HomologyResult, homology
from ..core._simplicial.laplacian import LaplacianResult, laplacian
from ..core._simplicial.persistence import PersistenceResult, persistent_homology


@dataclass(frozen=True)
class ComplexAnalysis:
    complex: SimplexTree
    homology: HomologyResult
    laplacians: dict[int, LaplacianResult]


@dataclass(frozen=True)
class PointCloudAnalysis:
    complex: SimplexTree
    persistence: PersistenceResult
    snapshots: dict[float, ComplexAnalysis]


def analyze_complex(complex_, *, max_dimension=1, field=2, scale=None,
                    tol=1e-10, max_dense_entries=10_000_000):
    """Ordinary homology and Laplacians for exactly the supplied complex."""
    maximum_dimension = nonnegative_int(max_dimension, "max_dimension")
    complex_ = as_complex(complex_)
    if scale is not None:
        complex_ = complex_.at(scale)
    homology_result = homology(complex_, max_dimension=maximum_dimension, field=field, tol=tol,
                      max_dense_entries=max_dense_entries)
    spectra = {q: laplacian(complex_, q, tol=tol, max_dense_entries=max_dense_entries)
               for q in range(maximum_dimension + 1)}
    return ComplexAnalysis(complex_, homology_result, spectra)


def analyze_points(points, *, complex_type="alpha", max_homology_dimension=1,
                   scales=(), field=2, tol=1e-10, max_dense_entries=10_000_000,
                   **construction_options):
    """Build once, compute persistence, optionally analyze fixed snapshots.

    Builds through dimension q+1 to retain deaths in Hq. In the alpha case,
    the builder still processes higher-dimensional Delaunay cofaces before
    truncating. scales use the selected construction's units.
    """
    maximum_homology_dimension = nonnegative_int(max_homology_dimension, "max_homology_dimension")
    if "max_dimension" in construction_options:
        raise ValueError("Use max_homology_dimension; construction dimension is chosen automatically")
    complex_ = build_complex(points, complex_type=complex_type, max_dimension=maximum_homology_dimension + 1,
                             **construction_options)
    persistence_result = persistent_homology(complex_, max_dimension=maximum_homology_dimension, field=field)
    snapshots = {float(scale): analyze_complex(complex_, max_dimension=maximum_homology_dimension, field=field,
                                          scale=scale, tol=tol, max_dense_entries=max_dense_entries) for scale in scales}
    return PointCloudAnalysis(complex_, persistence_result, snapshots)
