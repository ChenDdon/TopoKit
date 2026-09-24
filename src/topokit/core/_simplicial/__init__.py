"""Simplicial objects and mathematical algorithms; no geometry constructors."""

from ._validation import ComplexityLimitError
from .algebra import boundary_matrix
from .complex import Simplex, SimplexTree, SimplicialComplex
from .homology import HomologyResult, homology
from .laplacian import (
    LaplacianResult, laplacian, laplacian_filtration, laplacian_matrix,
    persistent_laplacian, persistent_laplacian_at, persistent_laplacian_matrix,
    spectrum, summarize_spectrum,
)
from .persistence import PersistenceInterval, PersistenceResult, persistent_homology

__version__ = "0.2.0"
__all__ = [
    "ComplexityLimitError", "Simplex", "SimplexTree", "SimplicialComplex", "boundary_matrix",
    "homology", "persistent_homology", "laplacian", "laplacian_matrix", "laplacian_filtration",
    "persistent_laplacian", "persistent_laplacian_at", "persistent_laplacian_matrix",
    "spectrum", "summarize_spectrum",
    "HomologyResult", "PersistenceResult", "PersistenceInterval", "LaplacianResult",
]
