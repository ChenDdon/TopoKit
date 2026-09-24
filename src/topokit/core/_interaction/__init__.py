"""Interaction homology and optional real Laplacians of filtered complexes."""

__version__ = "0.2.0"
DEFINITION_ID = "interaction-ordered-tensor-quotient-v1"

from .interaction import (
    FilteredInteractionChainComplex,
    InteractionConstructionDiagnostics,
    InteractionDegree,
    InteractionKey,
    build_interaction_chain_complex,
)
from .model import (
    FilteredSimplicialComplex,
    Simplex,
    SimplicialComplexBuilder,
)
from .laplacian import (
    InteractionLaplacianEngine,
    LaplacianDiagnostics,
    LaplacianResult,
    LaplacianSpectrum,
    compute_interaction_laplacian,
    compute_persistent_interaction_laplacian,
    signed_boundary_matrix,
)
from .persistence import (
    ReductionBackend,
    compute_persistence,
)
from .results import PersistenceDiagnostics, PersistenceInterval, PersistenceResult


__all__ = [
    "FilteredInteractionChainComplex",
    "FilteredSimplicialComplex",
    "InteractionConstructionDiagnostics",
    "InteractionDegree",
    "InteractionKey",
    "InteractionLaplacianEngine",
    "LaplacianDiagnostics",
    "LaplacianResult",
    "LaplacianSpectrum",
    "PersistenceDiagnostics",
    "PersistenceInterval",
    "PersistenceResult",
    "ReductionBackend",
    "Simplex",
    "SimplicialComplexBuilder",
    "build_interaction_chain_complex",
    "compute_interaction_laplacian",
    "compute_persistent_interaction_laplacian",
    "compute_persistence",
    "DEFINITION_ID",
    "signed_boundary_matrix",
    "__version__",
]
