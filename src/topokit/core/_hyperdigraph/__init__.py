"""Native sequence-hyperdigraph topology.

The public API is intentionally small: construct a static or filtered
hyperdigraph, then compute homology, persistence, or the corresponding
ordinary and persistent Laplacians.
"""

from .chain_complex import FilteredChainComplex, build_chain_complex, build_filtered_chain_complex, compute_homology
from .model import DEFINITION_ID, FilteredHyperdigraph, FilteredSequenceHyperdigraph, Hyperdigraph, SequenceHyperdigraph, WeightedHyperedge
from .laplacian import RealOmegaBackend, compute_laplacian, compute_laplacian_filtration, compute_persistent_laplacian
from .persistence import compute_persistence
from .path_compatible import ScoreDirectedEdgeFiltration, compute_path_compatible_persistence
from .results import Chain, DimensionDiagnostics, HomologyDiagnostics, HomologyResult, LaplacianDiagnostics, LaplacianDimensionResult, LaplacianFiltrationResult, LaplacianResult, LaplacianSnapshot, PersistenceDiagnostics, PersistenceInterval, PersistenceResult, PersistentLaplacianResult, RealChainDimensionDiagnostics, RealMatrix


__version__ = "0.6.0"


__all__ = ['Chain', 'DEFINITION_ID', 'DimensionDiagnostics', 'FilteredChainComplex', 'FilteredHyperdigraph', 'FilteredSequenceHyperdigraph', 'HomologyDiagnostics', 'HomologyResult', 'Hyperdigraph', 'LaplacianDiagnostics', 'LaplacianDimensionResult', 'LaplacianFiltrationResult', 'LaplacianResult', 'LaplacianSnapshot', 'PersistenceDiagnostics', 'PersistenceInterval', 'PersistenceResult', 'PersistentLaplacianResult', 'RealChainDimensionDiagnostics', 'RealMatrix', 'RealOmegaBackend', 'ScoreDirectedEdgeFiltration', 'SequenceHyperdigraph', 'WeightedHyperedge', 'build_chain_complex', 'build_filtered_chain_complex', 'compute_homology', 'compute_laplacian', 'compute_laplacian_filtration', 'compute_persistence', 'compute_persistent_laplacian', 'compute_path_compatible_persistence']
