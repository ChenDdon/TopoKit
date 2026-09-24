"""Six-layer topology toolkit. Shared records and convenience imports only."""

from .data import PointCloud
from .results import (HomologyResult, PersistenceInterval, PersistenceResult,
                      SpectrumResult, StationaryResult, Topology)
from .exceptions import ResourceLimitError
from .core import homology, persistence, laplacian, persistent_laplacian, laplacians, persistent_laplacians


def from_points(points, *, kind="simplicial", **options):
    """Convenience alias; object construction is loaded only when requested."""
    from .builders import from_points as build
    return build(points, kind=kind, **options)

__version__ = "0.3.0"
__all__ = ["PointCloud", "Topology", "HomologyResult", "PersistenceInterval",
           "PersistenceResult", "SpectrumResult", "StationaryResult", "ResourceLimitError", "__version__",
           "from_points", "homology", "persistence", "laplacian", "persistent_laplacian",
           "laplacians", "persistent_laplacians"]
