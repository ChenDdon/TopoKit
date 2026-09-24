"""Reproducible one-object workflows; batch/HPC callers can compose these."""
from dataclasses import dataclass, field
import math
import os
import platform
import time
import numpy as np
import scipy
from .. import core as api
from .._validation import dimension
from ..results import Topology, PersistenceResult


@dataclass(frozen=True)
class AnalysisResult:
    topology: Topology
    persistence: PersistenceResult
    snapshots: dict
    metadata: dict = field(default_factory=dict)


def analyze(topology, *, max_dimension=2, scales=(), field=2,
            return_eigenvectors=False, return_matrices=False,
            k=None, max_dense_entries=4_000_000, tol=1e-10):
    """Persistence plus ordinary snapshot Hq/Lq for explicitly chosen scales.

    Changing scales never changes the filtration or its barcode. Pairwise
    persistent Laplacians are not invoked implicitly by this workflow.
    Spectra are real-valued; homology/persistence use the requested prime field.
    """
    maximum_dimension = dimension(max_dimension)
    supplied_scales = tuple(scales)
    if any(isinstance(value, (bool, np.bool_)) for value in supplied_scales):
        raise ValueError("snapshot scales cannot be booleans")
    requested_scales = tuple(float(value) for value in supplied_scales)
    if any(not math.isfinite(value) for value in requested_scales) or len(set(requested_scales)) != len(requested_scales):
        raise ValueError("snapshot scales must be finite and distinct")
    start_time = time.perf_counter()
    persistence_result = api.persistence(topology, max_dimension=maximum_dimension, field=field)
    snapshots = {}
    for scale in requested_scales:
        snapshot_homology = api.homology(topology, max_dimension=maximum_dimension, scale=scale, field=field)
        spectra = {q: api.laplacian(topology, dimension=q, scale=scale, k=k,
                                   return_matrix=return_matrices, return_eigenvectors=return_eigenvectors,
                                   max_dense_entries=max_dense_entries, tol=tol)
                   for q in range(maximum_dimension + 1)}
        snapshots[scale] = {"homology": snapshot_homology, "laplacians": spectra}
    return AnalysisResult(topology, persistence_result, snapshots,
                          {"elapsed_seconds": time.perf_counter()-start_time,
                           "snapshot_scales": requested_scales, "max_dimension": maximum_dimension,
                           "python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "platform": platform.platform(),
                           "thread_environment": {key: os.environ.get(key) for key in
                                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                           "spectral_mode": "ordinary_snapshots", "field": persistence_result.field})
