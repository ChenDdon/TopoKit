"""Analysis-only native sequence-hyperdigraph homology and real spectra."""

from dataclasses import asdict
import numbers
import numpy as np
from scipy import sparse

from ..results import Topology, HomologyResult, PersistenceInterval, PersistenceResult, SpectrumResult
from ..exceptions import ResourceLimitError
from .._validation import dimension as _dimension, validate_query
from ._spectral import EigenSolution, solve_symmetric
from ._hyperdigraph import (
    DEFINITION_ID, Hyperdigraph, FilteredHyperdigraph, WeightedHyperedge,
    compute_homology as _homology, compute_persistence as _persistence,
)
from ._hyperdigraph.spectral import (
    EmbeddedOperator, ordinary_operator, pairwise_operator, reference_operator, labelled_basis,
)
from ._hyperdigraph.l0_sweep import L0Accumulator, supports_incremental_l0


def _unwrap(obj, q):
    q = _dimension(q)
    if isinstance(obj, Topology):
        if obj.kind != "hyperdigraph":
            raise TypeError("a hyperdigraph Topology is required")
        metadata, native = dict(obj.metadata), obj.native
        if q > metadata.get("max_analysis_dimension", q):
            raise ValueError("requested dimension exceeds construction; rebuild with a larger max_dimension")
    else:
        native, metadata = obj, {"definition_id": DEFINITION_ID}
    if not isinstance(native, (Hyperdigraph, FilteredHyperdigraph)):
        raise TypeError("supply a native Hyperdigraph, FilteredHyperdigraph, or hyperdigraph Topology")
    metadata.setdefault("scale_units", "user_defined")
    metadata.setdefault("route", "hyperdigraph")
    metadata.setdefault("coordinate_units", "user_defined")
    return native, metadata


def _snapshot(native, metadata, scale):
    value = validate_query(metadata, scale)
    if isinstance(native, FilteredHyperdigraph):
        if value is None:
            value = max(native.thresholds(), default=metadata.get("filtration_start", 0.0))
        return native.snapshot(value), value
    if value is not None:
        raise ValueError("scale applies only to a filtered hyperdigraph")
    return native, None


def _chain_preflight(native, q, budget):
    """Conservative dense-bitset storage bound; never prune to meet it."""
    budget = _dimension(budget, "max_chain_bytes")
    hyperedge_counts = [len(native.weighted_hyperedges(d)) if isinstance(native, FilteredHyperdigraph)
              else native.number_of_hyperedges(d) for d in range(q + 2)]
    storage_bits = sum(n * n + n * (hyperedge_counts[d - 1] if d else 0) for d, n in enumerate(hyperedge_counts))
    estimated_bytes = (storage_bits + 7) // 8 + 128 * sum(hyperedge_counts)
    if estimated_bytes > budget:
        raise ResourceLimitError(
            f"Conservative native GF(2) chain-storage bound is {estimated_bytes:,} bytes, "
            f"exceeding max_chain_bytes={budget:,}; use a larger explicit budget "
            "or construct a separately declared smaller object"
        )
    return estimated_bytes


def _gf2(field):
    valid = field in ("GF(2)", "F2") if isinstance(field, str) else (
        isinstance(field, numbers.Integral) and not isinstance(field, bool) and int(field) == 2
    )
    if not valid:
        raise ValueError("hyperdigraph homology and persistence currently support only GF(2)")


def homology(obj, max_dimension=2, scale=None, *, field=2, representatives=False, omega_backend="auto",
             max_chain_bytes=512_000_000):
    _gf2(field)
    native, metadata = _unwrap(obj, max_dimension)
    snapshot, value = _snapshot(native, metadata, scale)
    metadata["chain_storage_bound_bytes"] = _chain_preflight(snapshot, _dimension(max_dimension), max_chain_bytes)
    result = _homology(snapshot, _dimension(max_dimension), representatives=representatives,
                       omega_backend=omega_backend)
    metadata.update({"scale": value, "diagnostics": asdict(result.diagnostics)})
    if representatives:
        metadata["representatives"] = result.representatives
    return HomologyResult(result.betti_numbers, "GF(2)", metadata)


def persistence(obj, max_dimension=2, *, field=2, include_diagonal=False,
                omega_backend="auto", reduction_backend="auto", max_chain_bytes=512_000_000):
    _gf2(field)
    native, metadata = _unwrap(obj, max_dimension)
    if not isinstance(native, FilteredHyperdigraph):
        raise TypeError("persistence requires a filtered hyperdigraph")
    metadata["chain_storage_bound_bytes"] = _chain_preflight(native, _dimension(max_dimension), max_chain_bytes)
    result = _persistence(native, _dimension(max_dimension), include_diagonal=include_diagonal,
                          omega_backend=omega_backend, reduction_backend=reduction_backend)
    start = metadata.get("filtration_start", min(native.thresholds(), default=0.0))
    metadata.update({"filtration_start": start, "diagnostics": asdict(result.diagnostics),
                     "interval_convention": "[birth,death)",
                     "initial_stage_note": "birth at start does not recover a pre-truncation birth"})
    intervals = tuple(PersistenceInterval(x.dimension, x.birth, x.death, x.birth == start)
                      for x in result.intervals)
    return PersistenceResult(intervals, "GF(2)", _dimension(max_dimension), metadata)


def _spectrum(operator, q, metadata, *, k, return_eigenvectors, return_matrix,
              max_dense_entries, tol, scale=None, start=None, end=None, isolate_zeros=False):
    basis_size = operator.matrix.shape[0]
    if return_matrix and basis_size * basis_size > max_dense_entries:
        raise ResourceLimitError("Returning this dense embedded Laplacian exceeds max_dense_entries")
    explicit = isinstance(operator.matrix, np.ndarray)
    matrix = ((operator.matrix.copy() if explicit else operator.matrix.matmat(np.eye(basis_size)))
              if return_matrix else None)
    if isolate_zeros and k is None and not return_eigenvectors:
        # Only the guarded ordinary graph-L0 path uses this exact zero block.
        # Preserve the molecular workflow's existing incident-vertex solve.
        incident = np.flatnonzero(np.diag(operator.matrix) != 0)
        active = operator.matrix[np.ix_(incident, incident)]
        solved = solve_symmetric(active, max_dense_entries=max_dense_entries, tol=tol)
        values = np.sort(np.concatenate((np.zeros(basis_size - len(incident)), solved.values)))
        solution = EigenSolution(values, None, True, int(np.count_nonzero(np.abs(values) <= tol)), None)
        metadata["isolated_zeros_restored"] = basis_size - len(incident)
    else:
        solution = solve_symmetric(matrix if return_matrix else operator.matrix, k=k, return_eigenvectors=return_eigenvectors,
                                   max_dense_entries=max_dense_entries, tol=tol)
    metadata.update(operator.metadata)
    metadata.update({"definition_id": DEFINITION_ID, "basis_kind": "orthonormal_combinations",
                     "basis_note": "eigenvector rows index these labelled combinations of native ordered hyperedges",
                     "operator_dimension": basis_size,
                     "source_chain_dimension": basis_size,
                     "structural_absence": basis_size == 0,
                     "tol": tol, "tolerance": tol, "eigen_residual_max": solution.residual_max,
                     "residual_max": solution.residual_max,
                     "operator_kind": "pairwise_persistent" if start is not None else "ordinary",
                     "coefficient_field": "R", "complete_spectrum": solution.complete,
                     "field_note": "real nullities need not equal GF(2) Betti numbers"})
    return SpectrumResult(q, solution.values, solution.vectors, matrix,
                          labelled_basis(operator.basis, operator.hyperedges), scale, start, end,
                          "persistent" if start is not None else "ordinary",
                          solution.complete, solution.nullity, metadata)


class L0Sweep:
    """Incremental ordinary L0 of a defined filtered hyperdigraph.

    Every edge must have both singleton faces born at or before its birth.
    Edge births are filtration coordinates, not numerical weights. Reciprocal
    edges contribute separately; higher hyperedges are retained in the source
    object but do not enter ordinary L0. This is not a two-scale persistent
    Laplacian. Missing-face filtrations must use the general ``laplacian`` API.

    Scales are inclusive, finite, within the recorded domain and nondecreasing.
    The dense buffer is allocated once for all explicitly present singletons;
    its N*N entries must fit max_dense_entries. The edge schedule retains the
    existing max_sparse_entries guard on 2*edge_count and singleton_count.
    Returned matrices are independent copies in active ambient-vertex order.
    Instances are stateful and must not be advanced concurrently.
    """

    @staticmethod
    def supports(obj):
        native, _ = _unwrap(obj, 0)
        return isinstance(native, FilteredHyperdigraph) and supports_incremental_l0(native)

    def __init__(self, obj, *, max_dense_entries=4_000_000, max_sparse_entries=2_000_000):
        native, self._metadata = _unwrap(obj, 0)
        if not isinstance(native, FilteredHyperdigraph):
            raise TypeError("L0Sweep requires a filtered hyperdigraph")
        self.max_dense_entries = _dimension(max_dense_entries, "max_dense_entries")
        self.max_sparse_entries = _dimension(max_sparse_entries, "max_sparse_entries")
        self._state = L0Accumulator(native, self.max_dense_entries, self.max_sparse_entries)

    def _advance(self, scale):
        if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, numbers.Real):
            raise ValueError("scale must be a finite real number")
        value = validate_query(self._metadata, scale)
        matrix, labels = self._state.advance(value)
        return value, matrix, labels

    @property
    def diagnostics(self):
        """A detached record of actual edge insertions, excluding matrix copies."""
        return dict(self._state.diagnostics)

    def matrix_at(self, scale):
        """Return a fresh dense L0 matrix; no eigensolver is invoked."""
        _, matrix, _ = self._advance(scale)
        return matrix.copy()

    def basis_at(self, scale):
        """Return the singleton hyperedges indexing matrix_at(scale)."""
        _, _, labels = self._advance(scale)
        return labels

    def vertex_deleted_laplacian(self, scale, vertex, *, return_eigenvectors=False,
                                 return_matrix=False, tol=1e-10):
        """Ordinary L0 after independently removing one active vertex.

        Delete the vertex and all incident edges, including reciprocal edges,
        and recompute the remaining degrees. The sweep state is unchanged by
        deletion, so every call starts from the same undeleted graph at scale.
        ``vertex`` is an ambient vertex ID, not a matrix row number. A missing
        or not-yet-born vertex raises ValueError. All remaining singletons,
        including isolates, are retained. The complete spectrum is returned.

        This optional L0 operation changes neither higher-dimensional nor
        two-scale persistent Laplacian algorithms. Query scales still obey
        the sweep's nondecreasing rule.
        """
        from ._hyperdigraph.l0_sweep import delete_vertex_l0
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        value, matrix, labels = self._advance(scale)
        try:
            index = labels.index((vertex,))
        except ValueError as error:
            raise ValueError("vertex must be an active singleton ID") from error
        reduced = delete_vertex_l0(matrix, index)
        kept_labels = labels[:index] + labels[index + 1:]
        edge_count = self._state.edge_count - int(round(float(matrix[index, index])))
        operator = EmbeddedOperator(reduced, sparse.eye(len(kept_labels), format="csc"), kept_labels, {
            **self.diagnostics, "basis_backend": "face_closed_units",
            "upper_projection_backend": "identity_no_missing_faces",
            "omega_dimension": len(kept_labels), "upper_omega_dimension": edge_count,
            "upper_constraint_rank": 0, "ambient_hyperedge_count": len(kept_labels),
            "upper_hyperedge_count": edge_count,
            "operator_definition": "native_sequence_embedded_euclidean_laplacian",
            "vertex_deletion": "independent_induced_subgraph_recomputed_degrees",
            "deleted_vertex": vertex,
        })
        return _spectrum(operator, 0, dict(self._metadata), k=None,
                         return_eigenvectors=return_eigenvectors, return_matrix=return_matrix,
                         max_dense_entries=self.max_dense_entries, tol=tol, scale=value,
                         isolate_zeros=True)

    def laplacian(self, scale, *, return_eigenvectors=False, k=None,
                  return_matrix=False, tol=1e-10):
        """Use the unchanged symmetric eigensolver on the accumulated matrix.

        Complete eigenvalue-only queries retain the existing exact isolated-zero
        restoration. Eigenvector and partial queries use the full active basis.
        """
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if k is not None and not _dimension(k, "k"):
            raise ValueError("k must be positive")
        value, matrix, labels = self._advance(scale)
        operator = EmbeddedOperator(matrix, sparse.eye(len(labels), format="csc"), labels, {
            **self.diagnostics, "basis_backend": "face_closed_units",
            "upper_projection_backend": "identity_no_missing_faces",
            "omega_dimension": len(labels), "upper_omega_dimension": self._state.edge_count,
            "upper_constraint_rank": 0, "ambient_hyperedge_count": len(labels),
            "upper_hyperedge_count": self._state.edge_count,
            "operator_definition": "native_sequence_embedded_euclidean_laplacian",
        })
        return _spectrum(operator, 0, dict(self._metadata), k=k,
                         return_eigenvectors=return_eigenvectors, return_matrix=return_matrix,
                         max_dense_entries=self.max_dense_entries, tol=tol, scale=value,
                         isolate_zeros=True)


def laplacian(obj, dimension=0, scale=None, *, return_eigenvectors=False, k=None,
              return_matrix=False, max_dense_entries=4_000_000,
              max_sparse_entries=2_000_000, tol=1e-10, backend="auto"):
    q = _dimension(dimension, "dimension")
    dense_budget = _dimension(max_dense_entries, "max_dense_entries")
    sparse_budget = _dimension(max_sparse_entries, "max_sparse_entries")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    native, metadata = _unwrap(obj, q)
    snapshot, value = _snapshot(native, metadata, scale)
    if backend not in {"auto", "reference"}:
        raise ValueError("backend must be 'auto' or 'reference'")
    build_operator = ordinary_operator if backend == "auto" else reference_operator
    operator = build_operator(snapshot, q, max_dense_entries=dense_budget,
                     max_sparse_entries=sparse_budget, tol=tol)
    return _spectrum(operator, q, metadata, k=k, return_eigenvectors=return_eigenvectors,
                     return_matrix=return_matrix, max_dense_entries=dense_budget, tol=tol, scale=value)


def persistent_laplacian(obj, dimension=0, *, start, end, return_eigenvectors=False,
                         k=None, return_matrix=False, max_dense_entries=4_000_000,
                         max_sparse_entries=2_000_000, tol=1e-10, backend="auto"):
    q = _dimension(dimension, "dimension")
    dense_budget = _dimension(max_dense_entries, "max_dense_entries")
    sparse_budget = _dimension(max_sparse_entries, "max_sparse_entries")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    native, metadata = _unwrap(obj, q)
    if not isinstance(native, FilteredHyperdigraph):
        raise TypeError("a pairwise persistent Laplacian requires a filtered hyperdigraph")
    source_scale, target_scale = validate_query(metadata, start), validate_query(metadata, end)
    if source_scale is None or target_scale is None or source_scale > target_scale:
        raise ValueError("finite pairwise scales start <= end are required")
    if backend == "auto":
        operator = pairwise_operator(native, q, source_scale, target_scale, max_dense_entries=dense_budget,
                                     max_sparse_entries=sparse_budget, tol=tol)
    elif backend == "reference":
        operator = reference_operator(native.snapshot(source_scale), q, end_obj=native.snapshot(target_scale),
                                      max_dense_entries=dense_budget,
                                      max_sparse_entries=sparse_budget, tol=tol)
    else:
        raise ValueError("backend must be 'auto' or 'reference'")
    return _spectrum(operator, q, metadata, k=k, return_eigenvectors=return_eigenvectors,
                     return_matrix=return_matrix, max_dense_entries=dense_budget, tol=tol,
                     start=source_scale, end=target_scale)


__all__ = ['Hyperdigraph', 'FilteredHyperdigraph', 'WeightedHyperedge', 'L0Sweep',
           'homology', 'persistence', 'laplacian', 'persistent_laplacian']
