"""Real Hodge Laplacians and genuine inclusion-persistent Laplacians."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh

from ._validation import guard_dense, nonnegative_int, real, tolerance
from .algebra import boundary_matrix
from .complex import Simplex, as_complex


@dataclass(frozen=True)
class LaplacianResult:
    dimension: int
    simplices: tuple[Simplex, ...]
    matrix: object
    eigenvalues: np.ndarray
    betti_number: int
    kind: str
    tolerance: float

    @property
    def summary(self):
        return summarize_spectrum(self.eigenvalues, tol=self.tolerance)


def laplacian_matrix(complex_, dimension=0, *, sparse_output=True,
                     max_dense_entries=10_000_000):
    """Lq = d_q.T d_q + d_(q+1) d_(q+1).T in sorted simplex bases."""
    complex_ = as_complex(complex_)
    dimension = nonnegative_int(dimension, "dimension")
    down = boundary_matrix(complex_, dimension).astype(float)
    up = boundary_matrix(complex_, dimension + 1).astype(float)
    result = (down.T @ down + up @ up.T).tocsr()
    if sparse_output:
        return result
    guard_dense(result.shape, max_dense_entries)
    return result.toarray()


def _pair(source, target):
    source, target = as_complex(source), as_complex(target)
    if not set(source).issubset(target):
        raise ValueError("Persistent Laplacian requires source to be a subcomplex of target")
    return source, target


def persistent_laplacian_matrix(source, target, dimension=0, *, tol=1e-10,
                                sparse_output=True, max_dense_entries=10_000_000):
    """Lq(K,L) for K subset L, using the inherited Euclidean chain metric.

    Split d_(q+1)^L into rows A on Kq and B outside Kq. Allowed later chains
    satisfy Bx=0, so the persistent up term is A P_ker(B) A.T, NOT simply
    A A.T. A thin SVD of B gives P=I-Vr.T Vr without an explicit kernel basis.
    The down term is d_q^K.T d_q^K. At K=L this is the ordinary Laplacian.

    Dense work is guarded; this is a small/moderate-complex spectral backend.
    """
    source, target = _pair(source, target)
    dimension = nonnegative_int(dimension, "dimension")
    tol = tolerance(tol)
    source_basis = source.simplices(dimension)
    n = len(source_basis)
    if n == 0:
        return sparse.csr_matrix((0, 0)) if sparse_output else np.zeros((0, 0))
    down = boundary_matrix(source, dimension).astype(float)
    lower = (down.T @ down).tocsr()
    target_basis = target.simplices(dimension)
    target_index = {s: i for i, s in enumerate(target_basis)}
    source_rows = [target_index[s] for s in source_basis]
    source_row_set = set(source_rows)
    outside_rows = [i for i in range(len(target_basis)) if i not in source_row_set]
    boundary = boundary_matrix(target, dimension + 1).astype(float)
    # These are blocks A and B in the docstring, with all target columns retained.
    source_boundary = boundary[source_rows, :]
    outside_boundary = boundary[outside_rows, :]
    if outside_boundary.nnz == 0:
        result = (lower + source_boundary @ source_boundary.T).tocsr()
        if sparse_output:
            return result
        guard_dense(result.shape, max_dense_entries)
        return result.toarray()

    # Zero constraint rows need no SVD work.
    outside_boundary = outside_boundary[np.diff(outside_boundary.indptr) > 0, :]
    guard_dense(outside_boundary.shape, max_dense_entries)
    guard_dense((n, n), max_dense_entries)
    guard_dense((n, min(outside_boundary.shape)), max_dense_entries)
    _, singular_values, vt = linalg.svd(outside_boundary.toarray(), full_matrices=False)
    rank_cutoff = tol * max(float(singular_values[0]), 1.0) if singular_values.size else tol
    constraint_rowspace = vt[singular_values > rank_cutoff]
    correction = source_boundary @ constraint_rowspace.T
    result = (lower + source_boundary @ source_boundary.T).toarray() - correction @ correction.T
    result = (result + result.T) / 2
    return sparse.csr_matrix(result) if sparse_output else result


def spectrum(matrix, *, k=None, tol=1e-10, max_dense_entries=10_000_000):
    """Sorted eigenvalues, optionally the k smallest via a sparse solver.

    Partial spectra must not be used to infer total nullity. Significant
    negative values are rejected rather than silently relabeled as zeros.
    """
    tol = tolerance(tol)
    if sparse.issparse(matrix):
        matrix = matrix.astype(float).tocsr()
        finite = np.isfinite(matrix.data).all()
    else:
        matrix = np.asarray(matrix, dtype=float)
        finite = np.isfinite(matrix).all()
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or not finite:
        raise ValueError("matrix must be finite and square")
    n = matrix.shape[0]
    if k is not None:
        k = nonnegative_int(k, "k")
        if k == 0 or k > n:
            raise ValueError("k must lie between 1 and matrix size")
    if sparse.issparse(matrix):
        symmetry_difference = matrix - matrix.T
        symmetric = not symmetry_difference.nnz or np.max(np.abs(symmetry_difference.data)) <= tol
    else:
        symmetric = np.allclose(matrix, matrix.T, rtol=0, atol=tol)
    if not symmetric:
        raise ValueError("matrix must be symmetric")
    if n == 0:
        return np.empty(0)
    if k is not None and k < n - 1:
        if sparse.issparse(matrix) and matrix.nnz == 0:
            return np.zeros(k)
        values = eigsh(sparse.csr_matrix(matrix), k=k, which="SA", return_eigenvectors=False)
    else:
        guard_dense(matrix.shape, max_dense_entries)
        values = np.linalg.eigvalsh(matrix.toarray() if sparse.issparse(matrix) else matrix)
        if k is not None:
            values = values[:k]
    if values.size and np.min(values) < -tol:
        raise ValueError("Matrix has a negative eigenvalue beyond tol; check PSD or adjust numerical tolerance")
    values[np.abs(values) <= tol] = 0.0
    return np.sort(values)


def summarize_spectrum(eigenvalues, *, tol=1e-10):
    tol = tolerance(tol)
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("eigenvalues must be a finite vector")
    positive = values[values > tol]
    return {"num_eigenvalues": len(values), "num_zero": int(np.count_nonzero(np.abs(values) <= tol)),
            "min": float(values.min()) if values.size else None,
            "min_nonzero": float(positive.min()) if positive.size else None,
            "max": float(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
            "mean_nonzero": float(positive.mean()) if positive.size else None,
            "sum": float(values.sum())}


def _result(complex_, dimension, matrix, kind, tol, max_dense_entries):
    values = spectrum(matrix, tol=tol, max_dense_entries=max_dense_entries)
    return LaplacianResult(dimension, complex_.simplices(dimension), matrix, values,
                           int(np.count_nonzero(values == 0)), kind, tol)


def laplacian(complex_, dimension=0, *, scale=None, tol=1e-10,
              sparse_output=True, max_dense_entries=10_000_000):
    """Matrix, full spectrum, and real Betti number for one fixed complex."""
    complex_ = as_complex(complex_)
    if scale is not None:
        complex_ = complex_.at(scale)
    dimension = nonnegative_int(dimension, "dimension")
    tol = tolerance(tol)
    matrix = laplacian_matrix(complex_, dimension, sparse_output=sparse_output,
                              max_dense_entries=max_dense_entries)
    return _result(complex_, dimension, matrix, "ordinary", tol, max_dense_entries)


def persistent_laplacian(source, target, dimension=0, *, tol=1e-10,
                         sparse_output=True, max_dense_entries=10_000_000):
    """Matrix, full spectrum, and real persistent Betti number for K subset L."""
    source, target = _pair(source, target)
    dimension = nonnegative_int(dimension, "dimension")
    tol = tolerance(tol)
    matrix = persistent_laplacian_matrix(source, target, dimension, tol=tol,
                                         sparse_output=sparse_output, max_dense_entries=max_dense_entries)
    return _result(source, dimension, matrix, "persistent", tol, max_dense_entries)


def persistent_laplacian_at(complex_, source_scale, target_scale, dimension=0, **kwargs):
    """Two-scale persistent Laplacian of a single filtration."""
    source_scale = real(source_scale, "source_scale", finite=False)
    target_scale = real(target_scale, "target_scale", finite=False)
    if source_scale > target_scale:
        raise ValueError("source_scale must be <= target_scale")
    complex_ = as_complex(complex_)
    return persistent_laplacian(complex_.at(source_scale), complex_.at(target_scale), dimension, **kwargs)


def laplacian_filtration(complex_, scales, *, dimensions=(0, 1), **kwargs):
    """Ordinary snapshot Laplacians, intentionally distinct from persistence."""
    complex_ = as_complex(complex_)
    dimensions = tuple(nonnegative_int(q, "dimension") for q in dimensions)
    results = {}
    for scale in scales:
        scale = real(scale, "scale", finite=False)
        snapshot = complex_.at(scale)
        results[scale] = {q: laplacian(snapshot, q, **kwargs) for q in dimensions}
    return results
