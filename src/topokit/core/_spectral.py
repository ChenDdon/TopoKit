"""Budgeted real symmetric eigensolvers used by all three adapters."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from .._validation import dimension
from ..exceptions import ResourceLimitError


@dataclass(frozen=True)
class EigenSolution:
    values: np.ndarray
    vectors: np.ndarray | None
    complete: bool
    nullity: int | None
    residual_max: float | None


def solve_symmetric(matrix, *, k=None, return_eigenvectors=False,
                    max_dense_entries=4_000_000, tol=1e-10):
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be square")
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    dense_entry_limit = dimension(max_dense_entries, "max_dense_entries")
    n = matrix.shape[0]
    if k is not None:
        k = dimension(k, "k")
        if k == 0:
            raise ValueError("k must be positive")
    complete = k is None or k >= n
    if complete and n * n > dense_entry_limit:
        raise ResourceLimitError(f"Full spectrum needs a {n} x {n} dense operator; request k < {n} or raise max_dense_entries explicitly")
    if not n:
        return EigenSolution(np.empty(0), np.empty((0, 0)) if return_eigenvectors else None, True, 0, 0.0 if return_eigenvectors else None)
    if sparse.issparse(matrix):
        if np.iscomplexobj(matrix.data):
            raise ValueError("operator must be real symmetric")
        matrix = sparse.csr_matrix(matrix, dtype=float)
        difference = matrix - matrix.T
        magnitude = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
        if not np.all(np.isfinite(matrix.data)) or (difference.nnz and np.max(np.abs(difference.data)) > tol + 1e-9*magnitude):
            raise ValueError("Laplacian must be finite and symmetric")
    elif not isinstance(matrix, LinearOperator):
        if np.iscomplexobj(matrix):
            raise ValueError("operator must be real symmetric")
        matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=1e-9, atol=tol):
            raise ValueError("Laplacian must be finite and symmetric")
    elif not complete:
        # Opaque operators cannot be exhaustively validated without materializing
        # them. Native cores provide symmetric operators; these deterministic
        # probes additionally catch gross nonfinite/asymmetric implementations.
        probe_x = np.linspace(1.0, 2.0, n)
        probe_y = np.cos(np.arange(n, dtype=float))
        image_x, image_y = matrix @ probe_x, matrix @ probe_y
        if (not np.all(np.isfinite(image_x)) or not np.all(np.isfinite(image_y))
                or not np.isclose(probe_x @ image_y, image_x @ probe_y, rtol=1e-8, atol=tol*n)):
            raise ValueError("Laplacian operator failed finite/symmetric probes")
    if complete:
        if sparse.issparse(matrix):
            dense_matrix = matrix.toarray()
        elif isinstance(matrix, LinearOperator):
            dense_matrix = matrix.matmat(np.eye(n))
        else:
            dense_matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(dense_matrix)) or not np.allclose(dense_matrix, dense_matrix.T, rtol=1e-9, atol=tol):
            raise ValueError("Laplacian must be finite and symmetric")
        if return_eigenvectors:
            values, vectors = np.linalg.eigh(dense_matrix)
        else:
            values, vectors = np.linalg.eigvalsh(dense_matrix), None
    else:
        operator = matrix if sparse.issparse(matrix) or isinstance(matrix, LinearOperator) else sparse.csr_matrix(matrix)
        eigensolver_result = eigsh(operator, k=k, which="SM", tol=tol, return_eigenvectors=return_eigenvectors,
                                   v0=np.linspace(1.0, 2.0, n))
        if return_eigenvectors:
            values, vectors = eigensolver_result
        else:
            values, vectors = eigensolver_result, None
    eigenvalue_order = np.argsort(values)
    values = np.asarray(values)[eigenvalue_order]
    vectors = None if vectors is None else vectors[:, eigenvalue_order]
    residual_max = None
    if vectors is not None:
        residual_max = float(np.max(np.linalg.norm(matrix @ vectors - vectors * values, axis=0)))
    return EigenSolution(values, vectors, complete,
                         int(np.count_nonzero(np.abs(values) <= tol)) if complete else None,
                         residual_max)
