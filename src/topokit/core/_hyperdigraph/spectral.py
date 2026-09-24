"""Budgeted operators in the native real embedded chain spaces.

The upper term is evaluated as B P_ker(M) B^T, where M contains missing
deletion faces. This is the same Euclidean embedded-chain operator as an
explicit orthonormal Omega basis, without materializing that upper basis.
No homology field, filtration, hyperedge, or metric is changed.
"""

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu

from ...exceptions import ResourceLimitError
from .laplacian import _orthonormal_nullspace, _build_real_chain_complex, _persistent_up_boundary


@dataclass
class EmbeddedOperator:
    matrix: LinearOperator | np.ndarray
    basis: sparse.csc_matrix
    hyperedges: tuple
    metadata: dict


def _guard(entries, budget, name):
    if entries > budget:
        raise ResourceLimitError(
            f"{name} needs {entries:,} entries, exceeding the explicit budget "
            f"of {budget:,}; no hyperedges or dimensions were removed"
        )


def guarded_product(left, right, budget, name):
    """Bound sparse output support BEFORE numerical multiplication.

    Scalar-product counts give a cheap upper bound. If that is inconclusive,
    union structural row supports one output column at a time; this can stop
    as soon as the budget is exceeded without allocating the product.
    Cancellation can only reduce the actual output support.
    """
    left, right = left.tocsc(), right.tocsc()
    if left.shape[1] != right.shape[0]:
        raise ValueError("sparse product dimensions do not align")
    left_counts = np.diff(left.indptr)
    right_counts = np.bincount(right.indices, minlength=right.shape[0])
    count = int(np.dot(left_counts.astype(np.int64), right_counts.astype(np.int64)))
    if count > budget:
        total = 0
        for column in range(right.shape[1]):
            rows = set()
            for index in right.indices[right.indptr[column]:right.indptr[column + 1]]:
                rows.update(map(int, left.indices[left.indptr[index]:left.indptr[index + 1]]))
                _guard(total + len(rows), budget, name + " structural output bound")
            total += len(rows)
    return left @ right


def boundary_parts(obj, q, max_sparse_entries):
    """Signed deletion boundary separated into present and absent faces."""
    source_hyperedges = obj.directed_hyperedges(q)
    if q == 0:
        empty = sparse.csc_matrix((0, len(source_hyperedges)))
        return empty, empty
    _guard((q + 1) * len(source_hyperedges), max_sparse_entries, "Sparse boundary")
    lower_index = {edge: i for i, edge in enumerate(obj.directed_hyperedges(q - 1))}
    missing_index = {}
    # Keep signed sparse entries for allowed boundaries and forbidden faces apart.
    present_rows, present_columns, present_values = [], [], []
    missing_rows, missing_columns, missing_values = [], [], []
    for j, edge in enumerate(source_hyperedges):
        for i in range(q + 1):
            face = edge[:i] + edge[i + 1:]
            coefficient = (-1.0) ** i
            if face in lower_index:
                present_rows.append(lower_index[face])
                present_columns.append(j); present_values.append(coefficient)
            else:
                missing_rows.append(missing_index.setdefault(face, len(missing_index)))
                missing_columns.append(j); missing_values.append(coefficient)
    return (
        sparse.csc_matrix((present_values, (present_rows, present_columns)),
                          shape=(len(lower_index), len(source_hyperedges))),
        sparse.csc_matrix((missing_values, (missing_rows, missing_columns)),
                          shape=(len(missing_index), len(source_hyperedges))),
    )


def omega_basis(missing, max_dense_entries, max_sparse_entries, tol):
    """An explicit orthonormal basis, sparse for one-missing-face groups."""
    n = missing.shape[1]
    counts = np.diff(missing.indptr)
    if not missing.shape[0]:
        _guard(n, max_sparse_entries, "Omega identity basis")
        return sparse.eye(n, format="csc"), "face_closed_units"
    if np.max(counts, initial=0) <= 1:
        immediate = np.flatnonzero(counts == 0)
        groups = defaultdict(list)
        for j in np.flatnonzero(counts):
            offset = missing.indptr[j]
            groups[int(missing.indices[offset])].append((int(j), missing.data[offset]))
        predicted = len(immediate) + sum((len(g) - 1) * (len(g) + 2) // 2 for g in groups.values())
        _guard(predicted, max_sparse_entries, "Sparse orthonormal Omega basis")
        rows, cols, vals = list(map(int, immediate)), list(range(len(immediate))), [1.0] * len(immediate)
        column = len(immediate)
        for group in groups.values():
            # Signed Helmert contrasts form an orthonormal kernel of one row.
            for j in range(1, len(group)):
                scale = np.sqrt(j * (j + 1.0))
                for i in range(j):
                    row, sign = group[i]
                    rows.append(row); cols.append(column); vals.append(sign / scale)
                row, sign = group[j]
                rows.append(row); cols.append(column); vals.append(-j * sign / scale)
                column += 1
        return sparse.csc_matrix((vals, (rows, cols)), shape=(n, column)), "signed_helmert_groups"
    _guard(max(missing.shape[0] * n, n * n), max_dense_entries, "General Omega SVD")
    basis = _orthonormal_nullspace(missing.toarray(), n, tol)
    _guard(basis.size, max_sparse_entries, "General Omega basis")
    return sparse.csc_matrix(basis), "svd_constraint_nullspace"


class KernelProjection:
    """Euclidean orthogonal projection onto a sparse constraint kernel."""

    def __init__(self, missing, max_dense_entries, tol):
        self.n = missing.shape[1]
        self.constraints = None
        self.factor = None
        self.explicit_basis = None
        self.rank = 0
        if missing.shape[0] == 0:
            self.backend = "identity_no_missing_faces"
            return
        counts = np.diff(missing.indptr)
        if np.max(counts, initial=0) <= 2:
            # For signed incidence constraints, identify independent rows by
            # connected-component balance. A balanced unanchored component has
            # exactly one row dependency; a singleton constraint anchors it.
            adjacency = [[] for _ in range(missing.shape[0])]
            anchors = set()
            for j in range(self.n):
                column_start, column_end = missing.indptr[j:j + 2]
                rows, values = missing.indices[column_start:column_end], missing.data[column_start:column_end]
                if len(rows) == 1:
                    anchors.add(int(rows[0]))
                elif len(rows) == 2:
                    first_row, second_row = map(int, rows)
                    ratio = float(-values[0] / values[1])
                    adjacency[first_row].append((second_row, ratio))
                    adjacency[second_row].append((first_row, 1.0 / ratio))
            component_signs, independent_rows, fill_bound = {}, [], 0
            for root in range(len(adjacency)):
                if root in component_signs:
                    continue
                component_signs[root] = 1.0
                stack, component, anchored = [root], [], False
                while stack:
                    row = stack.pop()
                    component.append(row)
                    anchored = anchored or row in anchors
                    for neighbor, ratio in adjacency[row]:
                        value = component_signs[row] * ratio
                        if neighbor not in component_signs:
                            component_signs[neighbor] = value
                            stack.append(neighbor)
                        elif abs(component_signs[neighbor] - value) > tol:
                            anchored = True
                independent = component if anchored else component[1:]
                fill_bound += len(independent) ** 2
                independent_rows.extend(independent)
            _guard(fill_bound, max_dense_entries, "Constraint factorization fill bound")
            self.constraints = missing.tocsr()[sorted(independent_rows)].tocsc()
            self.rank = len(independent_rows)
            if self.rank:
                gram = (self.constraints @ self.constraints.T).tocsc()
                self.factor = splu(gram)
            self.backend = "signed_incidence_component_projection"
        else:
            _guard(max(missing.shape[0] * self.n, self.n ** 2), max_dense_entries, "General upper-domain SVD")
            self.explicit_basis = _orthonormal_nullspace(missing.toarray(), self.n, tol)
            self.rank = self.n - self.explicit_basis.shape[1]
            self.backend = "svd_kernel_projection"
        probe = np.linspace(1.0, 2.0, self.n)
        projected = self(probe)
        scale = max(1.0, np.linalg.norm(missing @ probe))
        if np.linalg.norm(missing @ projected) > 100.0 * tol * scale:
            raise RuntimeError("upper-domain projection does not satisfy missing-face constraints")
        if np.linalg.norm(self(projected) - projected) > 100.0 * tol * max(1.0, np.linalg.norm(projected)):
            raise RuntimeError("upper-domain projection is not numerically idempotent")

    def __call__(self, vector):
        if self.explicit_basis is not None:
            return self.explicit_basis @ (self.explicit_basis.T @ vector)
        if self.factor is None:
            return vector
        rhs = np.asarray(self.constraints @ vector)
        return vector - self.constraints.T @ self.factor.solve(rhs)


def ordinary_operator(obj, q, *, max_dense_entries=4_000_000,
                      max_sparse_entries=2_000_000, tol=1e-10):
    present, missing = boundary_parts(obj, q, max_sparse_entries)
    basis, backend = omega_basis(missing, max_dense_entries, max_sparse_entries, tol)
    upper, upper_missing = boundary_parts(obj, q + 1, max_sparse_entries)
    _guard(max(upper.shape[1], upper_missing.shape[0]), max_dense_entries, "Projection workspace")
    projector = KernelProjection(upper_missing, max_dense_entries, tol)
    down = guarded_product(present, basis, max_sparse_entries, "Embedded down boundary").tocsc()
    up = guarded_product(basis.T, upper, max_sparse_entries - down.nnz,
                         "Embedded combined boundary storage").tocsr()
    basis_size = basis.shape[1]

    def apply(v):
        return np.asarray(down.T @ (down @ v) + up @ projector(up.T @ v))

    def matmat(v):
        # A full eigensolver may pass an identity. Bound temporary dense
        # upper-domain blocks independently of the returned square basis matrix.
        output = np.empty((basis_size, v.shape[1]))
        block_width = max(1, min(32, max_dense_entries // max(1, upper.shape[1], upper_missing.shape[0])))
        for first_column in range(0, v.shape[1], block_width):
            output[:, first_column:first_column + block_width] = apply(
                v[:, first_column:first_column + block_width])
        return output

    operator = LinearOperator((basis_size, basis_size), matvec=apply, rmatvec=apply, matmat=matmat, dtype=float)
    return EmbeddedOperator(operator, basis, obj.directed_hyperedges(q), {
        "basis_backend": backend,
        "upper_projection_backend": projector.backend,
        "omega_dimension": basis_size,
        "upper_omega_dimension": upper.shape[1] - projector.rank,
        "upper_constraint_rank": projector.rank,
        "ambient_hyperedge_count": len(obj.directed_hyperedges(q)),
        "upper_hyperedge_count": upper.shape[1],
        "coefficient_field": "R",
        "operator_definition": "native_sequence_embedded_euclidean_laplacian",
    })


def pairwise_operator(filtration, q, start, end, *, max_dense_entries=4_000_000,
                      max_sparse_entries=2_000_000, tol=1e-10):
    if start == end:
        return ordinary_operator(filtration.snapshot(start), q,
                                 max_dense_entries=max_dense_entries,
                                 max_sparse_entries=max_sparse_entries, tol=tol)
    source_snapshot, target_snapshot = filtration.snapshot(start), filtration.snapshot(end)
    down_ambient, start_missing = boundary_parts(source_snapshot, q, max_sparse_entries)
    basis, backend = omega_basis(start_missing, max_dense_entries, max_sparse_entries, tol)
    down = guarded_product(down_ambient, basis, max_sparse_entries,
                           "Pairwise embedded down boundary").tocsc()
    upper, absent = boundary_parts(target_snapshot, q + 1, max_sparse_entries)
    target_index = {edge: i for i, edge in enumerate(target_snapshot.directed_hyperedges(q))}
    source_hyperedges = source_snapshot.directed_hyperedges(q)
    injection = sparse.csc_matrix((np.ones(len(source_hyperedges)),
                                  ([target_index[e] for e in source_hyperedges], range(len(source_hyperedges)))),
                                 shape=(len(target_index), len(source_hyperedges)))
    embedded = injection @ basis
    source_count = upper.shape[1]
    # Pairwise constraints also enforce boundary membership in start Omega.
    _guard(max((upper.shape[0] + absent.shape[0]) * source_count, source_count ** 2),
           max_dense_entries, "Pairwise upper-domain SVD")
    projected = guarded_product(embedded.T, upper, max_sparse_entries - down.nnz,
                                "Pairwise embedded combined boundary storage").toarray()
    outside = upper.toarray() - embedded @ projected
    constraints = np.vstack((absent.toarray(), outside))
    kernel = _orthonormal_nullspace(constraints, source_count, tol)
    up = projected @ kernel
    basis_size = basis.shape[1]

    def apply(v):
        return np.asarray(down.T @ (down @ v) + up @ (up.T @ v))

    operator = LinearOperator((basis_size, basis_size), matvec=apply, rmatvec=apply, matmat=apply, dtype=float)
    return EmbeddedOperator(operator, basis, source_hyperedges, {
        "basis_backend": backend, "upper_projection_backend": "pairwise_svd_restriction",
        "omega_dimension": basis_size, "persistent_upper_domain_dimension": kernel.shape[1],
        "coefficient_field": "R", "operator_definition": "native_pairwise_persistent_laplacian",
    })


def labelled_basis(basis, hyperedges):
    """Each label is the exact stored linear combination, not a single cell."""
    columns = basis.tocsc()
    return tuple(tuple((hyperedges[int(row)], float(value))
                       for row, value in zip(columns.indices[columns.indptr[j]:columns.indptr[j + 1]],
                                             columns.data[columns.indptr[j]:columns.indptr[j + 1]]))
                 for j in range(columns.shape[1]))


def reference_operator(obj, q, *, end_obj=None, max_dense_entries=4_000_000,
                       max_sparse_entries=2_000_000, tol=1e-10):
    """Forced original real-Omega/SVD construction for conformance checks."""
    for current in (obj,) if end_obj is None else (obj, end_obj):
        for dim in range(q + 2):
            present, missing = boundary_parts(current, dim, max_sparse_entries)
            n = present.shape[1]
            _guard(max(n * n, present.shape[0] * n, missing.shape[0] * n),
                   max_dense_entries, "Reference real-chain assembly")
    source_chain = _build_real_chain_complex(obj, q, omega_backend="svd", tolerance=tol)
    if end_obj is None:
        up = source_chain.boundaries[q + 1]
    else:
        target_chain = _build_real_chain_complex(end_obj, q, omega_backend="svd", tolerance=tol)
        _guard(target_chain.boundaries[q + 1].shape[1] ** 2, max_dense_entries, "Reference persistent upper-domain SVD")
        up = _persistent_up_boundary(source_chain, target_chain, q, tol)
    down = source_chain.boundaries[q]
    n = source_chain.omega_bases[q].shape[1]
    _guard(n * n, max_dense_entries, "Reference Laplacian")
    matrix = down.T @ down + up @ up.T
    operator = LinearOperator((n, n), matvec=lambda v: matrix @ v,
                             rmatvec=lambda v: matrix @ v, matmat=lambda v: matrix @ v, dtype=float)
    return EmbeddedOperator(operator, sparse.csc_matrix(source_chain.omega_bases[q]), source_chain.hyperedges[q], {
        "basis_backend": "reference_svd", "upper_projection_backend": "reference_svd",
        "coefficient_field": "R", "omega_dimension": n,
        "operator_definition": "native_sequence_embedded_euclidean_laplacian",
    })
