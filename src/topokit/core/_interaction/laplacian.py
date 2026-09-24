"""Real ordinary and pairwise persistent interaction Laplacians.

The interaction product generators are orthonormal.  If the signed boundary
at ``end`` is partitioned as ``B_(q+1)(end) = [A; N]``, where the rows of
``A`` already exist at ``start``, the persistent up operator is

``A P_ker(N) A.T = A A.T - (A V) (A V).T``.

Here the columns of ``V`` are an orthonormal basis of the row space of ``N``.
The latter formula preserves the inherited inner product without forming a
large nullspace basis or projector.  Constraints are decomposed into sparse
connected components before numerical work.  The down operator is always
``B_q(start).T B_q(start)``.  A sequence of ordinary snapshot Laplacians is
therefore distinct from a genuine pairwise persistent Laplacian.

NumPy and SciPy are optional runtime dependencies, imported only on a numerical
call.  Importing this module, or the package's GF(2) homology API, requires only
the Python standard library.  Install ``interaction-topology[laplacian]`` to
use the numerical API.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import inf, isfinite, isnan
from numbers import Integral, Real
from typing import Any

from .interaction import FilteredInteractionChainComplex, InteractionKey


DEFAULT_MAX_DENSE_BYTES = 256 * 1024**2


def _numerics() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from scipy import sparse
        from scipy.sparse import linalg
    except ImportError as error:
        raise ImportError(
            "Interaction Laplacians require the optional NumPy and SciPy "
            "dependencies. Install them with "
            "`python -m pip install 'interaction-topology[laplacian]'`. "
            "The GF(2) homology API does not require these packages."
        ) from error
    return np, sparse, linalg


def _nonnegative_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-negative finite real number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite real number")
    return result


def _threshold(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    result = float(value)
    if isnan(result):
        raise ValueError(f"{name} cannot be NaN")
    return result


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _check_budget(estimated: int, limit: int, operation: str) -> None:
    if estimated > limit:
        raise MemoryError(
            f"{operation} requires approximately {estimated / 1024**2:.2f} MiB "
            f"of numerical storage/workspace, exceeding max_dense_bytes="
            f"{limit} ({limit / 1024**2:.2f} MiB). Use a smaller problem or "
            "explicitly increase max_dense_bytes. For full matrix/spectrum "
            "requests, a matrix-free spectrum(k=...) may avoid this allocation."
        )


@dataclass(frozen=True, slots=True)
class LaplacianDiagnostics:
    """Construction details; ranks concern real numerical constraints.

    ``constraint_rank`` is ``None`` if an uncoupled constraint component was
    skipped: its rank is irrelevant to the operator and was not calculated.
    Likewise, nullable constraint counts are ``None`` when a structural or
    empty-space shortcut avoids evaluating them, rather than implying zero.
    ``correction_rank`` is the sum of stored row-space dimensions, not a
    claim about the rank of their images in the old chain space.  Constraint
    components of full column rank are removed outright and need no stored
    correction.
    ``max_dense_bytes`` is a conservative per-operation numerical workspace
    guard, not a process-wide RSS limit or a limit on the input chain object.
    """

    backend: str
    start_cell_count: int
    end_cell_count: int
    upper_cell_count: int
    down_boundary_nnz: int
    up_boundary_nnz: int
    constraint_shape: tuple[int, int]
    constraint_nnz: int | None
    constraint_components: int | None
    coupled_constraint_components: int
    constraint_rank: int | None
    correction_rank: int
    correction_storage_bytes: int
    rank_rtol: float | None
    max_dense_bytes: int
    coefficient_field: str = "R"


@dataclass(frozen=True, slots=True)
class LaplacianSpectrum:
    """Ordered eigenvalues with explicit full/partial-spectrum semantics.

    Values within ``zero_cutoff`` of zero are reported as exactly zero.
    ``nullity`` and ``spectral_gap`` are certified only for a complete
    numerical spectrum.  For a complete empty/all-zero spectrum the gap is
    ``0.0``; for a partial spectrum both fields are ``None``.
    ``zero_eigenvalue_count`` always counts only the returned eigenvalues.
    """

    eigenvalues: tuple[float, ...]
    is_complete: bool
    nullity: int | None
    zero_eigenvalue_count: int
    spectral_gap: float | None
    zero_cutoff: float


@dataclass(frozen=True, slots=True)
class _Correction:
    rows: Any
    values: Any


@dataclass(frozen=True, slots=True)
class LaplacianResult:
    """A lazy real symmetric operator in the ordered ``basis_keys`` basis.

    ``start == end`` denotes an ordinary snapshot.  No full square matrix is
    allocated until explicitly requested.  Sparse boundaries and small
    component-local row-space corrections implement matrix-vector products.
    Materialized matrices are independent copies; modifying one does not
    modify this result or its construction engine.
    """

    dimension: int
    start: float
    end: float
    basis_keys: tuple[InteractionKey, ...]
    diagnostics: LaplacianDiagnostics
    _down: Any = field(repr=False)
    _up: Any = field(repr=False)
    _corrections: tuple[_Correction, ...] = field(repr=False)
    _diagonal: Any = field(repr=False)
    _max_dense_bytes: int = field(repr=False)

    @property
    def size(self) -> int:
        return len(self.basis_keys)

    @property
    def shape(self) -> tuple[int, int]:
        return self.size, self.size

    def _apply(self, vector: Any) -> Any:
        np, _, _ = _numerics()
        values = np.asarray(vector, dtype=np.float64)
        if self._diagonal is not None:
            diagonal = self._diagonal if values.ndim == 1 else self._diagonal[:, None]
            return diagonal * values
        output = np.asarray(
            self._down.T @ (self._down @ values)
            + self._up @ (self._up.T @ values)
        )
        for correction in self._corrections:
            local_values = values[correction.rows]
            output[correction.rows] -= correction.values @ (
                correction.values.T @ local_values
            )
        return output

    def as_linear_operator(self) -> Any:
        """Return a SciPy ``LinearOperator`` without square-matrix assembly.

        Both vector and block-vector multiplication are supported.  This is
        the preferred interface for large ``L1`` operators and partial
        eigensolvers; real symmetric adjoint multiplication is identical.
        """

        np, _, linalg = _numerics()
        return linalg.LinearOperator(
            self.shape,
            matvec=self._apply,
            rmatvec=self._apply,
            matmat=self._apply,
            dtype=np.dtype(np.float64),
        )

    def _sparse_workspace_bound(self) -> int:
        np, _, _ = _numerics()
        if self._diagonal is not None:
            return 32 * self.size
        # A Gram product contributes at most support_size**2 entries per
        # source vector.  Python ints avoid overflow on a large sparse input.
        lower_sizes = np.diff(self._down.tocsr().indptr)
        upper_sizes = np.diff(self._up.tocsc().indptr)
        entries = min(
            self.size**2,
            sum(int(value) ** 2 for value in lower_sizes)
            + sum(int(value) ** 2 for value in upper_sizes)
            + sum(len(part.rows) ** 2 for part in self._corrections),
        )
        local_square = max(
            (len(part.rows) ** 2 for part in self._corrections), default=0
        )
        return (
            64 * entries
            + 16 * local_square
            + self.diagnostics.correction_storage_bytes
            + 32 * (self.size + 1)
        )

    def to_sparse(self) -> Any:
        """Materialize an independent SciPy CSC matrix, subject to the budget.

        A persistent correction can be dense within its local row block.
        "Sparse output" is therefore not a guarantee of low memory usage.
        Gram-product support bounds are checked before multiplication.
        """

        np, sparse, _ = _numerics()
        _check_budget(
            self._sparse_workspace_bound(), self._max_dense_bytes,
            "Laplacian sparse materialization",
        )
        if self._diagonal is not None:
            if self.size == 0:
                return sparse.csc_matrix((0, 0), dtype=np.float64)
            return sparse.diags(self._diagonal.copy(), format="csc")
        matrix = (self._down.T @ self._down + self._up @ self._up.T).tocsc()
        for correction in self._corrections:
            local_gram = correction.values @ correction.values.T
            rows = np.repeat(correction.rows, len(correction.rows))
            columns = np.tile(correction.rows, len(correction.rows))
            correction_matrix = sparse.coo_matrix(
                (local_gram.ravel(), (rows, columns)), shape=self.shape
            ).tocsc()
            matrix = matrix - correction_matrix
        matrix = ((matrix + matrix.T) * 0.5).tocsc()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix

    def to_dense(self) -> Any:
        """Materialize an independent float64 array, respecting the budget."""

        np, _, _ = _numerics()
        _check_budget(
            self._sparse_workspace_bound() + 16 * self.size**2,
            self._max_dense_bytes, "Laplacian dense materialization",
        )
        if self._diagonal is not None:
            return np.diag(self._diagonal)
        return self.to_sparse().toarray()

    def spectrum(
        self,
        *,
        k: int | None = None,
        zero_atol: Real = 1e-10,
        zero_rtol: Real = 1e-10,
        tol: Real = 0.0,
    ) -> LaplacianSpectrum:
        """Compute all eigenvalues or the ``k`` smallest algebraic values.

        ``k=None`` (or ``k >= size``) requests a complete spectrum; general
        operators use a budget-guarded dense symmetric eigensolver.  Smaller
        positive ``k`` uses SciPy ARPACK on the matrix-free operator.  ARPACK
        convergence failures propagate rather than silently returning an
        incomplete result.  Diagonal interaction ``L0`` uses exact diagonal
        sorting, including for partial requests.

        ``tol`` is the non-negative ARPACK convergence tolerance (zero means
        machine precision).  Zero classification is separate:
        ``zero_atol + zero_rtol * max(abs(returned eigenvalues))``.  A negative
        eigenvalue beyond this tolerance raises ``RuntimeError``.  A partial
        spectrum does not certify the full nullity or spectral gap.
        """

        np, _, linalg = _numerics()
        if k is not None:
            k = _integer(k, "k", 1)
        atol = _nonnegative_real(zero_atol, "zero_atol")
        rtol = _nonnegative_real(zero_rtol, "zero_rtol")
        solver_tol = _nonnegative_real(tol, "tol")
        complete = k is None or k >= self.size
        if self.size == 0:
            eigenvalues = np.empty(0, dtype=np.float64)
        elif self._diagonal is not None:
            _check_budget(32 * self.size, self._max_dense_bytes, "Diagonal spectrum")
            eigenvalues = np.sort(self._diagonal)
            if not complete:
                eigenvalues = eigenvalues[:k]
        elif complete:
            _check_budget(
                self._sparse_workspace_bound() + 64 * self.size**2,
                self._max_dense_bytes, "Full Laplacian eigenspectrum",
            )
            eigenvalues = np.linalg.eigvalsh(self.to_dense())
        else:
            assert k is not None
            ncv = min(self.size, max(2 * k + 1, 20))
            _check_budget(
                8 * (4 * self.size * ncv + 8 * self.size + 4 * ncv**2)
                + self.diagnostics.correction_storage_bytes,
                self._max_dense_bytes, "Partial Laplacian eigenspectrum",
            )
            initial_vector = np.random.default_rng(0).standard_normal(self.size)
            eigenvalues = np.sort(linalg.eigsh(
                self.as_linear_operator(), k=k, which="SA", ncv=ncv,
                tol=solver_tol, v0=initial_vector, return_eigenvectors=False,
            ))
        spectral_scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        zero_cutoff = atol + rtol * spectral_scale
        if np.any(eigenvalues < -zero_cutoff):
            raise RuntimeError(
                "The computed Laplacian spectrum contains a negative "
                "eigenvalue beyond the requested zero tolerance; check "
                "numerical conditioning or use a less stringent tolerance."
            )
        cleaned_eigenvalues = tuple(0.0 if abs(float(x)) <= zero_cutoff else float(x) for x in eigenvalues)
        zero_count = sum(value == 0.0 for value in cleaned_eigenvalues)
        positive_eigenvalues = tuple(value for value in cleaned_eigenvalues if value > 0.0)
        return LaplacianSpectrum(
            eigenvalues=cleaned_eigenvalues,
            is_complete=complete,
            nullity=zero_count if complete else None,
            zero_eigenvalue_count=zero_count,
            spectral_gap=(min(positive_eigenvalues) if positive_eigenvalues else 0.0) if complete else None,
            zero_cutoff=zero_cutoff,
        )


def _constraint_components(matrix: Any) -> tuple[tuple[list[int], list[int]], ...]:
    """Bipartite row/column components without forming a square adjacency."""

    parent: dict[int, int] = {}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while value != root:
            parent_node = parent[value]
            parent[value] = root
            value = parent_node
        return root

    for column in range(matrix.shape[1]):
        rows = matrix.indices[matrix.indptr[column]:matrix.indptr[column + 1]]
        if len(rows) == 0:
            continue
        first_row = int(rows[0])
        parent.setdefault(first_row, first_row)
        root = find(first_row)
        for item in rows[1:]:
            row = int(item)
            parent.setdefault(row, row)
            other_root = find(row)
            if root != other_root:
                parent[other_root] = root
    row_groups: dict[int, list[int]] = {}
    column_groups: dict[int, list[int]] = {}
    for row in parent:
        root = find(row)
        row_groups.setdefault(root, []).append(row)
    for column in range(matrix.shape[1]):
        column_start, column_end = matrix.indptr[column:column + 2]
        if column_start != column_end:
            root = find(int(matrix.indices[column_start]))
            column_groups.setdefault(root, []).append(column)
    return tuple(
        (sorted(rows), column_groups[root]) for root, rows in row_groups.items()
    )


class InteractionLaplacianEngine:
    """Reusable numerical workspace for a validated interaction chain.

    The input must have been built through one degree above every requested
    Laplacian.  Signed full CSC boundaries are assembled lazily and cached;
    filtration snapshots are prefix slices because each degree's births are
    sorted.  The engine does not copy the input factor/interaction indexes.

    ``rank_rtol=None`` uses the usual local SVD cutoff
    ``eps * max(component.shape) * largest_singular_value``.  A supplied
    relative tolerance in ``[0, 1)`` replaces the relative factor, with the
    machine-precision floor retained.  Rank decisions are over the reals,
    not ``GF(2)``.  ``max_dense_bytes`` bounds estimated numerical workspace
    before expensive allocations; it does not bound process-wide memory or
    the supplied chain and cached sparse boundary storage.
    """

    def __init__(
        self,
        chain: FilteredInteractionChainComplex,
        *,
        rank_rtol: Real | None = None,
        max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> None:
        if not isinstance(chain, FilteredInteractionChainComplex):
            raise TypeError("chain must be a FilteredInteractionChainComplex")
        self.chain = chain
        self.rank_rtol = (
            None if rank_rtol is None else _nonnegative_real(rank_rtol, "rank_rtol")
        )
        if self.rank_rtol is not None and self.rank_rtol >= 1.0:
            raise ValueError("rank_rtol must be less than 1")
        self.max_dense_bytes = _integer(max_dense_bytes, "max_dense_bytes", 1)
        self._boundaries: dict[int, Any] = {}
        _numerics()

    def _degree(self, degree: int, *, boundary: bool = False) -> int:
        value = _integer(degree, "degree" if boundary else "dimension")
        maximum = (
            self.chain.max_cell_degree if boundary
            else self.chain.max_homology_dimension
        )
        if value > maximum:
            raise ValueError(
                f"Requested degree {value} exceeds the constructed limit "
                f"{maximum}; build the interaction chain through the "
                "requested Laplacian dimension plus one."
            )
        return value

    def _count(self, degree: int, threshold: float) -> int:
        if degree < 0:
            return 0
        return bisect_right(self.chain.degrees[degree].births, threshold)

    def _boundary(self, degree: int) -> Any:
        np, sparse, _ = _numerics()
        cached = self._boundaries.get(degree)
        if cached is not None:
            return cached
        source_count = self.chain.number_of_cells(degree)
        if degree == 0:
            matrix = sparse.csc_matrix((0, source_count), dtype=np.float64)
        else:
            row_indices: list[int] = []
            coefficients: list[int] = []
            column_pointers = [0]
            for cell_index in range(source_count):
                for row, coefficient in self.chain.signed_boundary_entries(degree, cell_index):
                    row_indices.append(row)
                    coefficients.append(coefficient)
                column_pointers.append(len(row_indices))
            matrix = sparse.csc_matrix(
                (np.asarray(coefficients, dtype=np.float64),
                 np.asarray(row_indices, dtype=np.int64),
                 np.asarray(column_pointers, dtype=np.int64)),
                shape=(self.chain.number_of_cells(degree - 1), source_count),
            )
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
        self._boundaries[degree] = matrix
        return matrix

    def boundary_matrix(self, degree: int, *, filtration: Real = inf) -> Any:
        """Return an independent signed float64 CSC boundary at a snapshot.

        Columns and rows follow the corresponding degree's interaction-key
        prefixes at ``filtration``.  ``B0`` has shape ``(0, n0)``.  Empty
        quotient facets are omitted; surviving facets retain tensor/Koszul
        orientation signs.  Infinite thresholds are permitted, but NaN and
        boolean thresholds are not.
        """

        dimension = self._degree(degree, boundary=True)
        value = _threshold(filtration, "filtration")
        row_count = self._count(dimension - 1, value)
        column_count = self._count(dimension, value)
        return self._boundary(dimension)[:row_count, :column_count].tocsc(copy=True)

    def laplacian(self, dimension: int, *, filtration: Real = inf) -> LaplacianResult:
        """Build the ordinary ``L_q = B_q.T B_q + B_(q+1) B_(q+1).T``."""

        value = _threshold(filtration, "filtration")
        return self.persistent_laplacian(dimension, start=value, end=value)

    def _diagonal_l0(self, count: int, end: float) -> Any:
        np, _, _ = _numerics()
        _check_budget(16 * count, self.max_dense_bytes, "Interaction L0 diagonal")
        diagonal = np.zeros(count, dtype=np.float64)
        first_factor = self.chain.factors[0]
        for row, key in enumerate(self.chain.degrees[0].keys[:count]):
            vertex = first_factor.simplices[key[0]][0]
            diagonal[row] = sum(
                factor.births[edge] <= end
                for factor in self.chain.factors
                for edge in factor.postings(1, vertex)
            )
        diagonal.flags.writeable = False
        return diagonal

    def persistent_laplacian(
        self, dimension: int, *, start: Real, end: Real,
    ) -> LaplacianResult:
        """Build the genuine ``(start,end)`` persistent operator on old cells.

        The upper domain consists of end-time chains whose entire signed
        boundary lies in the start-time chain space.  New boundary rows are
        constraints on *linear combinations*, not a reason to discard their
        individual source columns.  The inherited orthonormal product-basis
        metric is retained by component-wise SVD row-space projection.

        For at least two factors, ``L0`` is exactly diagonal: every degree-1
        product has one surviving boundary vertex.  This path counts incident
        factor edges directly, without assembling ``B1`` or any constraints.
        """

        np, _, _ = _numerics()
        degree = self._degree(dimension)
        start_value = _threshold(start, "start")
        end_value = _threshold(end, "end")
        if start_value > end_value:
            raise ValueError("start must not exceed end")
        start_count = self._count(degree, start_value)
        end_count = self._count(degree, end_value)
        upper_count = self._count(degree + 1, end_value)
        basis_keys = self.chain.degrees[degree].keys[:start_count]

        if start_count == 0 or (degree == 0 and len(self.chain.factors) >= 2):
            diagonal = (
                np.zeros(0, dtype=np.float64) if start_count == 0
                else self._diagonal_l0(start_count, end_value)
            )
            diagonal.flags.writeable = False
            diagnostics = LaplacianDiagnostics(
                backend="empty" if start_count == 0 else "interaction_l0_diagonal",
                start_cell_count=start_count, end_cell_count=end_count,
                upper_cell_count=upper_count, down_boundary_nnz=0,
                up_boundary_nnz=int(diagonal.sum()),
                constraint_shape=(end_count - start_count, upper_count),
                constraint_nnz=(
                    upper_count - int(diagonal.sum())
                    if degree == 0 and len(self.chain.factors) >= 2 else None
                ),
                constraint_components=None,
                coupled_constraint_components=0, constraint_rank=None,
                correction_rank=0, correction_storage_bytes=0,
                rank_rtol=self.rank_rtol, max_dense_bytes=self.max_dense_bytes,
            )
            return LaplacianResult(
                degree, start_value, end_value, basis_keys, diagnostics,
                None, None, (), diagonal, self.max_dense_bytes,
            )

        lower_count = self._count(degree - 1, start_value)
        down = self._boundary(degree)[:lower_count, :start_count].tocsc(copy=True)
        upper_end = self._boundary(degree + 1)[:end_count, :upper_count]
        up = upper_end[:start_count, :].tocsc(copy=True)
        constraints = upper_end[start_count:, :].tocsc(copy=True)
        components = _constraint_components(constraints) if constraints.nnz else ()
        corrections: list[_Correction] = []
        eliminated_columns: list[int] = []
        cumulative_bytes = 0
        rank_sum = 0
        coupled_count = 0
        skipped_component = False
        used_svd = False
        for rows, columns in components:
            local_up = up[:, columns].tocsc()
            affected_rows = np.unique(local_up.indices)
            if affected_rows.size == 0:
                skipped_component = True
                continue
            coupled_count += 1
            local_up = local_up[affected_rows, :].tocsc()
            local_constraints = constraints[:, columns][rows, :].tocsc()
            row_count, column_count = local_constraints.shape
            svd_size = min(row_count, column_count)
            if column_count == 1:
                # A nonzero single-column constraint kills that source
                # coordinate exactly.  Removing it avoids subtracting two
                # identical Gram matrices and makes zero operators exact.
                eliminated_columns.extend(columns)
                rank_sum += 1
                continue
            if row_count == 1:
                # Exact row-space normalization avoids an SVD and a dense
                # vector whose length could be the whole upper chain space.
                _check_budget(
                    cumulative_bytes + 32 * len(affected_rows),
                    self.max_dense_bytes, "Single-row constraint projection",
                )
                norm = float(np.linalg.norm(local_constraints.data))
                correction_values = (local_up @ local_constraints.T).toarray() / norm
                rank = 1
            else:
                # Thin SVD stores at most min(r,c)*c row-space coefficients.
                # Account for the input, LAPACK copies/U/V/workspace and the
                # projected correction before requesting any dense array.
                estimated_bytes = 8 * (
                    4 * row_count * column_count
                    + 3 * svd_size * column_count
                    + 3 * row_count * svd_size
                    + 4 * svd_size**2
                    + 3 * len(affected_rows) * svd_size
                )
                _check_budget(
                    cumulative_bytes + estimated_bytes, self.max_dense_bytes,
                    "Persistent constraint-component SVD",
                )
                dense_constraints = local_constraints.toarray()
                left_vectors, singular_values, right_vectors_t = np.linalg.svd(dense_constraints, full_matrices=False)
                rank_floor = np.finfo(np.float64).eps * max(dense_constraints.shape)
                relative_cutoff = rank_floor if self.rank_rtol is None else max(rank_floor, self.rank_rtol)
                cutoff = relative_cutoff * float(singular_values[0]) if singular_values.size else 0.0
                rank = int(np.count_nonzero(singular_values > cutoff))
                used_svd = True
                if rank == column_count:
                    eliminated_columns.extend(columns)
                    rank_sum += rank
                    del dense_constraints, left_vectors, singular_values, right_vectors_t
                    continue
                correction_values = np.asarray(local_up @ right_vectors_t[:rank].T)
                del dense_constraints, left_vectors, singular_values, right_vectors_t
            rank_sum += rank
            correction_values = np.ascontiguousarray(correction_values, dtype=np.float64)
            affected_rows = np.asarray(affected_rows, dtype=np.int64)
            cumulative_bytes += correction_values.nbytes + affected_rows.nbytes
            _check_budget(
                cumulative_bytes, self.max_dense_bytes,
                "Accumulated persistent low-rank corrections",
            )
            correction_values.flags.writeable = False
            affected_rows.flags.writeable = False
            corrections.append(_Correction(affected_rows, correction_values))

        original_up_nnz = int(up.nnz)
        if eliminated_columns:
            keep_columns = np.ones(up.shape[1], dtype=bool)
            keep_columns[eliminated_columns] = False
            up = up[:, keep_columns].tocsc()
        diagonal = None
        if down.nnz == 0 and up.nnz == 0:
            _check_budget(
                16 * start_count, self.max_dense_bytes, "Zero-operator diagonal",
            )
            diagonal = np.zeros(start_count, dtype=np.float64)
            diagonal.flags.writeable = False
        backend = (
            "zero_operator" if diagonal is not None else
            "component_svd_projection" if used_svd else
            "component_structural_projection" if corrections else
            "full_rank_constraint_elimination" if eliminated_columns else
            "sparse_boundary_gram"
        )
        diagnostics = LaplacianDiagnostics(
            backend=backend, start_cell_count=start_count,
            end_cell_count=end_count, upper_cell_count=upper_count,
            down_boundary_nnz=int(down.nnz), up_boundary_nnz=original_up_nnz,
            constraint_shape=constraints.shape,
            constraint_nnz=int(constraints.nnz),
            constraint_components=len(components),
            coupled_constraint_components=coupled_count,
            constraint_rank=None if skipped_component else rank_sum,
            correction_rank=sum(part.values.shape[1] for part in corrections),
            correction_storage_bytes=cumulative_bytes,
            rank_rtol=self.rank_rtol, max_dense_bytes=self.max_dense_bytes,
        )
        return LaplacianResult(
            degree, start_value, end_value, basis_keys, diagnostics,
            down, up, tuple(corrections), diagonal, self.max_dense_bytes,
        )


def compute_interaction_laplacian(
    chain: FilteredInteractionChainComplex,
    *,
    dimension: int,
    filtration: Real = inf,
    rank_rtol: Real | None = None,
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> LaplacianResult:
    """Convenience ordinary Laplacian; reuse an engine for many snapshots."""

    return InteractionLaplacianEngine(
        chain, rank_rtol=rank_rtol, max_dense_bytes=max_dense_bytes,
    ).laplacian(dimension, filtration=filtration)


def compute_persistent_interaction_laplacian(
    chain: FilteredInteractionChainComplex,
    *,
    dimension: int,
    start: Real,
    end: Real,
    rank_rtol: Real | None = None,
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
) -> LaplacianResult:
    """Convenience genuine pairwise persistent interaction Laplacian."""

    return InteractionLaplacianEngine(
        chain, rank_rtol=rank_rtol, max_dense_bytes=max_dense_bytes,
    ).persistent_laplacian(dimension, start=start, end=end)


def signed_boundary_matrix(
    chain: FilteredInteractionChainComplex,
    degree: int,
    *,
    filtration: Real = inf,
) -> Any:
    """Return the signed real CSC interaction boundary at ``filtration``.

    Unlike a GF(2) support matrix this includes both ordinary simplex signs
    and tensor-product Koszul signs.  Degree zero has no augmented boundary.
    """

    return InteractionLaplacianEngine(chain).boundary_matrix(
        degree, filtration=filtration,
    )


__all__ = [
    "DEFAULT_MAX_DENSE_BYTES",
    "InteractionLaplacianEngine",
    "LaplacianDiagnostics",
    "LaplacianResult",
    "LaplacianSpectrum",
    "compute_interaction_laplacian",
    "compute_persistent_interaction_laplacian",
    "signed_boundary_matrix",
]
