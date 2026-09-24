"""Ordinary and persistent hyperdigraph Laplacians over the reals.

The module implements the sequence-hyperdigraph construction of Chen, Liu,
Wu, and Wei.  Explicit directed hyperedges form orthonormal ambient bases.  A
real orthonormal basis is then constructed for

``Omega_p = {x in F_p : boundary(x) is in F_(p-1)}``.

Two persistent operations are intentionally kept distinct:

``compute_laplacian_filtration``
    evaluates the ordinary Laplacian independently at selected snapshots;

``compute_persistent_laplacian``
    constructs the genuine two-parameter operator for one pair ``a <= b``.

Low dimensions avoid a general null-space decomposition whenever possible.
``L_0`` uses the signed directed-incidence matrix when all endpoint faces are
present.  ``L_1`` uses a structural QR basis when every explicit 2-hyperedge
has at most one missing edge face.  Missing-face cases outside this certified
regime, and non-face-closed dimensions above two, use an SVD null space.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import Literal, cast

import numpy as np

from .model import (
    DEFINITION_ID,
    FilteredSequenceHyperdigraph,
    Hyperedge,
    SequenceHyperdigraph,
)
from .results import (
    LaplacianDiagnostics,
    LaplacianDimensionResult,
    LaplacianFiltrationResult,
    LaplacianResult,
    LaplacianSnapshot,
    PersistentLaplacianResult,
    RealChainDimensionDiagnostics,
    RealMatrix,
)


RealOmegaBackend = Literal["auto", "svd"]
_DEFAULT_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class _PreparedRealBoundary:
    present: np.ndarray
    missing: np.ndarray
    missing_face_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RealChainComplex:
    hyperedges: tuple[tuple[Hyperedge, ...], ...]
    omega_bases: tuple[np.ndarray, ...]
    boundaries: tuple[np.ndarray, ...]
    diagnostics: tuple[RealChainDimensionDiagnostics, ...]
    ignored_hyperedges_above: int


def _validate_max_dimension(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_dimension must be a non-negative integer")
    if value < 0:
        raise ValueError("max_dimension must be non-negative")


def _validate_backend(value: str) -> RealOmegaBackend:
    if value not in {"auto", "svd"}:
        raise ValueError("omega_backend must be 'auto' or 'svd'")
    return cast(RealOmegaBackend, value)


def _validate_tolerance(value: Real | None) -> float:
    if value is None:
        return _DEFAULT_TOLERANCE
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("tolerance must be a positive finite real number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError("tolerance must be a positive finite real number")
    return result


def _validate_return_matrices(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError("return_matrices must be a boolean")


def _finite_threshold(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _matrix_rank(matrix: np.ndarray, tolerance: float) -> int:
    if matrix.size == 0 or min(matrix.shape, default=0) == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    scale = max(float(singular_values[0]), 1.0)
    numerical_floor = (
        np.finfo(np.float64).eps
        * max(matrix.shape)
        * max(float(singular_values[0]), 1.0)
    )
    cutoff = max(tolerance * scale, numerical_floor)
    return int(np.count_nonzero(singular_values > cutoff))


def _orthonormal_nullspace(
    matrix: np.ndarray, column_count: int, tolerance: float
) -> np.ndarray:
    """Return an orthonormal basis of the right null space."""

    if column_count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if matrix.shape[0] == 0:
        return np.eye(column_count, dtype=np.float64)
    if matrix.shape[1] != column_count:
        raise RuntimeError("null-space matrix has an inconsistent column count")
    if np.linalg.norm(matrix, ord="fro") <= tolerance:
        return np.eye(column_count, dtype=np.float64)

    _left, singular_values, right_transpose = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size:
        scale = max(float(singular_values[0]), 1.0)
        numerical_floor = (
            np.finfo(np.float64).eps
            * max(matrix.shape)
            * max(float(singular_values[0]), 1.0)
        )
        cutoff = max(tolerance * scale, numerical_floor)
        rank = int(np.count_nonzero(singular_values > cutoff))
    else:
        rank = 0
    return np.ascontiguousarray(right_transpose[rank:, :].T)


def _prepare_real_boundary(
    dimension: int,
    hyperedges: tuple[tuple[Hyperedge, ...], ...],
) -> _PreparedRealBoundary:
    source = hyperedges[dimension]
    if dimension == 0:
        return _PreparedRealBoundary(
            present=np.zeros((0, len(source)), dtype=np.float64),
            missing=np.zeros((0, len(source)), dtype=np.float64),
            missing_face_counts=tuple(0 for _ in source),
        )

    lower_index = {
        hyperedge: index for index, hyperedge in enumerate(hyperedges[dimension - 1])
    }
    missing_index: dict[Hyperedge, int] = {}
    present_entries: list[tuple[int, int, float]] = []
    missing_entries: list[tuple[int, int, float]] = []
    missing_counts: list[int] = []

    for column, hyperedge in enumerate(source):
        count = 0
        for removed in range(len(hyperedge)):
            face = hyperedge[:removed] + hyperedge[removed + 1 :]
            coefficient = -1.0 if removed % 2 else 1.0
            row = lower_index.get(face)
            if row is None:
                missing_row = missing_index.setdefault(face, len(missing_index))
                missing_entries.append((missing_row, column, coefficient))
                count += 1
            else:
                present_entries.append((row, column, coefficient))
        missing_counts.append(count)

    present = np.zeros((len(hyperedges[dimension - 1]), len(source)), dtype=np.float64)
    missing = np.zeros((len(missing_index), len(source)), dtype=np.float64)
    for row, column, coefficient in present_entries:
        present[row, column] = coefficient
    for row, column, coefficient in missing_entries:
        missing[row, column] = coefficient
    return _PreparedRealBoundary(present, missing, tuple(missing_counts))


def _single_missing_face_basis(
    prepared: _PreparedRealBoundary,
) -> np.ndarray:
    """Construct a signed kernel basis when each column has <=1 constraint."""

    source_count = prepared.missing.shape[1]
    unconstrained_columns = [
        column
        for column, count in enumerate(prepared.missing_face_counts)
        if count == 0
    ]
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for column, count in enumerate(prepared.missing_face_counts):
        if count == 0:
            continue
        rows = np.flatnonzero(prepared.missing[:, column])
        if len(rows) != 1:
            raise RuntimeError("single-missing-face basis received a general column")
        row = int(rows[0])
        groups[row].append((column, float(prepared.missing[row, column])))

    basis_count = len(unconstrained_columns) + sum(
        max(0, len(entries) - 1) for entries in groups.values()
    )
    if basis_count == 0:
        return np.zeros((source_count, 0), dtype=np.float64)

    candidates = np.zeros((source_count, basis_count), dtype=np.float64)
    basis_column = 0
    for column in unconstrained_columns:
        candidates[column, basis_column] = 1.0
        basis_column += 1
    for entries in groups.values():
        entries.sort(key=lambda item: item[0])
        pivot_column, pivot_coefficient = entries[0]
        for column, coefficient in entries[1:]:
            # c_p*(-c_j) + c_j*c_p = 0, including the orientation signs.
            candidates[pivot_column, basis_column] = -coefficient
            candidates[column, basis_column] = pivot_coefficient
            basis_column += 1
    if basis_column != basis_count:
        raise RuntimeError("single-missing-face basis has the wrong dimension")
    orthonormal, _triangular = np.linalg.qr(candidates, mode="reduced")
    return np.ascontiguousarray(orthonormal)


def _construct_omega_basis(
    dimension: int,
    prepared: _PreparedRealBoundary,
    backend: RealOmegaBackend,
    tolerance: float,
) -> tuple[np.ndarray, str]:
    source_count = prepared.missing.shape[1]
    if dimension == 0:
        return np.eye(source_count, dtype=np.float64), "unit_f0"
    if prepared.missing.shape[0] == 0:
        name = "signed_incidence_units" if dimension == 1 else "face_closed_units"
        return np.eye(source_count, dtype=np.float64), name

    maximum_missing = max(prepared.missing_face_counts, default=0)
    if backend == "auto" and dimension <= 2 and maximum_missing <= 1:
        basis = _single_missing_face_basis(prepared)
        selected_backend = "single_missing_face_structural_qr"
    else:
        basis = _orthonormal_nullspace(prepared.missing, source_count, tolerance)
        selected_backend = "svd_constraint_nullspace"

    if basis.shape[1]:
        gram_error = np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord="fro")
        if gram_error > 100.0 * tolerance * max(1, basis.shape[1]):
            raise RuntimeError("Omega basis is not numerically orthonormal")
    residual = prepared.missing @ basis
    if np.linalg.norm(residual, ord="fro") > 100.0 * tolerance * max(
        1.0, np.linalg.norm(prepared.missing, ord="fro")
    ):
        raise RuntimeError("Omega basis does not satisfy the missing-face constraints")
    return basis, selected_backend


def _build_real_chain_complex(
    hyperdigraph: SequenceHyperdigraph,
    max_dimension: int,
    *,
    omega_backend: RealOmegaBackend,
    tolerance: float,
) -> _RealChainComplex:
    through_dimension = max_dimension + 1
    hyperedges = tuple(
        hyperdigraph.directed_hyperedges(dimension)
        for dimension in range(through_dimension + 1)
    )
    prepared = tuple(
        _prepare_real_boundary(dimension, hyperedges)
        for dimension in range(through_dimension + 1)
    )

    omega_bases_list: list[np.ndarray] = []
    diagnostics: list[RealChainDimensionDiagnostics] = []
    for dimension, prepared_boundary in enumerate(prepared):
        basis, selected_backend = _construct_omega_basis(
            dimension, prepared_boundary, omega_backend, tolerance
        )
        omega_bases_list.append(basis)
        diagnostics.append(
            RealChainDimensionDiagnostics(
                dimension=dimension,
                hyperedge_count=len(hyperedges[dimension]),
                omega_dimension=basis.shape[1],
                constraint_row_count=prepared_boundary.missing.shape[0],
                constraint_rank=_matrix_rank(prepared_boundary.missing, tolerance),
                max_missing_face_count=max(prepared_boundary.missing_face_counts, default=0),
                omega_backend=selected_backend,
            )
        )
    omega_bases = tuple(omega_bases_list)

    boundaries: list[np.ndarray] = [
        np.zeros((0, omega_bases[0].shape[1]), dtype=np.float64)
    ]
    for dimension in range(1, through_dimension + 1):
        boundary = (
            omega_bases[dimension - 1].T
            @ prepared[dimension].present
            @ omega_bases[dimension]
        )
        boundary = np.ascontiguousarray(boundary)
        ambient_boundary = prepared[dimension].present @ omega_bases[dimension]
        projection_error = ambient_boundary - omega_bases[dimension - 1] @ boundary
        if np.linalg.norm(projection_error, ord="fro") > 100.0 * tolerance * max(
            1.0, np.linalg.norm(ambient_boundary, ord="fro")
        ):
            raise RuntimeError("a boundary left the lower embedded chain space")
        boundaries.append(boundary)

    for dimension in range(2, len(boundaries)):
        composed = boundaries[dimension - 1] @ boundaries[dimension]
        scale = max(
            1.0,
            np.linalg.norm(boundaries[dimension - 1], ord="fro")
            * np.linalg.norm(boundaries[dimension], ord="fro"),
        )
        if np.linalg.norm(composed, ord="fro") > 100.0 * tolerance * scale:
            raise RuntimeError("constructed real boundaries do not satisfy d^2 = 0")

    ignored = sum(
        hyperdigraph.number_of_hyperedges(dimension)
        for dimension in hyperdigraph.dimensions
        if dimension > through_dimension
    )
    return _RealChainComplex(
        hyperedges=hyperedges,
        omega_bases=omega_bases,
        boundaries=tuple(boundaries),
        diagnostics=tuple(diagnostics),
        ignored_hyperedges_above=ignored,
    )


def _immutable_matrix(matrix: np.ndarray) -> RealMatrix:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _laplacian_dimension_result(
    dimension: int,
    down_boundary: np.ndarray,
    up_boundary: np.ndarray,
    tolerance: float,
    return_matrix: bool,
) -> LaplacianDimensionResult:
    laplacian = down_boundary.T @ down_boundary + up_boundary @ up_boundary.T
    laplacian = np.ascontiguousarray((laplacian + laplacian.T) * 0.5)
    if laplacian.shape[0] == 0:
        eigenvalues_array = np.zeros(0, dtype=np.float64)
    else:
        eigenvalues_array = np.linalg.eigvalsh(laplacian)
    scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues_array))) if eigenvalues_array.size else 0.0,
    )
    cutoff = tolerance * scale
    cleaned: list[float] = []
    for raw in eigenvalues_array:
        value = float(raw)
        if abs(value) <= cutoff:
            cleaned.append(0.0)
        elif value < 0.0:
            if value < -100.0 * cutoff:
                raise RuntimeError("a computed Laplacian has a negative eigenvalue")
            cleaned.append(0.0)
        else:
            cleaned.append(value)
    eigenvalues = tuple(cleaned)
    positives = tuple(value for value in eigenvalues if value > 0.0)
    return LaplacianDimensionResult(
        dimension=dimension,
        eigenvalues=eigenvalues,
        nullity=sum(value == 0.0 for value in eigenvalues),
        omega_dimension=laplacian.shape[0],
        down_rank=_matrix_rank(down_boundary, tolerance),
        up_rank=_matrix_rank(up_boundary, tolerance),
        up_domain_dimension=up_boundary.shape[1],
        smallest_positive_eigenvalue=min(positives) if positives else None,
        matrix=_immutable_matrix(laplacian) if return_matrix else None,
    )


def _low_dimensional_backend(chain: _RealChainComplex, max_dimension: int) -> str:
    parts: list[str] = []
    if len(chain.diagnostics) > 1:
        parts.append(
            "signed_incidence_L0"
            if chain.diagnostics[1].omega_backend == "signed_incidence_units"
            else "embedded_boundary_L0"
        )
    if max_dimension >= 1 and len(chain.diagnostics) > 2:
        omega2 = chain.diagnostics[2].omega_backend
        if omega2 == "single_missing_face_structural_qr":
            parts.append("single_missing_face_L1")
        elif omega2 == "face_closed_units":
            parts.append("face_closed_L1")
        else:
            parts.append("general_nullspace_L1")
    return "+".join(parts) if parts else "empty_low_dimensional_complex"


def _higher_dimensional_backend(chain: _RealChainComplex, max_dimension: int) -> str:
    if max_dimension < 2:
        return "not_requested"
    backends = {
        diagnostic.omega_backend
        for diagnostic in chain.diagnostics[3:]
        if diagnostic.constraint_row_count
    }
    return "svd_normal_way" if backends else "face_closed_or_empty_normal_way"


def compute_laplacian(
    hyperdigraph: SequenceHyperdigraph,
    max_dimension: int | None = None,
    *,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> LaplacianResult:
    """Compute ordinary hyperdigraph Laplacians at one snapshot.

    Matrices are represented in orthonormal bases of the real ``Omega``
    spaces.  Their nullities are therefore the real Betti numbers.
    """

    if not isinstance(hyperdigraph, SequenceHyperdigraph):
        raise TypeError("hyperdigraph must be a Hyperdigraph")
    if max_dimension is None:
        max_dimension = max(0, hyperdigraph.max_dimension)
    _validate_max_dimension(max_dimension)
    selected_backend = _validate_backend(omega_backend)
    numerical_tolerance = _validate_tolerance(tolerance)
    _validate_return_matrices(return_matrices)

    chain = _build_real_chain_complex(
        hyperdigraph,
        max_dimension,
        omega_backend=selected_backend,
        tolerance=numerical_tolerance,
    )
    dimensions = tuple(
        _laplacian_dimension_result(
            dimension,
            chain.boundaries[dimension],
            chain.boundaries[dimension + 1],
            numerical_tolerance,
            return_matrices,
        )
        for dimension in range(max_dimension + 1)
    )
    return LaplacianResult(
        dimensions=dimensions,
        max_dimension=max_dimension,
        diagnostics=LaplacianDiagnostics(
            definition_id=DEFINITION_ID,
            coefficient_field="R",
            tolerance=numerical_tolerance,
            chain_dimensions=chain.diagnostics,
            low_dimensional_backend=_low_dimensional_backend(chain, max_dimension),
            higher_dimensional_backend=_higher_dimensional_backend(
                chain, max_dimension
            ),
            ignored_hyperedges_above=chain.ignored_hyperedges_above,
            notes=(
                "Matrices use orthonormal Omega coordinates and the inherited "
                "explicit-hyperedge inner product.",
                "Nullities are Betti numbers over R; field-sensitive results may "
                "differ from the package's GF(2) homology module.",
            ),
        ),
    )


def _embedded_inclusion(
    start_chain: _RealChainComplex,
    end_chain: _RealChainComplex,
    dimension: int,
    tolerance: float,
) -> np.ndarray:
    start_edges = start_chain.hyperedges[dimension]
    end_index = {
        edge: index for index, edge in enumerate(end_chain.hyperedges[dimension])
    }
    start_basis = start_chain.omega_bases[dimension]
    embedded = np.zeros(
        (len(end_chain.hyperedges[dimension]), start_basis.shape[1]),
        dtype=np.float64,
    )
    for row, edge in enumerate(start_edges):
        try:
            target = end_index[edge]
        except KeyError as exc:
            raise RuntimeError("filtration snapshots are not nested") from exc
        embedded[target, :] = start_basis[row, :]

    end_basis = end_chain.omega_bases[dimension]
    inclusion = end_basis.T @ embedded
    residual = embedded - end_basis @ inclusion
    if np.linalg.norm(residual, ord="fro") > 100.0 * tolerance * max(
        1.0, np.linalg.norm(embedded, ord="fro")
    ):
        raise RuntimeError("the start Omega space did not embed into the end space")
    if inclusion.shape[1]:
        gram_error = np.linalg.norm(
            inclusion.T @ inclusion - np.eye(inclusion.shape[1]), ord="fro"
        )
        if gram_error > 100.0 * tolerance * max(1, inclusion.shape[1]):
            raise RuntimeError("persistent inclusion is not numerically isometric")
    return np.ascontiguousarray(inclusion)


def _persistent_up_boundary(
    start_chain: _RealChainComplex,
    end_chain: _RealChainComplex,
    dimension: int,
    tolerance: float,
) -> np.ndarray:
    inclusion = _embedded_inclusion(start_chain, end_chain, dimension, tolerance)
    end_boundary = end_chain.boundaries[dimension + 1]
    projected = inclusion @ (inclusion.T @ end_boundary)
    outside_start = end_boundary - projected
    pair_domain = _orthonormal_nullspace(
        outside_start, end_boundary.shape[1], tolerance
    )
    persistent_boundary = inclusion.T @ end_boundary @ pair_domain
    return np.ascontiguousarray(persistent_boundary)


def compute_persistent_laplacian(
    filtration: FilteredSequenceHyperdigraph,
    start: Real,
    end: Real,
    max_dimension: int = 1,
    *,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> PersistentLaplacianResult:
    """Compute the genuine persistent Laplacians for one pair ``start <= end``.

    In dimension ``p``, the up-domain is

    ``{x in Omega_(p+1)(end) : d x is in Omega_p(start)}``.

    The nullity is the rank of the persistent homology map from ``start`` to
    ``end`` over the reals.
    """

    if not isinstance(filtration, FilteredSequenceHyperdigraph):
        raise TypeError("filtration must be a FilteredHyperdigraph")
    _validate_max_dimension(max_dimension)
    start_value = _finite_threshold(start, "start")
    end_value = _finite_threshold(end, "end")
    if start_value > end_value:
        raise ValueError("start must not exceed end")
    selected_backend = _validate_backend(omega_backend)
    numerical_tolerance = _validate_tolerance(tolerance)
    _validate_return_matrices(return_matrices)

    start_chain = _build_real_chain_complex(
        filtration.snapshot(start_value),
        max_dimension,
        omega_backend=selected_backend,
        tolerance=numerical_tolerance,
    )
    end_chain = _build_real_chain_complex(
        filtration.snapshot(end_value),
        max_dimension,
        omega_backend=selected_backend,
        tolerance=numerical_tolerance,
    )

    dimensions: list[LaplacianDimensionResult] = []
    for dimension in range(max_dimension + 1):
        up_boundary = _persistent_up_boundary(
            start_chain, end_chain, dimension, numerical_tolerance
        )
        dimensions.append(
            _laplacian_dimension_result(
                dimension,
                start_chain.boundaries[dimension],
                up_boundary,
                numerical_tolerance,
                return_matrices,
            )
        )

    ignored = sum(
        len(filtration.weighted_hyperedges(dimension))
        for dimension in filtration.dimensions
        if dimension > max_dimension + 1
    )
    return PersistentLaplacianResult(
        start=start_value,
        end=end_value,
        dimensions=tuple(dimensions),
        max_dimension=max_dimension,
        diagnostics=LaplacianDiagnostics(
            definition_id=DEFINITION_ID,
            coefficient_field="R",
            tolerance=numerical_tolerance,
            chain_dimensions=start_chain.diagnostics,
            end_chain_dimensions=end_chain.diagnostics,
            low_dimensional_backend=(
                "pairwise_restriction+"
                + _low_dimensional_backend(end_chain, max_dimension)
            ),
            higher_dimensional_backend=_higher_dimensional_backend(
                end_chain, max_dimension
            ),
            ignored_hyperedges_above=ignored,
            notes=(
                "This is the pairwise persistent operator, not an ordinary "
                "snapshot spectrum.",
                "Its kernel dimension is the persistent Betti number over R.",
            ),
        ),
    )


def _selected_thresholds(
    filtration: FilteredSequenceHyperdigraph,
    max_dimension: int,
    thresholds: Iterable[Real] | None,
) -> tuple[float, ...]:
    if thresholds is None:
        return filtration.thresholds(max_dimension + 1)
    if isinstance(thresholds, (str, bytes, bytearray)):
        raise TypeError("thresholds must be an iterable of finite real numbers")
    return tuple(
        sorted({_finite_threshold(value, "each threshold") for value in thresholds})
    )


def compute_laplacian_filtration(
    filtration: FilteredSequenceHyperdigraph,
    max_dimension: int = 1,
    *,
    thresholds: Iterable[Real] | None = None,
    tolerance: Real | None = None,
    return_matrices: bool = False,
    omega_backend: RealOmegaBackend = "auto",
) -> LaplacianFiltrationResult:
    """Evaluate ordinary Laplacian spectra along a filtration.

    This snapshot scan reproduces the feature sequence produced by the legacy
    package.  Use :func:`compute_persistent_laplacian` when a pairwise
    persistent operator and persistent Betti number are required.
    """

    if not isinstance(filtration, FilteredSequenceHyperdigraph):
        raise TypeError("filtration must be a FilteredHyperdigraph")
    _validate_max_dimension(max_dimension)
    selected_backend = _validate_backend(omega_backend)
    numerical_tolerance = _validate_tolerance(tolerance)
    _validate_return_matrices(return_matrices)
    values = _selected_thresholds(filtration, max_dimension, thresholds)

    snapshots: list[LaplacianSnapshot] = []
    previous_signature: tuple[int, ...] | None = None
    previous_result: LaplacianResult | None = None
    for threshold in values:
        signature = tuple(
            sum(
                record.birth <= threshold
                for record in filtration.weighted_hyperedges(dimension)
            )
            for dimension in range(max_dimension + 2)
        )
        if signature == previous_signature and previous_result is not None:
            result = previous_result
        else:
            result = compute_laplacian(
                filtration.snapshot(threshold),
                max_dimension,
                tolerance=numerical_tolerance,
                return_matrices=return_matrices,
                omega_backend=selected_backend,
            )
            previous_signature = signature
            previous_result = result
        snapshots.append(LaplacianSnapshot(threshold, result))
    return LaplacianFiltrationResult(tuple(snapshots), max_dimension)


__all__ = ['RealOmegaBackend', 'compute_laplacian', 'compute_laplacian_filtration', 'compute_persistent_laplacian']
