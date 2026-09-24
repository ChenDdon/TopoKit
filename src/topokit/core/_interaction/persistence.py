"""Streaming column reduction for persistent interaction homology over GF(2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from numbers import Real
from typing import Literal

from .interaction import (
    FilteredInteractionChainComplex,
)
from .results import PersistenceDiagnostics, PersistenceInterval, PersistenceResult


ReductionBackend = Literal["auto", "int", "sparse"]


@dataclass(frozen=True, slots=True)
class _ReducedBoundary:
    zero_columns: frozenset[int]
    deaths: dict[int, int]
    backend: str
    column_additions: int


def _reduce_integer_columns(
    chain: FilteredInteractionChainComplex, degree: int
) -> _ReducedBoundary:
    columns_by_pivot: dict[int, int] = {}
    death_columns: dict[int, int] = {}
    zero_columns: set[int] = set()
    column_additions = 0
    for column_index in range(len(chain.degrees[degree].keys)):
        column = 0
        for row in chain.boundary_rows(degree, column_index):
            column ^= 1 << row
        while column:
            pivot = column.bit_length() - 1
            pivot_column = columns_by_pivot.get(pivot)
            if pivot_column is None:
                columns_by_pivot[pivot] = column
                death_columns[pivot] = column_index
                break
            column ^= pivot_column
            column_additions += 1
        if not column:
            zero_columns.add(column_index)
    return _ReducedBoundary(frozenset(zero_columns), death_columns, "int", column_additions)


def _reduce_sparse_columns(
    chain: FilteredInteractionChainComplex, degree: int
) -> _ReducedBoundary:
    columns_by_pivot: dict[int, frozenset[int]] = {}
    death_columns: dict[int, int] = {}
    zero_columns: set[int] = set()
    column_additions = 0
    for column_index in range(len(chain.degrees[degree].keys)):
        column = set(chain.boundary_rows(degree, column_index))
        while column:
            pivot = max(column)
            pivot_column = columns_by_pivot.get(pivot)
            if pivot_column is None:
                columns_by_pivot[pivot] = frozenset(column)
                death_columns[pivot] = column_index
                break
            column.symmetric_difference_update(pivot_column)
            column_additions += 1
        if not column:
            zero_columns.add(column_index)
    return _ReducedBoundary(frozenset(zero_columns), death_columns, "sparse", column_additions)


def _reduce_boundary(
    chain: FilteredInteractionChainComplex,
    degree: int,
    backend: ReductionBackend,
) -> _ReducedBoundary:
    column_count = len(chain.degrees[degree].keys)
    if degree == 0 or not column_count:
        return _ReducedBoundary(
            zero_columns=frozenset(range(column_count)),
            deaths={},
            backend="trivial",
            column_additions=0,
        )
    row_count = len(chain.degrees[degree - 1].keys)
    selected = backend
    if selected == "auto":
        # Python integers are exceptionally fast for moderate row spaces, but
        # their storage is dense up to the highest bit.  Sparse sets avoid that
        # cost once the row space becomes large.
        selected = "int" if row_count <= 8192 else "sparse"
    if selected == "int":
        return _reduce_integer_columns(chain, degree)
    if selected == "sparse":
        return _reduce_sparse_columns(chain, degree)
    raise ValueError("backend must be 'auto', 'int', or 'sparse'")


def compute_persistence(
    chain: FilteredInteractionChainComplex,
    *,
    include_diagonal: bool = False,
    minimum_persistence: Real = 0.0,
    backend: ReductionBackend = "auto",
) -> PersistenceResult:
    """Reduce the filtered interaction boundary and return barcode intervals.

    ``minimum_persistence`` removes finite intervals whose width is less than
    or equal to the supplied tolerance.  Its default of zero removes only
    exact diagonal intervals.  ``include_diagonal=True`` overrides this filter.
    """

    if not isinstance(include_diagonal, bool):
        raise TypeError("include_diagonal must be a boolean")
    if isinstance(minimum_persistence, bool) or not isinstance(
        minimum_persistence, Real
    ):
        raise TypeError("minimum_persistence must be a finite non-negative real")
    persistence_threshold = float(minimum_persistence)
    if not isfinite(persistence_threshold) or persistence_threshold < 0.0:
        raise ValueError("minimum_persistence must be a finite non-negative real")
    if backend not in {"auto", "int", "sparse"}:
        raise ValueError("backend must be 'auto', 'int', or 'sparse'")

    reductions = tuple(
        _reduce_boundary(chain, degree, backend)
        for degree in range(chain.max_homology_dimension + 2)
    )

    intervals: list[PersistenceInterval] = []
    finite_pairs = 0
    for dimension in range(chain.max_homology_dimension + 1):
        birth_columns = reductions[dimension].zero_columns
        death_columns = reductions[dimension + 1].deaths
        if any(pivot not in birth_columns for pivot in death_columns):
            raise RuntimeError(
                "a persistence death pivot did not correspond to a cycle birth"
            )
        for birth_index in sorted(birth_columns):
            death_index = death_columns.get(birth_index)
            birth = chain.degrees[dimension].births[birth_index]
            death = (
                chain.degrees[dimension + 1].births[death_index]
                if death_index is not None
                else inf
            )
            if death < birth:
                raise RuntimeError("the interaction filtration produced a backwards interval")
            if death_index is not None:
                finite_pairs += 1
            if include_diagonal or death - birth > persistence_threshold:
                intervals.append(
                    PersistenceInterval(
                        dimension=dimension,
                        birth=birth,
                        death=death,
                        birth_index=birth_index,
                        death_index=death_index,
                    )
                )

    intervals.sort(
        key=lambda interval: (
            interval.dimension,
            interval.birth,
            interval.death,
            interval.birth_index if interval.birth_index is not None else -1,
        )
    )
    diagnostics = PersistenceDiagnostics(
        coefficient_field="GF(2)",
        interaction_cell_counts=tuple(
            len(layer.keys) for layer in chain.degrees
        ),
        reduction_backends=tuple(reduction.backend for reduction in reductions),
        reduction_column_count=sum(
            len(chain.degrees[degree].keys)
            for degree in range(1, len(reductions))
        ),
        column_addition_count=sum(
            reduction.column_additions for reduction in reductions
        ),
        finite_pair_count=finite_pairs,
    )
    return PersistenceResult(
        intervals=tuple(intervals),
        max_dimension=chain.max_homology_dimension,
        diagnostics=diagnostics,
    )


__all__ = [
    "ReductionBackend",
    "compute_persistence",
]
