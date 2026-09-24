"""Bit-packed linear algebra over :math:`GF(2)`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def iter_set_bits(vector: int) -> Iterable[int]:
    if vector < 0:
        raise ValueError("GF(2) vectors must be nonnegative integers")
    work = vector
    while work:
        least = work & -work
        yield least.bit_length() - 1
        work ^= least


class XorBasis:
    """Incremental echelon basis using the highest set bit as pivot."""

    __slots__ = ("_pivots",)

    def __init__(self, vectors: Iterable[int] = ()) -> None:
        self._pivots: dict[int, int] = {}
        for vector in vectors:
            self.add(vector)

    @property
    def rank(self) -> int:
        return len(self._pivots)

    @property
    def basis_vectors(self) -> tuple[int, ...]:
        return tuple(self._pivots[pivot] for pivot in sorted(self._pivots, reverse=True))

    def reduce(self, vector: int) -> int:
        if vector < 0:
            raise ValueError("GF(2) vectors must be nonnegative integers")
        work = vector
        while work:
            pivot = work.bit_length() - 1
            known = self._pivots.get(pivot)
            if known is None:
                break
            work ^= known
        return work

    def insert(self, vector: int) -> int | None:
        reduced = self.reduce(vector)
        if not reduced:
            return None
        pivot = reduced.bit_length() - 1
        self._pivots[pivot] = reduced
        return pivot

    def add(self, vector: int) -> bool:
        return self.insert(vector) is not None


class CoordinateSolver:
    """Express ambient vectors in an incrementally supplied independent basis."""

    __slots__ = ("_pivots",)

    def __init__(self) -> None:
        self._pivots: dict[int, tuple[int, int]] = {}

    def add(self, vector: int, basis_index: int) -> None:
        work = vector
        coordinates = 1 << basis_index
        while work:
            pivot = work.bit_length() - 1
            known = self._pivots.get(pivot)
            if known is None:
                self._pivots[pivot] = (work, coordinates)
                return
            work ^= known[0]
            coordinates ^= known[1]
        raise ValueError("basis vectors supplied to CoordinateSolver must be independent")

    def solve(self, vector: int) -> int:
        work = vector
        coordinates = 0
        while work:
            pivot = work.bit_length() - 1
            known = self._pivots.get(pivot)
            if known is None:
                raise ValueError("vector is outside the supplied basis span")
            work ^= known[0]
            coordinates ^= known[1]
        return coordinates


def rank(vectors: Iterable[int]) -> int:
    return XorBasis(vectors).rank


def apply_columns(selection: int, columns: Sequence[int]) -> int:
    result = 0
    for index in iter_set_bits(selection):
        if index >= len(columns):
            raise ValueError("selection contains a coordinate outside the column space")
        result ^= columns[index]
    return result


def transpose_columns(columns: Sequence[int], number_of_rows: int) -> tuple[int, ...]:
    rows = [0] * number_of_rows
    for column_index, column in enumerate(columns):
        for row_index in iter_set_bits(column):
            if row_index >= number_of_rows:
                raise ValueError("column contains a coordinate outside the row space")
            rows[row_index] ^= 1 << column_index
    return tuple(rows)


def nullspace_basis(
    equation_rows: Iterable[int], number_of_columns: int
) -> tuple[int, ...]:
    """Return a deterministic basis for ``{x: A x = 0}``."""

    if isinstance(number_of_columns, bool) or not isinstance(number_of_columns, int):
        raise TypeError("number_of_columns must be a non-negative integer")
    if number_of_columns < 0:
        raise ValueError("number_of_columns must be non-negative")
    if number_of_columns == 0:
        return ()

    mask = (1 << number_of_columns) - 1
    rows = list(dict.fromkeys(row & mask for row in equation_rows if row & mask))
    pivot_columns: list[int] = []
    # Rows before this cursor have fixed pivots in the left-to-right reduction.
    pivot_row_index = 0
    for column in range(number_of_columns):
        selected_row = next(
            (
                index
                for index in range(pivot_row_index, len(rows))
                if (rows[index] >> column) & 1
            ),
            None,
        )
        if selected_row is None:
            continue
        rows[pivot_row_index], rows[selected_row] = rows[selected_row], rows[pivot_row_index]
        pivot = rows[pivot_row_index]
        for row_index, row in enumerate(rows):
            if row_index != pivot_row_index and ((row >> column) & 1):
                rows[row_index] = row ^ pivot
        pivot_columns.append(column)
        pivot_row_index += 1
        if pivot_row_index == len(rows):
            break

    reduced_rows = rows[:pivot_row_index]
    pivot_set = set(pivot_columns)
    basis: list[int] = []
    for free_column in range(number_of_columns):
        if free_column in pivot_set:
            continue
        vector = 1 << free_column
        for row, pivot_column in zip(reduced_rows, pivot_columns):
            if (row & vector).bit_count() & 1:
                vector |= 1 << pivot_column
        basis.append(vector)

    expected_nullity = number_of_columns - pivot_row_index
    if len(basis) != expected_nullity or rank(basis) != expected_nullity:
        raise RuntimeError("internal GF(2) nullspace construction failed")
    return tuple(basis)


def quotient_representatives(
    cycles: Iterable[int], boundaries: Iterable[int]
) -> tuple[int, ...]:
    span = XorBasis(boundaries)
    representatives: list[int] = []
    for cycle in cycles:
        if span.add(cycle):
            representatives.append(cycle)
    return tuple(representatives)


__all__ = [
    "CoordinateSolver",
    "XorBasis",
    "apply_columns",
    "iter_set_bits",
    "nullspace_basis",
    "quotient_representatives",
    "rank",
    "transpose_columns",
]
