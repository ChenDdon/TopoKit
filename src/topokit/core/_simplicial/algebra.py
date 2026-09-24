"""Signed simplicial boundaries and native prime-field column reduction."""

import numpy as np
from scipy import sparse

from ._validation import guard_dense, nonnegative_int, prime
from .complex import as_complex, facets


def boundary_matrix(complex_, dimension, *, field=None, sparse_output=True,
                    max_dense_entries=10_000_000):
    """Return d_q: C_q -> C_(q-1), using sorted simplex bases.

    d_0 has zero rows (unreduced homology). field=None keeps signed integer
    entries; a prime field returns residues in [0,p). CSR is the default.
    Bases are available as complex_.simplices(q) and complex_.simplices(q-1).
    """
    complex_ = as_complex(complex_)
    dimension = nonnegative_int(dimension, "dimension")
    if field is not None:
        field = prime(field)
    column_basis = complex_.simplices(dimension)
    row_basis = complex_.simplices(dimension - 1) if dimension else ()
    row_index = {s: i for i, s in enumerate(row_basis)}
    row_indices, column_indices, values = [], [], []
    for j, simplex in enumerate(column_basis):
        for k, face in enumerate(facets(simplex)):
            row_indices.append(row_index[face])
            column_indices.append(j)
            value = (-1) ** k
            values.append(value if field is None else value % field)
    matrix = sparse.csr_matrix((values, (row_indices, column_indices)),
                               shape=(len(row_basis), len(column_basis)), dtype=np.int64)
    if sparse_output:
        return matrix
    guard_dense(matrix.shape, max_dense_entries)
    return matrix.toarray()


def reduce_column(column, pivots, field):
    """Reduce against normalized pivot columns, then retain a new pivot.

    GF(2) columns are Python integer bitsets (XOR); other prime fields use
    sparse coefficient dictionaries and modular inverses, not float ranks.
    Returns the pivot index or None for a zero column.
    """
    if field == 2:
        while column:
            pivot_row = column.bit_length() - 1
            if pivot_row not in pivots:
                pivots[pivot_row] = column
                return pivot_row
            column ^= pivots[pivot_row]
    else:
        while column:
            pivot_row = max(column)
            if pivot_row not in pivots:
                pivot_inverse = pow(column[pivot_row], -1, field)
                pivots[pivot_row] = {i: (v * pivot_inverse) % field for i, v in column.items()}
                return pivot_row
            pivot_coefficient = column[pivot_row]
            for i, value in pivots[pivot_row].items():
                updated_coefficient = (column.get(i, 0) - pivot_coefficient * value) % field
                if updated_coefficient:
                    column[i] = updated_coefficient
                else:
                    column.pop(i, None)
    return None


def boundary_column(simplex, row_index, field):
    if field == 2:
        column_bits = 0
        for face in facets(simplex):
            column_bits |= 1 << row_index[face]
        return column_bits
    return {row_index[face]: (-1) ** i % field for i, face in enumerate(facets(simplex))}


def boundary_rank(complex_, dimension, field):
    row_basis = complex_.simplices(dimension - 1) if dimension else ()
    if not row_basis:
        return 0
    row_index = {s: i for i, s in enumerate(row_basis)}
    pivot_columns = {}
    for simplex in complex_.simplices(dimension):
        reduce_column(boundary_column(simplex, row_index, field), pivot_columns, field)
    return len(pivot_columns)
