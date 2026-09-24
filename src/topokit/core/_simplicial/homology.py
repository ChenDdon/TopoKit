"""Ordinary simplicial homology over a chosen prime field or the reals."""

from dataclasses import dataclass

import numpy as np

from ._validation import nonnegative_int, prime, tolerance
from .algebra import boundary_matrix, boundary_rank
from .complex import as_complex


@dataclass(frozen=True)
class HomologyResult:
    betti_numbers: tuple[int, ...]
    chain_dimensions: tuple[int, ...]
    boundary_ranks: tuple[int, ...]
    field: int | str
    euler_characteristic: int


def homology(complex_, *, max_dimension=None, field=2, scale=None,
             tol=1e-10, max_dense_entries=10_000_000):
    """Compute Betti numbers from ranks of consecutive signed boundaries.

    field=2 (default) or another prime uses exact modular arithmetic.
    field='real' uses numerical SVD ranks and corresponds to real Laplacian
    nullities, which can differ from finite-field Betti numbers with torsion.
    No integral torsion decomposition or representative cycles is returned.
    """
    complex_ = as_complex(complex_)
    if scale is not None:
        complex_ = complex_.at(scale)
    maximum_dimension = (max(complex_.dimension, 0) if max_dimension is None
                         else nonnegative_int(max_dimension, "max_dimension"))
    tol = tolerance(tol)
    if field != "real":
        field = prime(field)
    chain_dimensions = tuple(len(complex_.simplices(q)) for q in range(maximum_dimension + 1))
    boundary_ranks = [0]
    for q in range(1, maximum_dimension + 2):
        if field == "real":
            matrix = boundary_matrix(complex_, q, sparse_output=False, max_dense_entries=max_dense_entries)
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            rank_cutoff = tol * max(float(singular_values[0]), 1.0) if singular_values.size else tol
            boundary_ranks.append(int(np.count_nonzero(singular_values > rank_cutoff)))
        else:
            boundary_ranks.append(boundary_rank(complex_, q, field))
    betti_numbers = tuple(chain_dimensions[q] - boundary_ranks[q] - boundary_ranks[q + 1] for q in range(maximum_dimension + 1))
    euler_characteristic = sum((-1) ** (len(s) - 1) for s in complex_)
    return HomologyResult(betti_numbers, chain_dimensions, tuple(boundary_ranks), field, euler_characteristic)
