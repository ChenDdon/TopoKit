from itertools import combinations

import numpy as np
import pytest
from scipy import linalg, sparse

from topokit.core._simplicial import (
    ComplexityLimitError, SimplexTree, boundary_matrix, homology,
    laplacian, laplacian_filtration, laplacian_matrix, persistent_laplacian,
    persistent_laplacian_at, persistent_laplacian_matrix, spectrum,
)
from topokit.builders._simplicial import graph_complex


def test_known_spectra():
    edge = SimplexTree([(0, 1)])
    np.testing.assert_allclose(laplacian(edge, 0).eigenvalues, [0, 2])
    np.testing.assert_allclose(laplacian(edge, 1).eigenvalues, [2])
    cycle = graph_complex([(0, 1), (1, 2), (2, 0)])
    np.testing.assert_allclose(laplacian(cycle, 1).eigenvalues, [0, 3, 3])
    disk = SimplexTree([(0, 1, 2)])
    np.testing.assert_allclose(laplacian(disk, 1).eigenvalues, [3, 3, 3])


def test_intermediate_vertex_schur_effect():
    source = SimplexTree([(0,), (2,)])
    target = SimplexTree([(0, 1), (1, 2)])
    result = persistent_laplacian(source, target, 0)
    np.testing.assert_allclose(result.matrix.toarray(), [[.5, -.5], [-.5, .5]])
    np.testing.assert_allclose(result.eigenvalues, [0, 1])
    assert result.betti_number == 1
    one = persistent_laplacian(SimplexTree([(0,)]), target)
    np.testing.assert_allclose(one.eigenvalues, [0])


def test_new_diagonal_must_cancel_in_persistent_chain():
    source = graph_complex([(0, 1), (1, 2), (2, 3), (3, 0)])
    target = SimplexTree([(0, 1, 2), (0, 2, 3)])
    result = persistent_laplacian(source, target, 1)
    np.testing.assert_allclose(result.eigenvalues, [2, 2, 2, 4])
    assert result.betti_number == 0
    assert laplacian(source, 1).betti_number == 1


def _rank(matrix):
    return int(np.linalg.matrix_rank(matrix, tol=1e-9)) if matrix.size else 0


def _kernel(matrix):
    # Some LAPACK builds cannot SVD a matrix with zero rows/columns.
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1])
    if matrix.shape[1] == 0:
        return np.empty((0, 0))
    return linalg.null_space(matrix)


def _real_persistent_betti(source, target, q):
    down = boundary_matrix(source, q, sparse_output=False)
    cycles = _kernel(down)
    source_basis, target_basis = source.simplices(q), target.simplices(q)
    lifted = np.zeros((len(target_basis), cycles.shape[1]))
    for i, simplex in enumerate(source_basis):
        lifted[target_basis.index(simplex)] = cycles[i]
    boundaries = boundary_matrix(target, q + 1, sparse_output=False)
    return _rank(np.column_stack((boundaries, lifted))) - _rank(boundaries)


def _nullspace_laplacian(source, target, q):
    source_basis, target_basis = source.simplices(q), target.simplices(q)
    rows = [target_basis.index(s) for s in source_basis]
    outside = [i for i in range(len(target_basis)) if i not in rows]
    boundary = boundary_matrix(target, q + 1, sparse_output=False)
    z = _kernel(boundary[outside])
    effective = boundary[rows] @ z
    down = boundary_matrix(source, q, sparse_output=False)
    return down.T @ down + effective @ effective.T


@pytest.mark.parametrize("seed", range(12))
def test_random_psd_nullity_and_independent_nullspace_formula(seed):
    rng = np.random.default_rng(seed)
    tree = SimplexTree({(v,): 0 for v in range(7)})
    for size in range(2, 5):
        for s in combinations(range(7), size):
            if rng.random() < .15:
                tree.insert(s, int(rng.integers(1, 4)))
    for q in range(3):
        ordinary = laplacian(tree, q)
        assert ordinary.betti_number == homology(tree, field="real", max_dimension=2).betti_numbers[q]
        same = persistent_laplacian(tree, tree, q)
        np.testing.assert_allclose(ordinary.matrix.toarray(), same.matrix.toarray())
        for a, b in [(0, 1), (1, 2), (1, 3), (2, 3)]:
            source, target = tree.at(a), tree.at(b)
            result = persistent_laplacian(source, target, q)
            np.testing.assert_allclose(result.matrix.toarray(), _nullspace_laplacian(source, target, q), atol=1e-10)
            assert result.betti_number == _real_persistent_betti(source, target, q)
            assert np.all(result.eigenvalues >= 0)


def test_scale_api_sparse_dense_empty_and_guard():
    tree = SimplexTree({(0, 1): 1, (1, 2): 1, (0, 2): 1, (0, 1, 2): 2})
    snapshots = laplacian_filtration(tree, [0, 1, 2], dimensions=[0, 1])
    assert snapshots[1][1].betti_number == 1
    assert snapshots[2][1].betti_number == 0
    assert persistent_laplacian_at(tree, 1, 2, 1).betti_number == 0
    np.testing.assert_array_equal(laplacian_matrix(tree, 1).toarray(), laplacian_matrix(tree, 1, sparse_output=False))
    assert laplacian([], 2).matrix.shape == (0, 0)
    assert persistent_laplacian([], tree, 1).betti_number == 0
    with pytest.raises(ComplexityLimitError):
        laplacian(tree, max_dense_entries=1)
    with pytest.raises(ComplexityLimitError):
        persistent_laplacian_matrix([(0,), (2,)], [(0, 1), (1, 2)], max_dense_entries=1)
    with pytest.raises(ValueError):
        persistent_laplacian([(0,)], [(1,)])
    with pytest.raises(ValueError):
        persistent_laplacian_at(tree, 2, 1)


def test_partial_spectrum_and_invalid_matrices():
    matrix = sparse.diags(np.arange(1, 21, dtype=float))
    np.testing.assert_allclose(spectrum(matrix, k=3), [1, 2, 3])
    np.testing.assert_array_equal(spectrum(sparse.csr_matrix((5, 5)), k=2), [0, 0])
    for matrix in [[[0, 1], [0, 0]], [[np.nan]], [[-1]]]:
        with pytest.raises(ValueError):
            spectrum(matrix)
