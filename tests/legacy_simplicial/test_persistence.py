from itertools import combinations

import numpy as np
import pytest

from topokit.core._simplicial import SimplexTree, boundary_matrix, homology, persistent_homology


def rref(matrix, field):
    """Independent dense row elimination for tiny validation fixtures."""
    matrix = np.array(matrix, dtype=np.int64) % field
    pivots = []
    row = 0
    for col in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[row:, col])
        if not len(candidates):
            continue
        j = row + candidates[0]
        matrix[[row, j]] = matrix[[j, row]]
        matrix[row] = matrix[row] * pow(int(matrix[row, col]), -1, field) % field
        for i in range(matrix.shape[0]):
            if i != row:
                matrix[i] = (matrix[i] - matrix[i, col] * matrix[row]) % field
        pivots.append(col)
        row += 1
        if row == matrix.shape[0]:
            break
    return matrix, pivots


def kernel(matrix, field):
    reduced, pivots = rref(matrix, field)
    free = [j for j in range(matrix.shape[1]) if j not in pivots]
    basis = np.zeros((matrix.shape[1], len(free)), dtype=np.int64)
    for i, col in enumerate(free):
        basis[col, i] = 1
        for row, pivot in enumerate(pivots):
            basis[pivot, i] = -reduced[row, col] % field
    return basis


def image_rank(source, target, q, field):
    down = boundary_matrix(source, q, sparse_output=False)
    cycles = kernel(down, field)
    source_basis, target_basis = source.simplices(q), target.simplices(q)
    lifted = np.zeros((len(target_basis), cycles.shape[1]), dtype=np.int64)
    for i, simplex in enumerate(source_basis):
        lifted[target_basis.index(simplex)] = cycles[i]
    boundaries = boundary_matrix(target, q + 1, sparse_output=False)
    return len(rref(np.column_stack((boundaries, lifted)), field)[1]) - len(rref(boundaries, field)[1])


def random_filtration(seed):
    rng = np.random.default_rng(seed)
    tree = SimplexTree({(v,): int(rng.integers(-1, 2)) for v in range(6)})
    for size in range(2, 5):
        for simplex in combinations(range(6), size):
            if rng.random() < .2:
                tree.insert(simplex, int(rng.integers(0, 4)))
    return tree


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("field", [2, 3, 7])
def test_persistence_against_independent_image_rank(seed, field):
    tree = random_filtration(seed)
    result = persistent_homology(tree, field=field, max_dimension=2)
    uncleared = persistent_homology(tree, field=field, max_dimension=2, clearing=False)
    for q in range(3):
        np.testing.assert_array_equal(result.diagram(q), uncleared.diagram(q))
    for a in [-1, 0, 1, 2, 3]:
        source = tree.at(a)
        assert result.betti_at(a) == homology(source, field=field, max_dimension=2).betti_numbers
        for b in range(max(a, 0), 4):
            for q in range(3):
                assert result.persistent_betti(q, a, b) == image_rank(source, tree.at(b), q, field)


def test_negative_vertex_births_half_open_intervals_and_h0_only():
    tree = SimplexTree({(0,): -3, (1,): -2, (2,): -1, (0, 1): 0, (1, 2): 2})
    result = persistent_homology(tree, max_dimension=0)
    np.testing.assert_array_equal(result.diagram(0), [[-3, np.inf], [-2, 0], [-1, 2]])
    assert result.betti_at(0) == (2,)
    assert result.betti_at(2) == (1,)
    assert result.persistent_betti(0, -2, 0) == 1
    with pytest.raises(ValueError):
        result.persistent_betti(1, 0, 2)


def test_zero_intervals_optional_and_empty_shape():
    tree = SimplexTree([(0, 1, 2)])
    result = persistent_homology(tree)
    assert result.diagram(1).shape == (0, 2)
    full = persistent_homology(tree, include_zero=True)
    assert len(full.diagram(0)) == 3
    assert len(full.diagram(1)) == 1
    assert full.betti_at(0) == (1, 0, 0)
    assert persistent_homology([]).betti_at(0) == (0,)


def test_clearing_actually_skips_columns():
    tree = SimplexTree([range(5)])
    result = persistent_homology(tree)
    assert result.diagnostics["cleared_columns"] > 0
    assert result.betti_at(0) == (1, 0, 0, 0, 0)
