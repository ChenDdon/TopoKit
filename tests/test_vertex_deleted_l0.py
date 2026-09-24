"""Independent induced-incidence oracles for the optional deletion operation."""
import itertools
import numpy as np
import pytest
from topokit.core import hyperdigraph as hd


@pytest.mark.parametrize('seed', range(12))
def test_deleted_l0_matches_incidence_and_full_reconstruction(seed):
    rng = np.random.default_rng(seed)
    vertices = ('a', 3, ('t', 1), 'd', 27, 'f', 'isolated')
    births = {v: float(rng.integers(0, 3)) for v in vertices}
    records = [((v,), birth) for v, birth in births.items()]
    for a, b in itertools.permutations(vertices[:-1], 2):
        if rng.random() < .4:
            records.append(((a, b), max(births[a], births[b]) + float(rng.integers(0, 2))))
    records += [(('a', 3, 'd'), 2.)]  # Higher hyperedges never enter ordinary L0.
    native = hd.FilteredHyperdigraph(vertices, records)
    sweep = hd.L0Sweep(native)
    for scale in (-1., 0., 1., 2., 3.):
        undeleted = sweep.matrix_at(scale)
        active = tuple(v[0] for v in sweep.basis_at(scale))
        for vertex in active:
            kept = tuple(v for v in active if v != vertex)
            edges = [r for r, birth in records if len(r) == 2 and birth <= scale and vertex not in r]
            B = np.zeros((len(kept), len(edges)))
            for j, (a, b) in enumerate(edges):
                B[kept.index(a), j] = -1
                B[kept.index(b), j] = 1
            expected = B @ B.T
            actual = sweep.vertex_deleted_laplacian(scale, vertex, return_matrix=True, return_eigenvectors=True)
            np.testing.assert_array_equal(actual.matrix, expected)
            np.testing.assert_allclose(actual.eigenvalues, np.linalg.eigvalsh(expected), atol=1e-11)
            np.testing.assert_allclose(expected @ actual.eigenvectors,
                                       actual.eigenvectors * actual.eigenvalues, atol=1e-10)
            fast = sweep.vertex_deleted_laplacian(scale, vertex)
            np.testing.assert_allclose(fast.eigenvalues, actual.eigenvalues, atol=1e-11)
            rebuilt = (hd.FilteredHyperdigraph(kept, [(e, 0.) for e in edges], include_all_vertices=True)
                       if kept else hd.FilteredHyperdigraph(vertices, []))
            reference = hd.laplacian(rebuilt, scale=0., backend='reference')
            np.testing.assert_allclose(fast.eigenvalues, reference.eigenvalues, atol=1e-10)
            assert fast.complete and fast.nullity == reference.nullity
            assert actual.metadata['upper_hyperedge_count'] == len(edges)
            np.testing.assert_array_equal(sweep.matrix_at(scale), undeleted)


def test_singleton_empty_missing_and_guards():
    sweep = hd.L0Sweep(hd.FilteredHyperdigraph(('v',), [(('v',), 1.)]))
    with pytest.raises(ValueError, match='active singleton'):
        sweep.vertex_deleted_laplacian(0., 'v')
    result = sweep.vertex_deleted_laplacian(1., 'v', return_matrix=True)
    assert result.matrix.shape == (0, 0) and len(result.eigenvalues) == 0
    assert result.nullity == 0 and result.metadata['structural_absence']
    for bad in (0, -1, np.nan, np.inf, True):
        with pytest.raises(ValueError, match='tol'):
            sweep.vertex_deleted_laplacian(1., 'v', tol=bad)
    with pytest.raises(ValueError, match='active singleton'):
        sweep.vertex_deleted_laplacian(1., 'missing')
    with pytest.raises(ValueError, match='nondecreasing'):
        sweep.vertex_deleted_laplacian(0., 'v')
