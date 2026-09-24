"""Independent dense-algebra audit of the optimized embedded-chain operators.

The oracles below do not call native chain assembly or its reference backend.
They build full signed deletion matrices directly from ordered hyperedges.
"""

import itertools

import numpy as np
import pytest
from scipy import linalg, sparse

from topokit import ResourceLimitError
from topokit.core import hyperdigraph as hd
from topokit.core._hyperdigraph.spectral import (
    KernelProjection, omega_basis, ordinary_operator, pairwise_operator,
)


def kernel(matrix):
    # This SciPy build does not accept empty LAPACK SVD dimensions. Also avoid
    # treating floating-point projection roundoff of an exactly zero constraint
    # as a nonzero matrix merely because a relative rank cutoff is used.
    if matrix.shape[1] == 0:
        return np.empty((0, 0))
    if matrix.shape[0] == 0 or np.linalg.norm(matrix) < 1e-9:
        return np.eye(matrix.shape[1])
    return linalg.null_space(matrix, rcond=1e-10)


def full_deletion(edges, extra_rows=()):
    rows = list(dict.fromkeys(extra_rows))
    for edge in edges:
        if len(edge) > 1:
            for position in range(len(edge)):
                face = edge[:position] + edge[position + 1:]
                if face not in rows:
                    rows.append(face)
    index = {row: i for i, row in enumerate(rows)}
    matrix = np.zeros((len(rows), len(edges)))
    for j, edge in enumerate(edges):
        if len(edge) > 1:
            for position in range(len(edge)):
                face = edge[:position] + edge[position + 1:]
                matrix[index[face], j] += (-1.) ** position
    return tuple(rows), matrix


def independent_omega(obj, q):
    edges = obj.directed_hyperedges(q)
    rows, deletion = full_deletion(edges)
    allowed = set(obj.directed_hyperedges(q - 1)) if q else set()
    absent = [i for i, face in enumerate(rows) if face not in allowed]
    constraints = deletion[absent]
    basis = kernel(constraints)
    return edges, deletion, basis


def independent_pair(source, target, q):
    source_edges, down, basis = independent_omega(source, q)
    upper_edges = target.directed_hyperedges(q + 1)
    rows, upper = full_deletion(upper_edges, extra_rows=source_edges)
    index = {edge: i for i, edge in enumerate(rows)}
    embed = np.zeros((len(rows), basis.shape[1]))
    for j, edge in enumerate(source_edges):
        embed[index[edge]] = basis[j]
    # This one full-boundary condition independently includes both missing
    # target faces and boundary membership in the earlier Omega space.
    constraints = upper - embed @ (embed.T @ upper)
    allowed = kernel(constraints)
    upper_map = embed.T @ upper @ allowed
    lower_map = down @ basis
    return basis, lower_map.T @ lower_map + upper_map @ upper_map.T


def assert_ambient_operator(operator, source, target, q):
    reference_basis, reference_matrix = independent_pair(source, target, q)
    matrix = operator.matrix.matmat(np.eye(operator.matrix.shape[0]))
    basis = operator.basis.toarray()
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2e-10)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-10)
    np.testing.assert_allclose(basis @ matrix @ basis.T,
                               reference_basis @ reference_matrix @ reference_basis.T,
                               atol=3e-8)
    if matrix.size:
        assert np.linalg.eigvalsh(matrix).min() >= -3e-8


@pytest.mark.parametrize("seed", range(8))
def test_signed_incidence_projection_matches_independent_pseudoinverse(seed):
    rng = np.random.default_rng(seed)
    constraints = np.zeros((6, 13))
    for column in range(constraints.shape[1]):
        count = int(rng.integers(0, 3))
        chosen = rng.choice(len(constraints), size=count, replace=False)
        constraints[chosen, column] = rng.choice([-1., 1.], size=count)
    projector = KernelProjection(sparse.csc_matrix(constraints), 10_000, 1e-10)
    actual = projector(np.eye(constraints.shape[1]))
    expected = np.eye(constraints.shape[1]) - np.linalg.pinv(constraints) @ constraints
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(actual, actual.T, atol=1e-10)
    np.testing.assert_allclose(actual @ actual, actual, atol=1e-10)
    np.testing.assert_allclose(constraints @ actual, 0., atol=1e-10)
    assert projector.rank == np.linalg.matrix_rank(constraints)


@pytest.mark.parametrize("anchored,unbalanced", [(False, False), (True, False),
                                                (False, True), (True, True)])
def test_balanced_unbalanced_and_anchored_signed_components(anchored, unbalanced):
    # Three two-row columns make a cycle; the last sign controls balance.
    matrix = np.array([[1., 0., -1. if not unbalanced else 1., 0.],
                       [-1., 1., 0., 0.], [0., -1., 1., 0.]])
    if anchored:
        matrix[0, 3] = 1.
    projector = KernelProjection(sparse.csc_matrix(matrix), 1000, 1e-10)
    assert projector.rank == (3 if anchored or unbalanced else 2)
    np.testing.assert_allclose(projector(np.eye(4)),
                               np.eye(4) - np.linalg.pinv(matrix) @ matrix, atol=1e-10)


def test_signed_helmert_basis_spans_exact_kernel_and_has_guarded_storage():
    constraints = sparse.csc_matrix([[1., -1., 1., -1., 0., 0.],
                                     [0., 0., 0., 0., -1., 0.]])
    basis, backend = omega_basis(constraints, 1000, 1000, 1e-10)
    assert backend == "signed_helmert_groups"
    assert basis.shape == (6, 4)
    np.testing.assert_allclose(constraints @ basis.toarray(), 0., atol=1e-12)
    np.testing.assert_allclose((basis.T @ basis).toarray(), np.eye(4), atol=1e-12)
    expected = np.eye(6) - np.linalg.pinv(constraints.toarray()) @ constraints.toarray()
    np.testing.assert_allclose((basis @ basis.T).toarray(), expected, atol=1e-12)
    with pytest.raises(ResourceLimitError):
        omega_basis(constraints, 1000, basis.nnz - 1, 1e-10)


def test_persistent_l0_late_intermediate_vertex_preserves_induced_chain_metric():
    filtration = hd.FilteredHyperdigraph((0, 1, 2), [
        ((0,), 0.), ((2,), 0.), ((1,), 1.), ((0, 1), 1.), ((1, 2), 1.)])
    result = hd.persistent_laplacian(filtration, 0, start=0., end=1.,
                                    return_matrix=True, return_eigenvectors=True)
    # The allowed late chain is (edge01 + edge12)/sqrt(2), so its
    # persistent boundary gives half a usual two-node graph Laplacian.
    np.testing.assert_allclose(result.matrix, [[.5, -.5], [-.5, .5]], atol=1e-10)
    np.testing.assert_allclose(result.eigenvalues, [0., 1.], atol=1e-10)
    assert result.nullity == 1
    assert result.basis == ((((0,), 1.),), (((2,), 1.),))
    assert_ambient_operator(pairwise_operator(filtration, 0, 0., 1.),
                            filtration.snapshot(0.), filtration.snapshot(1.), 0)


@pytest.mark.parametrize("seed", range(4))
def test_non_face_closed_pairs_match_independent_full_boundary_oracle(seed):
    rng = np.random.default_rng(seed)
    vertices = tuple(range(5))
    candidates = [edge for length in range(1, 6)
                  for edge in itertools.permutations(vertices, length)]
    records = [(edge, float(rng.integers(0, 3))) for edge in candidates if rng.random() < .22]
    filtration = hd.FilteredHyperdigraph(vertices, records)
    source, target = filtration.snapshot(0.), filtration.snapshot(2.)
    for q in range(4):
        assert_ambient_operator(ordinary_operator(source, q), source, source, q)
        assert_ambient_operator(pairwise_operator(filtration, q, 0., 2.), source, target, q)


def test_general_degree_three_source_basis_and_partial_spectrum_semantics():
    # Four tetrahedral sequences share one absent triangular face. Their
    # three-dimensional Omega space is a signed Helmert contrast space.
    upper = [(0, 1, 2, c) for c in range(3, 7)]
    present = set()
    for tetrahedron in upper:
        for length in range(1, 4):
            present.update(itertools.combinations(tetrahedron, length))
    present.remove((0, 1, 2))
    obj = hd.Hyperdigraph(tuple(range(7)), tuple(sorted(present)) + tuple(upper))
    operator = ordinary_operator(obj, 3)
    assert operator.basis.shape == (4, 3)
    assert_ambient_operator(operator, obj, obj, 3)
    full = hd.laplacian(obj, 3, return_eigenvectors=True)
    partial = hd.laplacian(obj, 3, k=1, return_eigenvectors=True)
    np.testing.assert_allclose(full.eigenvalues, [3., 3., 3.], atol=1e-10)
    np.testing.assert_allclose(partial.eigenvalues, [3.], atol=1e-10)
    assert full.complete and full.nullity == 0
    assert not partial.complete and partial.nullity is None
    assert len(partial.basis) == 3 and partial.eigenvectors.shape == (3, 1)


def test_embedded_sparse_fill_is_refused_for_ordinary_and_distinct_pair_times():
    cells = [edge for c in range(2, 7) for edge in [(0, c), (1, c), (0, 1, c)]]
    obj = hd.Hyperdigraph(tuple(range(7)), cells, include_all_vertices=True)
    # Raw boundary needs 15 entries, source Omega 14, but the embedded down
    # boundary requires 28. Equal objects at distinct times must not bypass it.
    with pytest.raises(ResourceLimitError):
        ordinary_operator(obj, 2, max_sparse_entries=20)
    with pytest.raises(ResourceLimitError):
        pairwise_operator(obj.as_filtered(0.), 2, 0., 1., max_sparse_entries=20)


def test_signed_factor_fill_bound_is_checked_before_factorization(monkeypatch):
    from topokit.core._hyperdigraph import spectral

    def forbidden(*args, **kwargs):
        raise AssertionError("factorization must not occur after a known fill-budget failure")

    incidence = np.zeros((8, 8))
    for column in range(8):
        incidence[column, column] = 1.
        incidence[(column + 1) % 8, column] = -1.
    monkeypatch.setattr(spectral, "splu", forbidden)
    with pytest.raises(ResourceLimitError, match="fill bound"):
        KernelProjection(sparse.csc_matrix(incidence), 48, 1e-10)
