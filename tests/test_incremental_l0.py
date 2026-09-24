"""Independent incidence oracles for incremental ordinary L0 construction."""
import itertools

import numpy as np
import pytest

from topokit import ResourceLimitError
from topokit.core import hyperdigraph as hd
from topokit.results import Topology
from topokit.workflows import laplacian_series


def incidence_oracle(native, scale):
    snapshot = native.snapshot(scale)
    vertices = snapshot.directed_hyperedges(0)
    index = {edge[0]: i for i, edge in enumerate(vertices)}
    edges = snapshot.directed_hyperedges(1)
    boundary = np.zeros((len(vertices), len(edges)))
    for column, (a, b) in enumerate(edges):
        boundary[index[a], column] = -1.
        boundary[index[b], column] = 1.
    return boundary @ boundary.T, vertices, len(edges)


@pytest.mark.parametrize("seed", range(12))
def test_all_scales_match_independent_incidence_and_reference(seed):
    rng = np.random.default_rng(seed)
    vertices = tuple(range(9))
    births = rng.integers(-1, 3, size=len(vertices))
    records = [((v,), float(births[v])) for v in vertices]
    for a, b in itertools.permutations(vertices, 2):
        if rng.random() < .35:
            records.append(((a, b), float(max(births[a], births[b]) + rng.integers(0, 3))))
    # Arbitrary higher hyperedges need not have all their pair faces present.
    records.extend([((0, 2, 4), 0.), ((1, 3, 5, 7), 2.)])
    native = hd.FilteredHyperdigraph(vertices, records)
    sweep = hd.L0Sweep(native)
    previous = None
    for scale in [-2., -1., -.5, 0., .5, 1., 2., 3., 4., 5.]:
        expected, basis, edges = incidence_oracle(native, scale)
        actual = sweep.matrix_at(scale)
        np.testing.assert_array_equal(actual, expected)
        assert sweep.basis_at(scale) == basis
        assert sweep.diagnostics["edges_inserted"] == edges
        assert sweep.diagnostics["edge_update_entries"] == 4 * edges
        np.testing.assert_array_equal(actual.sum(axis=1), np.zeros(len(actual)))
        assert np.trace(actual) == 2 * edges
        full = sweep.laplacian(scale, return_matrix=True, return_eigenvectors=True)
        np.testing.assert_array_equal(full.matrix, expected)
        np.testing.assert_allclose(full.matrix @ full.eigenvectors,
                                   full.eigenvectors * full.eigenvalues, atol=1e-11)
        reference = hd.laplacian(native, scale=scale, backend="reference")
        np.testing.assert_allclose(full.eigenvalues, reference.eigenvalues, atol=1e-10)
        assert full.nullity == reference.nullity
        if previous is not None:
            np.testing.assert_array_equal(previous[0], previous[1])
        previous = (actual, expected.copy())


def test_reciprocal_edges_ties_and_stable_mixed_vertex_labels():
    ids = ("last", 17, ("tuple", 3), "isolated")
    native = hd.FilteredHyperdigraph(ids, [
        ((ids[0], ids[1]), 1.), ((ids[1], ids[0]), 1.),
        ((ids[0], ids[2]), 1.), ((ids[1], ids[2]), 1.)], include_all_vertices=True)
    sweep = hd.L0Sweep(native)
    first = sweep.matrix_at(np.nextafter(1., 0.))
    np.testing.assert_array_equal(first, np.zeros((4, 4)))
    np.testing.assert_array_equal(sweep.matrix_at(1.),
        [[3, -2, -1, 0], [-2, 3, -1, 0], [-1, -1, 2, 0], [0, 0, 0, 0]])
    assert sweep.diagnostics["edges_added_at_scale"] == 4
    detached = sweep.matrix_at(1.)
    assert sweep.diagnostics["edges_added_at_scale"] == 0
    detached[:] = 99
    assert sweep.matrix_at(1.)[0, 0] == 3
    assert sweep.laplacian(1.).metadata["isolated_zeros_restored"] == 1
    assert sweep.basis_at(1.) == tuple((v,) for v in ids)


def test_vertex_birth_without_new_edge_and_empty_zero_chain():
    native = hd.FilteredHyperdigraph((0, 1, 2), [((0,), 0.), ((1,), 1.)])
    sweep = hd.L0Sweep(native)
    assert sweep.matrix_at(-1.).shape == (0, 0)
    assert sweep.laplacian(-1.).metadata["structural_absence"]
    assert sweep.matrix_at(0.).shape == (1, 1)
    assert sweep.matrix_at(1.).shape == (2, 2)
    assert sweep.diagnostics["edges_inserted"] == 0
    assert sweep.laplacian(1.).nullity == 2
    absent = hd.FilteredHyperdigraph((0, 1), [((0, 1, ), 0.)])
    assert not hd.L0Sweep.supports(absent)
    empty = hd.FilteredHyperdigraph((0, 1), [])
    assert hd.L0Sweep(empty).matrix_at(0.).shape == (0, 0)


def test_matrix_only_api_never_calls_an_eigensolver(monkeypatch):
    native = hd.FilteredHyperdigraph((0, 1), [((0, 1), .25)], include_all_vertices=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("matrix assembly must not solve eigenvalues")
    monkeypatch.setattr(hd, "solve_symmetric", forbidden)
    np.testing.assert_array_equal(hd.L0Sweep(native).matrix_at(.25), [[1, -1], [-1, 1]])


@pytest.mark.parametrize("scale", [None, True, np.bool_(False), "1", 1j, np.nan, np.inf, -np.inf])
def test_invalid_scales_do_not_advance_state(scale):
    native = hd.FilteredHyperdigraph((0, 1), [((0, 1), 1.)], include_all_vertices=True)
    sweep = hd.L0Sweep(native)
    with pytest.raises(ValueError):
        sweep.matrix_at(scale)
    assert sweep.diagnostics["edges_inserted"] == 0
    assert sweep.matrix_at(0.).shape == (2, 2)


def test_domain_direction_and_resource_guards():
    native = hd.FilteredHyperdigraph((0, 1), [((0, 1), 1.)], include_all_vertices=True)
    topology = Topology("hyperdigraph", native, metadata={"filtration_start": 0., "filtration_end": 2.})
    sweep = hd.L0Sweep(topology)
    for bad in [-1., 3.]:
        with pytest.raises(ValueError, match="domain"):
            sweep.matrix_at(bad)
    sweep.matrix_at(1.)
    with pytest.raises(ValueError, match="nondecreasing"):
        sweep.matrix_at(0.)
    for value in [True, -1, 3.5]:
        with pytest.raises(ValueError):
            hd.L0Sweep(native, max_dense_entries=value)
    with pytest.raises(ResourceLimitError, match="singleton buffer"):
        hd.L0Sweep(native, max_dense_entries=3)
    with pytest.raises(ResourceLimitError, match="schedule"):
        hd.L0Sweep(native, max_sparse_entries=1)
    for tol in [0., -1., np.inf, np.nan, True]:
        with pytest.raises(ValueError):
            sweep.laplacian(1., tol=tol)
    for k in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            sweep.laplacian(1., k=k)
    with pytest.raises(TypeError):
        hd.L0Sweep(native.snapshot(0.))


def test_partial_eigenspectrum_keeps_original_semantics():
    native = hd.FilteredHyperdigraph(tuple(range(5)),
        [((i, i+1), 1.) for i in range(4)], include_all_vertices=True)
    result = hd.L0Sweep(native).laplacian(1., k=2, return_eigenvectors=True)
    expected = hd.laplacian(native, scale=1., k=2, return_eigenvectors=True)
    np.testing.assert_allclose(result.eigenvalues, expected.eigenvalues, atol=1e-10)
    assert not result.complete and result.nullity is None
    assert result.eigenvectors.shape == (5, 2)


def test_missing_or_late_singleton_faces_use_general_workflow():
    native = hd.FilteredHyperdigraph((0, 1, 2), [
        ((0,), 0.), ((2,), 0.), ((1,), 2.), ((0, 1), 1.), ((1, 2), 1.)])
    assert not hd.L0Sweep.supports(native)
    with pytest.raises(ValueError, match="singleton faces"):
        hd.L0Sweep(native)
    series = laplacian_series(Topology("hyperdigraph", native), max_dimension=0, scales=[0., 1., 2.])
    for scale, row in series.items():
        expected = hd.laplacian(native, scale=scale, backend="reference")
        np.testing.assert_allclose(row[0].eigenvalues, expected.eigenvalues, atol=1e-10)
        assert "assembly_backend" not in row[0].metadata


def test_workflow_falls_back_when_only_requested_snapshots_fit_budget():
    native = hd.FilteredHyperdigraph(tuple(range(5)),
        [((i,), float(i)) for i in range(5)] + [((0, 1), 1.)])
    result = laplacian_series(Topology("hyperdigraph", native), max_dimension=0,
                              scales=[0., 1.], max_dense_entries=4)
    assert result[1.][0].eigenvalues.shape == (2,)
    assert "assembly_backend" not in result[1.][0].metadata


def test_mixed_degree_series_preserves_l1_l2_l3_and_persistent_paths(monkeypatch):
    vertices = tuple(range(5))
    records = [(edge, float(len(edge)-1)) for size in range(1, 5)
               for edge in itertools.combinations(vertices, size)]
    native = hd.FilteredHyperdigraph(vertices, records)
    topology = Topology("hyperdigraph", native, metadata={"max_analysis_dimension": 3})
    results = laplacian_series(topology, max_dimension=3, scales=[0., 1., 2., 3.], return_matrix=True)
    for scale, row in results.items():
        assert row[0].metadata["assembly_backend"] == "incremental_l0"
        for q in range(4):
            expected = hd.laplacian(native, q, scale=scale, backend="reference", return_matrix=True)
            np.testing.assert_allclose(row[q].eigenvalues, expected.eigenvalues, atol=1e-10)
            np.testing.assert_allclose(row[q].matrix, expected.matrix, atol=1e-10)
            if q:
                assert "assembly_backend" not in row[q].metadata
    def forbidden(*args, **kwargs):
        raise AssertionError("two-scale persistent operators must not use ordinary L0 sweep")
    monkeypatch.setattr(hd.L0Sweep, "__init__", forbidden)
    paired = laplacian_series(topology, max_dimension=3, mode="persistent", scale_pairs=[(0., 3.), (1., 3.)])
    for (start, end), row in paired.items():
        for q in range(4):
            expected = hd.persistent_laplacian(native, q, start=start, end=end, backend="reference")
            np.testing.assert_allclose(row[q].eigenvalues, expected.eigenvalues, atol=1e-10)
            assert row[q].kind == "persistent"


def test_explicit_reference_backend_bypasses_incremental_workflow():
    native = hd.FilteredHyperdigraph((0, 1), [((0, 1), 1.)], include_all_vertices=True)
    results = laplacian_series(Topology("hyperdigraph", native), max_dimension=0,
                              scales=[0., 1.], backend="reference")
    assert all("assembly_backend" not in row[0].metadata for row in results.values())
