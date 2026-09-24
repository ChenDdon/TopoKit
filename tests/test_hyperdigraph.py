import itertools
import numpy as np
import pytest

from topokit import PointCloud, ResourceLimitError
from topokit.core import hyperdigraph as hd
from topokit.builders import hyperdigraph as hbuild
from topokit.core._hyperdigraph.spectral import ordinary_operator, pairwise_operator, reference_operator


def test_uniform_weights_reciprocal_and_raw_clamped_births():
    cloud = PointCloud([[0, 0], [2, 0], [9, 0]], ids=["a", "b", "c"])
    obj = hbuild.from_points(cloud, filtration_start=5)
    assert obj.native.snapshot(5).directed_hyperedges(1) == (("a", "b"), ("b", "a"))
    assert all(obj.native.birth_of((v,)) == 5 for v in cloud.ids)
    assert obj.native.birth_of(("a", "b", "c")) == 7
    assert dict(obj.metadata["raw_births"])[("a", "b")] == 2
    assert obj.metadata["hyperedge_counts"] == (3, 4, 2, 0)
    result = hd.persistence(obj)
    assert result.betti_at(5) == hd.homology(obj, scale=5).betti_numbers
    with pytest.raises(ValueError):
        result.betti_at(4.9)
    with pytest.raises(ValueError):
        hd.laplacian(obj, scale=4.9)


def test_direction_uses_weights_and_exact_ties_only():
    obj = hbuild.from_points(PointCloud([[0, 0], [1, 0], [3, 0]], weights=[1, 1, 2]))
    assert set(obj.native.snapshot(3).directed_hyperedges(1)) == {(0, 1), (1, 0), (1, 2)}
    near = hbuild.from_points(PointCloud([[0, 0], [1, 0]], weights=[1, 1 + 1e-14]))
    assert near.native.snapshot(2).directed_hyperedges(1) == ((0, 1),)


def test_digraph_is_explicit_one_dimensional_not_expanded():
    graph = hbuild.from_digraph([(0, 1), (1, 2), (0, 2)], vertices=[0, 1, 2])
    assert graph.directed_hyperedges(2) == ()
    assert hd.homology(graph).betti_numbers == (1, 1, 0)
    reciprocal = hbuild.from_digraph([(0, 1), (1, 0)])
    assert hd.homology(reciprocal, 1).betti_numbers == (1, 1)
    assert hd.laplacian(reciprocal, 1).nullity == 1


def _assert_operator_matches_reference(obj, q):
    auto = ordinary_operator(obj, q)
    ref = reference_operator(obj, q)
    a = auto.matrix.matmat(np.eye(auto.matrix.shape[0]))
    b = ref.matrix.matmat(np.eye(ref.matrix.shape[0]))
    np.testing.assert_allclose(a, a.T, atol=1e-9)
    np.testing.assert_allclose(np.linalg.eigvalsh(a), np.linalg.eigvalsh(b), atol=2e-8)
    # Compare the actual ambient operators, not just spectral coincidences.
    aq, bq = auto.basis.toarray(), ref.basis.toarray()
    np.testing.assert_allclose(aq @ a @ aq.T, bq @ b @ bq.T, atol=2e-8)
    if len(a):
        assert np.linalg.eigvalsh(a)[0] > -2e-8
    np.testing.assert_allclose(aq.T @ aq, np.eye(aq.shape[1]), atol=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_general_explicit_hyperdigraph_operators_match_forced_real_svd(seed):
    rng = np.random.default_rng(seed)
    vertices = tuple(range(4))
    candidates = [edge for length in range(1, 5) for edge in itertools.permutations(vertices, length)]
    edges = tuple(edge for edge in candidates if rng.random() < 0.45)
    obj = hd.Hyperdigraph(vertices, edges)
    for q in range(3):
        _assert_operator_matches_reference(obj, q)


@pytest.mark.parametrize("seed", range(5))
def test_reciprocal_small_clouds_match_reference_at_multiple_scales(seed):
    cloud = PointCloud(np.random.default_rng(seed).normal(size=(5, 3)))
    obj = hbuild.from_points(cloud)
    times = obj.native.thresholds()
    for scale in (times[len(times) // 2], times[-1]):
        for q in range(3):
            _assert_operator_matches_reference(obj.native.snapshot(scale), q)


def test_labeled_basis_eigenvectors_and_residuals():
    # The missing shortcut makes Omega1 a nontrivial combination of two edges.
    obj = hd.Hyperdigraph((0, 1), ((0, 1), (1, 0)))
    result = hd.laplacian(obj, 1, return_eigenvectors=True, return_matrix=True)
    assert len(result.basis) == 1
    assert {edge for edge, _ in result.basis[0]} == {(0, 1), (1, 0)}
    assert result.metadata["basis_kind"] == "orthonormal_combinations"
    np.testing.assert_allclose(result.matrix @ result.eigenvectors,
                               result.eigenvectors * result.eigenvalues, atol=1e-10)
    assert result.metadata["eigen_residual_max"] < 1e-9


def test_pairwise_exact_restriction_matches_reference_and_barcode_rank():
    obj = hd.FilteredHyperdigraph((0, 1, 2),
        (((0, 1), 1), ((1, 2), 1), ((0, 2), 2), ((0, 1, 2), 3)),
        include_all_vertices=True)
    for start, end in ((0, 1), (1, 2), (2, 2), (2, 3)):
        for q in range(3):
            got = hd.persistent_laplacian(obj, q, start=start, end=end, return_eigenvectors=True)
            ref = hd.persistent_laplacian(obj, q, start=start, end=end, backend="reference")
            np.testing.assert_allclose(got.eigenvalues, ref.eigenvalues, atol=1e-9)
            # This fixture is torsion-free; do not generalize GF(2)=R.
            barcode = hd.persistence(obj)
            expected = sum(x.dimension == q and x.birth <= start and x.death > end for x in barcode.intervals)
            assert got.nullity == expected
            assert got.kind == "persistent"
            assert got.metadata["operator_kind"] == "pairwise_persistent"


def test_resource_refusal_and_partial_result_are_explicit():
    obj = hbuild.from_points(PointCloud([[0, 0], [1, 0], [2, 0], [3, 0]]))
    with pytest.raises(ResourceLimitError):
        hbuild.from_points(obj.cloud, max_hyperedges=3)
    with pytest.raises(ResourceLimitError):
        hd.laplacian(obj, 0, max_dense_entries=1)
    partial = hd.laplacian(obj, 0, k=1)
    assert not partial.complete
    assert partial.nullity is None
    with pytest.raises(ValueError):
        hd.homology(obj, max_dimension=3)


def test_singleton_and_invalid_inputs():
    obj = hbuild.from_points(PointCloud([[2, 4]], ids=["one"]), filtration_start=3)
    assert hd.homology(obj).betti_numbers == (1, 0, 0)
    assert hd.laplacian(obj, 2).eigenvalues.shape == (0,)
    with pytest.raises(ValueError):
        hbuild.from_points(PointCloud([[0, 0], [1, 1]]), filtration_start=-1)
    with pytest.raises(ValueError):
        hbuild.from_points(PointCloud([[0, 0], [0, 0]]))


@pytest.mark.parametrize("field", [2, np.int64(2), "GF(2)", "F2"])
def test_shared_field_argument_accepts_only_the_supported_gf2_aliases(field):
    obj = hbuild.from_points(PointCloud([[0, 0], [1, 0]]))
    assert hd.homology(obj, field=field).field == "GF(2)"
    assert hd.persistence(obj, field=field).field == "GF(2)"


@pytest.mark.parametrize("field", [3, 5, "R", "real", "GF(3)", True, False, None, 2.0])
def test_unsupported_coefficient_fields_are_not_silently_ignored(field):
    obj = hbuild.from_points(PointCloud([[0, 0], [1, 0]]))
    with pytest.raises(ValueError, match=r"only GF\(2\)"):
        hd.homology(obj, field=field)
    with pytest.raises(ValueError, match=r"only GF\(2\)"):
        hd.persistence(obj, field=field)


def test_shared_units_and_spectral_metadata_contract():
    cloud = PointCloud([[0, 0], [1, 0]], metadata={"coordinate_units": "angstrom"})
    obj = hbuild.from_points(cloud)
    bars = hd.persistence(obj, field=2)
    assert bars.field == "GF(2)"
    assert bars.metadata["scale_units"] == "coordinate_distance"
    assert bars.metadata["coordinate_units"] == "angstrom"
    spectrum = hd.laplacian(obj, return_eigenvectors=True)
    assert spectrum.metadata["coefficient_field"] == "R"
    assert spectrum.metadata["scale_units"] == "coordinate_distance"
    assert spectrum.metadata["residual_max"] < 1e-9
    assert spectrum.metadata["tolerance"] == 1e-10
    assert hd.homology(obj.native).metadata["scale_units"] == "user_defined"
    assert hd.persistence(obj.native).metadata["scale_units"] == "user_defined"
    assert hd.laplacian(obj.native.snapshot(1)).metadata["scale_units"] == "user_defined"


def test_sparse_product_resource_preflight_catches_fill_before_assembly():
    # Five triples share absent edge (0,1). The sparse Omega basis has 14
    # entries, but its embedded down boundary has 28, exceeding this budget.
    vertices = tuple(range(7))
    edges = tuple((a, c) for c in range(2, 7) for a in (0, 1))
    triples = tuple((0, 1, c) for c in range(2, 7))
    obj = hd.Hyperdigraph(vertices, edges + triples, include_all_vertices=True)
    with pytest.raises(ResourceLimitError, match="structural output bound"):
        hd.laplacian(obj, dimension=2, max_sparse_entries=20)
    filtration = obj.as_filtered(0)
    # Unequal query times with identical snapshots must be guarded too.
    with pytest.raises(ResourceLimitError, match="structural output bound"):
        hd.persistent_laplacian(filtration, dimension=2, start=0, end=1,
                               max_sparse_entries=20)


def test_analysis_core_has_no_construction_shortcuts():
    from topokit.core import _hyperdigraph as native
    assert not hasattr(hd, "from_points")
    assert not hasattr(hd, "from_digraph")
    assert not hasattr(native, "hyperdigraph_from_adjacency_matrix")
    assert not hasattr(native, "build_point_cloud_support")
    assert not hasattr(native.ScoreDirectedEdgeFiltration, "from_point_cloud")
    assert callable(hbuild.from_points)
    assert callable(hbuild.from_digraph)
