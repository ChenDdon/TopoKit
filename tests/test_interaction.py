"""Public two-factor contracts, geometry-to-identity mapping, and spectra."""
from itertools import combinations
from math import sqrt
import numpy as np
import pytest

from topokit import PointCloud, ResourceLimitError, Topology
from topokit.core import interaction
from topokit.builders import interaction as interaction_builder, simplicial as simplicial_builder
from topokit.core import _interaction as core


def triangle():
    return {simplex: 0.0 for size in (1, 2, 3)
            for simplex in combinations(("a", "b", "c"), size)}


def assert_eigenpairs(result):
    matrix = result.matrix.toarray()
    np.testing.assert_allclose(matrix @ result.eigenvectors,
                               result.eigenvectors * result.eigenvalues, atol=1e-9)
    np.testing.assert_allclose(result.eigenvectors.T @ result.eigenvectors,
                               np.eye(len(result.eigenvalues)), atol=1e-9)
    assert len(result.basis) == matrix.shape[0]


def test_subset_versus_whole_keeps_whole_factor_and_stable_ids():
    whole = PointCloud([[0, 0], [2, 0], [1, sqrt(3)], [1, sqrt(3) / 3]],
                       ids=("b0", "b1", "b2", "other"), weights=(1, 2, 3, 4))
    subset = whole.subset(("b2", "b0", "b1"))
    obj = interaction_builder.from_points(subset, whole,
          overlap_pairs=[(label, label) for label in subset.ids])
    first, second = obj.native.factors
    assert first.number_of_simplices == 7
    assert second.number_of_simplices == 13
    assert len(obj.metadata["factor_to_global"][0]) == 3
    assert len(obj.metadata["factor_to_global"][1]) == 4
    shared = set(obj.metadata["factor_to_global"][0].values())
    other = obj.metadata["factor_to_global"][1]["other"]
    assert other not in shared
    assert any(other in simplex for simplex in second.simplices)
    assert all(set(obj.native.interaction_vertices(key)).issubset(shared)
               for layer in obj.native.degrees for key in layer.keys)
    raw0, raw1 = obj.metadata["raw_factor_filtrations"]
    # Same shared edge can have different grades in subset/full alpha.
    edge0 = next(birth for simplex, birth in raw0 if set(simplex) == {"b0", "b1"})
    edge1 = next(birth for simplex, birth in raw1 if set(simplex) == {"b0", "b1"})
    assert edge0 == pytest.approx(1)
    assert edge1 == pytest.approx(4 / 3)
    assert obj.metadata["coupling"] == "maximum_factor_birth"


def test_self_mode_is_exactly_two_identical_factors_and_ignores_weights():
    points = [[0, 0], [2, 0], [0, 2]]
    a = interaction_builder.from_points(PointCloud(points, ids=("x", "y", "z"), weights=(1, 2, 3)))
    b = interaction_builder.from_points(PointCloud(points, ids=("x", "y", "z"), weights=(100, -2, 0)))
    assert len(a.native.factors) == 2
    assert a.metadata["overlap_count"] == 3
    assert a.native.factors[0].simplices == a.native.factors[1].simplices
    np.testing.assert_allclose(interaction.persistence(a).as_array(),
                               interaction.persistence(b).as_array())
    assert not a.metadata["point_weights_used"]


@pytest.mark.parametrize("pairs", [None, [("a", "x"), ("a", "y")],
    [("a", "x"), ("b", "x")], [("absent", "x")], [("a",)], [(True, "x")]])
def test_invalid_overlap_is_rejected(pairs):
    a = PointCloud([[0], [1]], ids=("a", "b"))
    b = PointCloud([[0], [1]], ids=("x", "y"))
    with pytest.raises(ValueError):
        interaction_builder.from_points(a, b, overlap_pairs=pairs)


def test_same_local_labels_do_not_infer_overlap():
    a = PointCloud([[0], [1]], ids=(0, 1))
    obj = interaction_builder.from_points(a, a, overlap_pairs=[])
    assert obj.native.number_of_cells() == 0
    assert obj.metadata["empty_overlap"]
    assert interaction.homology(obj).betti_numbers == (0, 0, 0)
    spectrum = interaction.laplacian(obj, return_eigenvectors=True, return_matrix=True)
    assert spectrum.eigenvectors.shape == (0, 0)
    assert spectrum.nullity == 0


def test_start_clamps_all_cells_and_preserves_raw_births():
    obj = interaction_builder.from_points([[0], [1]], filtration_start=1.0)
    assert all(birth == 1 for factor in obj.native.factors for birth in factor.births)
    assert {birth for _, birth in obj.metadata["raw_factor_filtrations"][0]} == {0, 0.25}
    assert interaction.homology(obj, scale=1).betti_numbers == (0, 1, 0)
    assert all(item.at_initial_stage for item in interaction.persistence(obj).intervals)
    with pytest.raises(ValueError):
        interaction.homology(obj, scale=0.5)


def test_independent_factor_dimensions_and_builder_options():
    points = [[0, 0], [1, 0], [0, 1]]
    obj = interaction_builder.from_points(points, factor_max_dimensions=(1, 2),
                                  factor_options=({"geometry_tolerance": 1e-11}, {}))
    assert tuple(f.max_dimension for f in obj.native.factors) == (1, 2)
    assert obj.metadata["factor_options"][0]["geometry_tolerance"] == 1e-11
    with pytest.raises(ValueError):
        interaction_builder.from_points(points, factor_options=({}, {"complex_type": "rips"}))
    with pytest.raises(ValueError):
        interaction_builder.from_points(points, factor_options=({}, {}, {}))


def test_h2_l2_and_labelled_eigenvectors():
    obj = interaction_builder.from_complexes(triangle(), triangle())
    assert interaction.homology(obj).betti_numbers == (0, 0, 1)
    result = interaction.laplacian(obj, 2, return_eigenvectors=True, return_matrix=True)
    assert result.nullity == 1
    assert result.matrix.shape == (15, 15)
    assert obj.native.max_cell_degree == 3
    assert_eigenpairs(result)
    assert all(label in {"a", "b", "c"} for pair in result.basis
               for simplex in pair for label in simplex)
    partial = interaction.laplacian(obj, 2, k=2, return_eigenvectors=True)
    assert partial.eigenvectors.shape == (15, 2)
    assert partial.nullity is None and not partial.complete


def test_genuine_two_time_projection_and_source_basis():
    first = {(v,): 0 for v in range(4)}
    first.update({edge: 0 for edge in ((0, 1), (0, 3))})
    first.update({edge: 1 for edge in ((0, 2), (1, 2), (2, 3))})
    first.update({triangle: 1 for triangle in ((0, 1, 2), (0, 2, 3))})
    obj = interaction_builder.from_complexes(first, {(0,): 0}, max_dimension=1)
    result = interaction.persistent_laplacian(obj, 1, start=0, end=1,
                                              return_matrix=True, return_eigenvectors=True)
    np.testing.assert_allclose(result.matrix.toarray(), [[1.5, 0.5], [0.5, 1.5]], atol=1e-12)
    np.testing.assert_allclose(result.eigenvalues, [1, 2], atol=1e-12)
    assert result.kind == "persistent" and result.start == 0 and result.end == 1
    assert_eigenpairs(result)
    assert result.basis == (((0, 1), (0,)), ((0, 3), (0,)))


def test_factor_topologies_align_external_ids_not_local_rows():
    a = simplicial_builder.from_points(PointCloud([[0], [1]], ids=("x", "y")))
    b = simplicial_builder.from_points(PointCloud([[1], [2]], ids=("y", "z")))
    obj = interaction_builder.from_complexes(a, b)
    assert obj.metadata["overlap_pairs"] == (("y", "y"),)


def test_preflight_budget_precedes_interaction_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("interaction constructor must not be reached")
    monkeypatch.setattr(core, "build_interaction_chain_complex", forbidden)
    with pytest.raises(ResourceLimitError, match="upper bound"):
        interaction_builder.from_complexes(triangle(), triangle(), max_cells=1)


def test_operator_and_dimension_guards():
    obj = interaction_builder.from_complexes(triangle(), triangle())
    with pytest.raises(ResourceLimitError):
        interaction.laplacian(obj, 2, k=2, return_matrix=True, max_dense_entries=10)
    with pytest.raises(ResourceLimitError):
        interaction.laplacian(obj, 2, max_dense_entries=10)
    with pytest.raises(ResourceLimitError):
        interaction.laplacian(obj, 0, max_dense_bytes=1)
    with pytest.raises(ValueError):
        interaction.laplacian(obj, 3)
    with pytest.raises(ValueError):
        interaction.persistence(obj, field=3)
    with pytest.raises(ValueError):
        interaction.persistent_laplacian(obj, 1, start=1, end=0)
    factor = obj.native.factors[0]
    invalid = Topology("interaction", core.build_interaction_chain_complex((factor,) * 3, max_homology_dimension=2))
    with pytest.raises(ValueError, match="exactly two"):
        interaction.persistence(invalid)


def test_start_clamp_does_not_repair_invalid_explicit_filtration():
    invalid = {(0,): 2, (1,): 0, (0, 1): 1}
    with pytest.raises(ValueError, match="face is born after"):
        interaction_builder.from_complexes(invalid, {(0,): 0}, filtration_start=3)
    with pytest.raises(ValueError, match="finite real"):
        interaction_builder.from_complexes({(0,): True}, {(0,): 0})


def test_builder_never_runs_homology_or_spectral_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("construction must not run analysis")
    monkeypatch.setattr(core, "compute_persistence", forbidden)
    monkeypatch.setattr(core, "InteractionLaplacianEngine", forbidden)
    assert interaction_builder.from_points([[0], [1]]).kind == "interaction"
    assert interaction_builder.from_complexes(triangle(), triangle()).kind == "interaction"


def test_core_accepts_native_chain_without_builder_metadata():
    factor_builder = interaction.FactorComplexBuilder()
    factor_builder.insert((0,), -3)
    factor_builder.insert((1,), -3)
    factor_builder.insert((0, 1), -1, with_faces=False)
    factor = factor_builder.freeze()
    chain = core.build_interaction_chain_complex((factor, factor), max_homology_dimension=1)
    assert interaction.homology(chain, max_dimension=1).betti_numbers == (0, 1)
    result = interaction.laplacian(chain, 1, scale=-1,
                                   return_matrix=True, return_eigenvectors=True)
    assert_eigenpairs(result)
    assert result.metadata["input_mode"] == "explicit_chain"
    assert not hasattr(interaction, "from_points")
    assert not hasattr(interaction, "from_complexes")


def test_construct_and_analyze_workflow_preserves_native_scalar_grades():
    from topokit.workflows.interaction import compute_interaction_persistence
    factor_builder = interaction.FactorComplexBuilder()
    factor_builder.insert((0,), -3)
    factor_builder.insert((1,), -3)
    factor_builder.insert((0, 1), -1, with_faces=False)
    factor = factor_builder.freeze()
    answer = compute_interaction_persistence((factor, factor), max_dimension=1)
    assert [(i.dimension, i.birth, i.death) for i in answer.intervals] == [
        (0, -3, -1), (0, -3, -1), (1, -1, float("inf"))]
    with pytest.raises(ValueError, match="exactly two"):
        compute_interaction_persistence((factor,) * 3, max_dimension=1)
    with pytest.raises(ResourceLimitError):
        compute_interaction_persistence((factor, factor), max_dimension=1, max_cells=1)


def test_higher_overlap_is_unordered_validation_only_and_preserves_input_order():
    points_a = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    points_b = points_a[[2, 0, 1]].copy()
    a = PointCloud(points_a, ids=("a", "b", "c"))
    b = PointCloud(points_b, ids=("z", "x", "y"))
    pairs = [("a", "x"), ("b", "y"), ("c", "z")]
    plain = interaction_builder.from_points(a, b, overlap_pairs=pairs)
    annotated = interaction_builder.from_points(
        a, b, overlap_vertices=pairs,
        overlap_simplices=[(("b", "a"), ("x", "y")), (("c", "b", "a"), ("y", "x", "z"))])
    assert annotated.metadata["overlap_simplices_role"] == "validation_annotation_only"
    assert annotated.metadata["overlap_vertices"] == tuple(pairs)
    assert annotated.native.degrees == plain.native.degrees
    np.testing.assert_array_equal(interaction.persistence(annotated).as_array(),
                                  interaction.persistence(plain).as_array())
    np.testing.assert_array_equal(a.points, points_a)
    np.testing.assert_array_equal(b.points, points_b)
    assert a.ids == ("a", "b", "c") and b.ids == ("z", "x", "y")
    np.testing.assert_array_equal(annotated.cloud[1].points, points_b)
    assert annotated.cloud[1].ids == b.ids


@pytest.mark.parametrize("records", [
    [(("a",), ("x",))], [(("a", "a"), ("x", "y"))],
    [(("a", "b"), ("x", "missing"))], [(("a", "b"), ("x", "z"))],
    [(("a", "b"), ("x", "y", "z"))], [("ab", "xy")],
    [(("a", "b"), ("x", "y")), (("b", "a"), ("y", "x"))],
])
def test_invalid_higher_overlap_records_are_rejected(records):
    a = PointCloud([[0, 0], [1, 0], [0, 1]], ids=("a", "b", "c"))
    b = PointCloud([[0, 0], [1, 0], [0, 1]], ids=("x", "y", "z"))
    with pytest.raises(ValueError):
        interaction_builder.from_points(a, b, overlap_vertices=list(zip(a.ids, b.ids)),
                                         overlap_simplices=records)


def test_annotation_cannot_insert_alpha_triangle_missing_from_whole_factor():
    whole = PointCloud([[0, 0], [2, 0], [1, sqrt(3)], [1, sqrt(3) / 3]],
                       ids=("b0", "b1", "b2", "other"))
    subset = whole.subset(("b0", "b1", "b2"))
    with pytest.raises(ValueError, match="absent from an independently"):
        interaction_builder.from_points(subset, whole,
            overlap_vertices=[(label, label) for label in subset.ids],
            overlap_simplices=[(subset.ids, subset.ids)])


def test_overlap_alias_conflicts_and_self_mode_full_overlap():
    with pytest.raises(ValueError, match="only one"):
        interaction_builder.from_points([[0], [1]], [[0], [1]],
                                         overlap_pairs=[], overlap_vertices=[])
    with pytest.raises(ValueError, match="self-interaction"):
        interaction_builder.from_points([[0], [1]], overlap_vertices=[])
    obj = interaction_builder.from_points([[0], [1]], overlap_simplices=[((1, 0), (0, 1))])
    assert obj.metadata["overlap_vertices"] == ((0, 0), (1, 1))


def test_factor_flag_connections_cutoffs_and_weight_overrides_are_independent():
    cloud = PointCloud([[0, 0], [1, 0], [0, 1]], ids=("x", "y", "z"), weights=(1, 2, 3))
    obj = interaction_builder.from_points(cloud, factor_options=(
        {"complex_type": "flag", "bonds": [("x", "y"), ("x", "z"), ("y", "z")],
         "cutoff": 1.5, "weights": {"x": 4, "y": 5, "z": 6}},
        {"complex_type": "graph", "bonds": [("x", "y"), ("x", "z"), ("y", "z")],
         "cutoff": 1.0}))
    assert tuple(f.number_of_simplices for f in obj.native.factors) == (7, 5)
    assert tuple(f.max_dimension for f in obj.native.factors) == (2, 1)
    assert obj.metadata["factor_scale_units"] == ("edge_length", "edge_length")
    assert obj.metadata["factor_construction_metadata"][0]["connection_support"] == "supplied"
    assert tuple(obj.cloud[0].weights) == (4, 5, 6)
    assert tuple(obj.cloud[1].weights) == (1, 2, 3)
    assert tuple(cloud.weights) == (1, 2, 3)
    assert not obj.metadata["point_weights_used"]


def test_reused_schedule_preserves_independent_factor_births_and_exact_default():
    a = PointCloud([[0], [1]], ids=("x", "y"))
    b = PointCloud([[0], [2]], ids=("x", "y"))
    exact = interaction_builder.from_points(a, b, overlap_vertices=[("x", "x"), ("y", "y")])
    sampled = interaction_builder.from_points(a, b, overlap_vertices=[("x", "x"), ("y", "y")],
        filtration_a=(0, 0.25, 1), progression=(5, 8, 11))
    factors = sampled.native.factors
    assert factors[0].births[factors[0].simplex_id((0, 1))] == 8
    assert factors[1].births[factors[1].simplex_id((0, 1))] == 11
    assert sampled.metadata["filtration_b_reused"]
    assert sampled.metadata["factor_thresholds"] == ((0, 0), (0.25, 0.25), (1, 1))
    assert sampled.metadata["coupling"] == "maximum_first_observed_stage"
    assert sampled.metadata["filtration_exactness"] == "sampled_one_parameter_path"
    assert exact.metadata["filtration_exactness"] == "exact_common_scalar_domain"
    assert exact.native.factors[0].births[-1] == 0.25
    result = interaction.persistence(sampled)
    assert "predate" in result.metadata["initial_stage_semantics"]
    assert result.metadata["right_censored"]
    assert "not proven essentiality" in result.metadata["terminal_interval_semantics"]


@pytest.mark.parametrize("a,b,progression", [
    ([], None, None), ([0, 1], [0], None), ([1, 0], None, None),
    ([0, 1], [1, 0], None), ([0, 1], None, [0]),
    ([0, 1], None, [0, 0]), ([0, 1], None, [1, 0]),
    ([0, float("inf")], None, None), ([0, float("nan")], None, None),
    ([False, 1], None, None), ("01", None, None),
])
def test_paired_schedule_validation(a, b, progression):
    with pytest.raises(ValueError):
        interaction_builder.from_points([[0], [1]], filtration_a=a,
                                         filtration_b=b, progression=progression)


def test_schedule_requires_first_array_and_allows_holding_one_factor_fixed():
    with pytest.raises(ValueError, match="require filtration_a"):
        interaction_builder.from_points([[0], [1]], filtration_b=(0, 1))
    obj = interaction_builder.from_points([[0], [2]], filtration_a=(0, 0, 1),
                                          filtration_b=(0, 1, 1))
    assert obj.metadata["factor_thresholds"] == ((0, 0), (0, 1), (1, 1))
    obj.native.validate_boundary()
    obj.native.validate_signed_boundary()


def test_pair_snapshots_use_full_independent_ranges_without_scalar_extrapolation():
    from topokit.workflows.interaction import homology_at_pair
    a = PointCloud([[0], [2]], ids=("x", "y"))
    b = PointCloud([[0], [4]], ids=("x", "y"))
    obj = interaction_builder.from_points(a, b, overlap_vertices=[("x", "x"), ("y", "y")],
                                          factor_options=({"max_scale": 1}, {"max_scale": 4}))
    assert obj.metadata["factor_filtration_ranges"] == ((0, 1), (0, 4))
    assert obj.metadata["filtration_end"] == 1
    assert obj.native.factors[1].number_of_simplices == 2
    assert all(i.birth <= 1 for i in interaction.persistence(obj).intervals)
    result = homology_at_pair(obj, 1, 4)
    assert result.betti_numbers == (0, 1, 0)
    assert result.metadata["factor_scale_pair"] == (1, 4)
    assert obj.native.factors[1].number_of_simplices == 2  # no mutation
    with pytest.raises(ValueError, match="outside its constructed"):
        interaction_builder.snapshot_at_pair(obj, 1.01, 4)
    with pytest.raises(ValueError, match="outside its constructed"):
        interaction_builder.snapshot_at_pair(obj, 1, 4.01)
    with pytest.raises(ValueError):
        interaction.homology(obj, scale=4)


def test_factor_ranges_and_mixed_units_require_explicit_paired_progression():
    obj = interaction_builder.from_points([[0], [2]],
        factor_options=({"complex_type": "alpha", "filtration_range": (3, 4)},
                        {"complex_type": "rips", "filtration_range": (0, 2)}),
        filtration_a=(3, 4), filtration_b=(0, 2))
    assert obj.metadata["factor_filtration_ranges"] == ((3, 4), (0, 2))
    assert obj.metadata["factor_scale_units"] == ("squared_radius", "edge_length")
    assert obj.metadata["factor_thresholds"] == ((3, 0), (4, 2))
    with pytest.raises(ValueError, match="outside its constructed"):
        interaction_builder.regrade_pair_filtration(obj, (2, 4), (0, 2))
    with pytest.raises(ValueError, match="no common scalar domain"):
        interaction_builder.from_points([[0], [2]],
            factor_options=({"filtration_range": (3, 4)}, {"filtration_range": (0, 2)}))


def test_regrading_reuses_original_factor_grades_not_previous_quantization():
    obj = interaction_builder.from_points([[0], [2]])
    coarse = interaction_builder.regrade_pair_filtration(obj, (0, 2))
    fine = interaction_builder.regrade_pair_filtration(coarse, (0, 0.5, 1, 2))
    direct = interaction_builder.regrade_pair_filtration(obj, (0, 0.5, 1, 2))
    assert fine.native.degrees == direct.native.degrees
    assert fine.native.factors[0].births[-1] == 2  # original alpha birth 1, not coarse stage 1
    assert coarse.metadata["progression"] == (0, 1)


def test_pair_workflow_uses_genuine_persistent_projection_and_source_basis():
    from topokit.workflows.interaction import (laplacian_at_pair,
        persistent_laplacian_between_pairs, persistence_along_pairs)
    first = {(v,): 0 for v in range(4)}
    first.update({edge: 0 for edge in ((0, 1), (0, 3))})
    first.update({edge: 1 for edge in ((0, 2), (1, 2), (2, 3))})
    first.update({face: 1 for face in ((0, 1, 2), (0, 2, 3))})
    obj = interaction_builder.from_complexes(first, {(0,): 0}, max_dimension=1, max_scale=2)
    result = persistent_laplacian_between_pairs(obj, 1, source_pair=(0, 0), target_pair=(1, 2),
                                                return_matrix=True, return_eigenvectors=True)
    np.testing.assert_allclose(result.matrix.toarray(), [[1.5, 0.5], [0.5, 1.5]], atol=1e-12)
    assert result.basis == (((0, 1), (0,)), ((0, 3), (0,)))
    assert result.metadata["source_factor_scales"] == (0, 0)
    assert result.metadata["target_factor_scales"] == (1, 2)
    assert_eigenpairs(result)
    ordinary = laplacian_at_pair(obj, 0, 0, dimension=1, return_matrix=True, return_eigenvectors=True)
    assert_eigenpairs(ordinary)
    assert not np.allclose(result.matrix.toarray(), ordinary.matrix.toarray())
    bars = persistence_along_pairs(obj, (0, 1), (0, 2), max_dimension=1)
    assert bars.metadata["filtration_parameters"] == 1 and bars.metadata["right_censored"]
    with pytest.raises(ValueError, match="nondecreasing"):
        persistent_laplacian_between_pairs(obj, 1, source_pair=(0, 2), target_pair=(1, 1))
    with pytest.raises(ValueError, match="exactly two"):
        persistent_laplacian_between_pairs(obj, source_pair=(0,), target_pair=(1, 1))


def test_empty_early_factor_snapshot_is_a_valid_zero_chain_and_operator():
    from topokit.workflows.interaction import homology_at_pair, persistent_laplacian_between_pairs
    obj = interaction_builder.from_complexes({("x",): 2}, {("x",): 0}, max_scale=3)
    snapshot = interaction_builder.snapshot_at_pair(obj, 1, 1)
    assert snapshot.native.factors[0].number_of_simplices == 0
    assert snapshot.native.factors[0].max_dimension == -1
    assert snapshot.native.number_of_cells() == 0
    snapshot.native.validate_signed_boundary()
    assert homology_at_pair(obj, 1, 1).betti_numbers == (0, 0, 0)
    assert interaction.persistence(snapshot).intervals == ()
    result = persistent_laplacian_between_pairs(obj, source_pair=(1, 1), target_pair=(2, 2),
                                                return_matrix=True, return_eigenvectors=True)
    assert result.matrix.shape == (0, 0)
    assert result.eigenvectors.shape == (0, 0)
    assert result.nullity == 0
    assert homology_at_pair(obj, 2, 2).betti_numbers == (1, 0, 0)


def test_pair_construction_guard_runs_before_interaction_allocation(monkeypatch):
    obj = interaction_builder.from_complexes(triangle(), triangle())
    def forbidden(*args, **kwargs):
        raise AssertionError("interaction cells must not be allocated")
    monkeypatch.setattr(core, "build_interaction_chain_complex", forbidden)
    with pytest.raises(ResourceLimitError, match="upper bound"):
        interaction_builder.regrade_pair_filtration(obj, (0, 1), max_cells=1)


def test_sampled_metadata_is_serializable_without_native_objects():
    import json
    obj = interaction_builder.from_points([[0], [2]], filtration_a=(0, 1))
    assert json.loads(json.dumps(obj.metadata))["factor_thresholds"] == [[0.0, 0.0], [1.0, 1.0]]


@pytest.mark.parametrize("unordered", [{0, 1}, frozenset((0, 1)), {0: "zero", 1: "one"}])
def test_paired_thresholds_never_consume_unordered_container_keys(unordered):
    from topokit.workflows.interaction import persistent_laplacian_between_pairs
    obj = interaction_builder.from_points([[0], [2]])
    with pytest.raises(ValueError):
        interaction_builder.regrade_pair_filtration(obj, unordered)
    with pytest.raises(ValueError):
        interaction_builder.regrade_pair_filtration(obj, (0, 1), unordered)
    with pytest.raises(ValueError):
        interaction_builder.regrade_pair_filtration(obj, (0, 1), progression=unordered)
    with pytest.raises(ValueError):
        persistent_laplacian_between_pairs(obj, source_pair=unordered, target_pair=(1, 1))
    with pytest.raises(ValueError):
        persistent_laplacian_between_pairs(obj, source_pair=(0, 0), target_pair=unordered)


@pytest.mark.parametrize("pairs", [{0: 0, 1: 1}, {(0, 0), (1, 1)}, [{0, 1}]])
def test_vertex_correspondence_requires_ordered_pair_records(pairs):
    with pytest.raises(ValueError):
        interaction_builder.from_points([[0], [1]], [[0], [1]], overlap_vertices=pairs)


def test_explicit_factor_topology_domains_are_preserved_for_pair_queries():
    a = simplicial_builder.from_points(PointCloud([[0], [2]], ids=("x", "y")),
                                       filtration_range=(1, 2))
    b = simplicial_builder.from_points(PointCloud([[0], [4]], ids=("x", "y")),
                                       filtration_range=(0, 4))
    obj = interaction_builder.from_complexes(a, b)
    assert obj.metadata["filtration_range"] == (1, 2)
    assert obj.metadata["factor_filtration_ranges"] == ((1, 2), (0, 4))
    with pytest.raises(ValueError, match="outside its constructed"):
        interaction_builder.snapshot_at_pair(obj, 0.5, 1)
    final = interaction_builder.snapshot_at_pair(obj, 2, 4)
    assert interaction.homology(final).betti_numbers == (0, 1, 0)
    with pytest.raises(ValueError):
        interaction.homology(obj, scale=0.5)


def test_shared_filtration_range_clamps_all_cells_for_exact_and_sampled_routes():
    exact = interaction_builder.from_points([[0], [1]], filtration_range=(1, 2))
    assert exact.metadata["factor_filtration_ranges"] == ((1, 2), (1, 2))
    assert all(birth == 1 for factor in exact.native.factors for birth in factor.births)
    sampled = interaction_builder.from_points([[0], [1]], filtration_range=(1, 2),
                                              filtration_a=(1, 2))
    assert all(birth == 0 for factor in sampled.native.factors for birth in factor.births)
    assert sampled.metadata["factor_thresholds"] == ((1, 1), (2, 2))
