"""Dataset-fitted barcode bins, scientific schemas, and ragged grid contracts."""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import pytest

from topokit import PersistenceInterval, PersistenceResult, ResourceLimitError
from topokit.postprocessing import FeatureResult, PersistenceVectorizer, histogram_features
from topokit.serialization import load_result, save_result
from topokit.workflows.ml import feature_matrix


def bars(records=((0, 0., 1.),), *, maximum=2, field="GF(2)", **metadata):
    source = {"route": "simplicial", "definition_id": "example-v1", "coordinate_units": "angstrom",
              "scale_units": "squared_radius", "filtration_start": 0., "filtration_end": math.inf,
              "start_policy": "clamp", "complex_type": "alpha", **metadata}
    return PersistenceResult(tuple(PersistenceInterval(*item) for item in records), field=field,
                             max_dimension=maximum, metadata=source)


def test_explicit_arrays_override_all_range_parameters_and_preserve_legacy_arrays():
    result = bars(((0, .2, 1.), (0, .4, math.inf), (1, 1., 2.)))
    args = {"birth_edges": [0, 1, 2], "death_edges": [0, 1, 2]}
    previous = histogram_features(result, **args)
    actual = histogram_features(result, min_value=math.nan, max_value=math.inf, step=-1, **args)
    np.testing.assert_array_equal(actual.values, previous.values)
    assert actual.finite_histograms.shape == (3, 2, 2)
    assert actual.essential_histograms.shape == (3, 2)
    assert actual.out_of_range.shape == (3, 2)
    assert actual.values.ndim == 1
    assert actual.grids[0].shape == (2, 2)


def test_range_bins_include_exact_max_and_default_to_recorded_start():
    source = bars(((0, 3, 5), (0, 5, math.inf)), filtration_start=3)
    result = histogram_features(source, max_value=5, step=.6)
    np.testing.assert_allclose(result.metadata["birth_edges"], [3, 3.6, 4.2, 4.8, 5])
    assert result.finite_histograms.sum() == result.essential_histograms.sum() == 1
    assert result.out_of_range.sum() == 0
    negative = histogram_features(bars(((0, -1, 1),), filtration_start=-2), max_value=1, step=1)
    np.testing.assert_array_equal(negative.metadata["birth_edges"], [0, 1])
    assert negative.out_of_range[0, 0] == 1
    override = histogram_features(source, min_value=0, max_value=5, step=1)
    assert override.metadata["birth_edges"][0] == 0


def test_one_result_function_does_not_silently_fit_an_individual_range():
    with pytest.raises(ValueError, match="finite.*range|max_value"):
        histogram_features(bars())


def test_fit_uses_global_training_maximum_freezes_schema_and_records_holdout_overflow():
    training = [bars(((0, 0, 1),)), bars(((0, 0, 2), (1, .5, 1.5)))]
    original = deepcopy(training)
    vectorizer = PersistenceVectorizer(step=.5).fit(iter(training))
    np.testing.assert_array_equal(vectorizer.birth_edges_, [0, .5, 1, 1.5, 2])
    assert vectorizer.fit_metadata_["fitted_finite_max"] == 2
    assert vectorizer.fit_metadata_["fit_scope"] == "training"
    assert vectorizer.n_samples_fit_ == 2
    matrix = vectorizer.transform(iter(training))
    assert matrix.shape == (2, 3 * (16 + 4 + 2))
    holdout = bars(((0, 0, 10), (0, 3, math.inf)))
    record = vectorizer.transform_results([holdout])[0]
    assert record.out_of_range[0].tolist() == [1, 1]
    assert record.values.shape == matrix[0].shape
    assert record.metadata["fitted_finite_max"] == 2
    assert record.metadata["learned_from_data"] is True
    assert record.metadata["source_metadata"] == holdout.metadata
    np.testing.assert_array_equal(vectorizer.birth_edges_, [0, .5, 1, 1.5, 2])
    assert training == original
    assert vectorizer.transform([]).shape == (0, matrix.shape[1])


def test_descriptive_scope_is_explicit_and_does_not_replace_training_default():
    training, full = [bars(((0, 0, 1),))], [bars(((0, 0, 1),)), bars(((0, 0, 3),))]
    fitted = PersistenceVectorizer(step=1).fit(training)
    descriptive = PersistenceVectorizer(step=1, fit_scope="descriptive").fit(full)
    assert fitted.birth_edges_[-1] == 1
    assert descriptive.birth_edges_[-1] == 3
    assert descriptive.transform_results(full)[0].metadata["fit_scope"] == "descriptive"
    with pytest.raises(ValueError, match="fit_scope"):
        PersistenceVectorizer(fit_scope="validation").fit(training)


def test_declared_finite_end_and_finite_essential_births_contribute_to_fit_range():
    finite_end = bars(((0, 0, math.inf),), filtration_end=4)
    fitted = PersistenceVectorizer(step=1).fit([finite_end])
    assert fitted.birth_edges_[-1] == 4
    assert fitted.transform_results([finite_end])[0].essential_histograms.sum() == 1
    born_late = bars(((0, 0, math.inf), (1, 2, math.inf)))
    fitted = PersistenceVectorizer(step=1).fit([born_late])
    assert fitted.birth_edges_[-1] == 2
    assert fitted.transform_results([born_late])[0].essential_histograms.sum() == 2
    empty_bounded = PersistenceVectorizer(step=1).fit([bars((), filtration_end=2)])
    assert empty_bounded.transform([bars((), filtration_end=2)]).sum() == 0


@pytest.mark.parametrize("result", [bars(()), bars(((0, 0, math.inf),)),
                                    bars(((0, 3, math.inf),), filtration_start=3)])
def test_no_positive_finite_range_requires_explicit_maximum(result):
    with pytest.raises(ValueError, match="finite|max_value"):
        PersistenceVectorizer(step=1).fit([result])
    fixed = PersistenceVectorizer(max_value=5, step=1).fit([result])
    assert fixed.transform([result]).shape == (1, fixed.n_features_out_)
    assert fixed.fit_metadata_["learned_from_data"] is False


def test_per_degree_arrays_and_range_specs_produce_ragged_grids_and_flat_vectors(tmp_path):
    result = bars(((0, .2, 1), (1, 1.1, 2), (1, 1.5, math.inf), (2, 0, 4)))
    birth = {0: [0, 2], 1: {"min_value": 0, "max_value": 2, "step": 1}, 2: [0, 1, 2, 3]}
    death = {0: {"edges": [0, 1, 2], "step": -1}, 1: [0, 1, 2], 2: [0, 3]}
    feature = histogram_features(result, birth_edges=birth, death_edges=death)
    assert isinstance(feature.finite_histograms, tuple)
    assert isinstance(feature.essential_histograms, tuple)
    assert [feature.grids[q].shape for q in (0, 1, 2)] == [(1, 2), (2, 2), (3, 1)]
    assert feature.values.size == (2 + 1 + 2) + (4 + 2 + 2) + (3 + 3 + 2)
    assert feature.values.sum() == 4
    assert feature.out_of_range[2, 0] == 1
    assert list(feature.metadata["birth_edges"]) == [0, 1, 2]
    vectorizer = PersistenceVectorizer(birth_edges=birth, death_edges=death).fit([result])
    np.testing.assert_array_equal(vectorizer.transform([result])[0], feature.values)
    grids = vectorizer.transform([result], output="grids")
    assert grids[0][1].shape == (2, 2)
    assert len(vectorizer.transform_results([result])) == 1
    restored = load_result(save_result(feature, tmp_path / "ragged.json"))
    assert isinstance(restored, FeatureResult)
    np.testing.assert_array_equal(restored.values, feature.values)
    for q in feature.grids:
        np.testing.assert_array_equal(restored.grids[q], feature.grids[q])
    np.testing.assert_array_equal(feature_matrix([restored, feature])[0], feature.values)


def test_per_degree_range_parameters_share_one_inferred_global_maximum():
    result = bars(((0, 0, 1), (1, 0, 3), (2, 1, math.inf)))
    fitted = PersistenceVectorizer(step={0: 1, 1: .5, 2: 1.5}).fit([result])
    assert [array[-1] for array in fitted.birth_edges_.values()] == [3, 3, 3]
    assert [len(array) for array in fitted.birth_edges_.values()] == [4, 7, 3]
    explicit = PersistenceVectorizer(max_value={0: 1, 1: 3, 2: 2}, step=1).fit([result])
    assert [array[-1] for array in explicit.birth_edges_.values()] == [1, 3, 2]
    assert explicit.fit_metadata_["learned_from_data"] is False


@pytest.mark.parametrize("key,other", [
    ("route", "hyperdigraph"), ("coordinate_units", "nm"), ("definition_id", "other-v2"),
    ("scale_units", "distance"), ("filtration_start", 1), ("filtration_end", 4),
    ("start_policy", "translate"), ("complex_type", "rips"), ("orientation", "opposite"),
    ("connection_support", "explicit"), ("cutoff_distance", 5), ("graph_expansion", "flag"),
    ("object_type", "digraph"), ("construction", "another-rule"),
])
def test_fit_and_transform_reject_mixed_scientific_schema(key, other):
    first, changed = bars(), bars(**{key: other})
    fitted = PersistenceVectorizer(max_value=2, step=1).fit([first])
    with pytest.raises(ValueError, match=key):
        fitted.transform([changed])
    with pytest.raises(ValueError, match=key):
        PersistenceVectorizer(max_value=2, step=1).fit([first, changed])
    left = histogram_features(first, max_value=2, step=1)
    right = histogram_features(changed, max_value=2, min_value=0, step=1)
    with pytest.raises(ValueError, match="schema"):
        feature_matrix([left, right])


def test_field_and_missing_declarations_are_not_silently_mixed():
    first, other = bars(), bars(field="GF(3)")
    fitted = PersistenceVectorizer(max_value=2, step=1).fit([first])
    with pytest.raises(ValueError, match="field"):
        fitted.transform([other])
    unknown = replace(first, metadata={k: v for k, v in first.metadata.items() if k != "coordinate_units"})
    with pytest.raises(ValueError, match="coordinate_units"):
        fitted.transform([unknown])
    with pytest.raises(ValueError, match="disagrees"):
        fitted.transform([bars(coefficient_field="GF(3)")])


def test_source_provenance_can_differ_and_fit_schema_is_independent_of_mutation():
    first = bars(source_file="one", connection_pairs=((0, 1),), retained_connection_count=1)
    second = bars(source_file="two", connection_pairs=((1, 2), (2, 3)), retained_connection_count=2)
    birth = {0: [0, 1, 2], 1: [0, 2], 2: [0, 1, 2, 3]}
    fitted = PersistenceVectorizer(birth_edges=birth, death_edges=[0, 1, 2]).fit([first, second])
    before = fitted.transform([first])
    birth[0][1] = .25
    fitted.birth_edges_[0] = np.array([0, 4, 5])
    fitted.schema_["source"]["coordinate_units"] = "changed-public-copy"
    fitted.fit_metadata_["fit_scope"] = "changed-public-copy"
    np.testing.assert_array_equal(fitted.transform([first]), before)
    record = fitted.transform_results([second])[0]
    assert record.metadata["fit_scope"] == "training"
    assert record.metadata["source_metadata"]["source_file"] == "two"
    record.metadata["schema"]["source"]["coordinate_units"] = "mutated-result"
    assert fitted.transform([first]).shape == before.shape
    fitted.set_params(step=2)
    with pytest.raises(RuntimeError, match="fit"):
        fitted.transform([first])


@pytest.mark.parametrize("options", [
    {"max_value": math.inf}, {"max_value": math.nan}, {"max_value": True},
    {"max_value": 0}, {"step": 0}, {"step": -1}, {"step": math.inf}, {"step": True},
    {"birth_edges": {0: [0, 1]}}, {"step": {0: 1}}, {"birth_edges": {True: [0, 1]}},
])
def test_invalid_or_incomplete_bins_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        histogram_features(bars(), **{"max_value": 2, "step": 1, **options})


@pytest.mark.parametrize("record", [(0, math.nan, 1), (0, math.inf, math.inf), (0, 2, 1),
                                    (0, 0, -math.inf), (0, 0, math.nan), (3, 0, 1), (True, 0, 1)])
def test_invalid_interval_records_are_not_silently_discarded(record):
    with pytest.raises(ValueError):
        histogram_features(bars([record]), max_value=2, step=1)


def test_histogram_memory_guard_precedes_allocation(monkeypatch):
    def no_histogram(*args, **kwargs):
        pytest.fail("histogram allocation should not be reached")
    monkeypatch.setattr(np, "histogram2d", no_histogram)
    with pytest.raises(ResourceLimitError, match="feature count"):
        histogram_features(bars(), max_value=10, step=.1, max_features=1000)
    with pytest.raises(ResourceLimitError, match="edge count"):
        histogram_features(bars(), max_value=1e200, step=1e-200)


def test_normalization_conserves_separate_channels_for_ragged_grids():
    record = bars(((0, 0, 1), (0, 2, 4), (0, 0, math.inf), (0, 4, math.inf),
                   (1, 0, 1), (1, 0, .01)))
    feature = histogram_features(record, birth_edges={0: [0, 2], 1: [0, 1, 2]},
                                 death_edges=[0, 2], dimensions=(0, 1), min_persistence=.1, normalize=True)
    assert feature.grids[0].sum() == .25
    assert feature.essential_histograms[0].sum() == .25
    np.testing.assert_array_equal(feature.out_of_range[0], [.25, .25])
    assert feature.grids[1].sum() == 1
    assert feature.values.sum() == 2
    assert len(record.intervals) == 6


def paired_bars(**overrides):
    return bars(filtration_a=(0., 1., 2.), filtration_b=(0., .5, 1.), progression=(0., 1., 2.),
        **{"filtration_end": 2., "scale_units": "progression_stage", "route": "interaction",
           "factor_filtration_ranges": ((0., 3.), (0., 3.)), "factor_scale_units": ("squared_radius", "squared_radius"),
           "filtration_exactness": "sampled_one_parameter_path", "factor_max_dimensions": (3, 3),
           "factor_coordinate_units": ("angstrom", "angstrom"), **overrides})


@pytest.mark.parametrize("key,value", [
    ("factor_filtration_ranges", ((0., 5.), (0., 3.))),
    ("factor_scale_units", ("squared_radius", "distance")),
    ("filtration_exactness", "exact_common_scalar_domain"), ("factor_max_dimensions", (2, 3)),
    ("factor_coordinate_units", ("angstrom", "nm")),
])
def test_paired_interaction_physical_semantics_are_part_of_feature_schema(key, value):
    first = paired_bars()
    other = paired_bars(**{key: value})
    fitted = PersistenceVectorizer(step=1).fit([first])
    with pytest.raises(ValueError, match=key):
        fitted.transform([other])
    with pytest.raises(ValueError, match=key):
        PersistenceVectorizer(step=1).fit([first, other])


@pytest.mark.parametrize("key,value", [("filtration_a", (0., 2., 4.)),
                                       ("filtration_b", (0., 1., 2.)), ("progression", (0., 2., 3.))])
def test_equal_stage_ids_do_not_equate_different_paired_physical_paths(key, value):
    first = paired_bars()
    other = replace(first, metadata={**first.metadata, key: value})
    fitted = PersistenceVectorizer(step=1).fit([first])
    with pytest.raises(ValueError, match=key):
        fitted.transform([other])
    with pytest.raises(ValueError, match=key):
        PersistenceVectorizer(step=1).fit([first, other])


def test_factor_construction_schemas_are_sanitized_and_share_spectral_semantics():
    from topokit import SpectrumResult
    from topokit.postprocessing._barcode_bins import source_schema
    options = ({"complex_type": "alpha", "max_dimension": 3, "bonds": [(0, 1)], "max_simplices": 100},) * 2
    construction = ({"complex_type": "alpha", "connection_support": "delaunay", "geometry_weighted": False,
                     "cutoff_distance": None, "raw_filtration": (((0,), 0.),), "point_ids": [0, 1]},) * 2
    first = paired_bars(factor_options=options, factor_construction_metadata=construction)
    second = paired_bars(
        factor_options=tuple({**item, "bonds": [(2, 3)], "max_simplices": 1000} for item in options),
        factor_construction_metadata=tuple({**item, "raw_filtration": (((4,), 5.),),
                                           "point_ids": [2, 3]} for item in construction))
    fitted = PersistenceVectorizer(step=1).fit([first, second])
    assert fitted.transform([second]).shape[0] == 1
    schema = fitted.schema_["source"]
    assert "bonds" not in schema["factor_options"][0]
    assert "max_simplices" not in schema["factor_options"][0]
    assert "raw_filtration" not in schema["factor_construction_metadata"][0]
    spectral = SpectrumResult(0, np.array([0., 1.]), metadata=first.metadata)
    assert source_schema(spectral, field="R") == {**schema, "field": "R"}
    for key, value in [("complex_type", "rips"), ("geometry_tolerance", 1e-4)]:
        changed = replace(second, metadata={**second.metadata,
            "factor_options": ({**options[0], key: value}, options[1])})
        with pytest.raises(ValueError, match="factor_options"):
            fitted.transform([changed])
    for key, value in [("connection_support", "supplied"), ("cutoff_distance", 3), ("graph_expansion", "flag")]:
        changed = replace(second, metadata={**second.metadata,
            "factor_construction_metadata": ({**construction[0], key: value}, construction[1])})
        with pytest.raises(ValueError, match="factor_construction_metadata"):
            fitted.transform([changed])
