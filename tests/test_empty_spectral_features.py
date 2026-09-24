"""Fixed-width spectral features for structurally absent chain groups."""

import numpy as np
import pytest

from topokit import builders, core
from topokit.postprocessing import summarize_spectra, summarize_spectrum
from topokit.results import SpectrumResult
from topokit.serialization import load_result, save_result
from topokit.workflows.ml import feature_matrix


STATISTICS = (
    "min", "max", "mean", "std", "median", "sum", "count",
    "zero_count", "positive_count", "energy", "spectral_entropy",
    "variance", "moment_2", "laplacian_energy",
)


def _absent_spectrum(dimension=2, **overrides):
    options = {
        "dimension": dimension,
        "eigenvalues": np.empty(0),
        "basis": (),
        "nullity": 0,
        "kind": "persistent",
        "start": 0.0,
        "end": 1.0,
        "metadata": {
            "operator_dimension": 0,
            "source_chain_dimension": 0,
            "structural_absence": True,
        },
    }
    options.update(overrides)
    return SpectrumResult(**options)


def test_complete_empty_operator_has_explicit_auditable_zero_fill():
    eigenvalues = np.empty(0)
    result = _absent_spectrum(eigenvalues=eigenvalues)

    preserved = summarize_spectrum(result, statistics=STATISTICS)
    assert np.isnan(preserved.values[:5]).all()
    assert np.isnan(preserved.values[11:13]).all()
    np.testing.assert_array_equal(
        preserved.values[[5, 6, 7, 8, 9, 10, 13]], 0.0
    )
    assert preserved.metadata["empty_operator_policy"] == "preserve"
    assert preserved.metadata["structural_absence"]
    assert preserved.metadata["zero_filled_statistics"] == ()

    filled = summarize_spectrum(
        result, statistics=STATISTICS, empty_operator_policy="zero"
    )
    np.testing.assert_array_equal(filled.values, np.zeros(len(STATISTICS)))
    assert filled.names == tuple(f"L2:{name}" for name in STATISTICS)
    assert filled.metadata["empty_operator_policy"] == "zero"
    assert filled.metadata["structural_absence"]
    assert filled.metadata["operator_dimension"] == 0
    assert filled.metadata["source_chain_dimension"] == 0
    assert filled.metadata["zero_filled_statistics"] == STATISTICS
    assert filled.metadata["nonfinite_statistics"] == ()
    sequence = summarize_spectra(
        (result,), statistics=STATISTICS, empty_operator_policy="zero"
    )
    np.testing.assert_array_equal(sequence[0].values, filled.values)
    np.testing.assert_array_equal(result.eigenvalues, eigenvalues)


def test_zero_policy_keeps_custom_callbacks_authoritative_on_empty_operator():
    calls = []

    def defined_on_empty(values):
        calls.append(values.copy())
        return 7.0

    result = _absent_spectrum(dimension=1)
    custom = summarize_spectrum(
        result,
        statistics={"custom": defined_on_empty},
        empty_operator_policy="zero",
    )
    np.testing.assert_array_equal(custom.values, [7.0])
    assert len(calls) == 1 and calls[0].shape == (0,)
    assert custom.metadata["structural_absence"] is True
    assert custom.metadata["zero_filled_statistics"] == ()

    unconfirmed = summarize_spectrum(
        SpectrumResult(1, np.empty(0)),
        statistics={"custom": defined_on_empty},
        empty_operator_policy="zero",
    )
    np.testing.assert_array_equal(unconfirmed.values, [7.0])
    assert len(calls) == 2 and calls[1].shape == (0,)
    assert unconfirmed.metadata["structural_absence"] is False
    assert unconfirmed.metadata["zero_filled_statistics"] == ()

    class UnhashableCallback:
        __hash__ = None

        def __call__(self, values):
            return 3.0

    unhashable = summarize_spectrum(
        result,
        statistics={"unhashable": UnhashableCallback()},
        empty_operator_policy="zero",
    )
    np.testing.assert_array_equal(unhashable.values, [3.0])

    with pytest.raises(ValueError, match="real scalar"):
        summarize_spectrum(
            result,
            statistics={"invalid": lambda values: np.ones(2)},
            empty_operator_policy="zero",
        )


def test_zero_fill_does_not_mask_nonempty_zero_modes_or_partial_results():
    nonempty_zero = summarize_spectrum(
        SpectrumResult(1, np.zeros(2)),
        statistics=("min", "mean", "zero_count"),
        empty_operator_policy="zero",
    )
    assert np.isnan(nonempty_zero.values[:2]).all()
    assert nonempty_zero.values[2] == 2
    assert not nonempty_zero.metadata["structural_absence"]
    assert nonempty_zero.metadata["zero_filled_statistics"] == ()

    partial = summarize_spectrum(
        SpectrumResult(1, np.empty(0), complete=False),
        statistics=("min", "zero_count", "laplacian_energy"),
        allow_partial=True,
        empty_operator_policy="zero",
    )
    assert np.isnan(partial.values).all()
    assert not partial.metadata["structural_absence"]
    assert partial.metadata["zero_filled_statistics"] == ()


@pytest.mark.parametrize(
    "overrides",
    (
        {"metadata": {"operator_dimension": 1, "source_chain_dimension": 0,
                      "structural_absence": True}},
        {"metadata": {"operator_dimension": 0, "source_chain_dimension": 1,
                      "structural_absence": True}},
        {"metadata": {"operator_dimension": 0, "source_chain_dimension": 0,
                      "structural_absence": False}},
        {"metadata": {}},
        {"basis": ("edge",)},
        {"matrix": np.eye(1)},
        {"eigenvectors": np.empty((1, 0))},
        {"nullity": None},
        {"nullity": 1},
    ),
)
def test_zero_fill_rejects_unconfirmed_or_inconsistent_empty_records(overrides):
    with pytest.raises(ValueError, match="core-confirmed complete 0 x 0 operator"):
        summarize_spectrum(
            _absent_spectrum(dimension=1, **overrides),
            empty_operator_policy="zero",
        )


def test_default_preserve_does_not_reinterpret_legacy_metadata_as_a_contract():
    legacy = SpectrumResult(
        1,
        np.empty(0),
        metadata={
            "operator_dimension": "application-specific",
            "source_chain_dimension": -1,
            "structural_absence": "unknown",
        },
    )
    summary = summarize_spectrum(legacy, statistics=("min", "count"))
    assert np.isnan(summary.values[0])
    assert summary.values[1] == 0.0
    assert summary.metadata["operator_dimension"] == 0
    assert summary.metadata["source_chain_dimension"] == 0
    assert summary.metadata["structural_absence"] is False


@pytest.mark.parametrize("policy", (None, True, "", "nan", "ZERO"))
def test_empty_operator_policy_is_closed_and_explicit(policy):
    with pytest.raises(ValueError, match="empty_operator_policy"):
        summarize_spectrum(
            SpectrumResult(0, np.empty(0)), empty_operator_policy=policy
        )


def test_zero_filled_and_nonempty_blocks_stack_with_one_feature_schema():
    options = {
        "statistics": ("min", "max", "mean", "std", "zero_count"),
        "positive_only": False,
        "empty_operator_policy": "zero",
    }
    empty = summarize_spectrum(
        _absent_spectrum(dimension=1),
        **options,
    )
    nonempty = summarize_spectrum(
        SpectrumResult(
            1, np.array([0.0, 2.0]), kind="persistent", start=0.0, end=1.0
        ),
        **options,
    )

    matrix = feature_matrix((empty, nonempty))
    assert matrix.shape == (2, 5)
    np.testing.assert_array_equal(matrix[0], 0.0)
    np.testing.assert_allclose(matrix[1], [0.0, 2.0, 1.0, 1.0, 1.0])

    preserved = summarize_spectrum(
        SpectrumResult(
            1, np.array([0.0, 2.0]), kind="persistent", start=0.0, end=1.0
        ),
        statistics=options["statistics"],
        positive_only=False,
    )
    with pytest.raises(ValueError, match="empty_operator_policy"):
        feature_matrix((nonempty, preserved))


def test_structural_zero_fill_roundtrip_preserves_the_feature_receipt(tmp_path):
    summary = summarize_spectrum(
        _absent_spectrum(),
        statistics=("min", "mean", "moment_2"),
        empty_operator_policy="zero",
    )

    restored = load_result(save_result(summary, tmp_path / "summary.json"))
    np.testing.assert_array_equal(restored.values, [0.0, 0.0, 0.0])
    assert restored.metadata["empty_operator_policy"] == "zero"
    assert restored.metadata["structural_absence"] is True
    assert restored.metadata["zero_filled_statistics"] == (
        "min", "mean", "moment_2"
    )


@pytest.mark.parametrize("route", ("simplicial", "hyperdigraph", "interaction"))
@pytest.mark.parametrize("persistent", (False, True))
def test_empty_source_chain_zero_fills_across_all_routes(route, persistent):
    options = {"kind": route, "max_dimension": 1, "max_scale": 2.0}
    if route == "hyperdigraph":
        options["cutoff"] = 2.0
    topology = builders.from_points([[0.0], [1.0]], **options)

    spectra = (
        core.persistent_laplacians(
            topology, max_dimension=1, start=0.0, end=2.0,
            return_matrix=True, return_eigenvectors=True,
        )
        if persistent else
        core.laplacians(
            topology, max_dimension=1, scale=0.0, return_matrix=True,
            return_eigenvectors=True,
        )
    )
    assert tuple(spectra) == (0, 1)
    spectrum = spectra[1]
    assert spectrum.complete and spectrum.nullity == 0
    assert spectrum.eigenvalues.shape == (0,)
    assert spectrum.eigenvectors.shape == (0, 0)
    assert spectrum.basis == ()
    assert spectrum.matrix.shape == (0, 0)
    assert spectrum.metadata["operator_dimension"] == 0
    assert spectrum.metadata["source_chain_dimension"] == 0
    assert spectrum.metadata["structural_absence"] is True

    if persistent:
        target = core.laplacian(
            topology, dimension=1, scale=2.0, return_matrix=True
        )
        target_size = len(target.basis)
        assert spectrum.start == 0.0 and spectrum.end == 2.0
        assert target_size > 0
        assert target.metadata["operator_dimension"] == target_size
        assert target.metadata["source_chain_dimension"] == target_size
        assert target.metadata["structural_absence"] is False
        assert target.matrix.shape == (target_size, target_size)

    summaries = {
        degree: summarize_spectrum(
            item,
            statistics=STATISTICS,
            positive_only=False,
            empty_operator_policy="zero",
        )
        for degree, item in spectra.items()
    }
    assert all(item.values.shape == (len(STATISTICS),) for item in summaries.values())
    assert all(np.isfinite(item.values).all() for item in summaries.values())
    summary = summaries[1]
    np.testing.assert_array_equal(summary.values, np.zeros(len(STATISTICS)))
    assert summary.metadata["zero_filled_statistics"] == STATISTICS


@pytest.mark.parametrize("route", ("simplicial", "hyperdigraph", "interaction"))
def test_partial_spectrum_metadata_retains_full_operator_dimension(route):
    options = {"kind": route, "max_dimension": 0, "max_scale": 2.0}
    if route == "hyperdigraph":
        options["cutoff"] = 2.0
    topology = builders.from_points([[0.0], [1.0], [2.0]], **options)

    spectrum = core.laplacian(topology, dimension=0, scale=2.0, k=1)
    assert not spectrum.complete and spectrum.nullity is None
    assert spectrum.eigenvalues.shape == (1,)
    assert spectrum.metadata["operator_dimension"] == len(spectrum.basis) == 3
    assert spectrum.metadata["source_chain_dimension"] == 3
    assert spectrum.metadata["structural_absence"] is False
