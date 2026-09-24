"""Definitions and scope contracts for predefined spectral information."""

import math

import numpy as np
import pytest

from topokit.postprocessing import (
    laplacian_energy,
    spectral_energy,
    spectral_moment,
    spectral_moments,
    spectral_variance,
    summarize_spectrum,
)
from topokit.results import SpectrumResult


def test_direct_spectral_statistics_have_distinct_explicit_definitions():
    values = np.array([0.0, 1.0, 3.0])
    original = values.copy()

    assert spectral_variance(values) == pytest.approx(14 / 9)
    assert spectral_moment(values, 0) == 1
    assert spectral_moment(values, 1) == pytest.approx(4 / 3)
    assert spectral_moment(values, 2) == pytest.approx(10 / 3)
    assert spectral_moment(values, 3) == pytest.approx(28 / 3)
    np.testing.assert_allclose(spectral_moments(values), [4 / 3, 10 / 3, 28 / 3, 82 / 3])
    np.testing.assert_allclose(spectral_moments(values, orders=(0, 2)), [1, 10 / 3])
    assert spectral_energy(values) == 4
    assert laplacian_energy(values) == pytest.approx(10 / 3)
    np.testing.assert_array_equal(values, original)


def test_spectral_variance_and_moments_have_explicit_empty_and_order_policies():
    assert math.isnan(spectral_variance([]))
    assert math.isnan(spectral_moment([], 0))
    assert laplacian_energy([]) == 0
    for order in (-1, 1.5, True, 1 + 0j, "2"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            spectral_moment([1, 2], order)
    for orders in ((), (1, 1), {1, 2}, {1: "first"}, "12", True, 2):
        with pytest.raises(ValueError, match="nonempty ordered sequence"):
            spectral_moments([1, 2], orders)


def test_spectral_moments_preserves_requested_order_and_empty_spectrum_policy():
    np.testing.assert_allclose(spectral_moments([0, 2], (3, 1)), [4, 1])
    assert np.isnan(spectral_moments([], (4, 2))).all()


def test_scaled_statistics_avoid_spurious_overflow_for_constant_large_values():
    values = np.array([1e308, 1e308])
    assert spectral_variance(values) == 0
    assert laplacian_energy(values) == 0
    assert spectral_moment(values, 1) == 1e308
    assert np.isposinf(spectral_moment(values, 2))


def test_power_rescaling_avoids_spurious_overflow_for_finite_variance():
    high = 1e160
    low = high * (1.0 - 2.0 ** -20)
    expected = ((high - low) / 2.0) ** 2

    result = spectral_variance([low, high])

    assert np.isfinite(result)
    assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("baseline", (1e155, 1e160))
def test_centered_statistics_preserve_adjacent_large_float_differences(baseline):
    adjacent = np.nextafter(baseline, np.inf)
    difference = adjacent - baseline

    assert spectral_variance([baseline, adjacent]) == (difference / 2.0) ** 2
    assert laplacian_energy([baseline, adjacent]) == difference


def test_centered_statistics_fall_back_when_translation_overflows():
    values = [-1e308, 1e308]

    assert np.isposinf(spectral_variance(values))
    assert np.isposinf(laplacian_energy(values))


def test_power_rescaling_preserves_moment_sign_overflow_and_underflow():
    assert spectral_moment([-2.0], 3) == -8.0
    assert np.isneginf(spectral_moment([-1e200], 3))
    negative_underflow = spectral_moment([-1e-200], 3)
    assert negative_underflow == 0.0
    assert math.copysign(1.0, negative_underflow) == -1.0


def test_predefined_summary_names_include_aliases_and_first_four_moments():
    summary = summarize_spectrum(
        SpectrumResult(0, np.array([0.0, 1.0, 3.0])),
        statistics=(
            "variance", "spectral_variance",
            "moment_1", "spectral_moment_1", "moment_2", "moment_3", "moment_4",
            "laplacian_energy",
        ),
        positive_only=False,
    )

    assert summary.names == (
        "L0:variance", "L0:spectral_variance",
        "L0:moment_1", "L0:spectral_moment_1", "L0:moment_2", "L0:moment_3",
        "L0:moment_4", "L0:laplacian_energy",
    )
    np.testing.assert_allclose(
        summary.values,
        [14 / 9, 14 / 9, 4 / 3, 4 / 3, 10 / 3, 28 / 3, 82 / 3, 10 / 3],
    )
    assert summary.metadata["spectral_moment_orders"] == (
        ("moment_1", 1), ("spectral_moment_1", 1),
        ("moment_2", 2), ("moment_3", 3), ("moment_4", 4),
    )


def test_builtin_laplacian_energy_uses_full_spectrum_despite_positive_default():
    values = np.array([-1e-12, 1.0, 3.0])
    summary = summarize_spectrum(
        SpectrumResult(1, values),
        statistics=("variance", "moment_2", "laplacian_energy"),
    )

    # Variance and moments obey positive_only=True, while Laplacian energy needs
    # every eigenvalue (including tolerance-resolved numerical zeros).
    np.testing.assert_allclose(summary.values, [1.0, 5.0, 10 / 3])
    assert summary.metadata["statistic_scopes"] == {
        "variance": "positive_eigenvalues_from_full_spectrum",
        "moment_2": "positive_eigenvalues_from_full_spectrum",
        "laplacian_energy": "full_operator_spectrum",
    }
    assert summary.metadata["laplacian_energy_available"]
    np.testing.assert_array_equal(values, [-1e-12, 1.0, 3.0])


def test_partial_summary_never_claims_full_operator_laplacian_energy():
    summary = summarize_spectrum(
        SpectrumResult(0, np.array([0.0, 2.0]), complete=False),
        statistics=("variance", "moment_2", "laplacian_energy"),
        allow_partial=True,
    )

    assert summary.values[0] == 0
    assert summary.values[1] == 4
    assert math.isnan(summary.values[2])
    assert summary.metadata["statistic_scopes"]["laplacian_energy"] == "unavailable_partial_spectrum"
    assert not summary.metadata["laplacian_energy_available"]
    assert summary.metadata["nonfinite_statistics"] == ("laplacian_energy",)


def test_custom_mapping_of_laplacian_energy_keeps_full_spectrum_contract():
    summary = summarize_spectrum(
        SpectrumResult(0, np.array([0.0, 1.0, 3.0])),
        statistics={"centered_energy": laplacian_energy},
    )
    assert summary.values[0] == pytest.approx(10 / 3)
    assert summary.metadata["statistic_scopes"] == {
        "centered_energy": "full_operator_spectrum"
    }


def test_feature_matrix_rejects_same_name_with_a_different_statistic_scope():
    from topokit.workflows.ml import feature_matrix

    result = SpectrumResult(0, np.array([0.0, 1.0, 3.0]))
    builtin = summarize_spectrum(result, statistics=("laplacian_energy",))
    selected_only = summarize_spectrum(
        result,
        statistics={"laplacian_energy": lambda values: laplacian_energy(values)},
    )

    with pytest.raises(ValueError, match="statistic_scopes"):
        feature_matrix((builtin, selected_only))
