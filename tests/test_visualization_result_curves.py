"""Reusable persistence and spectral-summary curve views."""

import math

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from topokit import visualization as viz
from topokit.postprocessing import VectorResult
from topokit.results import PersistenceInterval, PersistenceResult


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def bars():
    return PersistenceResult(
        (
            PersistenceInterval(0, 0.0, 1.5, at_initial_stage=True),
            PersistenceInterval(0, 0.0, math.inf, at_initial_stage=True),
            PersistenceInterval(1, 0.75, 2.0),
        ),
        max_dimension=1,
        metadata={
            "filtration_start": 0.0,
            "filtration_end": 2.0,
            "scale_units": "distance",
        },
    )


def test_barcode_initial_stage_markers_are_optional_without_changing_bars(bars):
    marked = viz.plot_barcodes(bars)
    assert sum(line.get_marker() == "o" for line in marked.lines) == 2
    assert marked._topokit_view["initial_stage_markers"] is True

    figure, axis = plt.subplots()
    returned = viz.plot_barcodes(bars, ax=axis, mark_initial_stage=False)
    assert returned is axis
    assert sum(line.get_marker() == "o" for line in axis.lines) == 0
    interval_lines = [line for line in axis.lines if line.get_marker() in (None, "None", "")]
    assert sum(float(line.get_xdata()[0]) == 0.0 for line in interval_lines) == 2
    assert axis._topokit_view["initial_stage_markers"] is False
    with pytest.raises(ValueError, match="boolean"):
        viz.plot_barcodes(bars, mark_initial_stage=1)


def test_betti_curves_query_supplied_persistence_result_and_axes(bars):
    figure, axis = plt.subplots()
    returned = viz.plot_betti_curves(
        bars, [0.0, 1.0, 2.0], dimensions=(0, 1), ax=axis,
        title="Persistent ranks", xlabel="Distance",
    )
    assert returned is axis
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [2, 2, 1])
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), [0, 1, 0])
    assert [line.get_label() for line in axis.lines] == ["H0", "H1"]
    assert axis.get_xlabel() == "Distance"
    assert axis._topokit_view == {
        "kind": "betti_curves", "dimensions": (0, 1),
        "scales": (0.0, 1.0, 2.0),
    }


@pytest.mark.parametrize("scales", ([0, 0], [0, math.inf], [True, 1]))
def test_betti_curves_reject_ambiguous_scales(bars, scales):
    with pytest.raises(ValueError, match="scales"):
        viz.plot_betti_curves(bars, scales)
    with pytest.raises(ValueError, match="recorded filtration domain"):
        viz.plot_betti_curves(bars, [0, 3])


def _summary(scale, values, *, names=None, dimension=0):
    return VectorResult(
        np.asarray(values, dtype=float),
        names or ("L0:min", "L0:max", "L0:mean", "L0:laplacian_energy"),
        {"scale": float(scale), "scale_units": "distance", "dimension": dimension},
    )


def test_spectral_summary_series_plots_min_max_mean_and_energy_with_nan_gap():
    summaries = (
        _summary(0.0, [0.0, 2.0, 1.0, 2.0]),
        _summary(1.0, [math.nan, 4.0, 2.0, 5.0]),
        _summary(2.0, [0.0, 6.0, 3.0, 7.0]),
    )
    axis, energy_axis = viz.plot_spectral_summary_series(summaries)

    assert axis.figure is energy_axis.figure
    assert [line.get_label() for line in axis.lines] == ["minimum", "maximum", "mean"]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [0, math.nan, 0], equal_nan=True)
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), [2, 4, 6])
    np.testing.assert_array_equal(axis.lines[2].get_ydata(), [1, 2, 3])
    np.testing.assert_array_equal(energy_axis.lines[0].get_ydata(), [2, 5, 7])
    assert axis._topokit_view["statistics"] == ("min", "max", "mean")
    assert axis._topokit_view["energy"] == "laplacian_energy"
    axis.figure.canvas.draw()


def test_spectral_summary_series_reuses_both_supplied_axes():
    summaries = (_summary(0, [0, 1, .5, 1]), _summary(1, [0, 2, 1, 2]))
    figure, axis = plt.subplots()
    supplied_energy_axis = axis.twinx()
    returned_axis, returned_energy_axis = viz.plot_spectral_summary_series(
        summaries, scales=[2, 3], ax=axis, energy_ax=supplied_energy_axis,
        title="L0 summaries", xlabel="Cutoff [angstrom]",
    )
    assert returned_axis is axis and returned_energy_axis is supplied_energy_axis
    assert axis.get_title() == "L0 summaries"
    assert axis.get_xlabel() == "Cutoff [angstrom]"


def test_spectral_summary_series_rejects_inconsistent_or_missing_data():
    valid = _summary(0, [0, 1, .5, 1])
    with pytest.raises(ValueError, match="nonempty"):
        viz.plot_spectral_summary_series(())
    with pytest.raises(ValueError, match="ordered schema"):
        viz.plot_spectral_summary_series((
            valid, _summary(1, [0, 1], names=("L0:min", "L0:max")),
        ))
    with pytest.raises(ValueError, match="finite, or NaN"):
        viz.plot_spectral_summary_series((_summary(0, [0, 1, .5, math.inf]),))
    with pytest.raises(ValueError, match="present in every summary"):
        viz.plot_spectral_summary_series((valid,), statistics=("variance",))
    with pytest.raises(ValueError, match="strictly increasing"):
        viz.plot_spectral_summary_series((valid, valid), scales=[1, 1])
