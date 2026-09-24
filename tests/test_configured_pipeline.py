"""0.3 feature-grid rendering and complete configured example."""
import numpy as np
import pytest
from topokit.results import PersistenceInterval, PersistenceResult
from topokit.postprocessing import histogram_features
from topokit.serialization import save_result, load_result


def test_ragged_degree_grids_roundtrip_and_display(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from topokit.visualization import plot_features
    bars = PersistenceResult((PersistenceInterval(0, 0., 1.),
                              PersistenceInterval(1, .5, 1.5)), max_dimension=1)
    features = histogram_features(bars, birth_edges={0: [0, 1, 2], 1: [0, 2]},
                                  death_edges={0: [0, 2], 1: [0, 1, 2]})
    restored = load_result(save_result(features, tmp_path / "ragged.json"))
    assert isinstance(restored.finite_histograms, tuple)
    assert restored.grids[0].shape == (2, 1)
    assert restored.grids[1].shape == (1, 2)
    np.testing.assert_array_equal(features.values, restored.values)
    for q in (0, 1):
        axis = plot_features(restored, dimension=q)
        axis.figure.canvas.draw()
        plt.close(axis.figure)


def test_configured_example_all_routes_h2_l2(tmp_path):
    from examples.configured_pipeline import run
    target, shape = run(tmp_path / "configured.json")
    data = load_result(target)
    assert shape[0] == 2 and shape[1] > 0
    assert data["paired_interaction_bars"].max_dimension == 2
    assert data["interaction_persistent_L2"].kind == "persistent"
    assert data["interaction_snapshot_L2"].kind == "ordinary"
    assert data["bin_fit"]["fit_scope"] == "training"
    assert data["positive_spectral_features"].values.shape == (5,)
