import numpy as np
import pytest
import topokit as tk
from topokit.workflows import compact_analysis
from topokit.postprocessing import histogram_features
from topokit.serialization import save_result, load_result
from topokit.workflows import analyze


@pytest.fixture
def cloud():
    return tk.PointCloud([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                         ids=("a", "b", "c", "d"), weights=(1., 1., 2., 2.))


@pytest.mark.parametrize("kind", ["simplicial", "hyperdigraph", "interaction"])
def test_three_routes_to_features_exports_and_spectra(cloud, kind, tmp_path):
    topology = tk.from_points(cloud, kind=kind, max_dimension=2)
    result = analyze(topology, max_dimension=2, scales=(0.5, 1.5), return_eigenvectors=True)
    assert result.persistence.max_dimension == 2
    for snapshot in result.snapshots.values():
        assert len(snapshot["homology"].betti_numbers) == 3
        for q, spectrum in snapshot["laplacians"].items():
            assert spectrum.dimension == q and spectrum.kind == "ordinary"
            assert spectrum.eigenvectors.shape == (len(spectrum.basis), len(spectrum.eigenvalues))
            assert spectrum.metadata.get("residual_max", 0) < 1e-7
    feature = histogram_features(result.persistence, birth_edges=[0, 1, 2, 4], death_edges=[0, 1, 2, 4])
    assert np.all(np.isfinite(feature.values))
    document = load_result(save_result(compact_analysis(result), tmp_path / f"{kind}.json"))
    assert document["topology"]["kind"] == kind
    assert document["persistence"].intervals == result.persistence.intervals
    assert document["persistence"].metadata["scale_units"] == result.persistence.metadata["scale_units"]
    assert document["persistence"].betti_at(0.5) == result.persistence.betti_at(0.5)


@pytest.mark.parametrize("kind", ["simplicial", "hyperdigraph", "interaction"])
def test_nonzero_start_and_snapshot_independence(cloud, kind):
    topology = tk.from_points(cloud, kind=kind, max_dimension=2, filtration_start=0.6)
    before = tk.persistence(topology).as_array()
    result = analyze(topology, scales=(0.6, 1.5))
    np.testing.assert_array_equal(before, result.persistence.as_array())
    assert all(interval.birth >= 0.6 for interval in result.persistence.intervals)
    with pytest.raises(ValueError):
        tk.laplacian(topology, scale=0.5)
    eigen = tk.laplacian(topology, dimension=0, scale=1.5)
    assert eigen.eigenvectors is None


@pytest.mark.parametrize("kind", ["simplicial", "hyperdigraph", "interaction"])
def test_pairwise_persistent_operators_are_opt_in(cloud, kind):
    topology = tk.from_points(cloud, kind=kind, max_dimension=2)
    for q in (0, 1, 2):
        ordinary = tk.laplacian(topology, dimension=q, scale=1.5, return_matrix=True)
        same_time = tk.persistent_laplacian(topology, dimension=q, start=1.5, end=1.5, return_matrix=True)
        np.testing.assert_allclose(same_time.eigenvalues, ordinary.eigenvalues, atol=1e-8)


def test_visualizations_consume_results(cloud, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    from topokit import visualization as viz
    import matplotlib.pyplot as plt
    topology = tk.from_points(cloud)
    bars = tk.persistence(topology)
    spectrum = tk.laplacian(topology, 0, scale=1.5, return_eigenvectors=True)
    features = histogram_features(bars, birth_edges=[0, 1, 2], death_edges=[0, 1, 2])
    axes = [viz.plot_points(cloud), viz.plot_simplices(cloud, [("a", "b", "c")]),
            viz.plot_hyperedges(cloud, [("a", "b"), ("b", "a"), ("a", "b", "c")]),
            viz.plot_barcodes(bars), viz.plot_spectrum(spectrum), viz.plot_eigenvector(spectrum),
            viz.plot_features(features, dimension=2)]
    for index, axis in enumerate(axes):
        path = tmp_path / f"view-{index}.png"
        axis.figure.savefig(path)
        assert path.read_bytes().startswith(b"\x89PNG")
        plt.close(axis.figure)
