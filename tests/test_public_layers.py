"""Integration of explicit layers and user extensions, without hidden analysis."""
from pathlib import Path
import numpy as np
import pytest
from topokit import PointCloud, builders, core, postprocessing, readers
from topokit.core import simplicial
from topokit.workflows import compact_analysis, analyze
from topokit.serialization import load_result, save_result
from examples.data_fixture import load_demo_cloud


def test_custom_builder_and_reader_compose_into_defined_object(tmp_path, monkeypatch):
    registry = readers.ReaderRegistry()
    def custom_reader(path):
        return PointCloud([[0.], [1.]], ids=("a", "b"), metadata={"measurement": Path(path).name})
    registry.register("custom", custom_reader, extensions=("custom",))
    cloud = registry.read(tmp_path / "sample.custom")
    def recipe(data):
        from topokit.builders.simplicial import from_points
        return from_points(data, max_dimension=1)
    monkeypatch.setattr(builders, "_CUSTOM_BUILDERS", {})
    builders.register_builder("custom-recipe", recipe)
    topology = builders.build(cloud, kind="custom-recipe")
    assert core.homology(topology, max_dimension=1).betti_numbers == (1, 0)
    assert topology.metadata["point_metadata"]["measurement"] == "sample.custom"
    with pytest.raises(ValueError, match="exists"):
        builders.register_builder("simplicial", recipe)
    with pytest.raises(ValueError):
        builders.register_builder("", recipe)
    builders.register_builder("bad", lambda data: data)
    with pytest.raises(TypeError, match="Topology"):
        builders.build(cloud, kind="bad")


def test_dataset_is_example_only_and_subset_metadata_stays_aligned():
    cloud = load_demo_cloud()
    assert len(cloud) == 24
    subset = cloud.subset([cloud.ids[2], cloud.ids[0]])
    assert subset.metadata["columns"]["id"] == subset.ids
    assert len(subset.metadata["columns"]["selected"]) == 2
    assert len(cloud.metadata["columns"]["id"]) == 24
    assert subset.metadata["selection_parent_ids"] == cloud.ids


def test_spectral_summary_roundtrip_is_independent_of_math(monkeypatch, tmp_path):
    obj = simplicial.SimplicialComplex([(0, 1), (1, 2), (0, 2)])
    spectrum = core.laplacian(obj)
    def forbidden(*args, **kwargs):
        raise AssertionError("postprocessing must not invoke the math core")
    monkeypatch.setattr(core, "laplacian", forbidden)
    vector = postprocessing.summarize_spectrum(spectrum)
    restored = load_result(save_result(vector, tmp_path / "features.json"))
    assert restored.names == vector.names
    np.testing.assert_allclose(restored.values, vector.values)


def test_documentation_tracks_current_release():
    import topokit
    base = Path(__file__).resolve().parents[1]
    assert topokit.__version__ in (base / "README.md").read_text()
    assert topokit.__version__ in (base / "markdown/CHANGELOG.md").read_text()
    for name in ("ARCHITECTURE", "NOTATION", "CONTRACTS", "CHANGELOG", "MIGRATION_0_2", "VALIDATION"):
        assert (base / "markdown" / f"{name}.md").is_file()


@pytest.mark.parametrize("route", ["simplicial", "hyperdigraph", "interaction"])
def test_coordinate_units_and_route_reach_features_and_ml_checks(route):
    from topokit.workflows.ml import feature_matrix
    features = []
    for units in ("angstrom", "nm"):
        cloud = PointCloud([[0.], [1.]], metadata={"coordinate_units": units})
        obj = builders.from_points(cloud, kind=route, max_dimension=0)
        bars = core.persistence(obj, max_dimension=0)
        vector = postprocessing.histogram_features(bars, birth_edges=[0, 2], death_edges=[0, 2])
        assert vector.metadata["coordinate_units"] == units
        assert vector.metadata["route"] == route
        features.append(vector)
    with pytest.raises(ValueError, match="coordinate_units"):
        feature_matrix(features)


def test_interaction_refuses_incompatible_coordinate_units():
    from topokit.builders import interaction
    a = PointCloud([[0.], [1.]], metadata={"coordinate_units": "angstrom"})
    b = PointCloud([[0.], [1.]], metadata={"coordinate_units": "nm"})
    with pytest.raises(ValueError, match="coordinate_units"):
        interaction.from_points(a, b, overlap_pairs=[(0, 0), (1, 1)])
