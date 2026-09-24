"""Run with an installed wheel from outside the repository: python -I PATH."""
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np


class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"simplicial_topo", "hyperdigraph_topo", "interaction_topology",
                                     "gudhi", "ripser", "dionysus", "gtda"}:
            raise AssertionError(f"wheel attempted forbidden runtime import: {fullname}")
        return None


sys.meta_path.insert(0, BlockLegacy())
import topokit
from topokit import builders, core, readers, postprocessing
from topokit.workflows import analyze, compact_analysis
from topokit.serialization import save_result

assert "site-packages" in str(Path(topokit.__file__))
assert not (Path(topokit.__file__).parent / "datasets").exists()
route_names = ("simplicial", "hyperdigraph", "interaction")
for namespace, prefix in ((core, "topokit.core"), (builders, "topokit.builders")):
    qualified = tuple(f"{prefix}.{route}" for route in route_names)
    assert not any(name in sys.modules for name in qualified)
    for index, (route, module_name) in enumerate(zip(route_names, qualified)):
        module = getattr(namespace, route)
        assert module.__name__ == module_name
        assert sys.modules[module_name] is module
        assert all(name not in sys.modules for name in qualified[index + 1:])
assert callable(builders.simplicial.from_graph)
assert callable(builders.hyperdigraph.from_digraph)
assert callable(builders.interaction.from_complexes)
fixture = Path(sys.argv[2])
cloud = readers.read_csv(fixture)
assert len(cloud) == 24
selected = cloud.subset([label for label, selected in
                         zip(cloud.ids, cloud.metadata["columns"]["selected"]) if selected == "1"])
result = {}
for route, scales in {"simplicial": (1, 3, 5), "hyperdigraph": (1, 2, 3), "interaction": (1, 3, 5)}.items():
    if route == "interaction":
        obj = builders.build(selected, kind=route, cloud_b=cloud,
                             overlap_pairs=[(label, label) for label in selected.ids])
    else:
        obj = builders.build(cloud, kind=route)
    analysis = analyze(obj, max_dimension=2, scales=scales)
    features = postprocessing.histogram_features(analysis.persistence, birth_edges=[0, 1, 3, 6], death_edges=[0, 1, 3, 6])
    assert len(features.values) and analysis.persistence.metadata["scale_units"]
    save_result(compact_analysis(analysis), Path(sys.argv[1]) / f"{route}.json")
    result[route] = {"intervals": len(analysis.persistence.intervals), "features": len(features.values)}

# Exercise 0.3 APIs from the installed wheel, not just retained 0.2 paths.
from topokit.builders import simplicial, hyperdigraph, interaction
from topokit.workflows import laplacian_series
from topokit.workflows.interaction import persistent_laplacian_between_pairs
tiny = topokit.PointCloud([[0.], [1.], [2.]], ids=("a", "b", "c"),
                         metadata={"labels": ("B", "N", "B"), "coordinate_units": "example_length"})
weighted = readers.assign_element_weights(tiny)
np.testing.assert_allclose(weighted.weights, [2.04, 3.04, 2.04])
bonds = builders.bonds_from_adjacency([[0, 1, 0], [1, 0, 1], [0, 1, 0]], ids=tiny.ids)
graph = simplicial.from_points(tiny, complex_type="graph", bonds=bonds, cutoff=1., max_dimension=1)
directed = hyperdigraph.from_points(weighted, object_type="digraph", bonds=bonds, max_dimension=1)
assert core.homology(graph, max_dimension=1).betti_numbers == (1, 0)
assert core.homology(directed, max_dimension=0).betti_numbers == (1,)
bars = core.persistence(graph, max_dimension=1)
encoder = postprocessing.PersistenceVectorizer(step=.5, dimensions=(0, 1))
assert encoder.fit_transform([bars, bars]).shape[0] == 2
assert len(laplacian_series(graph, max_dimension=1)) >= 1
sweep = core.hyperdigraph.L0Sweep(directed)
np.testing.assert_array_equal(sweep.matrix_at(1.), [[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
assert sweep.laplacian(1.).nullity == 1
assert laplacian_series(directed, max_dimension=1)[1.][0].metadata['assembly_backend'] == 'incremental_l0'
pair = interaction.from_points(tiny, max_dimension=1)
paired = persistent_laplacian_between_pairs(pair, dimension=1, source_pair=(0., 0.), target_pair=(1., 2.))
assert paired.kind == "persistent" and paired.eigenvectors is None and paired.matrix is None
empty_persistent = core.persistent_laplacian(
    graph, dimension=1, start=0., end=1.,
)
assert empty_persistent.complete and empty_persistent.eigenvalues.size == 0
assert empty_persistent.metadata["operator_dimension"] == 0
empty_features = postprocessing.summarize_spectrum(
    empty_persistent,
    statistics=("min", "max", "mean", "std", "zero_count"),
    empty_operator_policy="zero",
)
np.testing.assert_array_equal(empty_features.values, np.zeros(5))
assert empty_features.metadata["zero_filled_statistics"] == (
    "min", "max", "mean", "std", "zero_count",
)
result["configured_0_3"] = "passed"
print(json.dumps({"installed_package": str(Path(topokit.__file__)), "python": sys.executable,
                  "runtime_import_isolation": "passed", "three_route_demo": "passed",
                  "summary": result}, indent=2))
