"""Small coordinate-only example of the 0.3 construction/feature contracts.

Run from the project root: python examples/configured_pipeline.py
No molecule interpretation, label prediction, or performance claim is made.
"""
from pathlib import Path
import numpy as np

from topokit import PointCloud, core, postprocessing
from topokit.builders import simplicial, hyperdigraph, interaction
from topokit.serialization import save_result
from topokit.workflows import laplacian_series
from topokit.workflows.interaction import laplacian_at_pair, persistent_laplacian_between_pairs


def run(output=None):
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1.5, 0.], [0., 0., 1.]])
    cloud = PointCloud(points, ids=("a", "b", "c", "d"), weights=[0., 1., 1., 2.],
                       metadata={"coordinate_units": "example_length"})
    # Simplicial geometry ignores weights; directed constructions use them for orientation.
    alpha = simplicial.from_points(cloud, max_dimension=2)
    graph = simplicial.from_points(cloud, complex_type="graph", max_dimension=2,
                                   bonds=[("a", "b"), ("b", "c"), ("c", "a")], cutoff=3.)
    directed = hyperdigraph.from_points(cloud, object_type="digraph", max_dimension=2,
                                        bonds=[("a", "b"), ("b", "c")])
    # b and c have equal weights: both b→c and c→b remain present.
    hyper = hyperdigraph.from_points(cloud, max_dimension=2)
    subset = cloud.subset(("a", "b", "c"))
    pair = interaction.from_points(subset, cloud, max_dimension=2,
                                   overlap_vertices=[(label, label) for label in subset.ids])
    path = interaction.regrade_pair_filtration(pair, [0., .5, 1., 3.], [0., 1., 2., 3.])

    # Fit once on two coordinate-only samples; transforms reuse those exact bins.
    other = PointCloud(points * 1.2, ids=cloud.ids, weights=cloud.weights, metadata=cloud.metadata)
    training_bars = [core.persistence(obj, max_dimension=2) for obj in
                     (alpha, simplicial.from_points(other, max_dimension=2))]
    encoder = postprocessing.PersistenceVectorizer(step=.5, dimensions=(0, 1, 2))
    X = encoder.fit_transform(training_bars)
    records = encoder.transform_results(training_bars)
    result = {
        "alpha_bars": training_bars[0], "fixed_bin_features": records,
        "bin_fit": encoder.fit_metadata_, "feature_matrix_shape": X.shape,
        "graph_homology": core.homology(graph, max_dimension=2),
        "digraph_homology": core.homology(directed, max_dimension=2),
        "hyperdigraph_bars": core.persistence(hyper, max_dimension=2),
        "paired_interaction_bars": core.persistence(path, max_dimension=2),
        "ordinary_L0_to_L2": laplacian_series(alpha, max_dimension=2, scales=[0., 1., 3.]),
        "interaction_snapshot_L2": laplacian_at_pair(pair, 1., 2., dimension=2),
        "interaction_persistent_L2": persistent_laplacian_between_pairs(
            pair, dimension=2, source_pair=(.5, 1.), target_pair=(1., 2.)),
    }
    spectrum = result["ordinary_L0_to_L2"][1.][2]
    result["positive_spectral_features"] = postprocessing.summarize_spectrum(spectrum)
    target = (Path(__file__).resolve().parent / "output/configured_pipeline.json"
              if output is None else Path(output))
    save_result(result, target)
    return target, X.shape


if __name__ == "__main__":
    destination, shape = run()
    print(f"Saved {destination}; fitted feature matrix shape={shape}")
