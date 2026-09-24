"""Three-route numerical point-cloud demonstration and portable result bundle."""
import json
from pathlib import Path
import time
import numpy as np
from topokit.builders import from_points
from topokit.postprocessing import histogram_features, summarize_spectra
from topokit.serialization import save_result
from topokit.workflows import analyze, compact_analysis
try:
    from .data_fixture import load_demo_cloud
except ImportError:  # Script execution from the examples directory.
    from data_fixture import load_demo_cloud


DEFAULT_SCALES = {"simplicial": (1.0, 3.0, 5.0),
                  "hyperdigraph": (1.0, 2.0, 3.0),
                  "interaction": (1.0, 3.0, 5.0)}


def _plots(cloud, topology, analysis, features, directory):
    from topokit import visualization as viz
    import matplotlib.pyplot as plt
    def save_figure(figure, name):
        labels = [label for ax in figure.axes for label in
                  ([ax.xaxis.label, ax.yaxis.label] + ([ax.zaxis.label] if hasattr(ax, "zaxis") else []))]
        if figure._suptitle is not None:
            labels.append(figure._suptitle)
        figure.savefig(directory / name, dpi=140, bbox_inches="tight", bbox_extra_artists=labels, pad_inches=0.15)
    def save(axis, name):
        save_figure(axis.figure, name)
        plt.close(axis.figure)
    save(viz.plot_points(cloud, title="24-point fixture; assigned uniform weights"), "points.png")
    save(viz.plot_barcodes(analysis.persistence), "barcodes.png")
    save(viz.plot_features(features, dimension=min(2, analysis.persistence.max_dimension)), "features.png")
    scale = list(analysis.snapshots)[len(analysis.snapshots)//2]
    dimension = min(2, analysis.persistence.max_dimension)
    save(viz.plot_spectrum(analysis.snapshots[scale]["laplacians"][dimension]), "spectrum.png")
    if topology.kind == "simplicial":
        cells = [tuple(cloud.ids[i] for i in simplex)
                 for simplex, birth in topology.native.get_filtration() if birth <= scale]
        save(viz.plot_simplices(cloud, cells, title=f"Alpha object at squared radius {scale:g}"), "topology.png")
    elif topology.kind == "hyperdigraph":
        edges = [edge for edge in topology.native.directed_hyperedges(1)
                 if topology.native.birth_of(edge) <= scale]
        save(viz.plot_hyperedges(cloud, edges, title=f"Directed 1-hyperedges at distance {scale:g}"), "topology.png")
    else:
        a, b = topology.cloud
        maps = topology.metadata["factor_vertex_id_maps"]
        cells = []
        for factor, mapping in zip(topology.native.factors, maps):
            cells.append([tuple(mapping[v] for v in simplex)
                          for simplex_id, simplex in enumerate(factor.simplices)
                          if factor.filtration(simplex_id) <= scale])
        figure, axes = viz.plot_interaction_factors(a, b, *cells, overlap_pairs=topology.metadata["overlap_pairs"])
        save_figure(figure, "topology.png")
        plt.close(figure)


def run_demo(output, *, routes=("simplicial", "hyperdigraph", "interaction"),
             max_dimension=2, plots=False, max_dense_entries=4_000_000):
    """Run all requested routes, recording any failure without changing topology.

    Outputs are local only. A failed route remains failed in the report; the
    caller decides whether to raise budgets or change the scientific recipe.
    """
    routes = tuple(routes)
    if not routes or any(route not in DEFAULT_SCALES for route in routes):
        raise ValueError("routes must name simplicial, hyperdigraph, or interaction")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    cloud = load_demo_cloud()
    save_result(cloud, directory / "point_cloud.json")
    config = {"example": "point_cloud_24", "max_dimension": max_dimension,
              "weights": "assigned_uniform", "filtration_start": 0,
              "snapshot_scales": {route: DEFAULT_SCALES[route] for route in routes},
              "max_dense_entries": max_dense_entries, "return_eigenvectors": False,
              "interaction": "alpha(selected_ID_subset) versus alpha(full_cloud)",
              "weighted_alpha": False, "spectra": "ordinary_snapshots"}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = {"example": "point_cloud_24", "routes": {}}
    for route in routes:
        route_dir = directory / route
        route_dir.mkdir(parents=True, exist_ok=True)
        begun = time.perf_counter()
        try:
            if route == "interaction":
                subset = cloud.subset(cloud.metadata["subset_ids"])
                obj = from_points(subset, kind=route, cloud_b=cloud,
                                  overlap_pairs=[(label, label) for label in subset.ids],
                                  max_dimension=max_dimension)
            else:
                obj = from_points(cloud, kind=route, max_dimension=max_dimension)
            built = time.perf_counter()
            analysis = analyze(obj, max_dimension=max_dimension, scales=DEFAULT_SCALES[route],
                               max_dense_entries=max_dense_entries)
            features = histogram_features(analysis.persistence,
                                          birth_edges=np.linspace(0, 6, 13), death_edges=np.linspace(0, 6, 13))
            save_result(compact_analysis(analysis), route_dir / "analysis.json")
            save_result(features, route_dir / "features.json")
            spectra = [spectrum for snapshot in analysis.snapshots.values()
                       for spectrum in snapshot["laplacians"].values()]
            save_result(summarize_spectra(spectra), route_dir / "spectral_summaries.json")
            np.save(route_dir / "features.npy", features.values, allow_pickle=False)
            np.savetxt(route_dir / "intervals.csv", analysis.persistence.as_array(), delimiter=",",
                       header="dimension,birth,death", comments="")
            if plots:
                _plots(cloud, obj, analysis, features, route_dir)
            report["routes"][route] = {
                "status": "passed", "build_seconds": built-begun,
                "total_seconds": time.perf_counter()-begun,
                "interval_count": len(analysis.persistence.intervals),
                "feature_count": len(features.values),
                "scale_units": analysis.persistence.metadata.get("scale_units"),
                "snapshots": {str(scale): {"betti_GF2": list(snapshot["homology"].betti_numbers),
                                            "nullity_R": [s.nullity for s in snapshot["laplacians"].values()],
                                            "matrix_sizes": [len(s.basis) for s in snapshot["laplacians"].values()]}
                              for scale, snapshot in analysis.snapshots.items()}}
        except Exception as error:
            report["routes"][route] = {"status": "failed", "error_type": type(error).__name__,
                                       "error": str(error), "total_seconds": time.perf_counter()-begun}
        (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
