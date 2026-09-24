"""Explicit schematic examples of the object and reusable block views.

Run after installation: python examples/visualization_gallery.py
No topology is computed by the visualization layer. Coordinates below are
illustrative diagram layouts, not an experimental point cloud or measurements.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topokit import PointCloud, Topology
from topokit import visualization as viz
from topokit.core.simplicial import SimplicialComplex
from topokit.core.hyperdigraph import Hyperdigraph


def export_bundle(figure, output, name):
    for suffix in ("svg", "pdf", "png"):
        viz.save_figure(figure, output / f"{name}.{suffix}", dpi=240)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": "sans-serif", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        figure, axes = plt.subplots(2, 4, figsize=(9.2, 5.3))
        figure.subplots_adjust(left=.045, right=.96, top=.83, bottom=.07,
                               wspace=.26, hspace=.6)
        block_style = viz.PlotStyle(node_size=60, linewidth=1.15,
                                    face_alpha=.43, ribbon_width=11,
                                    ribbon_alpha=.55, arrow_size=14,
                                    label_size=10, label_offset=.07)
        viz.plot_building_blocks("simplicial", axes=axes[0], style=block_style)
        viz.plot_building_blocks("hyperdigraph", axes=axes[1], style=block_style)
        figure.suptitle("Topology building blocks", x=.045, ha="left", fontsize=16, fontweight="bold", y=.99)
        figure.text(.045, .915, "Filled simplex faces", fontsize=11, color="#234F7D")
        figure.text(.045, .45, "Ordered hyperedges · consecutive arrows", fontsize=11, color="#416C75")
        export_bundle(figure, output, "building_blocks")
        plt.close(figure)

        # Two tetrahedra joined by a triangular sheet; cells supplied explicitly.
        points = np.array([[0, 0, 0], [1.4, 0, 0], [.7, 1.3, 0], [.7, .45, 1.25],
                           [2.3, 1.05, 0], [3.0, .05, 0], [3.25, 1.4, 0], [2.9, .85, 1.1]])
        ids = tuple("ABCDEFGH")
        cloud = PointCloud(points, ids=ids, metadata={"coordinate_units": "schematic units"})
        simplices = [("A", "B", "C", "D"), ("B", "C", "E"), ("B", "E", "F"), ("E", "F", "G", "H")]
        sequences = [("A", "B"), ("B", "A"), ("B", "C", "D"),
                     ("C", "E", "F", "H"), ("F", "G"), ("G", "H")]
        lookup = cloud.index
        simplicial = Topology("simplicial",
            SimplicialComplex([tuple(lookup[label] for label in cell) for cell in simplices]),
            cloud, {"vertex_to_id": dict(enumerate(ids))})
        hyperdigraph = Topology("hyperdigraph",
            Hyperdigraph(ids, [(label,) for label in ids] + sequences), cloud)
        style = viz.PlotStyle(node_size=55, face_alpha=.35, ribbon_width=8,
                              ribbon_alpha=.5, arrow_size=13, padding=.15,
                              label_size=10, label_offset=.035)
        for view, suffix in (("projected", "objects_projected"), ("3d", "objects_3d")):
            figure = plt.figure(figsize=(10.2, 4.9))
            axes = [figure.add_subplot(1, 2, i+1, projection="3d" if view == "3d" else None) for i in range(2)]
            figure.subplots_adjust(left=.04, right=.96, top=.82, bottom=.14, wspace=.22)
            common = {"cloud": cloud, "style": style, "view": view,
                      "labels": True, "elev": 25, "azim": -75}
            viz.plot_simplicial_complex(simplicial, ax=axes[0], **common,
                                       title="Simplicial complex · filled faces")
            viz.plot_hyperdigraph(hyperdigraph, ax=axes[1], **common,
                                  title="Hyperdigraph · ordered sequences")
            figure.suptitle("Explicit topology objects", x=.04, y=.98, ha="left", fontsize=16, fontweight="bold")
            figure.text(.04, .06, "Illustrative coordinates • triangles and tetrahedra", color="#526170", fontsize=9)
            figure.text(.55, .06, "Reciprocal arrows remain separate • color marks degree", color="#526170", fontsize=9)
            export_bundle(figure, output, suffix)
            plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "visualization_gallery")
    args = parser.parse_args()
    print(main(args.output))
