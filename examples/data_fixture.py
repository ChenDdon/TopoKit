"""Example-specific fixture loading; the library has no bundled datasets."""
import json
from pathlib import Path
from topokit import PointCloud
from topokit.readers import read_csv


def load_demo_cloud():
    directory = Path(__file__).resolve().parent / "data"
    cloud = read_csv(directory / "point_cloud_24.csv")
    metadata = json.loads((directory / "point_cloud_24.json").read_text(encoding="utf-8"))
    metadata.update(cloud.metadata)
    metadata["subset_ids"] = tuple(label for label, selected in
                                    zip(cloud.ids, cloud.metadata["columns"]["selected"]) if selected == "1")
    return PointCloud(cloud.points, cloud.ids, cloud.weights, metadata)
