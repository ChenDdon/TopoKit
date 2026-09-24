"""Domain-independent point records. Weights are attributes, not alpha radii."""

from dataclasses import dataclass
from collections.abc import Mapping
import numbers
import numpy as np


def _coordinate_representatives(points):
    """First input row per exact coordinate, in input order, and row mapping."""
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    return np.sort(first), first[inverse]


def _selected_reader_metadata(metadata, selected_ids):
    """Update the documented molecular-reader view while retaining source data."""
    if "source_atom_count" not in metadata:
        return metadata
    selected_ids = tuple(selected_ids)
    selected_set = set(selected_ids)
    metadata["atom_count"] = len(selected_ids)
    metadata["selected_atom_count"] = len(selected_ids)
    metadata.setdefault("selection_source_ids", tuple(metadata.get("selection_parent_ids", ())))

    if "source_bonds" in metadata:
        source_bonds = tuple(metadata["source_bonds"])
        selected_bonds = tuple(
            bond for bond in source_bonds
            if len(tuple(bond.get("atoms", ()))) == 2
            and all(atom_id in selected_set for atom_id in bond["atoms"])
        )
        metadata["bonds"] = selected_bonds
        metadata["source_bond_count"] = len(source_bonds)
        metadata["selected_bond_count"] = len(selected_bonds)
        metadata["bond_selection_policy"] = "induced_by_selected_atom_ids"

    if "source_connectivity_records" in metadata:
        source_records = tuple(metadata["source_connectivity_records"])
        selected_records = []
        for record in source_records:
            if not record or record[0] not in selected_set:
                continue
            retained = (record[0],) + tuple(atom_id for atom_id in record[1:]
                                            if atom_id in selected_set)
            if len(retained) >= 2:
                selected_records.append(retained)
        metadata["connectivity_records"] = tuple(selected_records)
        metadata["source_connectivity_record_count"] = len(source_records)
        metadata["selected_connectivity_record_count"] = len(selected_records)
        metadata["connectivity_selection_policy"] = "induced_by_selected_atom_ids"
    return metadata


@dataclass(frozen=True, init=False, eq=False)
class PointCloud:
    points: np.ndarray
    ids: tuple
    weights: np.ndarray
    metadata: dict

    def __init__(self, points, ids=None, weights=None, metadata=None):
        coordinates = np.array(points, dtype=float, copy=True)
        if coordinates.ndim != 2 or coordinates.shape[1] == 0:
            raise ValueError("points must be a rectangular (n, ambient_dimension) array")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite")
        labels = tuple(range(len(coordinates))) if ids is None else tuple(ids)
        if len(labels) != len(coordinates):
            raise ValueError("ids must contain one label per point")
        if any(isinstance(label, bool) or not isinstance(label, (str, numbers.Integral)) for label in labels):
            raise TypeError("point IDs must be strings or integers (not booleans)")
        labels = tuple(int(label) if isinstance(label, numbers.Integral) else label for label in labels)
        if len(set(labels)) != len(labels):
            raise ValueError("point IDs must be unique")
        if isinstance(weights, Mapping):
            if set(weights) != set(labels):
                raise ValueError("weight mapping must contain exactly the point IDs")
            weights = [weights[label] for label in labels]
        weight_values = np.ones(len(labels), dtype=float) if weights is None else np.array(weights, dtype=float, copy=True)
        if weight_values.shape != (len(labels),) or not np.all(np.isfinite(weight_values)):
            raise ValueError("weights must contain one finite scalar per point")
        coordinates.setflags(write=False)
        weight_values.setflags(write=False)
        object.__setattr__(self, "points", coordinates)
        object.__setattr__(self, "ids", labels)
        object.__setattr__(self, "weights", weight_values)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def __len__(self):
        return len(self.ids)

    @property
    def index(self):
        return {label: index for index, label in enumerate(self.ids)}

    def subset(self, ids):
        """Select by stable IDs, preserving aligned attributes and source provenance.

        Molecular-reader metadata distinguishes the immutable source record
        from the selected view. ``source_atom_count``, raw declarations, and
        ``source_bonds`` remain unchanged; ``atom_count``,
        ``selected_atom_count``, and ``bonds`` describe this returned cloud.
        Current bonds/connectivity are the induced records whose endpoints are
        all selected. Other source-level records (cell, tags, properties, and
        non-atom lines) intentionally remain attached as provenance.
        """
        labels = tuple(ids)
        lookup = self.index
        try:
            indices = [lookup[label] for label in labels]
        except KeyError as error:
            raise ValueError(f"unknown point ID: {error.args[0]!r}") from error
        metadata = dict(self.metadata)
        # These documented reader fields are row-aligned. Unknown scientific
        # metadata remains untouched rather than guessing its interpretation.
        for key in ("labels", "extra_fields"):
            if key in metadata:
                if len(metadata[key]) != len(self):
                    raise ValueError(f"row-aligned metadata {key!r} has inconsistent length")
                metadata[key] = tuple(metadata[key][index] for index in indices)
        if "columns" in metadata:
            if any(len(column_values) != len(self) for column_values in metadata["columns"].values()):
                raise ValueError("row-aligned metadata columns have inconsistent length")
            metadata["columns"] = {key: tuple(column_values[index] for index in indices)
                                   for key, column_values in metadata["columns"].items()}
        metadata["selection_parent_ids"] = self.ids
        metadata = _selected_reader_metadata(metadata, labels)
        return PointCloud(self.points[indices], labels, self.weights[indices], metadata)

    def unique_coordinates(self):
        """Keep the first point at each exact coordinate, preserving input order.

        IDs, weights and documented row-aligned reader metadata come from that
        first point; they are not averaged. Near coordinates remain distinct.
        The original cloud is unchanged. Metadata records the retained input
        indices and the mapping from every original ID to its retained ID.
        Source bonds follow the existing induced-subset rule, not edge lifting.
        """
        retained, representatives = _coordinate_representatives(self.points)
        result = self.subset(tuple(self.ids[i] for i in retained))
        result.metadata["coordinate_deduplication"] = {
            "policy": "exact_keep_first", "input_count": len(self),
            "unique_count": len(retained), "removed_count": len(self)-len(retained),
            "retained_indices": tuple(int(i) for i in retained),
            "original_to_retained_id": {label: self.ids[int(i)]
                                        for label, i in zip(self.ids, representatives)},
        }
        return result


def as_point_cloud(value):
    return value if isinstance(value, PointCloud) else PointCloud(value)
