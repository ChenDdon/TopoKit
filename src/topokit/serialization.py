"""Portable, versioned numerical results; no pickle or arbitrary code loading."""

import dataclasses
import json
import math
from pathlib import Path
import numpy as np
from scipy import sparse
from . import results
from .data import PointCloud
from .postprocessing import FeatureResult, VectorResult

SCHEMA_VERSION = 1
_RESULT_TYPES = {cls.__name__: cls for cls in (
    results.HomologyResult, results.PersistenceInterval, results.PersistenceResult,
    results.SpectrumResult, FeatureResult, VectorResult)}


def to_jsonable(value):
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return {"__ndarray__": to_jsonable(value.tolist()), "dtype": str(value.dtype), "shape": list(value.shape)}
    if sparse.issparse(value):
        matrix = value.tocsr()
        return {"__csr__": True, "shape": list(matrix.shape), "data": to_jsonable(matrix.data),
                "indices": to_jsonable(matrix.indices), "indptr": to_jsonable(matrix.indptr)}
    if isinstance(value, PointCloud):
        return {"__type__": "PointCloud", "points": to_jsonable(value.points), "ids": to_jsonable(value.ids),
                "weights": to_jsonable(value.weights), "metadata": to_jsonable(value.metadata)}
    if isinstance(value, results.Topology):
        # The numerical core object is deliberately not pickled/rehydrated.
        return {"__type__": "TopologyDescription", "kind": value.kind,
                "metadata": to_jsonable(value.metadata), "cloud": to_jsonable(value.cloud)}
    if dataclasses.is_dataclass(value):
        return {"__type__": type(value).__name__, **{item.name: to_jsonable(getattr(value, item.name)) for item in dataclasses.fields(value)}}
    if isinstance(value, tuple):
        return {"__tuple__": [to_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        # Tagged mappings preserve tuple and integer IDs without string coercion.
        return {"__mapping__": [[to_jsonable(key), to_jsonable(item)] for key, item in value.items()]}
    raise TypeError(f"cannot serialize {type(value).__name__}; export numerical results, not live engines")


def from_jsonable(value):
    if isinstance(value, list):
        return [from_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__float__" in value:
        return float(value["__float__"])
    if "__ndarray__" in value:
        return np.asarray(from_jsonable(value["__ndarray__"]), dtype=value["dtype"]).reshape(value["shape"])
    if "__tuple__" in value:
        return tuple(from_jsonable(item) for item in value["__tuple__"])
    if "__mapping__" in value:
        return {from_jsonable(key): from_jsonable(item) for key, item in value["__mapping__"]}
    if "__csr__" in value:
        return sparse.csr_matrix((from_jsonable(value["data"]), from_jsonable(value["indices"]),
                                  from_jsonable(value["indptr"])), shape=tuple(value["shape"]))
    if "__type__" in value:
        type_name = value["__type__"]
        decoded_fields = {key: from_jsonable(item) for key, item in value.items() if key != "__type__"}
        if type_name == "PointCloud":
            return PointCloud(**decoded_fields)
        if type_name in _RESULT_TYPES:
            return _RESULT_TYPES[type_name](**decoded_fields)
        return {"type": type_name, **decoded_fields}
    return {key: from_jsonable(item) for key, item in value.items()}


def save_result(value, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "data": to_jsonable(value)}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def load_result(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported topokit result schema version")
    return from_jsonable(document["data"])
