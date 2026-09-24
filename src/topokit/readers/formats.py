"""Thin format adapters outside mathematical cores."""
import csv
import json
import math
from pathlib import Path
from ..data import PointCloud


_JSON_POINT_CLOUD_KEYS = frozenset({
    "schema", "schema_version", "coordinates", "points", "ids", "labels", "weights",
    "coordinate_units", "metadata",
})
_JSON_RESERVED_METADATA_KEYS = frozenset({
    "source_file", "format", "coordinate_units", "labels", "columns",
    "extra_fields", "selection_parent_ids", "source_atom_count",
    "selected_atom_count", "selection_source_ids", "json_schema",
    "json_schema_version", "json_coordinate_key",
})


def _units(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("coordinate_units must be a nonempty explicit label")
    return value.strip()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON objects must not repeat key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"JSON point clouds do not allow non-finite constant {value!r}")


def _json_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a JSON number (not a boolean)")
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError(f"{context} must be finite") from exc
    if not finite:
        raise ValueError(f"{context} must be finite")
    return value


def read_json(path, *, coordinate_units=None):
    """Read a native JSON point-cloud object.

    The canonical required ``coordinates`` member is a nonempty rectangular
    array of finite JSON numbers. Optional ``ids``, ``labels``, and ``weights`` arrays contain
    one item per point. ``coordinate_units`` is an explicit nonempty label and
    ``metadata`` is an object whose non-reserved keys are copied into the
    returned :class:`PointCloud` metadata. Optional identity fields are
    ``"schema": "topokit.point_cloud"`` and ``"schema_version": 1``.

    ``points`` is accepted as an ergonomic alias for ``coordinates``, but the
    two keys cannot occur together. A metadata/provenance-only JSON sidecar is
    not a point cloud and fails explicitly when it has neither coordinate key.
    An explicit ``coordinate_units=`` option must agree with an embedded units label; this
    reader never silently relabels or converts coordinates.
    """
    try:
        with Path(path).open(encoding="utf-8-sig") as handle:
            document = json.load(
                handle,
                object_pairs_hook=_unique_json_object,
                parse_constant=_invalid_json_constant,
            )
    except json.JSONDecodeError as exc:
        raise ValueError("JSON point cloud must contain valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("JSON point cloud root must be an object")
    coordinate_keys = tuple(key for key in ("coordinates", "points") if key in document)
    if not coordinate_keys:
        raise ValueError(
            "JSON point cloud requires a top-level 'coordinates' array; "
            "metadata/provenance-only JSON sidecars are not point clouds"
        )
    if len(coordinate_keys) != 1:
        raise ValueError("JSON point cloud cannot contain both 'coordinates' and 'points'")
    coordinate_key = coordinate_keys[0]
    unknown = set(document).difference(_JSON_POINT_CLOUD_KEYS)
    if unknown:
        raise ValueError(f"JSON point cloud contains unknown top-level keys: {sorted(unknown)}")
    if document.get("schema", "topokit.point_cloud") != "topokit.point_cloud":
        raise ValueError("JSON point-cloud schema must be 'topokit.point_cloud'")
    schema_version = document.get("schema_version", 1)
    if (isinstance(schema_version, bool) or not isinstance(schema_version, int)
            or schema_version != 1):
        raise ValueError("JSON point-cloud schema_version must be the integer 1")

    points = document[coordinate_key]
    if not isinstance(points, list) or not points:
        raise ValueError(f"JSON {coordinate_key!r} must be a nonempty array")
    if any(not isinstance(row, list) or not row for row in points):
        raise ValueError("every JSON point must be a nonempty coordinate array")
    dimension = len(points[0])
    if any(len(row) != dimension for row in points):
        raise ValueError(f"JSON {coordinate_key!r} must form a rectangular coordinate array")
    for row_index, row in enumerate(points):
        for column_index, value in enumerate(row):
            _json_number(value, f"JSON coordinate [{row_index}][{column_index}]")
    point_count = len(points)

    ids = document.get("ids")
    if "ids" in document:
        if not isinstance(ids, list) or len(ids) != point_count:
            raise ValueError("JSON 'ids' must contain one ID per point")
        if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in ids):
            raise ValueError("JSON point IDs must be strings or integers (not booleans)")

    labels = document.get("labels")
    if "labels" in document:
        if (not isinstance(labels, list) or len(labels) != point_count
                or any(not isinstance(value, str) for value in labels)):
            raise ValueError("JSON 'labels' must contain one string per point")

    weights = document.get("weights")
    if "weights" in document:
        if not isinstance(weights, list) or len(weights) != point_count:
            raise ValueError("JSON 'weights' must contain one number per point")
        for index, value in enumerate(weights):
            _json_number(value, f"JSON weight [{index}]")

    user_metadata = document.get("metadata", {})
    if not isinstance(user_metadata, dict):
        raise ValueError("JSON 'metadata' must be an object")
    conflicts = set(user_metadata).intersection(_JSON_RESERVED_METADATA_KEYS)
    if conflicts:
        raise ValueError(f"JSON metadata uses reader-reserved keys: {sorted(conflicts)}")

    embedded_units = (
        _units(document["coordinate_units"]) if "coordinate_units" in document else None
    )
    if coordinate_units is None:
        resolved_units = embedded_units or "unspecified"
    else:
        resolved_units = _units(coordinate_units)
        if embedded_units is not None and resolved_units != embedded_units:
            raise ValueError("coordinate_units option conflicts with the embedded JSON label")

    metadata = dict(user_metadata)
    metadata.update({
        "source_file": Path(path).name,
        "format": "json",
        "coordinate_units": resolved_units,
        "json_schema": "topokit.point_cloud",
        "json_schema_version": 1,
        "json_coordinate_key": coordinate_key,
    })
    if labels is not None:
        metadata["labels"] = tuple(labels)
    return PointCloud(points, ids=ids, weights=weights, metadata=metadata)


def read_csv(path, *, coordinate_columns=("x", "y", "z"), id_column="id", weight_column="weight",
             coordinate_units="input_units"):
    """Read selected columns, preserving every original column as metadata.

    Units are labels supplied by the caller, never inferred or converted. Use
    ``id_column=None`` or ``weight_column=None`` for PointCloud defaults.
    """
    coordinate_units = _units(coordinate_units)
    if isinstance(coordinate_columns, str):
        raise ValueError("coordinate_columns must be a sequence of column names")
    coordinate_columns = tuple(coordinate_columns)
    if (not coordinate_columns or any(not isinstance(name, str) or not name for name in coordinate_columns)
            or len(set(coordinate_columns)) != len(coordinate_columns)):
        raise ValueError("coordinate_columns must contain unique nonempty names")
    if any(name is not None and (not isinstance(name, str) or not name)
           for name in (id_column, weight_column)):
        raise ValueError("id_column and weight_column must be column names or None")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or any(not name.strip() for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV needs unique nonempty column headers")
        required = set(coordinate_columns) | {name for name in (id_column, weight_column) if name is not None}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"CSV is missing requested columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one point")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV rows must have exactly one field per header")
    try:
        points = [[float(row[column]) for column in coordinate_columns] for row in rows]
        weights = None if weight_column is None else [float(row[weight_column]) for row in rows]
    except ValueError as exc:
        raise ValueError("CSV coordinate and weight columns must contain numbers") from exc
    labels = None if id_column is None else [row[id_column] for row in rows]
    return PointCloud(points, labels, weights,
                      {"source_file": Path(path).name, "format": "csv", "coordinate_units": coordinate_units,
                       "coordinate_columns": coordinate_columns, "id_column": id_column,
                       "weight_column": weight_column,
                       "columns": {column: tuple(row[column] for row in rows) for column in columns}})


def read_xyz(path, *, coordinate_units="unspecified"):
    """Read a single XYZ record as points; labels are attributes, not chemistry."""
    coordinate_units = _units(coordinate_units)
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ requires a count and comment line")
    point_count = int(lines[0])
    rows = [line.split() for line in lines[2:] if line.strip()]
    if point_count <= 0 or len(rows) != point_count or any(len(row) < 4 for row in rows):
        raise ValueError("XYZ atom count/coordinate records are inconsistent (single frame only)")
    return PointCloud([[float(x) for x in row[1:4]] for row in rows],
                      metadata={"source_file": Path(path).name, "format": "xyz",
                                "labels": tuple(row[0] for row in rows),
                                "extra_fields": tuple(tuple(row[4:]) for row in rows),
                                "comment": lines[1], "coordinate_units": coordinate_units})
