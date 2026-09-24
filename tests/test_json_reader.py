"""Native JSON PointCloud schema and dispatch tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from topokit import readers


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_minimal_json_point_cloud_dispatches_with_safe_defaults(tmp_path):
    path = _write(tmp_path / "minimal.JSON", {"coordinates": [[0, 1], [2.5, -3]]})
    cloud = readers.read(path)
    np.testing.assert_array_equal(cloud.points, [[0, 1], [2.5, -3]])
    assert cloud.ids == (0, 1)
    np.testing.assert_array_equal(cloud.weights, [1, 1])
    assert cloud.metadata == {
        "source_file": "minimal.JSON",
        "format": "json",
        "coordinate_units": "unspecified",
        "json_schema": "topokit.point_cloud",
        "json_schema_version": 1,
        "json_coordinate_key": "coordinates",
    }


def test_full_json_schema_preserves_ids_labels_weights_units_and_metadata(tmp_path):
    document = {
        "schema": "topokit.point_cloud",
        "schema_version": 1,
        "coordinates": [[1, 2, 3], [4, 5, 6]],
        "ids": ["site-a", 7],
        "labels": ["C", "N"],
        "weights": [2.55, -1.25],
        "coordinate_units": " angstrom ",
        "metadata": {
            "experiment": "synthetic",
            "nested": {"temperature": 298, "tags": ["dry", "control"]},
        },
    }
    cloud = readers.read_json(_write(tmp_path / "cloud.json", document))
    assert cloud.ids == ("site-a", 7)
    assert cloud.metadata["labels"] == ("C", "N")
    np.testing.assert_array_equal(cloud.weights, [2.55, -1.25])
    assert cloud.metadata["coordinate_units"] == "angstrom"
    assert cloud.metadata["experiment"] == "synthetic"
    assert cloud.metadata["nested"] == document["metadata"]["nested"]
    selected = cloud.subset([7])
    assert selected.ids == (7,)
    assert selected.metadata["labels"] == ("N",)
    assert selected.metadata["nested"] == document["metadata"]["nested"]


def test_json_explicit_units_must_agree_with_embedded_label(tmp_path):
    path = _write(tmp_path / "units.json", {
        "coordinates": [[0, 0, 0]], "coordinate_units": "angstrom",
    })
    assert readers.read_json(path, coordinate_units=" angstrom ").metadata["coordinate_units"] == "angstrom"
    with pytest.raises(ValueError, match="conflicts"):
        readers.read_json(path, coordinate_units="nm")
    no_units = _write(tmp_path / "explicit.json", {"coordinates": [[0, 0, 0]]})
    assert readers.read_json(no_units, coordinate_units="nm").metadata["coordinate_units"] == "nm"


def test_json_points_alias_is_supported_but_cannot_duplicate_coordinates(tmp_path):
    alias = readers.read_json(_write(tmp_path / "alias.json", {"points": [[1, 2, 3]]}))
    np.testing.assert_array_equal(alias.points, [[1, 2, 3]])
    assert alias.metadata["json_coordinate_key"] == "points"
    both = _write(tmp_path / "both.json", {
        "coordinates": [[0, 0]], "points": [[0, 0]],
    })
    with pytest.raises(ValueError, match="cannot contain both"):
        readers.read_json(both)


def test_existing_json_provenance_sidecar_is_not_silently_treated_as_points():
    with pytest.raises(ValueError, match="provenance-only JSON sidecars are not point clouds"):
        readers.read(DATA / "point_cloud_24.json")


@pytest.mark.parametrize(("document", "message"), [
    ([], "root must be an object"),
    ({"metadata": {}}, "top-level 'coordinates'"),
    ({"coordinates": [[0]], "pointz": [[1]]}, "unknown top-level"),
    ({"schema": "other", "coordinates": [[0]]}, "schema must"),
    ({"schema_version": True, "coordinates": [[0]]}, "schema_version"),
    ({"schema_version": 1.0, "coordinates": [[0]]}, "schema_version"),
    ({"schema_version": 2, "coordinates": [[0]]}, "schema_version"),
    ({"coordinates": []}, "nonempty array"),
    ({"coordinates": "0"}, "nonempty array"),
    ({"coordinates": [[]]}, "nonempty coordinate"),
    ({"coordinates": [[0, 1], [2]]}, "rectangular"),
    ({"coordinates": [[False]]}, "not a boolean"),
    ({"coordinates": [["0"]]}, "JSON number"),
    ({"coordinates": [[0]], "ids": None}, "one ID per point"),
    ({"coordinates": [[0]], "ids": [False]}, "not booleans"),
    ({"coordinates": [[0]], "ids": [1.5]}, "strings or integers"),
    ({"coordinates": [[0], [1]], "ids": [1, 1]}, "unique"),
    ({"coordinates": [[0]], "labels": None}, "one string per point"),
    ({"coordinates": [[0]], "labels": [1]}, "one string per point"),
    ({"coordinates": [[0]], "weights": None}, "one number per point"),
    ({"coordinates": [[0]], "weights": [True]}, "not a boolean"),
    ({"coordinates": [[0]], "metadata": []}, "must be an object"),
    ({"coordinates": [[0]], "metadata": {"format": "mine"}}, "reader-reserved"),
    ({"coordinates": [[0]], "coordinate_units": None}, "nonempty explicit label"),
])
def test_json_schema_rejects_ambiguous_or_lossy_documents(tmp_path, document, message):
    path = _write(tmp_path / "invalid.json", document)
    with pytest.raises((TypeError, ValueError), match=message):
        readers.read_json(path)


@pytest.mark.parametrize("text, message", [
    ("{", "valid JSON"),
    ('{"coordinates": [[NaN]]}', "non-finite constant"),
    ('{"coordinates": [[1e999]]}', "must be finite"),
    ('{"coordinates": [[0]], "coordinates": [[1]]}', "must not repeat key"),
    ('{"coordinates": [[0]], "metadata": {"x": 1, "x": 2}}', "must not repeat key"),
])
def test_json_text_parser_rejects_invalid_nonfinite_and_duplicate_input(tmp_path, text, message):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        readers.read_json(path)
