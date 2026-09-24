"""Inference guards: feature identity, raw layout and failure behavior."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("alpha15_prediction", Path(__file__).parents[1] / "predict.py")
prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prediction)


@pytest.fixture
def features(tmp_path):
    (tmp_path / "samples").mkdir()
    (tmp_path / "records").mkdir()
    value = np.arange(27500, dtype=np.float32).reshape(10, 50, 55)
    np.save(tmp_path / "samples/example.npy", value)
    record = {"sample_id": "example", "recipe_id": "recipe", "status": "success",
              "output": {"sha256": prediction.sha256(tmp_path / "samples/example.npy")}}
    (tmp_path / "records/example.json").write_text(json.dumps(record))
    return tmp_path, value


def test_raw_c_order_preserved_without_scaling(features):
    root, value = features
    got = prediction.load_feature_matrix(root, ["example"], "recipe")
    np.testing.assert_array_equal(got, value.reshape(1, -1, order="C"))
    assert got.dtype == np.float32


@pytest.mark.parametrize("key,value", [("recipe_id", "other"), ("status", "failed"), ("sample_id", "wrong")])
def test_wrong_identity_blocks(features, key, value):
    root, _ = features
    p = root / "records/example.json"
    record = json.loads(p.read_text()); record[key] = value
    p.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity/checksum"):
        prediction.load_feature_matrix(root, ["example"], "recipe")


def test_corrupt_tensor_blocks_before_loading(features):
    root, _ = features
    (root / "samples/example.npy").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="identity/checksum"):
        prediction.load_feature_matrix(root, ["example"], "recipe")


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite", "fortran"])
def test_invalid_tensor_blocks_even_with_matching_checksum(features, kind):
    root, value = features
    if kind == "shape": value = value.reshape(50, 550)
    if kind == "dtype": value = value.astype(np.float64)
    if kind == "nonfinite": value[0, 0, 0] = np.nan
    if kind == "fortran": value = np.asfortranarray(value)
    path = root / "samples/example.npy"; np.save(path, value)
    p = root / "records/example.json"; record = json.loads(p.read_text())
    record["output"]["sha256"] = prediction.sha256(path); p.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Invalid feature tensor"):
        prediction.load_feature_matrix(root, ["example"], "recipe")


def test_id_cannot_escape_feature_directory(features):
    with pytest.raises(ValueError, match="Invalid sample ID"):
        prediction.load_feature_matrix(features[0], ["../../other"], "recipe")


def test_missing_feature_is_not_imputed(features):
    with pytest.raises(FileNotFoundError):
        prediction.load_feature_matrix(features[0], ["absent"], "recipe")
