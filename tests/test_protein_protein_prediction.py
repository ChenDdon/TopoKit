"""Integrity and inference checks for the public PPI Topo model contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from topokit.workflows.protein_protein_prediction import prediction as p
from topokit.workflows.protein_protein_prediction.features import RECIPE_ID, schema


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _refresh(folder):
    _json(folder / "run.json", {
        "status": "complete", "recipe_id": RECIPE_ID, "feature_count": 5040,
        "artifact_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                            for name in ("features.npy", "sample_ids.json", "feature_schema.json")}})


@pytest.fixture
def store(tmp_path):
    folder = tmp_path / "features"
    folder.mkdir()
    np.save(folder / "features.npy", np.arange(2 * 5040, dtype="<f4").reshape(2, 5040), allow_pickle=False)
    _json(folder / "sample_ids.json", ["second", "first"])
    _json(folder / "feature_schema.json", schema())
    _refresh(folder)
    return folder


def test_feature_store_preserves_order(store):
    ids, matrix = p.load_feature_store(store)
    assert ids == ("second", "first")
    assert matrix.shape == (2, 5040)
    assert matrix[1, 0] == 5040


@pytest.mark.parametrize("bad", ["float64", "nonfinite", "fortran", "dimension", "duplicate_ids", "schema"])
def test_reject_invalid_features_even_with_updated_hashes(store, bad):
    matrix = np.load(store / "features.npy")
    if bad == "float64":
        matrix = matrix.astype(np.float64)
    elif bad == "nonfinite":
        matrix[0, 0] = np.nan
    elif bad == "fortran":
        matrix = np.asfortranarray(matrix)
    elif bad == "dimension":
        matrix = matrix[:, :-1]
    elif bad == "duplicate_ids":
        _json(store / "sample_ids.json", ["same", "same"])
    else:
        changed = schema()
        changed["recipe_id"] = "different"
        _json(store / "feature_schema.json", changed)
    np.save(store / "features.npy", matrix, allow_pickle=False)
    _refresh(store)
    with pytest.raises(ValueError):
        p.load_feature_store(store)


def test_reject_artifact_tampering(store):
    _json(store / "sample_ids.json", ["changed", "first"])
    with pytest.raises(ValueError, match="checksum"):
        p.load_feature_store(store)


@pytest.mark.parametrize("change", [{"status": "partial"}, {"recipe_id": "wrong"}, {"artifact_sha256": {}}])
def test_reject_incomplete_store(store, change):
    run = json.loads((store / "run.json").read_text())
    run.update(change)
    _json(store / "run.json", run)
    with pytest.raises(ValueError):
        p.load_feature_store(store)


@pytest.fixture
def mock_bundle(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    folder = tmp_path / "models"
    folder.mkdir()
    x = np.arange(8 * 5040, dtype=np.float32).reshape(8, 5040)
    y = np.array([-1, -2, -3, -4, -5, -6, -7, -8], dtype=float)
    pipelines = []
    members = []
    for seed in range(3):
        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("gbdt", GradientBoostingRegressor(n_estimators=2, max_depth=1,
                                                                subsample=0.8, random_state=seed))])
        pipeline.fit(x, y)
        path = folder / f"seed_{seed}.joblib"
        joblib.dump(pipeline, path)
        members.append({"seed": seed, "file": path.name, "sha256": p._sha256(path)})
        pipelines.append(pipeline)
    params = pipelines[0]["gbdt"].get_params()
    params.pop("random_state")
    profile = {"profile_id": "test-only", "members": members, "software": p._runtime_versions(),
               "parameters": params, "training_complexes": len(y)}
    monkeypatch.setattr(p, "model_profile", lambda: profile)
    return folder, profile, pipelines


def test_predict_averages_fitted_pipelines_and_keeps_row_order(store, mock_bundle):
    folder, _, pipelines = mock_bundle
    x = np.load(store / "features.npy")
    expected = np.stack([model.predict(x) for model in pipelines], axis=1)
    result = p.predict_gbdt(store, folder)
    assert result.sample_ids == ("second", "first")
    np.testing.assert_array_equal(result.members, expected)
    np.testing.assert_array_equal(result.prediction, expected.mean(axis=1))
    assert result.model_profile_id == "test-only"


def test_all_hashes_checked_before_any_deserialization(store, mock_bundle, monkeypatch):
    import joblib
    folder, _, _ = mock_bundle
    (folder / "seed_2.joblib").write_bytes(b"tampered")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Must not deserialize any model"))
    with pytest.raises(ValueError, match="checksum"):
        p.predict_gbdt(store, folder)


def test_runtime_guard_precedes_deserialization(store, mock_bundle, monkeypatch):
    import joblib
    folder, profile, _ = mock_bundle
    profile["software"] = dict(profile["software"], sklearn="0.invalid")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Must not deserialize mismatched runtime"))
    with pytest.raises(ValueError, match="recorded PPI GBDT runtime"):
        p.predict_gbdt(store, folder)


def test_fitted_model_identity_guard(store, mock_bundle):
    folder, profile, _ = mock_bundle
    profile["training_complexes"] += 1
    with pytest.raises(ValueError, match="training identity"):
        p.predict_gbdt(store, folder)


def test_python_patch_compatibility_is_explicit(store, mock_bundle, monkeypatch):
    folder, profile, _ = mock_bundle
    versions = dict(profile["software"])
    versions["python"] = ".".join(versions["python"].split(".")[:2] + ["999"])
    monkeypatch.setattr(p, "_runtime_versions", lambda: versions)
    assert p.predict_gbdt(store, folder).prediction.shape == (2,)


def test_python_minor_mismatch_is_rejected(store, mock_bundle, monkeypatch):
    import joblib
    folder, profile, _ = mock_bundle
    versions = dict(profile["software"], python="0.0.0")
    monkeypatch.setattr(p, "_runtime_versions", lambda: versions)
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Must not deserialize mismatched Python"))
    with pytest.raises(ValueError, match="recorded PPI GBDT runtime"):
        p.predict_gbdt(store, folder)


def test_nonfinite_model_predictions_are_rejected(store, mock_bundle, monkeypatch):
    from sklearn.pipeline import Pipeline
    folder, _, _ = mock_bundle
    monkeypatch.setattr(Pipeline, "predict", lambda self, x: np.full(len(x), np.nan))
    with pytest.raises(ValueError, match="invalid predictions"):
        p.predict_gbdt(store, folder)


def test_packaged_model_profile_matches_feature_contract():
    profile = p.model_profile()
    assert profile["recipe_id"] == RECIPE_ID
    assert profile["feature_count"] == 5040
    assert profile["feature_shape"] == [10, 14, 36]
    assert [member["seed"] for member in profile["members"]] == [0, 1, 2]
    assert profile["aggregation"] == "arithmetic_mean"
    assert profile["weights"] == [1 / 3] * 3
    assert profile["target"]["unit"] == "kcal/mol"
    assert profile["target"]["transformation"] == "none; signed original values"
