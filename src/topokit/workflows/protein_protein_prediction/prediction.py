"""Inference with the frozen three-seed PPI Topo GBDT ensemble.

Optional scikit-learn and joblib imports occur only during prediction. Serialized
models are executable objects: pass only locally trusted model files. Their
bytes must match the package's pinned profile before any deserialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from .features import RECIPE_ID, schema


@dataclass(frozen=True)
class PredictionResult:
    """Signed binding free energy in kcal/mol, in the feature store's row order."""

    sample_ids: tuple[str, ...]
    prediction: np.ndarray
    members: np.ndarray
    model_profile_id: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def model_profile() -> dict[str, Any]:
    """Return the packaged model identity, hashes and compatible runtime."""
    return json.loads(resources.files(__package__).joinpath("model_profile.json").read_text(encoding="utf-8"))


def load_feature_store(folder: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read one completed, checksum-verified public PPI Topo feature batch."""
    folder = Path(folder)
    run = _read_json(folder / "run.json")
    if (not isinstance(run, dict) or run.get("status") != "complete"
            or run.get("recipe_id") != RECIPE_ID or run.get("feature_count") != 5040):
        raise ValueError("Require a complete PPI Topo feature store with the selected recipe")
    hashes = run.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Feature store is missing artifact checksums")
    for name in ("features.npy", "sample_ids.json", "feature_schema.json"):
        if hashes.get(name) != _sha256(folder / name):
            raise ValueError(f"Feature artifact checksum mismatch: {name}")
    expected_schema = (json.dumps(schema(), sort_keys=True, indent=2) + "\n").encode("utf-8")
    if (folder / "feature_schema.json").read_bytes() != expected_schema:
        raise ValueError("Feature schema differs from the selected PPI Topo recipe")
    ids = _read_json(folder / "sample_ids.json")
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Require nonempty, unique sample IDs")
    matrix = np.load(folder / "features.npy", allow_pickle=False)
    if (matrix.shape != (len(ids), 5040) or matrix.dtype != np.dtype("<f4")
            or not matrix.flags.c_contiguous or not np.isfinite(matrix).all()):
        raise ValueError("Require finite C-order float32 PPI Topo features with shape (N, 5040)")
    return tuple(ids), matrix


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for key, distribution in (("numpy", "numpy"), ("scipy", "scipy"),
                              ("sklearn", "scikit-learn"), ("joblib", "joblib")):
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise ImportError("PPI prediction requires the recorded runtime in ml/requirements.txt") from exc
    return versions


def _validate_pipeline(pipeline: Any, profile: dict[str, Any], seed: int) -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if (not isinstance(pipeline, Pipeline)
            or list(pipeline.named_steps) != ["scaler", "gbdt"]
            or not isinstance(pipeline["scaler"], StandardScaler)
            or not isinstance(pipeline["gbdt"], GradientBoostingRegressor)):
        raise ValueError("Expected a fitted StandardScaler + GBDT pipeline")
    scaler, estimator = pipeline["scaler"], pipeline["gbdt"]
    if (getattr(pipeline, "n_features_in_", None) != 5040
            or getattr(scaler, "n_features_in_", None) != 5040
            or getattr(estimator, "n_features_in_", None) != 5040
            or getattr(estimator, "n_estimators_", None) != profile["parameters"]["n_estimators"]
            or not np.all(np.asarray(getattr(scaler, "n_samples_seen_", -1)) == profile["training_complexes"])):
        raise ValueError("Fitted model dimensions or training identity differ from the selected profile")
    expected_parameters = dict(profile["parameters"], random_state=seed)
    if estimator.get_params() != expected_parameters:
        raise ValueError("Estimator parameters differ from the selected model profile")
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError("Expected the fitted default StandardScaler")


def predict_gbdt(feature_store: str | Path, models: str | Path) -> PredictionResult:
    """Predict signed DeltaG_bind (kcal/mol) using trusted external model files.

    ``feature_store`` is a complete batch from the public PPI exporter.
    ``models`` contains ``seed_0.joblib``, ``seed_1.joblib`` and ``seed_2.joblib``
    matching :func:`model_profile`. Features use canonical protein1/protein2
    order: descending observed ATOM residue count before cropping, with ties
    resolved by first chain appearance in the structure. Prediction preserves
    that feature order, with no partner-swap augmentation, refitting, imputation
    or target transformation.
    All file hashes, Python major/minor and exact dependency versions are checked
    before unpickling. Python patch releases within the recorded series are accepted.
    """
    ids, matrix = load_feature_store(feature_store)
    profile = model_profile()
    model_folder = Path(models)
    paths = []
    for member in profile["members"]:
        path = model_folder / member["file"]
        if _sha256(path) != member["sha256"]:
            raise ValueError(f"Model checksum mismatch: {member['file']}")
        paths.append(path)
    versions = _runtime_versions()
    expected = profile["software"]
    compatible_python = versions["python"].split(".")[:2] == expected["python"].split(".")[:2]
    if not compatible_python or any(versions[key] != expected[key] for key in expected if key != "python"):
        raise ValueError(f"Use the recorded PPI GBDT runtime before loading (Python patch releases accepted): expected {expected}; got {versions}")
    import joblib

    predictions = []
    for member, path in zip(profile["members"], paths):
        pipeline = joblib.load(path)
        _validate_pipeline(pipeline, profile, member["seed"])
        values = np.asarray(pipeline.predict(matrix), dtype=np.float64)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError("Model produced invalid predictions")
        predictions.append(values)
    members = np.stack(predictions, axis=1)
    prediction = members.mean(axis=1)
    if not np.isfinite(prediction).all():
        raise ValueError("Ensemble produced nonfinite predictions")
    return PredictionResult(ids, prediction, members, profile["profile_id"])
