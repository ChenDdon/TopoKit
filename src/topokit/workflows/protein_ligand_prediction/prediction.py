"""Predict pK with audited GBDT pipelines and their fitted scalers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _load_pipeline(folder, versions, schema_sha256, expected=None):
    import joblib

    receipt = read(folder / "MODEL.json")
    if versions != receipt["software"]:
        raise ValueError(f"Use the recorded GBDT runtime before loading: expected {receipt['software']}; got {versions}")
    if receipt.get("state") != "verified" or not receipt.get("pipeline_contains_fitted_scaler"):
        raise ValueError("Incomplete model bundle")
    required = {"pipeline.joblib", "scaler.npz", "feature_schema.json"}
    if not required.issubset(receipt["artifact_sha256"]):
        raise ValueError("Model/scaler/schema hashes are required")
    for name, digest in receipt["artifact_sha256"].items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError(f"Bundle artifact checksum mismatch: {name}")
    if schema_sha256 != receipt["feature_schema_sha256"]:
        raise ValueError("Feature schema differs from the fitted model")
    if expected is not None:
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"Ensemble member identity mismatch: {key}")
    pipeline = joblib.load(folder / "pipeline.joblib")
    if list(pipeline.named_steps) != ["scaler", "gbdt"]:
        raise ValueError("Unexpected pipeline steps")
    if (pipeline["scaler"].n_features_in_ != 27500
            or pipeline["gbdt"].n_features_in_ != 27500):
        raise ValueError("Model expects a different feature dimension")
    if expected is not None and pipeline["gbdt"].get_params() != receipt["parameters"]:
        raise ValueError("Estimator parameters differ from the member receipt")
    return receipt, pipeline


def _load_models(folder, versions, schema_sha256):
    """Accept an original single bundle or the final three-seed manifest."""
    if not (folder / "ENSEMBLE.json").exists():
        receipt, pipeline = _load_pipeline(folder, versions, schema_sha256)
        return receipt, [pipeline], None
    receipt = read(folder / "ENSEMBLE.json")
    if (receipt.get("format") != "topokit.gbdt_ensemble.v1"
            or receipt.get("state") != "verified"
            or receipt.get("aggregation") != "arithmetic_mean"
            or receipt.get("seeds") != [0, 1, 2]
            or receipt.get("weights") != [1 / 3] * 3
            or len(receipt.get("members", [])) != 3):
        raise ValueError("Require the complete equal-weight seeds 0/1/2 ensemble")
    if versions != receipt["software"]:
        raise ValueError("Use the recorded GBDT runtime before loading the ensemble")
    if schema_sha256 != receipt["feature_schema_sha256"]:
        raise ValueError("Feature schema differs from the fitted ensemble")
    pipelines = []
    common_parameters = None
    for seed, member in zip(receipt["seeds"], receipt["members"], strict=True):
        if member.get("seed") != seed or member.get("path") != f"seed_{seed}":
            raise ValueError("Ensemble member order/path mismatch")
        child = folder / member["path"]
        if sha256(child / "MODEL.json") != member["model_receipt_sha256"]:
            raise ValueError("Ensemble member receipt checksum mismatch")
        meta, pipeline = _load_pipeline(child, versions, schema_sha256, expected={
            "seed": seed, "training_set": receipt["training_set"], "recipe_id": receipt["recipe_id"]})
        params = dict(meta["parameters"])
        if params.pop("random_state") != seed:
            raise ValueError("Ensemble random_state differs from its declared seed")
        if common_parameters is not None and params != common_parameters:
            raise ValueError("Ensemble members use different hyperparameters")
        common_parameters = params
        pipelines.append(pipeline)
    return receipt, pipelines, receipt["seeds"]


def load_feature_matrix(features, sample_ids, recipe_id):
    import numpy as np

    features = Path(features)
    matrix = np.empty((len(sample_ids), 27500), dtype=np.float32)
    for i, sample in enumerate(sample_ids):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample):
            raise ValueError(f"Invalid sample ID: {sample!r}")
        path = features / "samples" / (sample + ".npy")
        record = read(features / "records" / (sample + ".json"))
        if (record.get("sample_id") != sample or record.get("recipe_id") != recipe_id
                or record.get("status") != "success"
                or record.get("output", {}).get("sha256") != sha256(path)):
            raise ValueError(f"Feature identity/checksum mismatch: {sample}")
        value = np.load(path, allow_pickle=False)
        if (value.shape != (10, 50, 55) or value.dtype != np.dtype("<f4")
                or not value.flags.c_contiguous or not np.isfinite(value).all()):
            raise ValueError(f"Invalid feature tensor: {sample}")
        matrix[i] = value.ravel(order="C")
    return matrix


def predict(model_dir, features, manifest, output, batch_size=128):
    import joblib
    import numpy as np
    import scipy
    import sklearn

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_dir, features, manifest, output = map(Path, (model_dir, features, manifest, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {output}")
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)
    receipt, pipelines, seeds = _load_models(model_dir, versions, sha256(features / "feature_schema.json"))
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["pdb_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Provide a nonempty manifest with unique pdb_id values")
    predictions = []
    for start in range(0, len(ids), batch_size):
        matrix = load_feature_matrix(features, ids[start:start + batch_size], receipt["recipe_id"])
        values = np.asarray([pipeline.predict(matrix) for pipeline in pipelines], dtype=np.float64)
        if values.shape != (len(pipelines), len(matrix)) or not np.isfinite(values).all():
            raise ValueError("Invalid member predictions")
        predictions.extend(np.mean(values, axis=0))
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pdb_id", "predicted_pK"])
        writer.writerows(zip(ids, predictions, strict=True))
    return dict(predictions=len(ids), training_set=receipt["training_set"], output=str(output),
                ensemble_size=len(pipelines), seeds=seeds)
