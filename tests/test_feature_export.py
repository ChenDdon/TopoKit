"""User manifests produce complete, prediction-compatible feature stores."""
import csv
import hashlib
import importlib.util
from importlib.resources import files
import json
from pathlib import Path

import numpy as np
import pytest

from topokit.workflows.protein_ligand_prediction import featurize, features
from topokit.workflows.protein_ligand_prediction.prediction import load_feature_matrix

SCRIPT = Path(__file__).resolve().parents[1] / "workflows/protein_ligand_prediction/extract_features.py"
SPEC = importlib.util.spec_from_file_location("feature_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def pair(tmp_path):
    protein = tmp_path / "protein.pdb"
    protein.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    ligand = tmp_path / "ligand.mol2"
    ligand.write_text("@<TRIPOS>MOLECULE\nsynthetic\n1 0 0 0 0\nSMALL\nNO_CHARGES\n"
                      "@<TRIPOS>ATOM\n1 C1 2.0 0.0 0.0 C.3 1 LIG 0.0\n")
    return {"sample_id": "custom_case", "protein_file": protein, "ligand_file": ligand}


def test_export_matches_public_api_and_prediction_loader(tmp_path):
    row = pair(tmp_path)
    out = tmp_path / "result"
    result = exporter.export_features([row], out)
    assert result["status"] == "complete"
    expected = featurize(row["protein_file"], row["ligand_file"])
    np.testing.assert_array_equal(np.load(out / "samples/custom_case.npy"), expected)
    assert (out / "feature_schema.json").read_bytes() == files(features.__package__).joinpath(
        "reference_schema.json").read_bytes()
    matrix = load_feature_matrix(out, [row["sample_id"]], result["recipe_id"])
    np.testing.assert_array_equal(np.load(out / "features.npy"), matrix)
    assert json.loads((out / "sample_ids.json").read_text()) == [row["sample_id"]]
    with (out / "manifest.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["pdb_id"] == row["sample_id"]
    record = json.loads((out / "records/custom_case.json").read_text())
    assert record["inputs"]["protein_file"]["sha256"] == hashlib.sha256(row["protein_file"].read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        exporter.export_features([row], out)


def test_failure_is_recorded_and_no_partial_matrix_is_published(tmp_path):
    row = pair(tmp_path)
    broken = {**row, "sample_id": "broken", "ligand_file": tmp_path / "missing.mol2"}
    out = tmp_path / "partial"
    result = exporter.export_features([row, broken], out)
    assert result["status"] == "failed" and result["failed_samples"] == ["broken"]
    assert (out / "samples/custom_case.npy").is_file()
    assert not (out / "features.npy").exists()
    assert not (out / "sample_ids.json").exists()
    assert json.loads((out / "records/broken.json").read_text())["status"] == "failed"


def test_manifest_paths_are_relative_to_csv_and_duplicates_rejected(tmp_path, monkeypatch):
    row = pair(tmp_path)
    manifest = tmp_path / "pairs.csv"
    text = "sample_id,protein_file,ligand_file\ncustom_case,protein.pdb,ligand.mol2\n"
    manifest.write_text(text)
    monkeypatch.chdir(tmp_path.parent)
    assert exporter.load_manifest(manifest) == [row]
    manifest.write_text(text + "custom_case,protein.pdb,ligand.mol2\n")
    with pytest.raises(ValueError, match="duplicate"):
        exporter.load_manifest(manifest)


def test_resource_failure_is_not_encoded_as_zeros(tmp_path):
    row = pair(tmp_path)
    out = tmp_path / "limited"
    result = exporter.export_features([row], out, max_dense_entries=1)
    assert result["status"] == "failed"
    record = json.loads((out / "records/custom_case.json").read_text())
    assert record["error"]["type"] == "ResourceLimitError"
    assert not (out / "samples/custom_case.npy").exists()


def test_alternate_conformer_warning_preserves_raw_recipe(tmp_path):
    row = pair(tmp_path)
    original = row["protein_file"].read_text().splitlines()[0]
    alternate = original[:16] + "A" + original[17:]
    row["protein_file"].write_text(alternate + "\nEND\n")
    assert "alternate-location" in exporter.input_warnings(row["protein_file"])[0]
    result = exporter.export_features([row], tmp_path / "warned")
    assert result["status"] == "complete" and row["sample_id"] in result["warnings"]
