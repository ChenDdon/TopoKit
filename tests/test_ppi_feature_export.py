"""Portable PPI manifests retain partner definitions, row order and failures."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from topokit.workflows.protein_protein_prediction import featurize, schema

SCRIPT = Path(__file__).resolve().parents[1] / "workflows/protein_protein_prediction/extract_features.py"
SPEC = importlib.util.spec_from_file_location("ppi_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def complex_row(tmp_path):
    path = tmp_path / "complex.pdb"
    atoms = [("A", "CA", "C", 0., 0., 0.), ("A", "C", "C", 1., 0., 0.),
             ("A", "N", "N", 0., 1., 0.), ("B", "CA", "C", 3., 0., 0.),
             ("B", "C", "C", 3., 1., 1.), ("B", "O", "O", 4., 0., 0.)]
    lines = [f"ATOM  {i:5d} {name:>4s} ALA {chain}{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
             for i, (chain, name, element, x, y, z) in enumerate(atoms, 1)]
    path.write_text("\n".join(lines) + "\nEND\n")
    return {"sample_id": "custom_pair", "structure_file": path,
            "partner_a_chains": ["A"], "partner_b_chains": ["B"]}


def test_export_matches_api_and_preserves_order_and_hashes(tmp_path):
    assert (SCRIPT.parent / "DEFAULT_RECIPE.json").read_bytes() == exporter.canonical_bytes(schema())
    row = complex_row(tmp_path)
    other = {**row, "sample_id": "second_pair"}
    output = tmp_path / "features"
    result = exporter.export_features([other, row], output)
    assert result["status"] == "complete" and result["feature_count"] == 5040
    expected = featurize(row["structure_file"], ["A"], ["B"])
    matrix = np.load(output / "features.npy", allow_pickle=False)
    np.testing.assert_array_equal(matrix, np.stack([expected.ravel(), expected.ravel()]))
    assert matrix.dtype == np.float32 and matrix.flags.c_contiguous
    assert json.loads((output / "sample_ids.json").read_text()) == ["second_pair", "custom_pair"]
    assert (output / "feature_schema.json").read_bytes() == exporter.canonical_bytes(schema())
    for name, digest in result["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    record = json.loads((output / "records/custom_pair.json").read_text())
    assert record["diagnostics"]["source_partner_chains"] == [["A"], ["B"]]
    assert record["input"]["sha256"] == exporter.sha256(row["structure_file"])
    with pytest.raises(FileExistsError):
        exporter.export_features([row], output)


def test_failed_sample_does_not_publish_partial_matrix(tmp_path):
    row = complex_row(tmp_path)
    broken = {**row, "sample_id": "missing_chain", "partner_b_chains": ["Z"]}
    out = tmp_path / "partial"
    result = exporter.export_features([row, broken], out)
    assert result["status"] == "failed" and result["failed_samples"] == ["missing_chain"]
    assert (out / "samples/custom_pair.npy").is_file()
    assert not (out / "features.npy").exists()
    assert not (out / "sample_ids.json").exists()
    assert json.loads((out / "records/missing_chain.json").read_text())["error"]["type"] == "ValueError"


def test_relative_manifest_paths_and_explicit_chain_policy(tmp_path, monkeypatch):
    row = complex_row(tmp_path)
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "structure_file", "partner_a_chains", "partner_b_chains", "allowed_missing_chains"])
        writer.writerow([row["sample_id"], "complex.pdb", "A;M", "B", "M"])
    monkeypatch.chdir(tmp_path.parent)
    rows = exporter.load_manifest(manifest)
    assert rows[0]["structure_file"] == row["structure_file"]
    assert rows[0]["partner_a_chains"] == ["A", "M"]
    result = exporter.export_features(rows, tmp_path / "explicit")
    assert result["status"] == "complete"
    manifest.write_text(manifest.read_text().replace("A;M,B,M", "A;M,B,Z"))
    with pytest.raises(ValueError, match="declared partner"):
        exporter.load_manifest(manifest)


@pytest.mark.parametrize("row", ["same,complex.pdb,A,A", "same,complex.pdb,A;,B", "same,complex.pdb,A,A;B"])
def test_invalid_partner_assignments_rejected(tmp_path, row):
    complex_row(tmp_path)
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("sample_id,structure_file,partner_a_chains,partner_b_chains\n" + row + "\n")
    with pytest.raises(ValueError):
        exporter.load_manifest(manifest)


def test_resource_failure_is_not_a_zero_feature_vector(tmp_path):
    row = complex_row(tmp_path)
    out = tmp_path / "limited"
    result = exporter.export_features([row], out, max_dense_entries=1)
    assert result["status"] == "failed"
    assert not (out / "samples/custom_pair.npy").exists()
    assert json.loads((out / "records/custom_pair.json").read_text())["error"]["type"] == "ResourceLimitError"


@pytest.mark.parametrize("ids", [("pair", "PAIR"), ("CON", "other"), ("aux", "other"), ("COM1", "other")])
def test_portable_sample_names_rejected_before_writing(tmp_path, ids):
    row = complex_row(tmp_path)
    rows = [{**row, "sample_id": value} for value in ids]
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        exporter.export_features(rows, out)
    assert not out.exists()
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("sample_id,structure_file,partner_a_chains,partner_b_chains\n" +
                        "".join(f"{value},complex.pdb,A,B\n" for value in ids))
    with pytest.raises(ValueError):
        exporter.load_manifest(manifest)
