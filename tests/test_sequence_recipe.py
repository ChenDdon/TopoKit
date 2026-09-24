"""CPZ routing, hash guards and export contracts without external model weights."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from topokit.workflows.protein_ligand_prediction import sequence


RECIPE_FOLDER = (Path(__file__).resolve().parents[1] / "workflows" /
                 "protein_ligand_prediction" / "sequence")


def test_profiles_keep_distinct_identity_and_historical_default():
    old = sequence.SequenceEncoder()
    selected = sequence.SequenceEncoder(ligand_profile="cpz")
    assert old.ligand_profile == "chembl27"
    assert old.recipe_id == sequence.RECIPE_ID == "sequence-esm2-t33-chembl27-bos-v1"
    assert selected.recipe_id == sequence.CPZ_RECIPE_ID == "sequence-esm2-t33-cpz-bos-v1"
    assert selected.protein_model is None and selected.ligand_model is None
    with pytest.raises(ValueError, match="ligand_profile"):
        sequence.SequenceEncoder(ligand_profile="auto")
    with pytest.raises(AttributeError):
        selected.ligand_profile = "chembl27"


@pytest.mark.parametrize("profile,label", [("chembl27", "ChEMBL27"), ("cpz", "CPZ")])
def test_untrusted_checkpoint_rejected_before_optional_imports(tmp_path, monkeypatch, profile, label):
    (tmp_path / "checkpoint_best.pt").write_bytes(b"not a trusted checkpoint")
    # An accidental import of either optional dependency now fails immediately.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "topokit.workflows.protein_ligand_prediction._chembl", None)
    encoder = sequence.SequenceEncoder(chembl_dir=tmp_path, ligand_profile=profile)
    with pytest.raises(ValueError, match=f"{label} checkpoint identity mismatch"):
        encoder.load_ligand()


def test_cpz_rejects_chembl_checkpoint_and_wrong_dictionary(tmp_path, monkeypatch):
    encoder = sequence.SequenceEncoder(chembl_dir=tmp_path, ligand_profile="cpz")
    monkeypatch.setattr(sequence, "sha256_file", lambda path: sequence.CHEMBL_SHA256)
    with pytest.raises(ValueError, match="CPZ checkpoint identity mismatch"):
        encoder.load_ligand()
    monkeypatch.setattr(sequence, "sha256_file", lambda path: (
        sequence.CPZ_SHA256 if path.name == "checkpoint_best.pt" else sequence.CHEMBL_DICT_SHA256))
    with pytest.raises(ValueError, match="CPZ vocabulary identity mismatch"):
        encoder.load_ligand()


@pytest.mark.parametrize("profile,checkpoint,dictionary", [
    ("chembl27", sequence.CHEMBL_SHA256, sequence.CHEMBL_DICT_SHA256),
    ("cpz", sequence.CPZ_SHA256, sequence.CPZ_DICT_SHA256),
])
def test_verified_profile_reuses_loader_and_freezes_model(tmp_path, monkeypatch, profile, checkpoint, dictionary):
    calls = []
    model = SimpleNamespace(requires_grad_=lambda value: calls.append(("grad", value)))
    config = SimpleNamespace(embed_dim=512, max_positions=256)
    def build(checkpoint_path, dict_path, device):
        calls.append((checkpoint_path, dict_path, device))
        return model, "vocab", config
    monkeypatch.setattr(sequence, "sha256_file", lambda path: (
        checkpoint if path.name == "checkpoint_best.pt" else dictionary))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda value: value))
    monkeypatch.setitem(sys.modules, "topokit.workflows.protein_ligand_prediction._chembl",
                        SimpleNamespace(build_model_from_checkpoint=build))
    encoder = sequence.SequenceEncoder(chembl_dir=tmp_path, ligand_profile=profile)
    assert encoder.load_ligand() is encoder
    assert encoder.ligand_model is model and encoder.ligand_config is config
    assert calls == [(str(tmp_path / "checkpoint_best.pt"), str(tmp_path / "dict.txt"), "cpu"),
                     ("grad", False)]


def test_recipe_asset_pins_match_installed_encoder():
    recipe = json.loads((RECIPE_FOLDER / "RECIPE.json").read_text())
    assert recipe["recipe_id"] == sequence.SequenceEncoder(ligand_profile="cpz").recipe_id
    assert recipe["ligand"]["asset_sha256"] == {
        "checkpoint_best.pt": sequence.CPZ_SHA256, "dict.txt": sequence.CPZ_DICT_SHA256}
    assert recipe["protein"]["asset_sha256"] == {
        "model.safetensors": sequence.ESM_SHA256, **sequence.ESM_METADATA}


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("sequence_extraction_runner",
                                                 RECIPE_FOLDER / "extract_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("record", [
    [], {"protein_chains": "ACDE", "smiles": "CC"},
    {"protein_chains": [], "smiles": "CC"},
    {"protein_chains": ["AC-D"], "smiles": "CC"},
    {"protein_chains": ["ACDE"], "smiles": " CC"},
])
def test_runner_rejects_ambiguous_or_invalid_input(runner, tmp_path, record):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runner.read_input(path)


def test_runner_exports_raw_features_and_token_receipt(runner, tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    record = {"id": "explicit-case", "protein_chains": ["ACDE", "ACDE"], "smiles": "CC"}
    input_path.write_text(json.dumps(record))
    (tmp_path / "dict.txt").write_text("C 1\n")
    output = tmp_path / "output"
    calls = []
    class FakeEncoder:
        recipe_id = sequence.CPZ_RECIPE_ID
        ligand_profile = "cpz"
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def encode(self, chains, smiles, **kwargs):
            calls.append((chains, smiles, kwargs))
            return np.arange(1792, dtype=np.float32), {"tokens_omitted": 0, "unknown_symbols": []}
    monkeypatch.setattr(sequence, "SequenceEncoder", FakeEncoder)
    monkeypatch.setattr(runner, "verify_assets", lambda *args: None)
    args = ["--input", str(input_path), "--esm-dir", str(tmp_path),
            "--cpz-dir", str(tmp_path), "--output-dir", str(output)]
    assert runner.main(args) == 0
    values = np.load(output / "features.npy", allow_pickle=False)
    np.testing.assert_array_equal(values, np.arange(1792, dtype=np.float32))
    receipt = json.loads((output / "RECEIPT.json").read_text())
    assert receipt["recipe_id"] == sequence.CPZ_RECIPE_ID
    assert receipt["chain_lengths"] == [4, 4]
    assert receipt["features"]["scaling"] == "none"
    assert receipt["features"]["sha256"] == sequence.sha256_file(output / "features.npy")
    assert receipt["input_sha256"] == sequence.sha256_file(output / "input.json")
    assert receipt["source_input_sha256"] == sequence.sha256_file(input_path)
    assert json.loads((output / "input.json").read_text()) == record
    assert calls[0]["ligand_profile"] == "cpz"
    assert calls[1] == (record["protein_chains"], "CC", {"allow_truncation": False, "allow_unknown": False})
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2


def test_validate_only_does_not_construct_encoder(runner, tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"protein_chains": ["ACDE"], "smiles": "CC"}))
    (tmp_path / "dict.txt").write_text("C 1\n")
    monkeypatch.setattr(runner, "verify_assets", lambda *args: None)
    def no_encoder(**kwargs):
        raise AssertionError("validate-only must not load models")
    monkeypatch.setattr(sequence, "SequenceEncoder", no_encoder)
    assert runner.main(["--input", str(input_path), "--esm-dir", str(tmp_path),
                        "--cpz-dir", str(tmp_path), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated_without_inference"
