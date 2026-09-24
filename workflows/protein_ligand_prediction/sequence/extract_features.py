#!/usr/bin/env python3
"""Extract one ESM-2/CPZ vector from explicit sequences and prepared SMILES."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np

from topokit.workflows.protein_ligand_prediction import sequence


def read_input(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Input must be a JSON object")
    chains = record.get("protein_chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError("protein_chains must be a nonempty list of sequences")
    for chain in chains:
        sequence.protein_windows(chain)
    smiles = record.get("smiles")
    if not isinstance(smiles, str) or not smiles or smiles != smiles.strip():
        raise ValueError("smiles must be a nonempty stripped string")
    if "id" in record and (not isinstance(record["id"], str) or not record["id"].strip()):
        raise ValueError("id, if supplied, must be a nonempty string")
    return record


def verify_assets(esm_dir, cpz_dir, recipe):
    for section, folder in (("protein", esm_dir), ("ligand", cpz_dir)):
        for name, expected in recipe[section]["asset_sha256"].items():
            path = Path(folder) / name
            if not path.is_file():
                raise ValueError(f"Missing {section} asset: {path}")
            if sequence.sha256_file(path) != expected:
                raise ValueError(f"{section} asset identity mismatch: {name}")


def read_vocabulary(path):
    # Mirrors the checkpoint's dictionary indexing without importing PyTorch.
    symbols = ["<s>", "<pad>", "</s>", "<unk>"]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if " " not in line:
            raise ValueError("Malformed ligand dictionary")
        token = line.rsplit(" ", 1)[0]
        if token not in symbols:
            symbols.append(token)
    return {token: index for index, token in enumerate(symbols)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON with protein_chains and prepared smiles; optional id")
    parser.add_argument("--esm-dir", type=Path, required=True,
                        help="Trusted local esm2_t33_650M_UR50D snapshot")
    parser.add_argument("--cpz-dir", type=Path, required=True,
                        help="Trusted local chembl27_pubchem_zinc_512 directory")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for features.npy, input.json and RECEIPT.json")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda:0")
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--allow-unknown", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Verify inputs/assets and report tokenization without loading models")
    args = parser.parse_args(argv)
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required for extraction")
    try:
        if args.output_dir is not None and args.output_dir.exists():
            raise ValueError("Output directory already exists; choose a new directory")
        record = read_input(args.input)
        recipe_path = Path(__file__).with_name("RECIPE.json")
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        verify_assets(args.esm_dir, args.cpz_dir, recipe)
        indices = read_vocabulary(args.cpz_dir / "dict.txt")
        _, token_receipt = sequence.ligand_tokens(
            record["smiles"], indices, allow_truncation=args.allow_truncation,
            allow_unknown=args.allow_unknown)
        if args.validate_only:
            print(json.dumps({"status": "validated_without_inference",
                              "recipe_id": recipe["recipe_id"],
                              "ligand_tokens": token_receipt}, indent=2))
            return 0
        encoder = sequence.SequenceEncoder(
            esm_dir=args.esm_dir, chembl_dir=args.cpz_dir,
            ligand_profile="cpz", device=args.device)
        if encoder.recipe_id != recipe["recipe_id"]:
            raise ValueError("Installed encoder does not match the repository recipe")
        features, token_receipt = encoder.encode(
            record["protein_chains"], record["smiles"],
            allow_truncation=args.allow_truncation, allow_unknown=args.allow_unknown)
        if features.shape != (1792,) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("Encoder did not produce finite float32[1792] features")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output = args.output_dir / "features.npy"
        np.save(output, features, allow_pickle=False)
        (args.output_dir / "input.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        packages = {}
        for name in ("topokit", "numpy", "torch", "transformers"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = "not-installed-as-distribution"
        receipt = {
            "status": "complete", "recipe_id": encoder.recipe_id,
            "ligand_profile": encoder.ligand_profile,
            "selected_model_strategy": "FS-AU", "prediction_performed": False,
            "source_input_sha256": sequence.sha256_file(args.input),
            "input_sha256": sequence.sha256_file(args.output_dir / "input.json"),
            "recipe_sha256": sequence.sha256_file(recipe_path),
            "sequence_implementation_sha256": sequence.sha256_file(sequence.__file__),
            "encoder_assets": {key: recipe[key]["asset_sha256"] for key in ("protein", "ligand")},
            "chain_lengths": [len(chain) for chain in record["protein_chains"]],
            "ligand_tokens": token_receipt,
            "allow_truncation": args.allow_truncation, "allow_unknown": args.allow_unknown,
            "device": args.device, "python": platform.python_version(), "packages": packages,
            "features": {"file": output.name, "shape": list(features.shape),
                         "dtype": str(features.dtype), "sha256": sequence.sha256_file(output),
                         "scaling": "none"},
        }
        (args.output_dir / "RECEIPT.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {output}: float32[1792], recipe={encoder.recipe_id}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
