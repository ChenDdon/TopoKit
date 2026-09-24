"""Export the installed, fixed FS-AN recipe for user-supplied structure pairs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy

import topokit
from topokit import readers
from topokit.workflows.protein_ligand_prediction import features as fs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path):
    """Resolve explicit paths relative to the manifest, never a dataset root."""
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "protein_file", "ligand_file"}.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest needs sample_id, protein_file, ligand_file columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest must contain at least one pair")
    result, seen = [], set()
    for row in rows:
        sample = row["sample_id"]
        if not sample or not re.fullmatch(r"[A-Za-z0-9_-]+", sample) or sample in seen:
            raise ValueError(f"Invalid or duplicate sample_id: {sample!r}")
        seen.add(sample)
        pair = {"sample_id": sample}
        for key in ("protein_file", "ligand_file"):
            if not row[key] or not row[key].strip():
                raise ValueError(f"Missing {key} for {sample}")
            candidate = (path.parent / row[key]).resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"{sample}: {key} does not exist: {candidate}")
            pair[key] = candidate
        result.append(pair)
    return result


def input_warnings(protein_file):
    """Report raw-row PDB caveats without choosing conformers or changing atoms."""
    cloud = readers.read_pdb(protein_file, coordinate_units="angstrom")
    columns = cloud.metadata["columns"]
    selected = [i for i, (record, element) in enumerate(zip(
        columns["record_name"], cloud.metadata["labels"]))
        if record == "ATOM" and element in fs.PROTEIN_BASE_ELEMENTS]
    alternate = sum(bool(columns["alternate_location"][i]) for i in selected)
    zero_occupancy = sum(columns["occupancy"][i] is not None
                         and columns["occupancy"][i] <= 0 for i in selected)
    warnings = []
    if alternate:
        warnings.append(f"{alternate} protein ATOM rows have alternate-location labels; "
                        "FS-AN retains raw rows and does not select a conformer")
    if zero_occupancy:
        warnings.append(f"{zero_occupancy} protein ATOM rows have nonpositive occupancy; "
                        "FS-AN does not filter by occupancy")
    return warnings


def export_features(rows, output, *, max_dense_entries=25_000_000,
                    max_simplices=1_000_000, manifest_path=None):
    """Write every row's outcome; consolidate only when the whole batch succeeds."""
    if not rows:
        raise ValueError("At least one pair is required")
    if max_dense_entries <= 0 or max_simplices <= 0:
        raise ValueError("Resource limits must be positive")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in ids):
        raise ValueError("Supply unique sample IDs containing letters, digits, underscores or hyphens")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    (output / "records").mkdir()
    # The guarded prediction loader hashes schema bytes, not a parsed dictionary.
    schema_bytes = files(fs.__package__).joinpath("reference_schema.json").read_bytes()
    (output / "feature_schema.json").write_bytes(schema_bytes)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("pdb_id", "protein_file", "ligand_file"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"pdb_id": row["sample_id"],
                             "protein_file": str(row["protein_file"]),
                             "ligand_file": str(row["ligand_file"])})
    receipt = {"status": "running", "strategy_id": fs.STRATEGY_ID,
               "recipe_id": fs.schema()["recipe_id"], "samples_requested": len(rows),
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "implementation": fs.implementation_receipt(),
               "runtime": {"python": platform.python_version(), "topokit": topokit.__version__,
                           "numpy": np.__version__, "scipy": scipy.__version__},
               "limits": {"max_dense_entries": max_dense_entries, "max_simplices": max_simplices},
               "failed_samples": [], "successful_samples": [], "warnings": {}}
    if manifest_path is not None:
        receipt["input_manifest"] = {"path": str(Path(manifest_path).resolve()),
                                     "sha256": sha256(manifest_path)}
    write_json(output / "run.json", receipt)
    for row in rows:
        sample = row["sample_id"]
        started = time.perf_counter()
        record = {"sample_id": sample, "recipe_id": receipt["recipe_id"], "status": "failed"}
        try:
            inputs = {key: {"path": str(Path(row[key]).resolve()), "sha256": sha256(row[key])}
                      for key in ("protein_file", "ligand_file")}
            record["inputs"] = inputs
            warnings = input_warnings(row["protein_file"])
            record["warnings"] = warnings
            if warnings:
                receipt["warnings"][sample] = warnings
                for warning in warnings:
                    print(f"{sample}: warning: {warning}", file=sys.stderr)
            selected = fs.read_selected_atoms(row["protein_file"], row["ligand_file"])
            tensor, present, counts = fs.compute(selected, max_dense_entries=max_dense_entries,
                                                 max_simplices=max_simplices)
            tensor = np.ascontiguousarray(tensor, dtype="<f4")
            if tensor.shape != (10, 50, 55) or not np.isfinite(tensor).all():
                raise ValueError("Feature encoder returned an invalid tensor")
            if any(sha256(row[key]) != value["sha256"] for key, value in inputs.items()):
                raise ValueError("Input structure changed during extraction")
            destination = output / "samples" / (sample + ".npy")
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, tensor, allow_pickle=False)
            temporary.replace(destination)
            record.update(status="success", output={"path": f"samples/{sample}.npy",
                          "sha256": sha256(destination), "shape": list(tensor.shape),
                          "dtype": "float32", "order": "C"},
                          counts=selected["counts"],
                          coordinate_deduplication=selected["coordinate_deduplication"],
                          channel_presence=present.tolist(), channel_atom_counts=counts.tolist())
            receipt["successful_samples"].append(sample)
            print(f"{sample}: wrote (10, 50, 55)", flush=True)
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            receipt["failed_samples"].append(sample)
            print(f"{sample}: failed: {error}", file=sys.stderr, flush=True)
        record["seconds"] = time.perf_counter() - started
        write_json(output / "records" / (sample + ".json"), record)
        write_json(output / "run.json", receipt)
    if not receipt["failed_samples"]:
        temporary = output / "features.npy.tmp"
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype="<f4", shape=(len(rows), 27500))
        for index, sample in enumerate(ids):
            matrix[index] = np.load(output / "samples" / (sample + ".npy"),
                                    allow_pickle=False).ravel(order="C")
        matrix.flush()
        del matrix
        temporary.replace(output / "features.npy")
        write_json(output / "sample_ids.json", ids)
        receipt["matrix"] = {"path": "features.npy", "shape": [len(rows), 27500],
                             "sha256": sha256(output / "features.npy")}
        receipt["status"] = "complete"
    else:
        receipt["status"] = "failed"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="CSV: sample_id,protein_file,ligand_file; paths relative to CSV")
    parser.add_argument("--output", required=True, type=Path, help="New directory; never overwritten")
    parser.add_argument("--max-dense-entries", type=int, default=25_000_000)
    parser.add_argument("--max-simplices", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        rows = load_manifest(args.manifest)
        result = export_features(rows, args.output, max_dense_entries=args.max_dense_entries,
                                 max_simplices=args.max_simplices, manifest_path=args.manifest)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"{result['status']}: {len(result['successful_samples'])}/{len(rows)} pairs; {args.output}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
