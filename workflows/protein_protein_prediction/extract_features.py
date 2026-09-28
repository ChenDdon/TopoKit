"""Export the predefined Topo PPI features for explicit structure/partner pairs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy
import topokit
from topokit.workflows.protein_protein_prediction import features as fs


RESERVED_IDS = {"CON", "PRN", "AUX", "NUL", *(f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10))}


def valid_sample_id(value):
    return (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
            and value.upper() not in RESERVED_IDS)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def _chains(value, field):
    chains = [part.strip() for part in (value or "").split(";")]
    if any(not part for part in chains) or len(set(chains)) != len(chains):
        raise ValueError(f"{field}: supply unique semicolon-separated chain IDs")
    return chains


def load_manifest(path):
    """Resolve structure paths against the CSV directory and preserve row order."""
    path = Path(path).resolve()
    required = {"sample_id", "structure_file", "partner_a_chains", "partner_b_chains"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest needs sample_id, structure_file, partner_a_chains, partner_b_chains")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest must contain at least one complex")
    result, seen = [], set()
    for row in rows:
        sample = row["sample_id"]
        if not valid_sample_id(sample) or sample.casefold() in seen:
            raise ValueError(f"Invalid, reserved or case-insensitively duplicate sample_id: {sample!r}")
        seen.add(sample.casefold())
        if (row.get("blocked_reason") or "").strip():
            raise ValueError(f"{sample}: resolve the recorded blocked_reason before extraction")
        value = row["structure_file"]
        if not value or not value.strip():
            raise ValueError(f"{sample}: missing structure_file")
        structure = (path.parent / value).resolve()
        if not structure.is_file():
            raise FileNotFoundError(f"{sample}: missing structure: {structure}")
        a, b = (_chains(row[key], key) for key in ("partner_a_chains", "partner_b_chains"))
        if set(a) & set(b):
            raise ValueError(f"{sample}: partner chains must be disjoint")
        allowed = (row.get("allowed_missing_chains") or "").strip()
        missing = _chains(allowed, "allowed_missing_chains") if allowed else []
        if not set(missing) <= set(a + b):
            raise ValueError(f"{sample}: allowed missing chains must be declared partner chains")
        empty = (row.get("empty_partner") or "").strip() or None
        if empty not in (None, "A", "B"):
            raise ValueError(f"{sample}: empty_partner must be A, B, or blank")
        result.append({"sample_id": sample, "structure_file": structure,
                       "partner_a_chains": a, "partner_b_chains": b,
                       "allowed_missing_chains": missing, "empty_partner": empty})
    return result


def export_features(rows, output, *, max_dense_entries=25_000_000,
                    max_simplices=1_000_000, manifest_path=None):
    """Write every outcome and consolidate features only when every row succeeds."""
    if not rows:
        raise ValueError("At least one complex is required")
    if any(type(value) is not int or value <= 0 for value in (max_dense_entries, max_simplices)):
        raise ValueError("Resource limits must be positive integers")
    ids = [row["sample_id"] for row in rows]
    if any(not valid_sample_id(s) for s in ids) or len({s.casefold() for s in ids}) != len(ids):
        raise ValueError("Supply case-insensitively unique, nonreserved sample IDs containing letters, digits, underscores or hyphens")
    schema = fs.schema()
    shape = tuple(schema["tensor"]["shape"])
    width = schema["tensor"]["features"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    (output / "records").mkdir()
    (output / "feature_schema.json").write_bytes(canonical_bytes(schema))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("sample_id", "structure_file", "partner_a_chains", "partner_b_chains",
                  "allowed_missing_chains", "empty_partner")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"sample_id": row["sample_id"], "structure_file": str(Path(row["structure_file"]).resolve()),
                             "partner_a_chains": ";".join(row["partner_a_chains"]),
                             "partner_b_chains": ";".join(row["partner_b_chains"]),
                             "allowed_missing_chains": ";".join(row.get("allowed_missing_chains", [])),
                             "empty_partner": row.get("empty_partner") or ""})
    receipt = {"status": "running", "recipe_id": schema["recipe_id"], "feature_count": width,
               "samples_requested": len(rows), "started_utc": datetime.now(timezone.utc).isoformat(),
               "implementation": fs.implementation_receipt(),
               "runtime": {"python": platform.python_version(), "topokit": topokit.__version__,
                           "numpy": np.__version__, "scipy": scipy.__version__},
               "limits": {"max_dense_entries": max_dense_entries, "max_simplices": max_simplices},
               "successful_samples": [], "failed_samples": []}
    if manifest_path is not None:
        receipt["input_manifest"] = {"path": str(Path(manifest_path).resolve()), "sha256": sha256(manifest_path)}
    write_json(output / "run.json", receipt)
    for row in rows:
        sample, started = row["sample_id"], time.perf_counter()
        record = {"sample_id": sample, "recipe_id": schema["recipe_id"], "status": "failed"}
        try:
            digest = sha256(row["structure_file"])
            record["input"] = {"path": str(Path(row["structure_file"]).resolve()), "sha256": digest}
            selected = fs.read_selected_atoms(row["structure_file"], row["partner_a_chains"], row["partner_b_chains"],
                                             allowed_missing_chains=row.get("allowed_missing_chains", ()),
                                             empty_partner=row.get("empty_partner"))
            record["diagnostics"] = selected["diagnostics"]
            tensor, present, counts = fs.compute(selected, max_dense_entries=max_dense_entries,
                                                max_simplices=max_simplices)
            tensor = np.ascontiguousarray(tensor, dtype="<f4")
            if tensor.shape != shape or not np.isfinite(tensor).all():
                raise ValueError("Feature encoder returned an invalid tensor")
            if sha256(row["structure_file"]) != digest:
                raise ValueError("Input structure changed during extraction")
            destination = output / "samples" / (sample + ".npy")
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, tensor, allow_pickle=False)
            temporary.replace(destination)
            record.update(status="success", output={"path": f"samples/{sample}.npy", "sha256": sha256(destination),
                          "shape": list(shape), "dtype": "float32", "order": "C"},
                          channel_presence=present.tolist(), channel_atom_counts=counts.tolist())
            receipt["successful_samples"].append(sample)
            print(f"{sample}: wrote {shape}", flush=True)
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            receipt["failed_samples"].append(sample)
            print(f"{sample}: failed: {error}", file=sys.stderr, flush=True)
        record["seconds"] = time.perf_counter() - started
        write_json(output / "records" / (sample + ".json"), record)
        write_json(output / "run.json", receipt)
    if not receipt["failed_samples"]:
        temporary = output / "features.npy.tmp"
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype="<f4", shape=(len(rows), width))
        for index, sample in enumerate(ids):
            matrix[index] = np.load(output / "samples" / (sample + ".npy"), allow_pickle=False).ravel(order="C")
        matrix.flush()
        del matrix
        temporary.replace(output / "features.npy")
        write_json(output / "sample_ids.json", ids)
        receipt["artifact_sha256"] = {name: sha256(output / name) for name in
                                      ("features.npy", "sample_ids.json", "feature_schema.json")}
        receipt["matrix"] = {"path": "features.npy", "shape": [len(rows), width],
                             "dtype": "float32", "order": "C", "sha256": sha256(output / "features.npy")}
        receipt["status"] = "complete"
    else:
        receipt["status"] = "failed"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with explicit structures and partner chain lists")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwritten")
    parser.add_argument("--max-dense-entries", type=int, default=25_000_000)
    parser.add_argument("--max-simplices", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        rows = load_manifest(args.manifest)
        result = export_features(rows, args.output, max_dense_entries=args.max_dense_entries,
                                 max_simplices=args.max_simplices, manifest_path=args.manifest)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"{result['status']}: {len(result['successful_samples'])}/{len(rows)} complexes; {args.output}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
