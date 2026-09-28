#!/usr/bin/env python3
"""Write PPI Topo ensemble predictions from a completed feature batch."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path, help="Completed PPI Topo feature directory")
    parser.add_argument("--models", required=True, type=Path, help="Trusted directory containing pinned seed_0/1/2.joblib files")
    parser.add_argument("--output", required=True, type=Path, help="New prediction CSV; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    from topokit.workflows.protein_protein_prediction import predict_gbdt

    result = predict_gbdt(args.features, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "predicted_dg_bind_kcal_mol", "seed_0", "seed_1", "seed_2", "model_profile_id"])
        for i, sample_id in enumerate(result.sample_ids):
            writer.writerow([sample_id, result.prediction[i], *result.members[i], result.model_profile_id])
    print(f"Wrote {len(result.sample_ids)} PPI predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
