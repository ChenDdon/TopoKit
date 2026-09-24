"""Predict pK using an FS-AN ensemble or a single pipeline with fitted scalers."""
import argparse
import json
from pathlib import Path
from topokit.workflows.protein_ligand_prediction.prediction import (
    load_feature_matrix, predict, read, sha256,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    print(json.dumps(predict(args.model_dir, args.features, args.manifest, args.output, args.batch_size)))
