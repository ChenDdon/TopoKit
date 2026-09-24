"""One fixed protein–ligand feature strategy and optional GBDT inference.

No datasets, fitted weights, scheduler or ablation switches are installed.
Trusted trained pipelines and their fitted scalers are supplied as separate
model bundles; the final predictor averages three explicitly seeded members.
"""
from .features import (STRATEGY_ID, compute, featurize, implementation_receipt,
                       read_selected_atoms, schema)
from .prediction import predict

__all__ = ["STRATEGY_ID", "schema", "implementation_receipt", "read_selected_atoms",
           "compute", "featurize", "predict"]
