"""Fixed PPI Topo features and optional externally supplied GBDT inference."""
from .features import (RECIPE_ID, compute, featurize, implementation_receipt,
                       read_selected_atoms, schema)

__all__ = ["RECIPE_ID", "schema", "implementation_receipt", "read_selected_atoms",
           "compute", "featurize", "predict_gbdt", "PredictionResult", "model_profile"]


def __getattr__(name):
    if name in {"predict_gbdt", "PredictionResult", "model_profile"}:
        from . import prediction
        return getattr(prediction, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
