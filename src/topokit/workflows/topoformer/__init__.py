"""Optional supervised TopoFormer over precomputed feature tensors.

Importing this namespace does not import PyTorch. Install ``topokit[dl]`` to
build a model or load a trained bundle. Dataset selection and ablations are
external application concerns; no training happens at import time.
"""

from .config import TopoFormerConfig
from .encoding import position_encoding, tokenize


def build_model(config=None):
    """Construct an independently initialized compact encoder (lazy PyTorch)."""
    from ._model import TopoFormerMini
    if config is None:
        config = TopoFormerConfig()
    if isinstance(config, dict):
        config = TopoFormerConfig(**config)
    return TopoFormerMini(config)


def save_bundle(*args, **kwargs):
    """Save weights, scaler arrays, schema and checksummed metadata."""
    from .bundle import save_bundle as save
    return save(*args, **kwargs)


def load_predictor(path, *, device="cpu"):
    """Load a verified model bundle, including its training-only normalizer."""
    from .bundle import Predictor
    return Predictor(path, device=device)


__all__ = ["TopoFormerConfig", "build_model", "position_encoding", "tokenize",
           "save_bundle", "load_predictor"]
