"""Scale-token ordering and the published TopoFormer position convention.

Independent NumPy implementation of the width-first 2D sine/cosine formula,
adapted to an n_scales x 1 patch grid. See the pinned upstream reference in the
application documentation. Positions are patch indices, not physical radii.
"""
import numpy as np


def tokenize(features):
    """(batch, statistic, scale, channel) -> (batch, scale, statistic*channel)."""
    x = np.asarray(features)
    if x.ndim != 4:
        raise ValueError("expected (batch, statistic, scale, channel)")
    return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)


def position_encoding(n_scales, d_model):
    """Frozen float32 positions, including the leading all-zero CLS row."""
    if n_scales <= 0 or d_model <= 0 or d_model % 4:
        raise ValueError("positive scales and d_model divisible by four required")
    quarter = d_model // 4
    frequencies = 1.0 / (10000 ** (np.arange(quarter, dtype=np.float64) / quarter))
    phase = np.arange(n_scales, dtype=np.float64)[:, None] * frequencies
    patches = np.concatenate((np.zeros_like(phase), np.ones_like(phase),
                              np.sin(phase), np.cos(phase)), axis=1)
    return np.concatenate((np.zeros((1, d_model)), patches), axis=0).astype(np.float32)
