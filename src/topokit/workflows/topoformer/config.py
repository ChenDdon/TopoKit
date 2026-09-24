"""Dependency-light configuration for the supervised compact model."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TopoFormerConfig:
    n_statistics: int = 10
    n_scales: int = 50
    n_channels: int = 55
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12

    def __post_init__(self):
        for name in ("n_statistics", "n_scales", "n_channels", "d_model",
                     "num_layers", "num_heads", "ff_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % 4 or self.d_model % self.num_heads:
            raise ValueError("d_model must divide into four position quarters and all heads")
        if not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 < self.layer_norm_eps < 1:
            raise ValueError("invalid layer_norm_eps")

    @property
    def feature_shape(self):
        return (self.n_statistics, self.n_scales, self.n_channels)

    def to_dict(self):
        return asdict(self)
