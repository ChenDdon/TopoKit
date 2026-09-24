"""Reusable visual settings; importing styles never loads Matplotlib."""
from dataclasses import dataclass, fields
from numbers import Real
import math


@dataclass(frozen=True)
class PlotStyle:
    """Object/block styling in points, except alpha, curvature and padding.

    ``node_size`` is marker area in points squared. ``label_offset`` and
    ``padding`` are fractions of the displayed span. No global rcParams are
    changed. Dimension colors are supplied separately as ``colors={q: color}``.
    """
    node_size: float = 44.0
    node_color: str = "#20262E"
    node_edge_color: str = "white"
    node_linewidth: float = 0.65
    linewidth: float = 1.0
    edge_color: str = "#36485B"
    face_alpha: float = 0.32
    ribbon_width: float = 7.0
    ribbon_alpha: float = 0.44
    arrow_size: float = 12.0
    arrow_color: str = "#263444"
    curvature: float = 0.18
    label_size: float = 9.0
    label_offset: float = 0.045
    padding: float = 0.18

    def __post_init__(self):
        colors = {"node_color", "node_edge_color", "edge_color", "arrow_color"}
        for item in fields(self):
            if item.name in colors:
                continue  # Color parsing stays lazy, at the plotting boundary.
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and nonnegative")
        for name in ("node_size", "arrow_size", "label_size"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("face_alpha", "ribbon_alpha"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must lie in [0, 1]")


__all__ = ["PlotStyle"]
