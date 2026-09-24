"""Explicit file export with editable vector text and local settings."""
from numbers import Real
from pathlib import Path
import math


def save_figure(figure, path, *, dpi=300, transparent=False):
    """Save a figure or axes as PNG/SVG/PDF/etc. and return the output Path.

    The filename extension selects the format. SVG text stays editable and
    PDF uses embedded TrueType fonts. Font settings are scoped to this save;
    the caller's rcParams and figure remain intact. Nothing is shown or closed.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, Real) or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")
    if not isinstance(transparent, bool):
        raise ValueError("transparent must be a boolean")
    if not callable(getattr(figure, "savefig", None)):
        figure = getattr(figure, "figure", None)
    if not callable(getattr(figure, "savefig", None)):
        raise TypeError("Supply a Matplotlib figure or axes")
    target = Path(path)
    if not target.suffix:
        raise ValueError("Supply a filename extension such as .png, .svg or .pdf")
    supported = figure.canvas.get_supported_filetypes()
    if target.suffix[1:].lower() not in supported:
        raise ValueError(f"Unsupported figure format: {target.suffix}")
    import matplotlib as mpl
    target.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        figure.savefig(target, format=target.suffix[1:].lower(), dpi=dpi,
                       bbox_inches="tight", transparent=transparent)
    return target


__all__ = ["save_figure"]
