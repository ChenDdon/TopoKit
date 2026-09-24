"""Filtration parameter validation; no geometry or topology is constructed here."""
import math
from collections.abc import Mapping, Set
from numbers import Real


def _bound(value, name, *, infinite=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a nonnegative real number")
    value = float(value)
    if math.isnan(value) or value < 0 or (not infinite and not math.isfinite(value)):
        raise ValueError(f"{name} must be nonnegative {'and finite' if not infinite else '(+inf allowed)'}")
    return value


def resolve_range(filtration_range=None, *, filtration_start=0.0, max_scale=math.inf):
    """Resolve an inclusive construction interval in the builder's native units.

    Alpha uses squared radius; Rips and geometric edge builders use distance.
    A range is distinct from a sequence of Laplacian observation scales.
    Nondefault legacy bounds must agree with an explicitly supplied range.
    """
    start = _bound(filtration_start, "filtration_start")
    end = _bound(max_scale, "max_scale", infinite=True)
    if filtration_range is not None:
        if isinstance(filtration_range, (str, bytes, Mapping, Set)):
            raise ValueError("filtration_range must contain exactly (start, end)")
        try:
            bounds = tuple(filtration_range)
        except TypeError as exc:
            raise ValueError("filtration_range must contain exactly (start, end)") from exc
        if len(bounds) != 2:
            raise ValueError("filtration_range must contain exactly (start, end)")
        range_start = _bound(bounds[0], "range start")
        range_end = _bound(bounds[1], "range end", infinite=True)
        if (start != 0 and start != range_start) or (end != math.inf and end != range_end):
            raise ValueError("filtration_range conflicts with filtration_start/max_scale")
        start, end = range_start, range_end
    if end < start:
        raise ValueError("filtration end must be at least its start")
    return start, end
