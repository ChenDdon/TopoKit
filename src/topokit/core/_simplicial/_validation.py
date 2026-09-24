"""Shared input validation and allocation guards."""

from math import isfinite, isnan, isqrt
from numbers import Integral, Real

from ...exceptions import ResourceLimitError


class ComplexityLimitError(ResourceLimitError):
    """A user-configurable size limit was reached before a large allocation."""


def nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def real(value, name, *, finite=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if isnan(value) or (finite and not isfinite(value)):
        raise ValueError(f"{name} must be {'finite' if finite else 'non-NaN'}")
    return value


def tolerance(value):
    value = real(value, "tolerance")
    if value <= 0:
        raise ValueError("tolerance must be positive")
    return value


def prime(value):
    value = nonnegative_int(value, "field")
    # Explicit bound keeps validation and coefficient arithmetic predictable.
    if value < 2 or value > 2147483647:
        raise ValueError("field must be a prime between 2 and 2147483647")
    if value != 2 and (value % 2 == 0 or any(value % d == 0 for d in range(3, isqrt(value) + 1, 2))):
        raise ValueError("field must be prime")
    return value


def guard_dense(shape, limit):
    limit = nonnegative_int(limit, "max_dense_entries")
    entry_count = 1
    for axis_size in shape:
        entry_count *= axis_size
    if entry_count > limit:
        raise ComplexityLimitError(
            f"Dense shape {shape} requires {entry_count:,} entries (limit {limit:,}); "
            "reduce the complex or explicitly increase max_dense_entries."
        )
