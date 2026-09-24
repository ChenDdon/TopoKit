"""Small public-boundary validators shared by adapters."""
import math
import numbers


def dimension(value, name="max_dimension"):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def start_scale(value):
    if isinstance(value, bool):
        raise ValueError("filtration_start must be a finite nonnegative number")
    validated_scale = float(value)
    if not math.isfinite(validated_scale) or validated_scale < 0:
        raise ValueError("filtration_start must be a finite nonnegative number")
    return validated_scale


def validate_query(metadata, scale):
    if scale is None:
        return None
    if isinstance(scale, bool):
        raise ValueError("scale must be a finite number")
    validated_scale = float(scale)
    if not math.isfinite(validated_scale):
        raise ValueError("scale must be finite")
    if validated_scale < metadata.get("filtration_start", -math.inf) or validated_scale > metadata.get("filtration_end", math.inf):
        raise ValueError("scale lies outside the constructed filtration domain")
    return validated_scale
