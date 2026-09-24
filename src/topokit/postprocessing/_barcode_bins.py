"""Finite bin schemas and scientific compatibility, without topology imports."""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real
import numpy as np
from ..exceptions import ResourceLimitError
from ..results import PersistenceResult

SEMANTIC_KEYS = (
    "route", "topology_kind", "definition_id", "scale_units", "coordinate_units",
    "filtration_start", "filtration_end", "filtration_parameters", "filtration_coordinate",
    "filtration_start_policy", "start_policy", "coupling", "construction_kind",
    "complex_type", "geometry_weighted", "weighted_alpha", "point_weights_role",
    "weight_role", "direction_rule", "sequence_rule", "filtration_kind", "filtration_schema",
    "construction", "object_type", "orientation", "connection_support", "connection_cutoff",
    "connection_limit", "interval_convention", "initial_stage_semantics", "initial_stage_note",
    "cutoff_distance", "cutoff_policy", "graph_expansion",
    "filtration_a", "filtration_b", "progression", "factor_filtration_ranges", "factor_scale_units",
    "filtration_exactness", "factor_max_dimensions", "factor_coordinate_units", "factor_count",
    "interaction_admission_rule", "overlap_simplices_role",
)
_SPEC_KEYS = {"edges", "min_value", "max_value", "step"}
_FACTOR_OPTION_KEYS = {"complex_type", "max_dimension", "max_simplex_dimension",
                       "geometry_tolerance", "duplicates"}
_FACTOR_CONSTRUCTION_KEYS = set(SEMANTIC_KEYS) | {"geometry_tolerance", "duplicates", "max_simplex_dimension"}


def finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def feature_budget(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_features must be a positive integer")
    return int(value)


def edges(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real bin edges")
    result = np.array(values, dtype=float, copy=True)
    if result.ndim != 1 or len(result) < 2 or not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be finite, strictly increasing bin edges")
    result.setflags(write=False)
    return result


def validate_result(result, dimensions):
    if not isinstance(result, PersistenceResult):
        raise TypeError("result must be a topokit PersistenceResult")
    if (isinstance(result.max_dimension, (bool, np.bool_))
            or not isinstance(result.max_dimension, Integral) or result.max_dimension < 0):
        raise ValueError("result.max_dimension must be a nonnegative integer")
    degrees = tuple(range(result.max_dimension + 1)) if dimensions is None else tuple(dimensions)
    if (any(isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral) or q < 0 for q in degrees)
            or len(set(degrees)) != len(degrees)):
        raise ValueError("dimensions must be unique nonnegative integers")
    if any(q > result.max_dimension for q in degrees):
        raise ValueError("requested feature dimension was not computed in the persistence result")
    if not isinstance(result.metadata, Mapping):
        raise TypeError("persistence metadata must be a mapping")
    for bar in result.intervals:
        if (isinstance(bar.dimension, (bool, np.bool_)) or not isinstance(bar.dimension, Integral)
                or not 0 <= bar.dimension <= result.max_dimension):
            raise ValueError("interval dimension is outside the computed dimensions")
        birth = finite_real(bar.birth, "interval birth")
        death = bar.death
        if (isinstance(death, (bool, np.bool_)) or not isinstance(death, Real)
                or math.isnan(death) or death < birth):
            raise ValueError("interval death must be finite or positive infinity, and at least birth")
    for key in ("field", "coefficient_field"):
        if key in result.metadata and result.metadata[key] != result.field:
            raise ValueError(f"persistence {key} metadata disagrees with result.field")
    return tuple(int(q) for q in degrees)


def source_schema(result, *, field=None):
    """Sample-independent scientific semantics for barcode or spectral records.

    ``field='R'`` supports SpectrumResult, whose coefficient field is implicit.
    Per-factor recipes are whitelisted: source IDs, cells, bonds, counts,
    weights, raw births, and resource budgets must not become column schemas.
    """
    scientific_schema = {"field": deepcopy(result.field if field is None else field),
              **{key: deepcopy(result.metadata[key]) for key in SEMANTIC_KEYS if key in result.metadata}}
    for name, allowed in (("factor_options", _FACTOR_OPTION_KEYS),
                          ("factor_construction_metadata", _FACTOR_CONSTRUCTION_KEYS)):
        if name not in result.metadata:
            continue
        records = result.metadata[name]
        if not isinstance(records, (tuple, list)) or any(not isinstance(item, Mapping) for item in records):
            raise TypeError(f"{name} must be a sequence of factor mappings")
        scientific_schema[name] = tuple({key: deepcopy(value) for key, value in item.items() if key in allowed}
                             for item in records)
    return scientific_schema


def same_value(left, right):
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right) and all(same_value(left[k], right[k]) for k in left))
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
                and len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def check_schema(result, reference):
    actual_schema = source_schema(result)
    for key in sorted(set(actual_schema) | set(reference)):
        if (key in actual_schema) != (key in reference) or not same_value(actual_schema.get(key), reference.get(key)):
            raise ValueError(f"persistence schema differs: {key}; fit/transform require matching scientific semantics")


def default_minimum(result):
    value = result.metadata.get("filtration_start")
    return max(0.0, finite_real(value, "filtration_start")) if value is not None else 0.0


def collection_maximum(results, degrees):
    maximum = None
    def observe(value):
        nonlocal maximum
        maximum = value if maximum is None else max(maximum, value)
    for result in results:
        for bar in result.intervals:
            if bar.dimension in degrees:
                observe(float(bar.birth))
                if math.isfinite(bar.death):
                    observe(float(bar.death))
        end = result.metadata.get("filtration_end")
        if end is not None:
            if isinstance(end, (bool, np.bool_)) or not isinstance(end, Real) or math.isnan(end):
                raise ValueError("filtration_end must be a real number, not NaN")
            if math.isfinite(end):
                observe(float(end))
    return maximum


def _degree_value(value, q, name):
    if not isinstance(value, Mapping):
        return value
    if any(isinstance(key, (bool, np.bool_)) or not isinstance(key, Integral) or key < 0 for key in value):
        raise ValueError(f"{name} mapping keys must be nonnegative homology degrees")
    if q not in value:
        raise ValueError(f"{name} has no specification for H{q}")
    return value[q]


def _range_edges(low, high, step, budget, name):
    low = finite_real(low, f"{name} min_value")
    if high is None:
        raise ValueError("no finite bin range is available; supply explicit edges or finite max_value and step")
    high = finite_real(high, f"{name} max_value")
    step = finite_real(step, f"{name} step")
    if high <= low or step <= 0:
        raise ValueError("bin max_value must exceed min_value and step must be positive; supply a finite range")
    ratio = (high - low) / step
    if not math.isfinite(ratio) or ratio > budget:
        raise ResourceLimitError("bin edge count exceeds max_features; increase step or declare a larger budget")
    # Include the exact maximum, with a shorter final bin if necessary.
    bin_count = max(1, math.ceil(ratio))
    candidate_edges = low + np.arange(bin_count, dtype=float) * step
    candidate_edges = candidate_edges[candidate_edges < high]
    return edges(np.append(candidate_edges, high), name)


def resolve_axes(degrees, *, birth_edges, death_edges, min_value, max_value, step,
                 default_min, inferred_max=None, max_features=1_000_000):
    budget = feature_budget(max_features)
    axes_by_kind, inferred_range = [], False
    for name, configured in (("birth_edges", birth_edges), ("death_edges", death_edges)):
        by_degree = {}
        for q in degrees:
            item = configured
            if isinstance(item, Mapping) and not set(item).issubset(_SPEC_KEYS):
                item = _degree_value(item, q, name)
            axis_spec = item if isinstance(item, Mapping) else {}
            if axis_spec and not set(axis_spec).issubset(_SPEC_KEYS):
                raise ValueError(f"unknown {name} specification keys: {set(axis_spec) - _SPEC_KEYS}")
            explicit_edges = axis_spec.get("edges") if isinstance(item, Mapping) else item
            if explicit_edges is not None:
                by_degree[q] = edges(explicit_edges, f"{name}[H{q}]")
                continue
            low = axis_spec["min_value"] if "min_value" in axis_spec else _degree_value(min_value, q, "min_value")
            high = axis_spec["max_value"] if "max_value" in axis_spec else _degree_value(max_value, q, "max_value")
            bin_width = axis_spec["step"] if "step" in axis_spec else _degree_value(step, q, "step")
            if low is None:
                low = default_min
            if high is None:
                high = inferred_max
                inferred_range = True
            by_degree[q] = _range_edges(low, high, bin_width, budget, f"{name}[H{q}]")
        axes_by_kind.append(by_degree)
    feature_count = sum((len(axes_by_kind[0][q]) - 1) * (len(axes_by_kind[1][q]) - 1)
                + len(axes_by_kind[0][q]) - 1 + 2 for q in degrees)
    if feature_count > budget:
        raise ResourceLimitError(f"barcode feature count {feature_count:,} exceeds max_features={budget:,}; no histograms allocated")
    return axes_by_kind[0], axes_by_kind[1], inferred_range


def compact_edges(mapping):
    """Preserve the legacy shared-array representation whenever axes match."""
    arrays = tuple(mapping.values())
    if arrays and all(np.array_equal(arrays[0], value) for value in arrays[1:]):
        return edges(arrays[0], "edges")
    return {q: edges(value, "edges") for q, value in mapping.items()}


def stack_or_tuple(arrays, empty_shape):
    if not arrays:
        return np.empty(empty_shape, dtype=float)
    return np.stack(arrays) if all(array.shape == arrays[0].shape for array in arrays) else tuple(arrays)
