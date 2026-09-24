"""Customizable numerical spectral summaries; no topology is recomputed."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
import numpy as np
from ..results import SpectrumResult
from ._metadata import source_semantics
from ._barcode_bins import source_schema


@dataclass(frozen=True)
class VectorResult:
    values: np.ndarray
    names: tuple[str, ...]
    metadata: dict = field(default_factory=dict)


def _real_values(values):
    if np.iscomplexobj(values):
        raise ValueError("eigenvalues must be real")
    eigenvalues = np.asarray(values, dtype=float)
    if eigenvalues.ndim != 1 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must be a finite one-dimensional array")
    return eigenvalues


def _tolerance(tol):
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")


def _validate_empty_operator_policy(policy):
    if not isinstance(policy, str) or policy not in {"preserve", "zero"}:
        raise ValueError("empty_operator_policy must be 'preserve' or 'zero'")


def _metadata_dimension(value):
    """Return a usable dimension without imposing new rules on old metadata."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) or value < 0):
        return None
    return int(value)


def _is_empty_optional_matrix(value):
    if value is None:
        return True
    try:
        return tuple(value.shape) == (0, 0)
    except (AttributeError, TypeError):
        return False


def _operator_metadata(result, eigenvalue_count):
    """Resolve dimensions and verify the core's structural-absence receipt.

    Older or hand-built ``SpectrumResult`` objects may use arbitrary metadata,
    so ordinary summarization must not start rejecting them merely because the
    new standardized keys are absent or malformed.  Structural zero encoding
    is stricter: its Boolean return is true only when both the mathematical
    envelope and all three core metadata fields agree on a 0 x 0 operator.
    """
    reported_operator_dimension = result.metadata.get("operator_dimension")
    operator_dimension = _metadata_dimension(reported_operator_dimension)
    if result.complete:
        # A complete spectrum itself fixes the full operator dimension.  Keep
        # the inferred value for backward-compatible records with no receipt.
        if operator_dimension != eigenvalue_count:
            operator_dimension = eigenvalue_count
    elif operator_dimension is not None and operator_dimension < eigenvalue_count:
        operator_dimension = None

    reported_source_dimension = result.metadata.get("source_chain_dimension")
    source_chain_dimension = _metadata_dimension(reported_source_dimension)
    if source_chain_dimension is None or (
            operator_dimension is not None
            and source_chain_dimension != operator_dimension):
        source_chain_dimension = operator_dimension

    reported_absence = result.metadata.get("structural_absence")
    receipt_says_absent = (
        isinstance(reported_absence, (bool, np.bool_))
        and bool(reported_absence)
        and _metadata_dimension(reported_operator_dimension) == 0
        and _metadata_dimension(reported_source_dimension) == 0
    )
    empty_matrix = _is_empty_optional_matrix(result.matrix)
    empty_eigenvectors = _is_empty_optional_matrix(result.eigenvectors)
    zero_nullity = (
        not isinstance(result.nullity, (bool, np.bool_))
        and isinstance(result.nullity, Integral)
        and int(result.nullity) == 0
    )
    structural_absence = bool(
        result.complete
        and eigenvalue_count == 0
        and isinstance(result.basis, tuple)
        and len(result.basis) == 0
        and empty_matrix
        and empty_eigenvectors
        and zero_nullity
        and receipt_says_absent
    )
    return operator_dimension, source_chain_dimension, structural_absence


def spectral_entropy(values, *, base=math.e, tol=1e-10):
    """Shannon entropy of trace-normalized nonnegative eigenvalues.

    For a graph Laplacian this is its von Neumann spectral entropy, not degree
    entropy. The zero operator has entropy 0 by explicit convention. Numerical
    negative values in [-tol, 0] are clipped only in this local calculation.
    """
    eigenvalues = _real_values(values)
    _tolerance(tol)
    if isinstance(base, (bool, np.bool_)) or not isinstance(base, Real) or not np.isfinite(base) or base <= 1:
        raise ValueError("logarithm base must be finite and exceed 1")
    if np.any(eigenvalues < -tol):
        raise ValueError("spectral entropy requires a positive-semidefinite spectrum")
    values = np.maximum(eigenvalues, 0)
    if not len(values) or values.max() == 0:
        return 0.0
    # Normalize by the maximum first to avoid overflow in the trace.
    scaled_eigenvalues = values / values.max()
    probabilities = scaled_eigenvalues[scaled_eigenvalues > 0] / scaled_eigenvalues.sum()
    # A subnormal value can underflow only at this second normalization.
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(base))


def _or_nan(function):
    return lambda values: float(function(values)) if len(values) else math.nan


def _scaled(function):
    """Homogeneous statistics without avoidable overflow in intermediate sums."""
    def calculate(values):
        if not len(values):
            return math.nan
        scale = float(np.max(np.abs(values)))
        return float(function(values / scale) * scale) if scale else 0.0
    return calculate


def _sum(values):
    if not len(values):
        return 0.0
    scale = float(np.max(np.abs(values)))
    with np.errstate(over="ignore"):
        return float(np.sum(values / scale) * scale) if scale else 0.0


def _rescale_power(coefficient, scale, order):
    """Restore a homogeneous power statistic after stable normalization."""
    if coefficient == 0 or scale == 0:
        return 0.0
    if order == 0 or scale == 1:
        return float(coefficient)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.float64(coefficient) * np.power(np.float64(scale), order)
    if np.isfinite(result) and result != 0:
        return float(result)

    # ``scale**order`` can overflow even when multiplication by a small
    # normalized coefficient brings the final statistic back into range.  Use
    # logarithms only for that exceptional path, leaving ordinary calculations
    # and their rounding behavior unchanged.  The signed zero is meaningful for
    # underflowed odd moments of negative spectra.
    log_magnitude = math.log(abs(float(coefficient)))
    try:
        log_magnitude += order * math.log(float(scale))
    except OverflowError:
        log_magnitude = math.inf if scale > 1 else -math.inf
    maximum_log = math.log(np.finfo(float).max)
    minimum_log = math.log(np.nextafter(0.0, 1.0))
    if log_magnitude > maximum_log:
        return math.copysign(math.inf, coefficient)
    if log_magnitude < minimum_log:
        return math.copysign(0.0, coefficient)
    try:
        magnitude = math.exp(log_magnitude)
    except OverflowError:
        magnitude = math.inf
    return math.copysign(magnitude, coefficient)


def _translated_scale(eigenvalues):
    """Return finite translated values and their range scale when possible."""
    # Variance and centered energy are translation invariant.  Subtracting one
    # observed value before normalization preserves representable differences
    # between adjacent large floats that division by their common magnitude
    # would round away.  Opposite-sign extremes can overflow on subtraction;
    # callers deliberately retain their magnitude-scaled fallback for that
    # case.
    with np.errstate(over="ignore", invalid="ignore"):
        translated = eigenvalues - eigenvalues[0]
    if not np.all(np.isfinite(translated)):
        return None
    return translated, float(np.max(np.abs(translated)))


def spectral_variance(values):
    """Return the population variance of a supplied real spectrum.

    The definition is ``mean((lambda - mean(lambda))**2)``. An empty spectrum
    has undefined variance and returns NaN. Translating before range scaling
    preserves small representable differences atop a large common offset. If
    translation overflows for widely separated opposite-sign values, the
    calculation falls back to magnitude scaling. A logarithmic fallback
    restores the squared scale when forming it directly would overflow; a
    genuinely unrepresentable variance still returns positive infinity.
    """
    eigenvalues = _real_values(values)
    if not len(eigenvalues):
        return math.nan
    translated_scale = _translated_scale(eigenvalues)
    if translated_scale is not None:
        translated, scale = translated_scale
        if scale == 0:
            return 0.0
        scaled_variance = float(np.var(translated / scale))
        return _rescale_power(scaled_variance, scale, 2)

    # Subtracting a reference can overflow for opposite-sign values near the
    # ends of the float range.  The previous magnitude normalization remains a
    # safe fallback and correctly yields infinity when the variance cannot be
    # represented.
    magnitude_scale = float(np.max(np.abs(eigenvalues)))
    if magnitude_scale == 0:
        return 0.0
    scaled_eigenvalues = eigenvalues / magnitude_scale
    scaled_variance = float(np.var(scaled_eigenvalues))
    return _rescale_power(scaled_variance, magnitude_scale, 2)


def _moment_order(order):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order < 0:
        raise ValueError("spectral moment order must be a nonnegative integer")
    return int(order)


def spectral_moment(values, order):
    """Return raw population spectral moment ``mean(lambda**order)``.

    ``order`` must be a nonnegative integer. The zeroth moment of a nonempty
    spectrum is one. Moments of an empty spectrum are undefined and return
    NaN. The calculation is normalized before exponentiation to avoid
    avoidable intermediate overflow; a genuinely out-of-range result may be
    infinite.
    """
    eigenvalues = _real_values(values)
    order = _moment_order(order)
    if not len(eigenvalues):
        return math.nan
    if order == 0:
        return 1.0
    scale = float(np.max(np.abs(eigenvalues)))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_moment = float(np.mean(np.power(eigenvalues / scale, order)))
    return _rescale_power(scaled_moment, scale, order)


def spectral_moments(values, orders=(1, 2, 3, 4)):
    """Return selected raw population moments as a one-dimensional array.

    ``orders`` must be a nonempty ordered iterable of unique nonnegative
    integers. Its order is preserved in the output; unordered sets and mappings
    are rejected rather than silently imposing an order.
    """
    if isinstance(orders, (str, bytes, Mapping, set, frozenset, bool, np.bool_)):
        raise ValueError(
            "orders must be a nonempty ordered sequence of unique nonnegative integers"
        )
    try:
        requested_orders = tuple(_moment_order(order) for order in orders)
    except TypeError as error:
        raise ValueError(
            "orders must be a nonempty ordered sequence of unique nonnegative integers"
        ) from error
    if not requested_orders or len(set(requested_orders)) != len(requested_orders):
        raise ValueError(
            "orders must be a nonempty ordered sequence of unique nonnegative integers"
        )
    eigenvalues = _real_values(values)
    return np.asarray([
        spectral_moment(eigenvalues, order) for order in requested_orders
    ], dtype=float)


def spectral_energy(values):
    """Return the uncentered spectral energy ``sum(abs(eigenvalues))``."""
    eigenvalues = _real_values(values)
    with np.errstate(over="ignore"):
        return float(np.sum(np.abs(eigenvalues)))


def laplacian_energy(values):
    """Return centered Laplacian spectral energy for a supplied full spectrum.

    This is ``sum(abs(lambda - mean(lambda)))`` and is distinct from
    :func:`spectral_energy`, which is ``sum(abs(lambda))``. For a combinatorial
    graph ``L0``, the mean eigenvalue is ``2m/n`` and this equals the classical
    graph Laplacian energy. For other Laplacians or dimensions it is an
    explicitly generalized centered spectral functional. The empty spectrum
    has energy zero by convention. This array-level function cannot determine
    whether a spectrum is complete; :func:`summarize_spectrum` enforces
    completeness for its ``"laplacian_energy"`` built-in. Translating before
    range scaling preserves small representable differences atop a large
    common offset; magnitude scaling is used if that subtraction overflows.
    """
    eigenvalues = _real_values(values)
    if not len(eigenvalues):
        return 0.0
    translated_scale = _translated_scale(eigenvalues)
    if translated_scale is not None:
        translated, scale = translated_scale
        if scale == 0:
            return 0.0
        scaled_eigenvalues = translated / scale
        scaled_energy = float(np.sum(np.abs(
            scaled_eigenvalues - np.mean(scaled_eigenvalues)
        )))
        return _rescale_power(scaled_energy, scale, 1)

    magnitude_scale = float(np.max(np.abs(eigenvalues)))
    if magnitude_scale == 0:
        return 0.0
    scaled_eigenvalues = eigenvalues / magnitude_scale
    scaled_energy = float(np.sum(np.abs(
        scaled_eigenvalues - np.mean(scaled_eigenvalues)
    )))
    return _rescale_power(scaled_energy, magnitude_scale, 1)


def _moment_statistic(order):
    def calculate(values):
        return spectral_moment(values, order)

    calculate._spectral_moment_order = order
    return calculate


_MOMENT_STATISTICS = {order: _moment_statistic(order) for order in range(1, 5)}


STATISTICS = {
    "min": _or_nan(np.min), "max": _or_nan(np.max),
    "mean": _scaled(np.mean), "std": _scaled(np.std),
    "median": _scaled(np.median), "sum": _sum,
    "count": lambda values: float(len(values)), "energy": spectral_energy,
    "spectral_entropy": spectral_entropy,
    "variance": spectral_variance, "spectral_variance": spectral_variance,
    "laplacian_energy": laplacian_energy,
}
for _order, _function in _MOMENT_STATISTICS.items():
    STATISTICS[f"moment_{_order}"] = _function
    STATISTICS[f"spectral_moment_{_order}"] = _function


def _zero_count(values):
    """Marker dispatched with the full spectrum and the resolved tolerance."""
    return float(np.count_nonzero(values == 0))


def _positive_count(values):
    return float(np.count_nonzero(values > 0))


STATISTICS.update(zero_count=_zero_count, positive_count=_positive_count)


def summarize_spectrum(result, *, statistics=("mean", "max", "min", "std", "zero_count"),
                       allow_partial=False, positive_only=True,
                       empty_operator_policy="preserve", tol=1e-10, atol=None,
                       rtol=0.0):
    """One vector from supplied eigenvalues, including ordinary/persistent Lq.

    ``statistics`` is an ordered collection of built-in names or an ordered
    mapping ``name -> callable(values)``. Each callable receives its own copy.
    Statistics and callbacks use positive eigenvalues by default. ``zero_count``
    and ``laplacian_energy`` instead always refer to the complete full spectrum,
    including zeros, and are NaN for opted-in partial spectra. This keeps
    centered Laplacian spectral energy distinct from a centered summary of only
    the returned positive modes. It equals classical graph Laplacian energy for
    a combinatorial graph ``L0``; elsewhere it is a generalized functional.
    Empty positive min/max/mean/std, variance, and moments are normally NaN,
    not a fabricated zero. ``empty_operator_policy="zero"`` is an explicit
    feature-encoding exception for predefined statistics: if and only if the
    result carries the core-confirmed receipt for a complete zero-dimensional
    operator, every requested built-in summary slot is set to zero. This keeps
    a finite fixed-width feature block without inserting a fake zero
    eigenvalue. It does not apply to an all-zero nonempty operator, an empty
    positive-mode selection, a partial spectrum, or a failed computation.
    Custom mappings remain authoritative and their callbacks are always run.

    Numerical zero means abs(lambda) <= max(atol, rtol * max(abs(lambda))).
    ``atol=None`` uses the legacy ``tol`` argument; rtol defaults to zero to
    match the core's absolute tolerance. More negative values raise an error,
    even when positive_only=True. Raw eigenvalues are never modified.
    """
    if not isinstance(result, SpectrumResult):
        raise TypeError("expected SpectrumResult")
    if not isinstance(allow_partial, (bool, np.bool_)) or not isinstance(positive_only, (bool, np.bool_)):
        raise ValueError("allow_partial and positive_only must be booleans")
    _validate_empty_operator_policy(empty_operator_policy)
    if not result.complete and not allow_partial:
        raise ValueError("partial spectrum: use allow_partial=True to summarize only supplied eigenvalues")
    _tolerance(tol)
    _tolerance(rtol)
    absolute_tolerance = tol if atol is None else atol
    _tolerance(absolute_tolerance)
    eigenvalues = _real_values(result.eigenvalues)
    operator_dimension, source_chain_dimension, structural_absence = (
        _operator_metadata(result, len(eigenvalues))
    )
    zero_threshold = max(float(absolute_tolerance), float(rtol) * float(np.max(np.abs(eigenvalues), initial=0.)))
    if not math.isfinite(zero_threshold):
        raise ValueError("resolved eigenvalue tolerance must be finite")
    if np.any(eigenvalues < -zero_threshold):
        raise ValueError("Laplacian summaries require a positive-semidefinite spectrum; negative eigenvalue exceeds tolerance")
    custom_statistics = isinstance(statistics, Mapping)
    if custom_statistics:
        statistic_functions = dict(statistics)
    else:
        names = tuple(statistics)
        if len(set(names)) != len(names) or any(name not in STATISTICS for name in names):
            raise ValueError("statistics must be unique known names or a callable mapping")
        statistic_functions = {name: STATISTICS[name] for name in names}
    if not statistic_functions or any(not isinstance(name, str) or not name or not callable(fn)
                            for name, fn in statistic_functions.items()):
        raise ValueError("provide at least one named callable statistic")
    selected_eigenvalues = eigenvalues[eigenvalues > zero_threshold] if positive_only else eigenvalues
    if (empty_operator_policy == "zero" and not custom_statistics
            and result.complete and len(eigenvalues) == 0
            and not structural_absence):
        raise ValueError(
            "empty_operator_policy='zero' requires a core-confirmed complete "
            "0 x 0 operator with empty basis/matrices, nullity=0, and matching "
            "operator_dimension/source_chain_dimension/structural_absence metadata"
        )
    structural_zero_fill = (
        structural_absence
        and empty_operator_policy == "zero"
        and not custom_statistics
    )
    summary_values, statistic_scopes = [], {}
    for name, function in statistic_functions.items():
        # Give entropy the caller's tolerance; arbitrary callbacks stay simple.
        if structural_zero_fill:
            value = 0.0
            statistic_scopes[name] = (
                "full_operator_spectrum"
                if (function is _zero_count or function is _positive_count
                    or function is laplacian_energy or not positive_only) else
                "positive_eigenvalues_from_full_spectrum"
            )
        elif function is _zero_count:
            value = float(np.count_nonzero(np.abs(eigenvalues) <= zero_threshold)) if result.complete else math.nan
            statistic_scopes[name] = ("full_operator_spectrum" if result.complete
                                      else "unavailable_partial_spectrum")
        elif function is _positive_count:
            value = float(np.count_nonzero(eigenvalues > zero_threshold))
            statistic_scopes[name] = ("full_operator_spectrum" if result.complete
                                      else "supplied_eigenvalues_only")
        elif function is laplacian_energy:
            if result.complete:
                # Apply the summary's numerical-zero convention locally without
                # mutating the authoritative stored eigenvalues.
                full_eigenvalues = eigenvalues.copy()
                full_eigenvalues[np.abs(full_eigenvalues) <= zero_threshold] = 0.0
                value = laplacian_energy(full_eigenvalues)
                statistic_scopes[name] = "full_operator_spectrum"
            else:
                value = math.nan
                statistic_scopes[name] = "unavailable_partial_spectrum"
        else:
            value = (spectral_entropy(selected_eigenvalues, tol=zero_threshold) if function is spectral_entropy
                     else function(selected_eigenvalues.copy()))
            statistic_scopes[name] = (
                "positive_eigenvalues_from_full_spectrum"
                if positive_only and result.complete else
                "supplied_positive_eigenvalues_only"
                if positive_only else
                "full_operator_spectrum"
                if result.complete else
                "supplied_eigenvalues_only"
            )
        if np.ndim(value) != 0 or np.iscomplexobj(value):
            raise ValueError(f"statistic {name!r} did not return a real scalar")
        summary_values.append(float(value))
    builtin_entropy = any(function is spectral_entropy for function in statistic_functions.values())
    builtin_variance = any(function is spectral_variance for function in statistic_functions.values())
    builtin_laplacian_energy = any(function is laplacian_energy for function in statistic_functions.values())
    moment_orders = tuple(
        (name, function._spectral_moment_order)
        for name, function in statistic_functions.items()
        if hasattr(function, "_spectral_moment_order")
    )
    entropy_policy = ({"normalization": "trace_of_selected_nonnegative_eigenvalues",
                       "logarithm_base": math.e, "zero_operator": "zero_entropy",
                       "negative_values": "clip_within_tolerance_else_reject"}
                      if builtin_entropy else None)
    empty_summary_policy = (
        "custom callbacks retain their declared empty-input behavior"
        if empty_operator_policy == "zero" and custom_statistics else
        "zero for predefined statistics on a core-confirmed complete zero-dimensional operator"
        if empty_operator_policy == "zero" else
        "NaN for undefined statistics; zero entropy convention"
    )
    return VectorResult(np.asarray(summary_values), tuple(f"L{result.dimension}:{name}" for name in statistic_functions),
                        {**source_semantics(result, field=result.metadata.get("coefficient_field", "R")),
                         "representation": "spectral_summary", "dimension": result.dimension,
                         "schema": {"source": source_schema(result, field=result.metadata.get("coefficient_field", "R"))},
                         "operator_kind": result.kind, "scale": result.scale,
                         "start": result.start, "end": result.end,
                         "complete_spectrum": result.complete, "positive_only": bool(positive_only),
                         "operator_dimension": operator_dimension,
                         "source_chain_dimension": source_chain_dimension,
                         "structural_absence": structural_absence,
                         "empty_operator_policy": empty_operator_policy,
                         "zero_filled_statistics": (
                             tuple(statistic_functions) if structural_zero_fill else ()
                         ),
                         "partial_scope": "full_operator" if result.complete else "supplied_eigenvalues_only",
                         "supplied_eigenvalues": len(eigenvalues), "selected_eigenvalues": len(selected_eigenvalues),
                         "statistics": tuple(statistic_functions), "statistic_scopes": statistic_scopes,
                         "tolerance": zero_threshold,
                         "absolute_tolerance": float(absolute_tolerance), "relative_tolerance": float(rtol),
                         "zero_count_semantics": ("real_ordinary_betti_number" if result.kind == "ordinary"
                                                  else "real_persistent_image_rank"),
                         "zero_count_available": bool(result.complete),
                         "observed_zero_count": int(np.count_nonzero(np.abs(eigenvalues) <= zero_threshold)),
                         "zero_count_caveat": "numerical real-field nullity at recorded tolerance; not finite-field homology",
                         "scale_units": result.metadata.get("scale_units"),
                         "entropy_definition": ("trace-normalized spectral Shannon entropy (natural log)"
                                                if builtin_entropy else None),
                         "entropy_policy": entropy_policy,
                         "spectral_variance_definition": (
                             "population central second moment over the statistic's recorded scope"
                             if builtin_variance else None),
                         "spectral_moment_definition": (
                             "raw population moment mean(lambda**order) over the statistic's recorded scope"
                             if moment_orders else None),
                         "spectral_moment_orders": moment_orders,
                         "laplacian_energy_definition": (
                             "centered spectral functional sum(abs(lambda - mean(lambda))) over the complete full operator spectrum; equals classical graph Laplacian energy for combinatorial graph L0"
                             if builtin_laplacian_energy else None),
                         "laplacian_energy_available": (
                             bool(result.complete) if builtin_laplacian_energy else None),
                         "nonfinite_statistics": tuple(name for name, value in zip(statistic_functions, summary_values)
                                                       if not np.isfinite(value)),
                         "empty_summary_policy": empty_summary_policy})


def summarize_spectra(results, **options):
    """Summarize snapshots/two-scale results, forwarding explicit policies."""
    return tuple(summarize_spectrum(result, **options) for result in results)


def graph_entropy(result, *, base=math.e, tol=1e-10):
    """Spectral entropy of a supplied complete graph L0 result.

    The caller is responsible for supplying a graph Laplacian. Higher-degree
    and partial spectra are rejected; use spectral_entropy for other PSD data.
    """
    if not isinstance(result, SpectrumResult) or result.dimension != 0 or not result.complete:
        raise ValueError("graph_entropy requires a complete degree-zero Laplacian spectrum")
    return spectral_entropy(result.eigenvalues, base=base, tol=tol)
