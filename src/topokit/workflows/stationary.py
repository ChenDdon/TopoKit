"""One-scale analysis assembled from public core and postprocessing layers."""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np

from .. import core
from ..postprocessing import summarize_spectrum
from ..results import StationaryResult, Topology


DEFAULT_STATISTICS = ("min", "max", "mean", "zero_count", "energy")


def _dimensions(value):
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        values = (int(value),)
    elif isinstance(value, (str, bytes, Mapping, bool, np.bool_)):
        raise ValueError("dimensions must be a nonempty sequence of unique nonnegative integers")
    else:
        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError(
                "dimensions must be a nonempty sequence of unique nonnegative integers"
            ) from error
    if (not values or any(isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral)
                          or q < 0 for q in values) or len(set(values)) != len(values)):
        raise ValueError("dimensions must be a nonempty sequence of unique nonnegative integers")
    return tuple(int(q) for q in values)


def analyze_stationary(topology, *, scale=None, dimensions=(0, 1), field=2,
                       statistics=DEFAULT_STATISTICS, positive_only=False,
                       atol=1e-10, rtol=0.0, return_eigenvectors=False,
                       return_matrices=False, max_dense_entries=4_000_000,
                       tol=1e-10):
    """Compute Hq and ordinary Lq spectra at one explicitly selected scale.

    This workflow does not compute persistence. ``dimensions`` controls the
    Laplacians to return; homology is computed through the largest requested
    degree. Summary values are derived from each returned spectrum with
    :func:`topokit.postprocessing.summarize_spectrum`.
    """
    if not isinstance(topology, Topology):
        raise TypeError("topology must be a TopoKit Topology")
    degrees = _dimensions(dimensions)
    if scale is not None and (isinstance(scale, (bool, np.bool_))
                              or not isinstance(scale, Real)
                              or not math.isfinite(float(scale))):
        raise ValueError("scale must be a finite real number or None")
    query_scale = None if scale is None else float(scale)
    homology = core.homology(
        topology, max_dimension=max(degrees), scale=query_scale, field=field,
    )
    spectra = {
        q: core.laplacian(
            topology, dimension=q, scale=query_scale,
            return_eigenvectors=return_eigenvectors,
            return_matrix=return_matrices,
            max_dense_entries=max_dense_entries, tol=tol,
        )
        for q in degrees
    }
    summaries = {}
    summary_names = None
    for q, spectrum in spectra.items():
        summary = summarize_spectrum(
            spectrum, statistics=statistics, positive_only=positive_only,
            atol=atol, rtol=rtol,
        )
        names = tuple(name.split(":", 1)[1] for name in summary.names)
        summary_names = names if summary_names is None else summary_names
        if names != summary_names:
            raise RuntimeError("stationary spectrum summaries have inconsistent schemas")
        values = dict(zip(names, map(float, summary.values)))
        if "min" in values and math.isfinite(values["min"]) and abs(values["min"]) <= atol:
            values["min"] = 0.0
        summaries[q] = values
    return StationaryResult(
        topology=topology,
        scale=query_scale,
        dimensions=degrees,
        homology=homology,
        spectra=spectra,
        summaries=summaries,
        metadata={
            "summary_statistics": summary_names or (),
            "positive_only": bool(positive_only),
            "absolute_tolerance": float(atol),
            "relative_tolerance": float(rtol),
            "field": homology.field,
            "operator_kind": "ordinary",
        },
    )


__all__ = ["DEFAULT_STATISTICS", "analyze_stationary"]
